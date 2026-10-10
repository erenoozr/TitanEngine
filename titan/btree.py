"""Stage 2: B+ Tree index on disk pages."""
import struct
import threading

from .pager import PAGE_SIZE, HDR, PageType, read_header, write_header

META = struct.Struct('<III')   # magic, root_page_id, num_pages (page 0, after the header)
MAGIC = 0x7174A9E0
SLOT = struct.Struct('<H')     # leaf slot array: record offset
LREC = struct.Struct('<qH')    # leaf record: key, value_len (+ value)
CHILD0 = struct.Struct('<I')   # internal node: leftmost child
ENTRY = struct.Struct('<qI')   # internal node: separator key, right child
INTERNAL_MAX = (PAGE_SIZE - HDR - CHILD0.size) // ENTRY.size
MAX_RECORD = 1024              # <= page/3 so both halves of a split always fit


def _slot_off(buf, i):
    return SLOT.unpack_from(buf, HDR + SLOT.size * i)[0]


def leaf_key(buf, i):
    return LREC.unpack_from(buf, _slot_off(buf, i))[0]


def leaf_value(buf, i):
    off = _slot_off(buf, i)
    n = LREC.unpack_from(buf, off)[1]
    return bytes(buf[off + LREC.size: off + LREC.size + n])


def leaf_search(buf, key):
    """In-page binary search (lower bound): (position, found)."""
    n = read_header(buf)[2]
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if leaf_key(buf, mid) < key:
            lo = mid + 1
        else:
            hi = mid
    return lo, lo < n and leaf_key(buf, lo) == key


def leaf_items(buf):
    out = []
    for i in range(read_header(buf)[2]):
        off = _slot_off(buf, i)
        k, n = LREC.unpack_from(buf, off)
        out.append((k, bytes(buf[off + LREC.size: off + LREC.size + n])))
    return out


def leaf_write(buf, items):
    """Slotted page: the slot array grows from the front, records from the back. False if it does not fit."""
    if HDR + sum(SLOT.size + LREC.size + len(v) for _, v in items) > PAGE_SIZE:
        return False
    off = PAGE_SIZE
    for i, (k, v) in enumerate(items):
        off -= LREC.size + len(v)
        LREC.pack_into(buf, off, k, len(v))
        buf[off + LREC.size: off + LREC.size + len(v)] = v
        SLOT.pack_into(buf, HDR + SLOT.size * i, off)
    h = read_header(buf)
    h[2], h[3] = len(items), off
    write_header(buf, h)
    return True


def _entry(buf, i):
    return ENTRY.unpack_from(buf, HDR + CHILD0.size + ENTRY.size * i)


def int_child(buf, key):
    """Binary search in an internal node: page_id of the child covering key."""
    lo, hi = 0, read_header(buf)[2]
    while lo < hi:
        mid = (lo + hi) // 2
        if _entry(buf, mid)[0] <= key:
            lo = mid + 1
        else:
            hi = mid
    return CHILD0.unpack_from(buf, HDR)[0] if lo == 0 else _entry(buf, lo - 1)[1]


def int_read(buf):
    return CHILD0.unpack_from(buf, HDR)[0], [_entry(buf, i) for i in range(read_header(buf)[2])]


def int_write(buf, child0, entries):
    CHILD0.pack_into(buf, HDR, child0)
    for i, e in enumerate(entries):
        ENTRY.pack_into(buf, HDR + CHILD0.size + ENTRY.size * i, *e)
    h = read_header(buf)
    h[2], h[3] = len(entries), HDR + CHILD0.size + ENTRY.size * len(entries)
    write_header(buf, h)


class BPlusTree:
    """Keys: int64, values: <= MAX_RECORD bytes. Every write operation logs the full
    images of the pages it modified to the WAL as one atomic group (physical redo)."""

    def __init__(self, pool, wal=None):
        self.pool, self.wal = pool, wal
        # shortcut: single tree-wide latch (no latch crabbing); move to per-page latches if write concurrency matters
        self.latch = threading.RLock()
        self._pins = {}
        self._dirty = set()
        meta = pool.fetch(0)
        try:
            if META.unpack_from(meta, HDR)[0] != MAGIC:
                raise ValueError('not a valid TitanEngine database file')
        finally:
            pool.unpin(0)

    @staticmethod
    def format(pager):
        meta = bytearray(PAGE_SIZE)
        write_header(meta, [0, PageType.META, 0, 0, 0, 0, 0])
        META.pack_into(meta, HDR, MAGIC, 1, 2)
        root = bytearray(PAGE_SIZE)
        write_header(root, [1, PageType.LEAF, 0, PAGE_SIZE, 0, 0, 0])
        pager.write_page(0, meta)
        pager.write_page(1, root)
        pager.sync()

    # --- per-operation page handling: every touched page stays pinned until the operation ends,
    # so a page holding unlogged changes can never be evicted and written to disk.
    def _page(self, pid):
        buf = self._pins.get(pid)
        if buf is None:
            buf = self._pins[pid] = self.pool.fetch(pid)
        return buf

    def _log(self, txid):
        if self._dirty and self.wal is not None:
            self.wal.log_pages(txid, [(pid, self._pins[pid]) for pid in sorted(self._dirty)])

    def _release(self):
        pins, dirty = self._pins, self._dirty
        self._pins, self._dirty = {}, set()
        for pid in pins:
            self.pool.unpin(pid, pid in dirty)

    def _meta(self):
        return list(META.unpack_from(self._page(0), HDR))

    def _set_meta(self, m):
        META.pack_into(self._page(0), HDR, *m)
        self._dirty.add(0)

    def _alloc(self, ptype):
        m = self._meta()
        pid = m[2]
        m[2] += 1
        self._set_meta(m)
        buf = self._pins[pid] = self.pool.new_page(pid)
        self._dirty.add(pid)
        write_header(buf, [pid, ptype, 0, PAGE_SIZE, 0, 0, 0])
        return pid, buf

    def _path(self, key):
        path = [self._meta()[1]]
        while read_header(self._page(path[-1]))[1] == PageType.INTERNAL:
            path.append(int_child(self._page(path[-1]), key))
        return path

    # --- public API
    def get(self, key):
        with self.latch:
            try:
                buf = self._page(self._path(key)[-1])
                i, found = leaf_search(buf, key)
                return leaf_value(buf, i) if found else None
            finally:
                self._release()

    def put(self, key, value, txid=0):
        if len(value) > MAX_RECORD:
            raise ValueError(f'record too large: {len(value)} > {MAX_RECORD}')
        with self.latch:
            try:
                self._put(key, value)
                self._log(txid)
            finally:
                self._release()

    def delete(self, key, txid=0):
        with self.latch:
            try:
                pid = self._path(key)[-1]
                buf = self._page(pid)
                i, found = leaf_search(buf, key)
                if found:
                    items = leaf_items(buf)
                    del items[i]
                    leaf_write(buf, items)
                    self._dirty.add(pid)
                    self._log(txid)
                return found
            finally:
                self._release()

    def _put(self, key, value):
        path = self._path(key)
        pid = path.pop()
        buf = self._page(pid)
        i, found = leaf_search(buf, key)
        items = leaf_items(buf)
        if found:
            items[i] = (key, value)
        else:
            items.insert(i, (key, value))
        self._dirty.add(pid)
        if leaf_write(buf, items):
            return
        # leaf split: down the middle by bytes
        sizes = [SLOT.size + LREC.size + len(v) for _, v in items]
        half, acc, cut = sum(sizes) / 2, 0, 0
        while acc < half:
            acc += sizes[cut]
            cut += 1
        cut = max(1, min(cut, len(items) - 1))
        rpid, rbuf = self._alloc(PageType.LEAF)
        h, rh = read_header(buf), read_header(rbuf)
        rh[4], rh[5], h[4] = h[4], pid, rpid
        if rh[4]:
            nbuf = self._page(rh[4])
            nh = read_header(nbuf)
            nh[5] = rpid
            write_header(nbuf, nh)
            self._dirty.add(rh[4])
        write_header(buf, h)
        write_header(rbuf, rh)
        assert leaf_write(buf, items[:cut]) and leaf_write(rbuf, items[cut:])
        self._insert_parent(path, pid, items[cut][0], rpid)

    def _insert_parent(self, path, left, key, right):
        if not path:  # root split: new root
            root, rbuf = self._alloc(PageType.INTERNAL)
            int_write(rbuf, left, [(key, right)])
            m = self._meta()
            m[1] = root
            self._set_meta(m)
            return
        pid = path.pop()
        buf = self._page(pid)
        child0, entries = int_read(buf)
        entries.insert(sum(1 for k, _ in entries if k <= key), (key, right))
        self._dirty.add(pid)
        if len(entries) <= INTERNAL_MAX:
            int_write(buf, child0, entries)
            return
        mid = len(entries) // 2
        up_key, up_child = entries[mid]
        npid, nbuf = self._alloc(PageType.INTERNAL)
        int_write(buf, child0, entries[:mid])
        int_write(nbuf, up_child, entries[mid + 1:])
        self._insert_parent(path, pid, up_key, npid)

    def range_scan(self, min_key, max_key, reverse=False):
        """Range [min_key, max_key] via the leaf chain (next/prev)."""
        with self.latch:
            try:
                pid = self._path(max_key if reverse else min_key)[-1]
            finally:
                self._release()
            out = []
            while pid:
                buf = self.pool.fetch(pid)
                try:
                    items, h = leaf_items(buf), read_header(buf)
                finally:
                    self.pool.unpin(pid)
                if reverse:
                    out += [kv for kv in reversed(items) if min_key <= kv[0] <= max_key]
                    if items and items[0][0] < min_key:
                        break
                    pid = h[5]
                else:
                    out += [kv for kv in items if min_key <= kv[0] <= max_key]
                    if items and items[-1][0] > max_key:
                        break
                    pid = h[4]
            return out

    def check(self):
        """Structural integrity check. Returns (key_count, height); raises AssertionError on violation."""
        with self.latch:
            meta = self.pool.fetch(0)
            root = META.unpack_from(meta, HDR)[1]
            self.pool.unpin(0)
            leaves, depths = [], set()

            def walk(pid, lo, hi, depth):
                buf = self.pool.fetch(pid)
                try:
                    h = read_header(buf)
                    assert h[0] == pid, f'page {pid} header says {h[0]}'
                    if h[1] == PageType.LEAF:
                        keys = [k for k, _ in leaf_items(buf)]
                    else:
                        assert h[1] == PageType.INTERNAL, f'page {pid} has corrupt type: {h[1]}'
                        child0, entries = int_read(buf)
                        keys = [k for k, _ in entries]
                finally:
                    self.pool.unpin(pid)
                assert all(a < b for a, b in zip(keys, keys[1:])), f'page {pid} is not sorted'
                assert all((lo is None or k >= lo) and (hi is None or k < hi) for k in keys), \
                    f'page {pid} is outside its parent bounds'
                if h[1] == PageType.LEAF:
                    leaves.append((pid, h[4], h[5], keys))
                    depths.add(depth)
                    return
                assert keys, f'internal node {pid} is empty'
                bounds = [lo] + keys + [hi]
                for i, c in enumerate([child0] + [c for _, c in entries]):
                    walk(c, bounds[i], bounds[i + 1], depth + 1)

            walk(root, None, None, 1)
            assert len(depths) == 1, f'leaves at different depths: {depths}'
            for i, (pid, nxt, prv, _) in enumerate(leaves):
                assert prv == (leaves[i - 1][0] if i else 0), f'leaf {pid} has a broken prev pointer'
                assert nxt == (leaves[i + 1][0] if i + 1 < len(leaves) else 0), f'leaf {pid} has a broken next pointer'
            return sum(len(ks) for *_, ks in leaves), depths.pop()
