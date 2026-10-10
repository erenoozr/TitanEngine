"""Stage 1: Disk pager and LRU buffer pool manager."""
import os
import struct
import threading
from dataclasses import dataclass
from enum import IntEnum

PAGE_SIZE = 4096
NO_PAGE = 0  # page 0 is the meta page and can never be a child/sibling, so it doubles as the null pointer


class PageType(IntEnum):
    META = 0
    LEAF = 1
    INTERNAL = 2


# page_id, page_type, item_count, free_space_offset, next_page_id, prev_page_id, page_lsn
HEADER = struct.Struct('<IBHHIIQ')
HDR = HEADER.size
LSN_OFF = HDR - 8


class CrashedError(Exception):
    """Raised by every I/O call after a simulated crash."""


def read_header(buf):
    return list(HEADER.unpack_from(buf, 0))


def write_header(buf, h):
    HEADER.pack_into(buf, 0, *h)


def get_lsn(buf):
    return struct.unpack_from('<Q', buf, LSN_OFF)[0]


def set_lsn(buf, lsn):
    struct.pack_into('<Q', buf, LSN_OFF, lsn)


class DiskPager:
    """Raw I/O layer that splits a file into fixed 4096-byte pages."""

    def __init__(self, path):
        self.f = open(path, 'r+b' if os.path.exists(path) else 'w+b')
        self.lock = threading.Lock()
        self.closed = False

    def _check(self):
        if self.closed:
            raise CrashedError('pager is closed')

    def read_page(self, pid):
        with self.lock:
            self._check()
            self.f.seek(pid * PAGE_SIZE)
            data = self.f.read(PAGE_SIZE)
        return bytearray(data.ljust(PAGE_SIZE, b'\0'))  # beyond end of file = zero page

    def write_page(self, pid, data):
        assert len(data) == PAGE_SIZE
        with self.lock:
            self._check()
            self.f.seek(pid * PAGE_SIZE)
            self.f.write(data)

    def sync(self):
        with self.lock:
            self._check()
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.f.close()


@dataclass(slots=True)
class Frame:
    data: bytearray
    dirty: bool = False
    pins: int = 0


class BufferPoolManager:
    """Fixed-capacity, thread-safe page cache with LRU eviction.
    Dict insertion order is the LRU order: head = oldest, tail = newest."""

    def __init__(self, pager, capacity=50, wal=None):
        self.pager, self.capacity, self.wal = pager, capacity, wal
        self.frames = {}
        self.lock = threading.RLock()
        self.closed = False

    def _check(self):
        if self.closed:
            raise CrashedError('buffer pool is closed')

    def fetch(self, pid):
        """Pins the page and returns its bytearray; the caller must unpin it when done."""
        with self.lock:
            self._check()
            fr = self.frames.pop(pid, None)
            if fr is None:
                self._make_room()
                fr = Frame(self.pager.read_page(pid))
            self.frames[pid] = fr
            fr.pins += 1
            return fr.data

    def new_page(self, pid):
        with self.lock:
            self._check()
            self._make_room()
            fr = Frame(bytearray(PAGE_SIZE), dirty=True, pins=1)
            self.frames[pid] = fr
            return fr.data

    def unpin(self, pid, dirty=False):
        with self.lock:
            if self.closed:
                return
            fr = self.frames[pid]
            fr.pins -= 1
            fr.dirty |= dirty

    def _make_room(self):
        if len(self.frames) < self.capacity:
            return
        for pid, fr in self.frames.items():
            if fr.pins == 0:
                self._write_back(pid, fr)
                del self.frames[pid]
                return
        raise RuntimeError('buffer pool full: all pages are pinned')

    def _write_back(self, pid, fr):
        if fr.dirty:
            if self.wal is not None:
                self.wal.flush(get_lsn(fr.data))  # WAL rule: log first, then the page
            self.pager.write_page(pid, fr.data)
            fr.dirty = False

    def flush_all(self):
        with self.lock:
            self._check()
            for pid, fr in self.frames.items():
                self._write_back(pid, fr)
            self.pager.sync()

    def crash(self):
        with self.lock:
            self.closed = True
            self.frames.clear()  # dirty pages are lost without reaching disk
