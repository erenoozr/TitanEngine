"""Stage 3: Write-ahead logging and ARIES-style crash recovery."""
import os
import struct
import threading
from dataclasses import dataclass
from enum import IntEnum

from .pager import CrashedError, get_lsn, set_lsn


class LogType(IntEnum):
    BEGIN = 1
    COMMIT = 2
    ABORT = 3
    INSERT = 4
    UPDATE = 5
    PAGE = 6   # physical redo: key = page_id, new_value = full page image


@dataclass
class LogRecord:
    lsn: int
    txid: int
    type: LogType
    key: int = 0
    old_value: bytes = b''
    new_value: bytes = b''


FILE_HDR = struct.Struct('<QI')          # base_lsn, next_txid (written at checkpoint)
REC = struct.Struct('<IIQIBqII')         # checksum, total_len, lsn, txid, type, key, old_len, new_len
_BODY = struct.Struct('<IQIBqII')
MASK = 0xFFFFFFFF


def _encode(lsn, txid, typ, key, old, new):
    rest = _BODY.pack(REC.size + len(old) + len(new), lsn, txid, typ, key, len(old), len(new)) + old + new
    return struct.pack('<I', sum(rest) & MASK) + rest


class WALManager:
    def __init__(self, path):
        fresh = not os.path.exists(path) or os.path.getsize(path) < FILE_HDR.size
        self.f = open(path, 'w+b' if fresh else 'r+b')
        if fresh:
            self.f.write(FILE_HDR.pack(1, 1))
            self.f.flush()
            os.fsync(self.f.fileno())
        self.lock = threading.Lock()
        self.buf = []
        self.closed = False
        self.records = self._scan()

    def _scan(self):
        self.f.seek(0)
        data = self.f.read()
        base_lsn, self.base_txid = FILE_HDR.unpack_from(data, 0)
        pos, recs = FILE_HDR.size, []
        while pos + REC.size <= len(data):
            cs, total, lsn, txid, typ, key, ol, nl = REC.unpack_from(data, pos)
            end = pos + total
            if total != REC.size + ol + nl or end > len(data) or sum(data[pos + 4:end]) & MASK != cs:
                break  # torn tail: everything after this is invalid
            o = pos + REC.size
            recs.append(LogRecord(lsn, txid, LogType(typ), key, data[o:o + ol], data[o + ol:end]))
            pos = end
        self.f.seek(pos)
        self.f.truncate()
        self.next_lsn = recs[-1].lsn + 1 if recs else base_lsn
        self.flushed_lsn = self.next_lsn - 1
        return recs

    def _check(self):
        if self.closed:
            raise CrashedError('WAL is closed')

    def append(self, txid, typ, key=0, old=b'', new=b''):
        with self.lock:
            self._check()
            lsn = self.next_lsn
            self.next_lsn += 1
            self.buf.append(_encode(lsn, txid, typ, key, old, new))
            return lsn

    def log_pages(self, txid, pages):
        """All page images of one B+ Tree operation under a single lock, so a flush can never split the group."""
        with self.lock:
            self._check()
            for pid, page in pages:
                lsn = self.next_lsn
                self.next_lsn += 1
                set_lsn(page, lsn)
                self.buf.append(_encode(lsn, txid, LogType.PAGE, pid, b'', bytes(page)))
            return self.next_lsn - 1

    def flush(self, upto=None):
        """Group commit: writes and fsyncs all pending records."""
        with self.lock:
            self._check()
            if upto is not None and upto <= self.flushed_lsn:
                return
            if self.buf:
                self.f.seek(0, 2)
                self.f.write(b''.join(self.buf))
                self.buf.clear()
                self.f.flush()
                os.fsync(self.f.fileno())
            self.flushed_lsn = self.next_lsn - 1

    def truncate(self, next_txid):
        """Checkpoint: resets the log once every page is on disk; LSNs stay monotonic."""
        with self.lock:
            self._check()
            assert not self.buf
            self.f.seek(0)
            self.f.truncate()
            self.f.write(FILE_HDR.pack(self.next_lsn, next_txid))
            self.f.flush()
            os.fsync(self.f.fileno())

    def crash(self):
        with self.lock:
            self.buf.clear()  # the un-fsynced log is lost
            if not self.closed:
                self.closed = True
                self.f.close()

    def close(self):
        self.crash()


def recover(wal, pool, tree):
    """ARIES: Analysis -> Redo (repeat history, physical) -> Undo (losers, logical)."""
    recs = wal.records
    # 1) Analysis: every transaction without COMMIT/ABORT is a loser
    open_tx = {}
    for r in recs:
        if r.type == LogType.BEGIN:
            open_tx[r.txid] = True
        elif r.type in (LogType.COMMIT, LogType.ABORT):
            open_tx.pop(r.txid, None)
    losers = set(open_tx)
    # 2) Redo: apply the image if the page LSN is older than the record
    redone = 0
    for r in recs:
        if r.type == LogType.PAGE:
            buf = pool.fetch(r.key)
            stale = get_lsn(buf) < r.lsn
            if stale:
                buf[:] = r.new_value
                redone += 1
            pool.unpin(r.key, stale)
    # 3) Undo: write old values back in reverse order. Absolute values make it idempotent,
    # so rerunning after a crash during recovery is safe.
    undone = 0
    for r in reversed(recs):
        if r.txid in losers and r.type in (LogType.INSERT, LogType.UPDATE):
            if r.old_value:
                tree.put(r.key, r.old_value, r.txid)
            else:
                tree.delete(r.key, r.txid)
            undone += 1
    for t in sorted(losers):
        wal.append(t, LogType.ABORT)
    return {'records': len(recs), 'redone_pages': redone, 'losers': len(losers), 'undone_ops': undone}
