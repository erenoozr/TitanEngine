"""Stage 5: ACID transaction manager (snapshot isolation + first-updater-wins)."""
import threading
from enum import Enum

from .btree import MAX_RECORD
from .mvcc import ReadView, Version, decode_chain, encode_chain, prune
from .wal import LogType

MAX_VALUE = 256


class TxState(Enum):
    ACTIVE = 'active'
    COMMITTED = 'committed'
    ABORTED = 'aborted'


class WriteConflict(Exception):
    """Write-write conflict; the transaction has already been aborted."""


class TransactionError(Exception):
    pass


class Transaction:
    def __init__(self, mgr, txid, view):
        self.mgr, self.txid, self.view = mgr, txid, view
        self.state = TxState.ACTIVE
        self.undo = []  # (key, previous raw chain | None)

    def _check(self):
        if self.state is not TxState.ACTIVE:
            raise TransactionError(f'transaction {self.txid} is {self.state.value}')

    def get(self, key):
        self._check()
        raw = self.mgr.tree.get(key)
        return self.view.visible(decode_chain(raw)) if raw else None

    def scan(self, min_key, max_key):
        self._check()
        out = []
        for k, raw in self.mgr.tree.range_scan(min_key, max_key):
            v = self.view.visible(decode_chain(raw))
            if v is not None:
                out.append((k, v))
        return out

    def put(self, key, value):
        self._check()
        if len(value) > MAX_VALUE:
            raise ValueError(f'value too large: {len(value)} > {MAX_VALUE}')
        tree, wal = self.mgr.tree, self.mgr.wal
        with tree.latch:
            raw = tree.get(key)
            chain = decode_chain(raw) if raw else []
            # first-updater-wins: conflict if we cannot see the newest version (still active or committed after us)
            if not (chain and chain[0].xmin != self.txid and not self.view.sees(chain[0].xmin)):
                if chain and chain[0].xmin == self.txid:
                    chain.pop(0)  # our own uncommitted version: nobody else sees it, overwrite
                elif chain:
                    chain[0].xmax = self.txid
                new = encode_chain(prune([Version(self.txid, 0, value)] + chain, self.mgr.low_water()))
                if len(new) > MAX_RECORD:
                    raise ValueError('version chain too long')
                wal.append(self.txid, LogType.UPDATE if raw else LogType.INSERT, key, raw or b'', new)
                tree.put(key, new, self.txid)
                self.undo.append((key, raw))
                return
        self.abort()
        with self.mgr.lock:
            self.mgr.stats['conflicts'] += 1
        raise WriteConflict(f'transaction {self.txid}: key {key} was written by another transaction')

    def commit(self):
        self._check()
        lsn = self.mgr.wal.append(self.txid, LogType.COMMIT)
        if self.undo:
            self.mgr.wal.flush(lsn)  # durability: do not return before the commit record is on disk
        self.mgr._finish(self, TxState.COMMITTED)

    def abort(self):
        self._check()
        tree = self.mgr.tree
        with tree.latch:
            for key, old in reversed(self.undo):
                if old is None:
                    tree.delete(key, self.txid)
                else:
                    tree.put(key, old, self.txid)
        # the ABORT record is logged before the lock (active state) is released
        self.mgr.wal.append(self.txid, LogType.ABORT)
        self.mgr._finish(self, TxState.ABORTED)


class TransactionManager:
    def __init__(self, tree, wal, next_txid):
        self.tree, self.wal = tree, wal
        self.lock = threading.Lock()
        self.next_txid = next_txid
        self.active = {}
        self.stats = {'committed': 0, 'aborted': 0, 'conflicts': 0}

    def begin(self):
        with self.lock:
            txid = self.next_txid
            self.next_txid += 1
            t = Transaction(self, txid, ReadView(txid, frozenset(self.active)))
            self.active[txid] = t
        self.wal.append(txid, LogType.BEGIN)
        return t

    def low_water(self):
        with self.lock:
            return min((t.view.low for t in self.active.values()), default=self.next_txid)

    def _finish(self, t, state):
        with self.lock:
            t.state = state
            self.active.pop(t.txid, None)
            self.stats[state.value] += 1
