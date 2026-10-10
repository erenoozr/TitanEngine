"""TitanEngine: entry point that wires all layers together."""
import os

from .btree import BPlusTree
from .pager import BufferPoolManager, CrashedError, DiskPager
from .txn import TransactionManager
from .wal import WALManager, recover


class TitanEngine:
    def __init__(self, path, pool_size=50):
        db = path + '.db'
        fresh = not os.path.exists(db) or os.path.getsize(db) == 0
        self.pager = DiskPager(db)
        if fresh:
            BPlusTree.format(self.pager)
        self.wal = WALManager(path + '.wal')
        self.pool = BufferPoolManager(self.pager, pool_size, self.wal)
        self.tree = BPlusTree(self.pool, self.wal)
        self.recovery = recover(self.wal, self.pool, self.tree)
        next_txid = max([self.wal.base_txid] + [r.txid + 1 for r in self.wal.records])
        self.wal.records = None
        self.txm = TransactionManager(self.tree, self.wal, next_txid)
        self.crashed = False
        self.checkpoint()

    def begin(self):
        if self.crashed:
            raise CrashedError('engine has crashed')
        return self.txm.begin()

    def checkpoint(self):
        """Sharp checkpoint: with no active transactions, flush every page and reset the WAL."""
        with self.tree.latch, self.txm.lock:
            assert not self.txm.active, 'cannot checkpoint while transactions are active'
            self.wal.flush()
            self.pool.flush_all()
            self.wal.truncate(self.txm.next_txid)

    def check_integrity(self):
        return self.tree.check()

    def close(self):
        self.checkpoint()
        self.wal.close()
        self.pager.close()

    def crash(self):
        """Power failure: the un-fsynced WAL and dirty buffer pages are lost."""
        self.crashed = True
        self.wal.crash()
        self.pool.crash()
        self.pager.close()
