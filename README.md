# TitanEngine

A disk-based storage engine written in pure Python (3.10+, standard library only): a B+ Tree index, write-ahead logging with ARIES-style crash recovery, and MVCC with snapshot isolation.

All data is stored as 4096-byte binary pages in a `.db` file, and every change is logged first to a `.wal` file. Nothing is held only in memory.

## Architecture

| Module | Layer |
|---|---|
| `titan/pager.py` | `DiskPager` (fixed 4 KB pages) and a thread-safe `BufferPoolManager` with LRU eviction, dirty-page tracking and pinning |
| `titan/btree.py` | On-disk B+ Tree: slotted leaf pages, binary search within each page, leaf/internal node splits, a doubly linked leaf chain for forward/reverse `range_scan` |
| `titan/wal.py` | `WALManager` (checksummed binary records, group commit, torn-tail detection) and `recover()`: analysis, redo, undo |
| `titan/mvcc.py` | Per-key version chains (`xmin` / `xmax`), `ReadView` visibility rules, version garbage collection |
| `titan/txn.py` | `TransactionManager`: begin / commit / abort, write-write conflict detection (first-updater-wins) |
| `titan/engine.py` | `TitanEngine`: wires the layers together, checkpoints, simulated crashes |

**Page header:** `page_id`, `page_type`, `item_count`, `free_space_offset`, `next_page_id`, `prev_page_id`, `page_lsn`

**Log record:** `LSN`, `TxID`, `Type` (BEGIN, COMMIT, ABORT, INSERT, UPDATE, PAGE), `Key`, `Old_Value`, `New_Value`

### Recovery

- **Logging:** Each B+ Tree operation writes the full image of every page it modifies to the WAL as one atomic group. Pages stay pinned until they are logged, so a page with unlogged changes is never written to disk.
- **Redo:** Replays page images whose LSN is newer than the page on disk. This rebuilds the exact pre-crash tree, including splits made by uncommitted transactions.
- **Undo:** Logical. For transactions with no COMMIT or ABORT record, the previous version chain of each key they wrote (`Old_Value`) is written back in reverse order. Each step writes an absolute value, so undo can safely run again if recovery itself is interrupted.

### Concurrency

- **Isolation:** Snapshot isolation. A transaction sees only versions committed before it began, so it never sees dirty reads and its reads are repeatable.
- **Conflicts:** If a transaction tries to update a key whose newest version it cannot see, it is aborted with `WriteConflict`. This prevents lost updates.

## Usage

```python
from titan import TitanEngine, WriteConflict

db = TitanEngine('mydb')          # creates mydb.db / mydb.wal, recovers if needed
tx = db.begin()
tx.put(42, b'hello')
print(tx.get(42))                 # b'hello'
print(tx.scan(0, 100))            # [(42, b'hello')]
tx.commit()
db.close()                        # checkpoint + truncate WAL
```

Keys are 64-bit signed integers. Values are bytes, up to 256 bytes.

## Tests

```
python test_titan.py
```

- LRU eviction order and write-back of dirty pages
- B+ Tree with 6000 keys (height 3): lookups, forward and reverse range scans, updates, deletes, reopening from disk
- MVCC: no dirty reads, snapshot reads, write-write conflicts, rollback
- Recovery: an uncommitted transaction whose dirty pages were already flushed is rolled back after a crash
- **Chaos test:** 10 threads each run 500 bank-transfer transactions across 5000 accounts. A power failure is simulated at a random point (unflushed WAL and buffer pool contents are lost). After recovery the test checks:
  - B+ Tree structural integrity
  - conservation of the total balance (atomicity)
  - every acknowledged commit survived (durability)
  - no aborted write survived

## Limitations

- A single tree-wide latch serializes B+ Tree access. Transactions run concurrently, but no latch crabbing is implemented.
- Keys are not physically deleted, and there is no page merging.
- Checkpoints are sharp: they require no active transactions.
