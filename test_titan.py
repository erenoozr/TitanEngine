"""Stage 6: Unit tests + chaos/stress test (power failure simulation).
Run: python test_titan.py
"""
import os
import struct
import threading
import time

from titan import (BPlusTree, BufferPoolManager, DiskPager, TitanEngine, TxState, WriteConflict)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ACC = struct.Struct('<qI')  # balance, last writer txid


def fresh(name):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, name)
    for ext in ('.db', '.wal'):
        if os.path.exists(p + ext):
            os.remove(p + ext)
    return p


def rnd(n):
    return int.from_bytes(os.urandom(4), 'little') % n


def test_pager_lru():
    p = fresh('lru')
    pager = DiskPager(p + '.db')
    pool = BufferPoolManager(pager, capacity=3)
    for pid in (1, 2, 3):
        pool.fetch(pid)[100] = pid
        pool.unpin(pid, dirty=True)
    pool.fetch(1)
    pool.unpin(1)                       # 1 is now the most recent
    pool.fetch(4)
    pool.unpin(4)                       # LRU = 2 must be evicted
    assert list(pool.frames) == [3, 1, 4], list(pool.frames)
    assert pager.read_page(2)[100] == 2  # dirty page was written back on eviction
    assert pager.read_page(3)[100] == 0  # still only in memory
    pool.flush_all()
    assert pager.read_page(3)[100] == 3
    for pid in (5, 6, 7):
        pool.fetch(pid)
    try:
        pool.fetch(8)
        raise AssertionError('eviction must fail while every page is pinned')
    except RuntimeError:
        pass
    pager.close()


def test_btree():
    p = fresh('btree')
    pager = DiskPager(p + '.db')
    BPlusTree.format(pager)
    tree = BPlusTree(BufferPoolManager(pager, 50))
    N = 6000
    val = lambda k: struct.pack('<q', k) * 25  # 200 bytes: forces internal node splits
    for i in range(N):
        k = (i * 7919) % N  # shuffled order
        tree.put(k, val(k))
    assert all(tree.get(k) == val(k) for k in range(N))
    assert tree.get(N + 5) is None
    count, height = tree.check()
    assert count == N and height >= 3, (count, height)
    assert [k for k, _ in tree.range_scan(1000, 1999)] == list(range(1000, 2000))
    assert [k for k, _ in tree.range_scan(1000, 1999, reverse=True)] == list(range(1999, 999, -1))
    for k in range(0, N, 3):
        tree.put(k, b'x')
    for k in range(1, N, 3):
        tree.delete(k)
    assert tree.check()[0] == N - len(range(1, N, 3))
    tree.pool.flush_all()
    pager.close()
    pager = DiskPager(p + '.db')  # reopen from disk
    tree = BPlusTree(BufferPoolManager(pager, 50))
    assert tree.get(3) == b'x' and tree.get(4) is None and tree.get(5) == val(5)
    tree.check()
    pager.close()
    print(f'      B+ Tree: {N} keys, height {height}')


def test_mvcc():
    eng = TitanEngine(fresh('mvcc'))
    t0 = eng.begin()
    t0.put(1, b'a')
    t0.commit()
    t1, t2 = eng.begin(), eng.begin()
    t1.put(1, b'b')
    assert t1.get(1) == b'b'            # sees its own write
    assert t2.get(1) == b'a'            # no dirty read
    t1.commit()
    assert t2.get(1) == b'a'            # snapshot: does not see later commits
    t3 = eng.begin()
    assert t3.get(1) == b'b'
    try:
        t2.put(1, b'c')                 # t1 committed after t2 began -> lost update prevented
        raise AssertionError('conflict expected')
    except WriteConflict:
        assert t2.state is TxState.ABORTED
    t4, t5 = eng.begin(), eng.begin()
    t4.put(2, b'x')
    try:
        t5.put(2, b'y')                 # second writer aborts while the first is active
        raise AssertionError('conflict expected')
    except WriteConflict:
        pass
    t4.abort()                          # rollback: key 2 must not exist
    t6 = eng.begin()
    assert t6.get(2) is None and t6.get(1) == b'b'
    t6.put(2, b'z')
    t6.commit()
    t3.commit()
    eng.close()


def test_recovery_basic():
    p = fresh('rec')
    eng = TitanEngine(p, pool_size=10)
    t1 = eng.begin()
    for k in range(1500):
        t1.put(k, b'committed-%d' % k)
    t1.commit()
    t2 = eng.begin()
    for k in range(1500):
        t2.put(k, b'LOSER' * 20)        # small pool: dirty pages reach disk (steal)
    t3 = eng.begin()
    t3.put(99999, b'late')
    t3.commit()                          # also flushes t2's log records
    eng.crash()
    eng = TitanEngine(p, pool_size=10)
    r = eng.recovery
    assert r['losers'] == 1 and r['undone_ops'] == 1500, r
    t = eng.begin()
    assert all(t.get(k) == b'committed-%d' % k for k in range(1500))
    assert t.get(99999) == b'late'
    t.commit()
    assert eng.check_integrity()[0] == 1501
    eng.close()


def test_chaos():
    N_ACC, INIT, THREADS, TXNS = 5000, 1000, 10, 500
    p = fresh('chaos')
    eng = TitanEngine(p)
    t = eng.begin()
    for k in range(N_ACC):
        t.put(k, ACC.pack(INIT, t.txid))
    t.commit()
    setup = t.txid
    eng.checkpoint()

    total_txns = THREADS * TXNS
    crash_at = total_txns // 5 + rnd(total_txns * 3 // 5)  # random point between 20% and 80%
    lock = threading.Lock()
    started = [0]
    acked, committing, aborted, errors = [], set(), set(), []

    def pick():
        return rnd(20) if rnd(2) else rnd(N_ACC)  # half are hot keys: produces conflicts

    def worker():
        for _ in range(TXNS):
            with lock:
                started[0] += 1
                crash_now = started[0] == crash_at
            try:
                t = eng.begin()
                a, b = pick(), pick()
                while b == a:
                    b = pick()
                ba, bb = ACC.unpack(t.get(a))[0], ACC.unpack(t.get(b))[0]
                amt = 1 + rnd(100)
                t.put(a, ACC.pack(ba - amt, t.txid))
                t.put(b, ACC.pack(bb + amt, t.txid))
                if crash_now:
                    # as if another transaction's group commit flushed this half-done one: forces undo
                    eng.wal.flush()
                    eng.crash()
                    return
                if rnd(20) == 0:
                    aborted.add(t.txid)
                    t.abort()
                    continue
                committing.add(t.txid)
                t.commit()
                acked.append((t.txid, a, b))
            except WriteConflict:
                aborted.add(t.txid)
                if crash_now:
                    eng.crash()
                    return
            except Exception as e:
                if eng.crashed:
                    return
                errors.append(e)
                return

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    run_s = time.time() - t0
    assert not errors, errors
    assert eng.crashed
    stats = dict(eng.txm.stats)

    t0 = time.time()
    eng = TitanEngine(p)  # WAL recovery
    rec_s, rec = time.time() - t0, eng.recovery
    assert rec['redone_pages'] > 0
    keys, height = eng.check_integrity()
    assert keys == N_ACC, keys
    t = eng.begin()
    rows = t.scan(0, N_ACC - 1)
    t.commit()
    assert len(rows) == N_ACC
    bal = {k: ACC.unpack(v) for k, v in rows}
    assert sum(b for b, _ in bal.values()) == N_ACC * INIT, 'money created/destroyed: atomicity broken'
    ok_writers = {x for x, _, _ in acked} | committing | {setup}
    for k, (_, w) in bal.items():
        assert w in ok_writers and w not in aborted, f'account {k}: invalid writer {w}'
    for txid, a, b in acked:
        assert bal[a][1] >= txid and bal[b][1] >= txid, f'commit {txid} lost: durability broken'
    eng.close()

    eng = TitanEngine(p)  # second open after a clean shutdown
    assert eng.recovery['records'] == 0
    t = eng.begin()
    assert sum(ACC.unpack(v)[0] for _, v in t.scan(0, N_ACC - 1)) == N_ACC * INIT
    t.commit()
    eng.close()
    print(f'      crash point: {crash_at}/{total_txns} transactions, run {run_s:.1f}s; '
          f'committed {stats["committed"]}, aborted {stats["aborted"]} (conflicts {stats["conflicts"]})')
    print(f'      recovery {rec_s:.1f}s: {rec}; tree height {height}')


if __name__ == '__main__':
    for fn in (test_pager_lru, test_btree, test_mvcc, test_recovery_basic, test_chaos):
        t0 = time.time()
        fn()
        print(f'[OK] {fn.__name__} ({time.time() - t0:.1f}s)')
    print('All tests passed.')
