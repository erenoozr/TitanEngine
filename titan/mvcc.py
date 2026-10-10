"""Stage 4: MVCC version chains and snapshot isolation visibility.
Each key's B+ Tree value is a version chain, newest first."""
import struct
from dataclasses import dataclass

VERSION = struct.Struct('<IIH')  # xmin, xmax, value_len


@dataclass
class Version:
    xmin: int      # TxID that created the version
    xmax: int      # TxID that superseded (updated) the version, 0 = live
    value: bytes


def encode_chain(chain):
    return bytes([len(chain)]) + b''.join(
        VERSION.pack(v.xmin, v.xmax, len(v.value)) + v.value for v in chain)


def decode_chain(raw):
    out, pos = [], 1
    for _ in range(raw[0]):
        xmin, xmax, n = VERSION.unpack_from(raw, pos)
        pos += VERSION.size
        out.append(Version(xmin, xmax, bytes(raw[pos:pos + n])))
        pos += n
    return out


@dataclass(frozen=True)
class ReadView:
    """Snapshot taken when a transaction begins: it sees versions of transactions that
    started earlier and were not active at that moment (i.e. committed).
    Aborted transactions physically roll back their versions before leaving the active set."""
    txid: int
    active: frozenset

    @property
    def low(self):
        return min(self.active, default=self.txid)

    def sees(self, xid):
        return xid == self.txid or (xid < self.txid and xid not in self.active)

    def visible(self, chain):
        for v in chain:
            if self.sees(v.xmin) and not (v.xmax and self.sees(v.xmax)):
                return v.value
        return None


def prune(chain, low):
    """Drop old versions no active snapshot can see (GC)."""
    return [v for v in chain if not (v.xmax and v.xmax < low)]
