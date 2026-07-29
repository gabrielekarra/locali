"""Expert store for MiniMax-M2.5: positional reads through the pointer index.

Differs from expert_store.py in one way that matters: there is no experts.bin.
build_index.py addresses experts as (file, offset, nbytes) inside the ORIGINAL
safetensors shards, so this store reads from 54 files -- 27 of the 4-bit
snapshot for hot experts, 27 of the 2-bit pack for cold ones. Mixed precision is
therefore a property of the index, not of any pack this code has to build.

The rules from CLAUDE.md that this file exists to honour:
  - the ceiling is a HARD invariant: evict BEFORE inserting, never after
  - os.pread into buffers we own, never mmap -- WE manage residency
  - a failed or short read crashes loudly; there is no fallback path
"""

import os
import fcntl
import time
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

F_NOCACHE = 48
PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")
NP = {"U32": np.uint32, "I32": np.int32, "F16": np.float16,
      "BF16": np.uint16, "F32": np.float32, "U8": np.uint8}


class M25Store:
    def __init__(self, index_path, ceiling_gb=8.0, nocache=True, trace=False,
                 threads=8):
        idx = json.loads(Path(index_path).read_text())
        self.meta = idx["experts"]
        self.layers, self.E = idx["layers"], idx["num_experts"]
        self.top_k = idx["top_k"]
        self.ceiling = int(ceiling_gb * 1e9)
        self.cache = OrderedDict()          # (layer, expert) -> dict of mx arrays
        self.resident = 0
        self.fds, self._nocache = {}, nocache
        self.hits = self.misses = self.evictions = self.bytes_read = 0
        self.peak = 0
        # Routing trace, counted on every access whether it hits or misses:
        # the hot/cold split needs how often the ROUTER picks an expert, not how
        # often the cache happened to miss it.
        self.trace = {} if trace else None
        self.t_pread = self.t_convert = self.t_eval = 0.0
        # 8 is where measured throughput goes flat on this machine; see bw.py.
        self.pool = ThreadPoolExecutor(max_workers=threads)

    def _fd(self, root, shard):
        key = (root, shard)
        if key not in self.fds:
            fd = os.open(os.path.join(root, shard), os.O_RDONLY)
            if self._nocache:
                # Keep the unified buffer cache out of it: residency is ours to
                # account for, and a page-cache hit would make the metrics lie.
                fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[key] = fd
        return self.fds[key]

    def _pread(self, rec):
        """Bytes only. Runs on the pool: os.pread drops the GIL, so this is the
        half that parallelises. The mx conversion deliberately does not -- it
        stays on the calling thread, where mlx expects to be driven from."""
        root, shard, off, nbytes, _, _ = rec
        raw = os.pread(self._fd(root, shard), nbytes, off)
        if len(raw) != nbytes:
            raise IOError(f"short read {len(raw)}/{nbytes} at {shard}+{off}")
        return raw

    def _to_mx(self, rec, raw):
        *_, shape, dt = rec
        arr = mx.array(np.frombuffer(raw, dtype=NP[dt]).reshape(shape))
        return arr.view(mx.bfloat16) if dt == "BF16" else arr

    def tier(self, layer, expert):
        """'hot' (served from the 4-bit snapshot) or 'cold' (2-bit pack)."""
        return self.meta[f"L{layer}.E{expert}"]["tier"]

    def _entry_bytes(self, m):
        return sum(m[p][k][3] for p in PROJS for k in ARRS)

    def get_many(self, layer, experts):
        """Fetch a whole layer call's experts in ONE round of parallel preads.

        Serial reads measured 1.54 GB/s where the disk gives 4.0-4.66: the
        bottleneck was never the policy, it was one 2.65 MB pread in flight at a
        time. The router hands the whole routed set over at once, so there is no
        reason to discover the misses one at a time.

        Returns {expert: arrays}. Every expert asked for is resident on return,
        which is why the working set has to fit under the ceiling as a whole.
        """
        want = list(dict.fromkeys(experts))
        if self.trace is not None:
            t = self.trace.setdefault(layer, {})
            for e in want:
                t[e] = t.get(e, 0) + 1

        out, miss = {}, []
        for e in want:
            key = (layer, e)
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)   # protects it from this call's evictions
                out[e] = self.cache[key]
            else:
                self.misses += 1
                miss.append(e)
        if not miss:
            return out

        metas = {e: self.meta[f"L{layer}.E{e}"] for e in miss}
        needs = {e: self._entry_bytes(m) for e, m in metas.items()}
        total = sum(needs.values())
        held = sum(self._entry_bytes(self.meta[f"L{layer}.E{e}"]) for e in out)
        if total + held > self.ceiling:
            raise ValueError(
                f"working set for layer {layer} is {(total+held)/1e6:.0f} MB "
                f"({len(want)} experts) against a {self.ceiling/1e6:.0f} MB "
                f"ceiling; raise --ceiling-gb")

        # Evict FIRST, for the whole batch. Inserting and trimming afterwards
        # would breach the ceiling for the width of that window, which is the
        # whole invariant.
        while self.resident + total > self.ceiling:
            _, ev = self.cache.popitem(last=False)
            self.resident -= sum(v.nbytes for d in ev.values() for v in d.values())
            self.evictions += 1

        recs = [metas[e][p][k] for e in miss for p in PROJS for k in ARRS]
        t0 = time.perf_counter()
        raws = list(self.pool.map(self._pread, recs))
        t1 = time.perf_counter()
        vals, i = {}, 0
        for e in miss:
            vals[e] = {p: {} for p in PROJS}
            for p in PROJS:
                for k in ARRS:
                    vals[e][p][k] = self._to_mx(recs[i], raws[i])
                    i += 1
        t2 = time.perf_counter()
        mx.eval([v for d in vals.values() for pd in d.values() for v in pd.values()])
        self.t_pread += t1 - t0            # wall time of the batch, not thread-seconds
        self.t_convert += t2 - t1
        self.t_eval += time.perf_counter() - t2

        for e in miss:
            self.cache[(layer, e)] = vals[e]
            self.resident += needs[e]
            self.bytes_read += needs[e]
            out[e] = vals[e]
        self.peak = max(self.peak, self.resident)
        assert self.resident <= self.ceiling, "ceiling breached"
        return out

    def get(self, layer, expert):
        return self.get_many(layer, [expert])[expert]

    def stats(self):
        t = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / t if t else 0.0,
                "evictions": self.evictions, "bytes_read": self.bytes_read,
                "resident": self.resident, "peak": self.peak,
                "slots": len(self.cache), "t_pread": self.t_pread,
                "t_convert": self.t_convert, "t_eval": self.t_eval}

    def dump_trace(self, path):
        """Routing histogram in the shape build_index.py --trace expects."""
        assert self.trace is not None, "store was not built with trace=True"
        total = sum(sum(d.values()) for d in self.trace.values())
        out = {"counts": {str(l): {str(e): n for e, n in d.items()}
                          for l, d in self.trace.items()},
               "accesses": total, "layers": self.layers, "num_experts": self.E}
        Path(path).write_text(json.dumps(out))
        return total

    def close(self):
        self.pool.shutdown()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _self_check(index_path):
    """Two properties, both of which a bug would silently break:
    bytes returned must equal what mx.load gives, and the ceiling must hold
    under an access pattern designed to thrash it."""
    import random
    probe = M25Store(index_path)
    # The LARGEST entry in the layer, not E0's: a mixed index holds 7.96 MB hot
    # experts beside 4.42 MB cold ones, so slot arithmetic off the first one
    # under-counts and the test stops forcing the eviction it exists to force.
    one = max(probe._entry_bytes(probe.meta[f"L1.E{e}"]) for e in range(probe.E))
    probe.close()
    cap = 20 * one / 1e9              # 20 slots: tight enough for 400 reads to thrash
    st = M25Store(index_path, ceiling_gb=cap)
    assert st.ceiling // one < 40, "ceiling too loose to force eviction"

    random.seed(0)
    for _ in range(400):
        st.get(1, random.randrange(st.E))
        assert st.resident <= st.ceiling, (st.resident, st.ceiling)
    assert st.evictions > 0 and st.peak <= st.ceiling
    assert st.hits + st.misses == 400

    # LRU order: re-touching an expert must protect it from the next eviction.
    st2 = M25Store(index_path, ceiling_gb=cap)
    n = 0
    while st2.resident + st2._entry_bytes(st2.meta[f"L1.E{n}"]) <= st2.ceiling:
        st2.get(1, n)                # fill to capacity by bytes, not by slots:
        n += 1                       # mixed tiers mean slots are not the unit
    st2.get(1, 0)                    # refresh the oldest
    st2.get(1, n)                    # first insert that must evict
    assert (1, 0) in st2.cache, "LRU evicted the most recently used entry"
    assert (1, 1) not in st2.cache, "LRU did not evict the true oldest"

    # The batch path: same bytes as the serial one, ceiling still held, and
    # every expert asked for resident on return (the loop that follows a
    # get_many indexes the result directly and has no second chance to fetch).
    st3 = M25Store(index_path, ceiling_gb=cap)
    batch = list(range(20, 32))
    got = st3.get_many(1, batch)
    assert set(got) == set(batch), "get_many dropped an expert"
    assert all((1, e) in st3.cache for e in batch), "returned but not resident"
    assert st3.resident <= st3.ceiling, (st3.resident, st3.ceiling)
    assert mx.array_equal(got[25]["up_proj"]["weight"],
                          st.get(1, 25)["up_proj"]["weight"])
    for _ in range(30):                  # thrash it in batches
        st3.get_many(1, [random.randrange(st3.E) for _ in range(10)])
        assert st3.resident <= st3.ceiling, (st3.resident, st3.ceiling)
    assert st3.evictions > 0

    # A working set larger than the ceiling crashes and names the fix.
    try:
        st3.get_many(1, list(range(100)))
        raise AssertionError("oversized working set was not rejected")
    except ValueError as e:
        assert "ceiling" in str(e), e

    # Correctness against the framework loader.
    rec = st.meta["L1.E5"]["gate_proj"]["weight"]
    ref = mx.load(os.path.join(rec[0], rec[1]))
    want = ref["model.layers.1.block_sparse_moe.switch_mlp.gate_proj.weight"][5]
    assert mx.array_equal(st.get(1, 5)["gate_proj"]["weight"], want)

    print(f"self-check ok: ceiling held over 400 random reads "
          f"({st.evictions} evictions, peak {st.peak/1e6:.0f} MB of "
          f"{st.ceiling/1e6:.0f} MB), LRU order correct, bytes match mx.load")
    st.close()
    st2.close()
    st3.close()


if __name__ == "__main__":
    import sys
    _self_check(sys.argv[1] if len(sys.argv) > 1 else "models/m25.idx")
