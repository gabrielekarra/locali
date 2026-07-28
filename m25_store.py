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
import json
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

F_NOCACHE = 48
PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")
NP = {"U32": np.uint32, "I32": np.int32, "F16": np.float16,
      "BF16": np.uint16, "F32": np.float32, "U8": np.uint8}


class M25Store:
    def __init__(self, index_path, ceiling_gb=8.0, nocache=True):
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

    def _read(self, rec):
        root, shard, off, nbytes, shape, dt = rec
        raw = os.pread(self._fd(root, shard), nbytes, off)
        if len(raw) != nbytes:
            raise IOError(f"short read {len(raw)}/{nbytes} at {shard}+{off}")
        a = np.frombuffer(raw, dtype=NP[dt]).reshape(shape)
        arr = mx.array(a)
        return arr.view(mx.bfloat16) if dt == "BF16" else arr

    def tier(self, layer, expert):
        """'hot' (served from the 4-bit snapshot) or 'cold' (2-bit pack)."""
        return self.meta[f"L{layer}.E{expert}"]["tier"]

    def _entry_bytes(self, m):
        return sum(m[p][k][3] for p in PROJS for k in ARRS)

    def get(self, layer, expert):
        key = (layer, expert)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        m = self.meta[f"L{layer}.E{expert}"]
        need = self._entry_bytes(m)
        if need > self.ceiling:
            raise ValueError(f"one expert is {need}B, ceiling is {self.ceiling}B")
        # Evict FIRST. Inserting and trimming afterwards would breach the
        # ceiling for the width of that window, which is the whole invariant.
        while self.resident + need > self.ceiling:
            _, ev = self.cache.popitem(last=False)
            self.resident -= sum(v.nbytes for d in ev.values() for v in d.values())
            self.evictions += 1
        val = {p: {k: self._read(m[p][k]) for k in ARRS} for p in PROJS}
        mx.eval([v for d in val.values() for v in d.values()])
        self.cache[key] = val
        self.resident += need
        self.bytes_read += need
        self.peak = max(self.peak, self.resident)
        assert self.resident <= self.ceiling, "ceiling breached"
        return val

    def stats(self):
        t = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / t if t else 0.0,
                "evictions": self.evictions, "bytes_read": self.bytes_read,
                "resident": self.resident, "peak": self.peak,
                "slots": len(self.cache)}

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _self_check(index_path):
    """Two properties, both of which a bug would silently break:
    bytes returned must equal what mx.load gives, and the ceiling must hold
    under an access pattern designed to thrash it."""
    import random
    st = M25Store(index_path, ceiling_gb=0.3)
    one = st._entry_bytes(st.meta["L1.E0"])
    assert st.ceiling // one < 40, "ceiling too loose to force eviction"

    random.seed(0)
    for _ in range(400):
        st.get(1, random.randrange(st.E))
        assert st.resident <= st.ceiling, (st.resident, st.ceiling)
    assert st.evictions > 0 and st.peak <= st.ceiling
    assert st.hits + st.misses == 400

    # LRU order: re-touching an expert must protect it from the next eviction.
    st2 = M25Store(index_path, ceiling_gb=0.3)
    n = st2.ceiling // one
    for e in range(n):
        st2.get(1, e)
    st2.get(1, 0)                    # refresh the oldest
    st2.get(1, n)                    # forces one eviction
    assert (1, 0) in st2.cache, "LRU evicted the most recently used entry"
    assert (1, 1) not in st2.cache, "LRU did not evict the true oldest"

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


if __name__ == "__main__":
    import sys
    _self_check(sys.argv[1] if len(sys.argv) > 1 else "models/m25.idx")
