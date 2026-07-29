"""Expert cache as a preallocated arena in unified memory, gathered in place.

`m25_store.py` caches an expert as nine freshly allocated mlx arrays, built by
copying out of a freshly allocated `bytes` that `os.pread` returned. Measured,
that copy is 29% of the run, and it is not the memory bus: a warm memcpy on this
machine does 42 GB/s while the conversion managed 4-5, because every expert
allocates and faults new pages.

Apple Silicon makes the copy unnecessary. There is no device memory and no PCIe:
the GPU reads the same LPDDR the CPU writes. mlx hands out a writable memoryview
of its own buffers, and those buffers land 16 KB page-aligned -- which is what
`F_NOCACHE` wants in order to DMA straight into user memory instead of bouncing
through the kernel. So the bytes can go from the SSD controller into exactly the
memory `gather_qmm` will read, with nobody copying them.

Measured, F_NOCACHE, 1.18 MB blocks:

    pread -> bytes -> mx.array     1.89 GB/s
    preadv -> fresh mlx array      1.49 GB/s   (allocating per read is worse:
                                                mx.zeros writes the bytes the
                                                read is about to overwrite)
    preadv -> preallocated slab    2.62 GB/s   +39%, and no conversion at all

The arena is that slab. One per tier, since 4-bit and 2-bit experts cannot share
an [E, out, in] tensor, laid out exactly as mlx_lm's SwitchGLU expects so that
`gather_qmm` can select experts by slot index -- which also collapses 24 matvec
launches per layer into 6.

The ceiling stops being a rule enforced by bookkeeping and becomes a property of
the allocation: the arenas are all the memory there is, so residency cannot
exceed them. Eviction is a slot being reused, and the LRU order decides which.
"""

import fcntl
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.switch_layers import SwiGLU

SWIGLU = SwiGLU()

F_NOCACHE = 48
PAGE = 16384
PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases")
BITS = {"hot": 4, "cold": 2}
GROUP = 64
MXD = {"U32": mx.uint32, "I32": mx.int32, "F16": mx.float16,
       "BF16": mx.bfloat16, "F32": mx.float32, "U8": mx.uint8}


class ArenaStore:
    def __init__(self, index_path, ceiling_gb=8.0, hot_share=0.33, threads=8,
                 nocache=True):
        idx = json.loads(Path(index_path).read_text())
        self.meta = idx["experts"]
        self.layers, self.E = idx["layers"], idx["num_experts"]
        self.top_k = idx["top_k"]
        self.fds, self._nocache = {}, nocache
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.hits = self.misses = self.evictions = self.bytes_read = 0
        self.t_stall = 0.0

        # A sample expert per tier fixes the arena shapes. Every expert of a
        # tier has identical shapes -- that is what makes a stacked arena
        # possible at all.
        sample = {}
        for name, m in self.meta.items():
            t = m["tier"]
            if t not in sample:
                sample[t] = m
            if len(sample) == 2:
                break
        self.tiers = sorted(sample)

        self.arena, self.mv, self.item = {}, {}, {}
        self.slots, self.lru, self.free = {}, {}, {}
        share = {"hot": hot_share, "cold": 1.0 - hot_share}
        for t in self.tiers:
            m = sample[t]
            per = sum(m[p][k][3] for p in PROJS for k in ARRS)
            self.item[t] = per
            # +1 for the reserved zero slot at index 0: a masked-out entry in a
            # mixed-tier call gathers from it and contributes exactly 0.0, which
            # is what keeps the all-hot case bit-identical.
            n = max(2, int(ceiling_gb * 1e9 * share.get(t, 0.5) / per)) + 1
            self.slots[t] = n
            self.lru[t] = OrderedDict()
            self.free[t] = list(range(1, n))
            for p in PROJS:
                for k in ARRS:
                    _, _, _, nb, shape, dt = m[p][k]
                    a = mx.zeros((n, *shape), dtype=MXD[dt])
                    mx.eval(a)
                    # Fault every page in once, here, rather than one expert at
                    # a time during decode. This is the cost the old path paid
                    # over and over.
                    np.asarray(a.view(mx.uint16) if dt == "BF16" else a)[:] = 0
                    self.arena[(t, p, k)] = a
                    self.mv[(t, p, k)] = memoryview(a).cast("B")
        self.resident = sum(self.slots[t] * self.item[t] for t in self.tiers)

    def _fd(self, root, shard):
        key = (root, shard)
        if key not in self.fds:
            fd = os.open(os.path.join(root, shard), os.O_RDONLY)
            if self._nocache:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[key] = fd
        return self.fds[key]

    def tier(self, layer, expert):
        return self.meta[f"L{layer}.E{expert}"]["tier"]

    def _read_into(self, rec, dst):
        """DMA one tensor into its slot. No intermediate buffer exists."""
        root, shard, off, nbytes, _, _ = rec
        got = os.preadv(self._fd(root, shard), [dst], off)
        if got != nbytes:
            raise IOError(f"short read {got}/{nbytes} at {shard}+{off}")

    def _claim(self, t, key):
        """A slot for `key`, evicting the least recently used if none is free.

        Safe to overwrite because the caller has already evaluated the previous
        layer -- the router of layer L+1 consumes layer L's output, which forces
        it, so no pending graph still references a slot by the time this runs.
        """
        if self.free[t]:
            s = self.free[t].pop()
        else:
            _, s = self.lru[t].popitem(last=False)
            self.evictions += 1
        self.lru[t][key] = s
        return s

    def slots_for(self, layer, experts):
        """{expert: (tier, slot)}, every one resident on return."""
        placed, futs = self.submit(layer, experts)
        self.wait(futs)
        return placed

    def wait(self, futs):
        if not futs:
            return
        t0 = time.perf_counter()
        for f in futs:
            f.result()
        self.t_stall += time.perf_counter() - t0

    def submit(self, layer, experts):
        """Claim slots now, return before the bytes arrive.

        Slots are claimed synchronously so a later submit cannot steal one an
        earlier submit is still filling; only the reads are deferred.

        Nothing currently defers them. Splitting a layer's tokens into chunks
        and staggering the waits was tried and removed: it does hide disk (105s
        against 129s blocked at B=512) but the gathers get narrower and the
        penalty is larger than the hiding. Kept split from `slots_for` because
        the seam is where any future overlap has to go -- per tier rather than
        per token, so the token dimension stays wide.
        """
        want = list(dict.fromkeys(experts))
        out, miss = {}, []
        for e in want:
            key = (layer, e)
            t = self.tier(layer, e)
            if key in self.lru[t]:
                self.hits += 1
                self.lru[t].move_to_end(key)
                out[e] = (t, self.lru[t][key])
            else:
                self.misses += 1
                miss.append((e, t))

        need = {}
        for _, t in miss:
            need[t] = need.get(t, 0) + 1
        for t, n in need.items():
            if n > self.slots[t] - 1:
                raise ValueError(
                    f"layer {layer} needs {n} {t} experts but the arena has "
                    f"{self.slots[t]-1}; raise --ceiling-gb or --hot-share")

        jobs = []
        for e, t in miss:
            s = self._claim(t, (layer, e))
            out[e] = (t, s)
            m = self.meta[f"L{layer}.E{e}"]
            for p in PROJS:
                for k in ARRS:
                    rec = m[p][k]
                    nb = rec[3]
                    jobs.append((rec, self.mv[(t, p, k)][s * nb:(s + 1) * nb]))
            self.bytes_read += self.item[t]

        # Issue order is deliberately NOT sorted by disk offset. That was tried
        # -- at batch 32 a layer's reads are a sweep over its experts, so
        # ascending order looked free -- and measured 1.95 tok/s against 2.07
        # unsorted. With eight reads already in flight the drive does its own
        # scheduling and the sort only costs.
        futs = [self.pool.submit(self._read_into, *j) for j in jobs]
        return out, futs

    def stats(self):
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 0.0,
                "evictions": self.evictions, "bytes_read": self.bytes_read,
                "resident": self.resident, "peak": self.resident,
                "slots": {t: self.slots[t] for t in self.tiers},
                "t_stall": self.t_stall, "t_convert": 0.0, "t_eval": 0.0}

    def close(self):
        self.pool.shutdown()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class ArenaMoE:
    """MoE block over the arena, one gather_qmm per tier per projection.

    Mirrors mlx_lm's SwitchGLU arithmetic exactly -- same kernel, same shapes,
    same reduction order -- so that against an all-hot index this reproduces
    `blk(x)` bit for bit. That is a stronger check than the old path could make:
    it compared dequantize-then-matmul against gather_qmm, two different
    computations that differ by 7.8e-3 however correct the fetch is.

    Mixed tiers are handled by running both gathers over all [T, k] entries with
    the other tier's gates zeroed and its indices pointed at the reserved zero
    slot, which contributes exactly 0.0. Wasteful in FLOPs and free in launches,
    which is the right trade on hardware measured at 56 GFLOP/s of a matvec.
    """

    def __init__(self, store: ArenaStore, layer: int, gate, bias, top_k: int):
        self.store, self.layer, self.gate = store, layer, gate
        self.bias, self.top_k = bias, top_k

    def route(self, x):
        scores = mx.sigmoid(self.gate(x.astype(mx.float32)))
        inds = mx.argpartition(-(scores + self.bias), kth=self.top_k - 1,
                               axis=-1)[..., :self.top_k]
        g = mx.take_along_axis(scores, inds, axis=-1)
        return inds, g / (mx.sum(g, axis=-1, keepdims=True) + 1e-20)

    def __call__(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        T = flat.shape[0]
        inds, gates = self.route(x)
        ii = inds.reshape(-1, self.top_k).tolist()
        placed = self.store.slots_for(self.layer, [e for row in ii for e in row])
        sc = gates.reshape(T, self.top_k).astype(x.dtype)
        out = self._apply(flat, ii, placed, sc, shape[-1], x.dtype)
        return out.reshape(shape).astype(x.dtype)

    def _apply(self, flat, ii, placed, sc, d, dtype):
        """The gather itself, for one contiguous run of tokens.

        `sc` is already cast to x.dtype: mlx_lm casts the normalised scores
        BEFORE weighting, and swiglu is one fused op rather than a separate
        silu-then-multiply, which rounds differently. Both are copied exactly,
        and together they are why bit-identity against the framework holds.
        """
        T = len(ii)
        slot = {t: np.zeros((T, self.top_k), dtype=np.uint32)
                for t in self.store.tiers}
        mask = {t: np.zeros((T, self.top_k), dtype=np.float32)
                for t in self.store.tiers}
        for ti, row in enumerate(ii):
            for ki, e in enumerate(row):
                t, s = placed[e]
                slot[t][ti, ki] = s
                mask[t][ti, ki] = 1.0
        present = [t for t in self.store.tiers if mask[t].any()]

        if len(present) == 1:
            # One tier: the [T, k] grid is exactly the work required, and the
            # reduction is mlx_lm's own. This is the path `verify` exercises,
            # and the reason bit-identity survives at all.
            t = present[0]
            y = self._gather(t, mx.expand_dims(flat, (-2, -3)),
                             mx.array(slot[t])).squeeze(-2)      # [T, k, d]
            return (y * sc[..., None]).sum(axis=-2)

        # Both tiers. Computing the full [T, k] grid for each and masking the
        # other away costs nothing in launches and exactly 2x in FLOPs -- the
        # right trade at batch 1, where the machine measured 56 GFLOP/s and
        # launches dominated, and the wrong one at batch 256 where the disk has
        # slack and the gather binds. So take only each tier's own entries.
        #
        # Accumulated in float32. The masked path summed a token's k terms in
        # one reduction; this scatter-adds them, and eight sequential roundings
        # in bfloat16 cost 1.4e-3 relative -- bfloat16 working as specified, but
        # there is no reason to pay it when the accumulator can be wider.
        out = mx.zeros((T, d), dtype=mx.float32)
        gflat = sc.reshape(-1).astype(mx.float32)
        for t in present:
            ent = np.argwhere(mask[t] > 0)                       # [n, 2]
            rows = mx.array(ent[:, 0])
            idx = mx.array(slot[t][ent[:, 0], ent[:, 1]])[:, None]
            xin = mx.expand_dims(flat[rows], (-2, -3))           # [n, 1, 1, d]
            y = self._gather(t, xin, idx).reshape(len(ent), d)
            g = gflat[mx.array(ent[:, 0] * self.top_k + ent[:, 1])]
            out = out.at[rows].add(y.astype(mx.float32) * g[:, None])
        return out.astype(dtype)

    def _gather(self, t, xin, idx):
        A = lambda p, k: self.store.arena[(t, p, k)]
        qm = lambda v, p: mx.gather_qmm(
            v, A(p, "weight"), A(p, "scales"), A(p, "biases"),
            rhs_indices=idx, transpose=True, group_size=GROUP, bits=BITS[t])
        h = SWIGLU(qm(xin, "up_proj"), qm(xin, "gate_proj"))
        return qm(h, "down_proj")


def verify(snap, index_path, layer, tokens, ceiling_gb, hot_share):
    """Against an all-hot index the arena holds the same bytes the resident
    model does, so this must reproduce mlx_lm's own `blk(x)` EXACTLY.

    That is a stronger claim than the old path could make. m25_stream compares
    dequantize-then-matmul against gather_qmm -- genuinely different
    computations, stuck at 7.8e-3 however correct the fetch is -- so it had to
    build its own reference. Moving to gather_qmm removes the discrepancy: same
    kernel, same shapes, same reduction order, and the framework becomes the
    reference.
    """
    from transformers import AutoTokenizer
    from neuron_tail_live import build_n_layers

    snap = Path(snap)
    idx = json.loads(Path(index_path).read_text())
    tiers = {v["tier"] for v in idx["experts"].values()}
    exact = tiers == {"hot"}

    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(Path("eval/pride_prejudice.txt").read_text()[:8000])
                   ["input_ids"][:tokens])
    model, cfg = build_n_layers(snap, layer + 1)
    lay = model.model.layers[layer]
    blk = lay.block_sparse_moe

    h = model.model.embed_tokens(ids[None])
    for i in range(layer):
        h = model.model.layers[i](h, mask=None, cache=None)
    x = lay.post_attention_layernorm(
        h + lay.self_attn(lay.input_layernorm(h), None, None))

    ref = blk(x)                       # mlx_lm's own path, gather_qmm and all
    store = ArenaStore(index_path, ceiling_gb=ceiling_gb, hot_share=hot_share)
    moe = ArenaMoE(store, layer, blk.gate, blk.e_score_correction_bias,
                   blk.num_experts_per_tok)
    got = moe(x)
    mx.eval(ref, got)

    f32 = lambda a: a.astype(mx.float32)
    d = float(mx.max(mx.abs(f32(got) - f32(ref))))
    rel = float(mx.linalg.norm(f32(got) - f32(ref)) /
                (mx.linalg.norm(f32(ref)) + 1e-20))
    s = store.stats()
    print(f"layer {layer}, {tokens} tokens, ceiling {ceiling_gb} GB, "
          f"index tiers {sorted(tiers)}, slots {s['slots']}")
    print(f"  vs mlx_lm blk(x), SAME kernel: max abs {d:.3e}  relative {rel:.3e}")
    print(f"  hit {s['hit_rate']*100:.1f}%  read {s['bytes_read']/1e6:.0f} MB  "
          f"evictions {s['evictions']}  arena {s['resident']/1e9:.2f} GB")
    if exact:
        assert d == 0.0, f"arena diverged from the framework by {d:.3e}"
        print("  MATCH: bit-identical to mlx_lm's own block")
    else:
        print(f"  mixed index: {rel:.1%} is the 2-bit cold tier, expected")
    store.close()
    return d, rel


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25-allhot.idx")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=7)
    ap.add_argument("--ceiling-gb", type=float, default=1.0)
    ap.add_argument("--hot-share", type=float, default=0.33)
    a = ap.parse_args()
    verify(a.snap, a.index, a.layer, a.tokens, a.ceiling_gb, a.hot_share)


if __name__ == "__main__":
    main()
