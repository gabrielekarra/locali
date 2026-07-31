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
                 nocache=True, cache_policy="lru"):
        idx = json.loads(Path(index_path).read_text())
        if cache_policy not in ("lru", "slru-cold", "slru-all"):
            raise ValueError(
                "cache_policy must be 'lru', 'slru-cold', or 'slru-all'"
            )
        self.meta = idx["experts"]
        self.layers, self.E = idx["layers"], idx["num_experts"]
        self.top_k = idx["top_k"]
        self.cache_policy = cache_policy
        self.ceiling = int(ceiling_gb * 1e9)
        # Large expert-major reads leave enough useful work per tier to overlap
        # I/O and gather.  The original nine-read layout regresses under the same
        # scheduling because its many writes contend with the GPU for memory.
        self.tier_overlap = bool(idx.get("pack"))
        self.fds, self._nocache = {}, nocache
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.hits = self.misses = self.evictions = self.bytes_read = 0
        self.t_stall = 0.0
        # Cross-layer prefetch needs two guards that a purely synchronous store
        # does not. `pinned` holds the slots the CURRENT layer routed to: its
        # gather is still an unevaluated graph when the prefetch for L+1 claims
        # slots, so _claim must not hand one of them away. `inflight` maps a
        # resident key to the reads still filling it, because a prefetched
        # expert is in the LRU -- and therefore a hit -- before its bytes land,
        # and the gather must wait for them anyway.
        self.pinned = {}
        self.inflight = {}
        self.pf_keys = set()
        self.prefetched = self.prefetch_used = self.prefetch_wasted = 0

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
        self.slots, self.lru, self.protected, self.protected_cap = {}, {}, {}, {}
        self.free = {}
        share = (
            {self.tiers[0]: 1.0}
            if len(self.tiers) == 1
            else {"hot": hot_share, "cold": 1.0 - hot_share}
        )
        for t in self.tiers:
            m = sample[t]
            per = sum(m[p][k][3] for p in PROJS for k in ARRS)
            self.item[t] = per
            # Slot 0 is reserved: a masked-out mixed-tier entry gathers from it
            # and contributes exactly 0.0. It is part of the requested ceiling,
            # not an allocation hidden just beyond it.
            n = int(self.ceiling * share.get(t, 0.5) / per)
            if n < 2:
                raise ValueError(
                    f"{ceiling_gb:.2f} GB gives tier {t} no usable expert "
                    f"slot after its reserved zero slot; raise --ceiling-gb "
                    f"or adjust --hot-share"
                )
            self.slots[t] = n
            self.lru[t] = OrderedDict()
            self.protected[t] = OrderedDict()
            # SLRU reserves half the usable slots for entries that have been
            # requested at least twice. New demand and speculative prefetches
            # compete in probation first, so a one-pass scan cannot evict the
            # recurring half of the working set. Plain LRU gives protection no
            # reserved capacity and therefore follows the original path.
            self.protected_cap[t] = (
                (n - 1) // 2
                if (
                    self.cache_policy == "slru-all"
                    or self.cache_policy == "slru-cold" and t == "cold"
                )
                else 0
            )
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
        assert self.resident <= self.ceiling, (
            f"arena allocated {self.resident} bytes against "
            f"{self.ceiling}-byte ceiling"
        )

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

    def _read_many(self, recs, dsts):
        """DMA one contiguous expert into its nine discontiguous arena arrays."""
        root, shard, off, _, _, _ = recs[0]
        end = off
        for rec, dst in zip(recs, dsts):
            rroot, rshard, roff, nbytes, _, _ = rec
            if (rroot, rshard, roff) != (root, shard, end):
                raise ValueError("expert pack records are not contiguous")
            if len(dst) != nbytes:
                raise ValueError(f"arena slice is {len(dst)} bytes, need {nbytes}")
            end += nbytes
        want = end - off
        got = os.preadv(self._fd(root, shard), dsts, off)
        if got != want:
            raise IOError(f"short packed read {got}/{want} at {shard}+{off}")

    @staticmethod
    def _contiguous(recs):
        root, shard, end, _, _, _ = recs[0]
        for rec in recs:
            rroot, rshard, off, nbytes, _, _ = rec
            if (rroot, rshard, off) != (root, shard, end):
                return False
            end += nbytes
        return True

    def _claim(self, t, key):
        """A slot for `key`, evicting the least recently used if none is free.

        The synchronous path was safe to overwrite because the caller had
        already evaluated the previous layer -- the router of layer L+1 consumes
        layer L's output, which forces it, so no pending graph referenced a slot
        by the time this ran. Prefetch breaks that: it claims slots for L+1 while
        layer L's gather is still an unevaluated graph. Hence `pinned`, which
        holds exactly the slots the current layer routed to and is skipped when
        choosing a victim.

        Returns None when every slot of the tier is pinned, which only a
        best-effort prefetch can ask for and which it treats as "skip this one".
        """
        pin = self.pinned.get(t, ())
        probation_cap = self.slots[t] - 1 - self.protected_cap[t]
        # Fixed-size probation is intentional. At startup SLRU may leave the
        # protected half physically free until entries earn promotion; using
        # those slots for one-hit scans would reduce SLRU back to plain LRU.
        probation_full = len(self.lru[t]) >= probation_cap
        victim = None
        victim_segment = None
        if probation_full or not self.free[t]:
            victim = next(
                (k for k, v in self.lru[t].items() if v not in pin),
                None,
            )
            if victim is not None:
                victim_segment = self.lru[t]
        if victim is None and not probation_full and self.free[t]:
            s = self.free[t].pop()
        else:
            if victim is None:
                victim = next(
                    (k for k, v in self.protected[t].items() if v not in pin),
                    None,
                )
                if victim is not None:
                    victim_segment = self.protected[t]
            if victim is None:
                return None
            s = victim_segment.pop(victim)
            self.inflight.pop(victim, None)
            if victim in self.pf_keys:
                # Prefetched, then evicted before its layer ever ran: bytes and
                # a slot spent on a misprediction. The counter is the honest
                # price of the 78.5% and has to be reported next to it.
                self.pf_keys.discard(victim)
                self.prefetch_wasted += 1
            self.evictions += 1
        self.lru[t][key] = s
        return s

    def _slot(self, t, key):
        """Return a resident slot from either SLRU segment."""
        if key in self.protected[t]:
            return self.protected[t][key]
        return self.lru[t][key]

    def _touch(self, t, key):
        """Refresh an LRU hit or promote an SLRU probation hit."""
        if key in self.protected[t]:
            self.protected[t].move_to_end(key)
            return
        if self.protected_cap[t] == 0:
            self.lru[t].move_to_end(key)
            return
        slot = self.lru[t].pop(key)
        self.protected[t][key] = slot
        if len(self.protected[t]) > self.protected_cap[t]:
            old, old_slot = self.protected[t].popitem(last=False)
            self.lru[t][old] = old_slot

    def slots_for(self, layer, experts):
        """{expert: (tier, slot)}, every one resident on return."""
        placed, by_tier = self.submit(layer, experts)
        self.wait([f for futs in by_tier.values() for f in futs])
        return placed

    def wait(self, futs):
        if not futs:
            return
        t0 = time.perf_counter()
        for f in futs:
            f.result()
        self.t_stall += time.perf_counter() - t0

    def submit(self, layer, experts, prefetch=False):
        """Claim slots now, return before the bytes arrive.

        Slots are claimed synchronously so a later submit cannot steal one an
        earlier submit is still filling; only the reads are deferred.

        `prefetch=True` is the cross-layer path: layer L issues layer L+1's
        predicted experts while its own gather still has to run. It is
        best-effort in both directions -- it never evicts a slot the current
        layer is using, and it never raises when the arena is full -- and it
        does not touch hit/miss accounting, which must keep meaning "what the
        real router asked for".

        Splitting a layer's TOKENS into chunks and staggering the waits was
        tried and removed: it does hide disk (105s against 129s blocked at
        B=512) but the gathers get narrower and the penalty is larger than the
        hiding. This defers across layers instead, where the gather stays full
        width.
        """
        want = list(dict.fromkeys(experts))
        out, miss = {}, []
        waits = {t: [] for t in self.tiers}
        for e in want:
            key = (layer, e)
            t = self.tier(layer, e)
            if key in self.lru[t] or key in self.protected[t]:
                out[e] = (t, self._slot(t, key))
                if not prefetch:
                    self.hits += 1
                    self._touch(t, key)
                    if key in self.pf_keys:
                        self.pf_keys.discard(key)
                        self.prefetch_used += 1
                # A prefetched expert enters the LRU when its slot is claimed,
                # not when its bytes land. Resident is not the same as ready.
                futs = self.inflight.get(key)
                if futs:
                    if all(f.done() for f in futs):
                        del self.inflight[key]
                    elif not prefetch:
                        waits[t].extend(futs)
            else:
                if not prefetch:
                    self.misses += 1
                miss.append((e, t))

        if not prefetch:
            need = {}
            for _, t in miss:
                need[t] = need.get(t, 0) + 1
            for t, n in need.items():
                if n > self.slots[t] - 1:
                    raise ValueError(
                        f"layer {layer} needs {n} {t} experts but the arena has "
                        f"{self.slots[t]-1}; raise --ceiling-gb or --hot-share")

        jobs = {t: [] for t in self.tiers}
        issued = []
        for e, t in miss:
            s = self._claim(t, (layer, e))
            if s is None:            # every slot of the tier is pinned
                continue
            out[e] = (t, s)
            issued.append((e, t, s))
            m = self.meta[f"L{layer}.E{e}"]
            recs, dsts = [], []
            for p in PROJS:
                for k in ARRS:
                    rec = m[p][k]
                    nb = rec[3]
                    recs.append(rec)
                    dsts.append(self.mv[(t, p, k)][s * nb:(s + 1) * nb])
            key = (layer, e)
            if self._contiguous(recs):
                jobs[t].append((key, self._read_many, (recs, dsts)))
            else:
                jobs[t].extend(
                    (key, self._read_into, (rec, dst))
                    for rec, dst in zip(recs, dsts)
                )
            self.bytes_read += self.item[t]

        # Issue order is deliberately NOT sorted by disk offset. That was tried
        # -- at batch 32 a layer's reads are a sweep over its experts, so
        # ascending order looked free -- and measured 1.95 tok/s against 2.07
        # unsorted. With eight reads already in flight the drive does its own
        # scheduling and the sort only costs.
        # Queue a whole tier before the next.  ArenaMoE waits/evaluates in this
        # same order, so once the first tier is ready its full-width gather can
        # occupy the GPU while the pool fills the other tier's disjoint arrays.
        by_tier = {}
        for t in self.tiers:
            futs = []
            for key, fn, args in jobs[t]:
                f = self.pool.submit(fn, *args)
                self.inflight.setdefault(key, []).append(f)
                futs.append(f)
            # Reads a prefetch issued are already in flight; the real submit
            # still has to wait for them, so they join this layer's futures.
            by_tier[t] = futs + waits[t]

        if prefetch:
            self.prefetched += len(issued)
            self.pf_keys.update((layer, e) for e, _, _ in issued)
        else:
            # Pin what this layer routed to. The prefetch that runs next claims
            # slots while the gather below is still an unevaluated graph.
            pin = {t: set() for t in self.tiers}
            for t, s in out.values():
                pin[t].add(s)
            self.pinned = pin
        return out, by_tier

    def stats(self):
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 0.0,
                "evictions": self.evictions, "bytes_read": self.bytes_read,
                "resident": self.resident, "peak": self.resident,
                "slots": {t: self.slots[t] for t in self.tiers},
                "t_stall": self.t_stall, "t_convert": 0.0, "t_eval": 0.0,
                "prefetched": self.prefetched,
                "prefetch_used": self.prefetch_used,
                "prefetch_wasted": self.prefetch_wasted,
                "cache_policy": self.cache_policy,
                "protected": {
                    t: len(self.protected[t]) for t in self.tiers
                }}

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
        # Set by the engine to the NEXT layer's block when cross-layer prefetch
        # is on. Left None otherwise, which restores the synchronous path
        # exactly -- no prefetch submit, no pinning, nothing to wait on.
        self.nxt = None
        # And to the layer after that, when depth-2 lookahead is on. The router
        # at L+2 is 72.5% accurate on layer L's MoE input against 78.5% at L+1
        # (`crosslayer_probe.py --distances 1,2,3`; L+3 is 68.0%), so the decay
        # with distance is gentle and all three sit far above the 33.0% that
        # killed the same-layer next-token prefetch. The reason to want it is
        # queue depth: `depth_bw.py` measures 3.96 GB/s at the depth the
        # algorithm allows and 5.91 at depth 16.
        #
        # MEASURED, AND IT LOSES IN PYTHON -- default off. At the 9 GB / k=5
        # operating point, depth 2 with k2=4 does everything it was built to do:
        # hit 75.3% -> 76.7%, blocked on disk 10.8s -> 10.4s. And it still runs
        # 2.22 tok/s against 2.34, because the time OFF the disk grows 6.5s ->
        # 7.7s. That is 1984 extra `route` calls (62 layers x 32 tokens) at
        # ~0.6 ms each: a router matmul, an argpartition, an argsort, and a
        # `.tolist()` that forces a GPU sync. It pays 0.6 ms per layer of
        # interpreter to save 0.2 ms of disk.
        #
        # Same shape as the prefetch_k=8 result below, different currency --
        # bytes there, host overhead here. Keep it: in a native runtime the
        # router pass is nearly free and the sync disappears, so the 0.4s of
        # disk it removes would come without the 1.2s that currently buys it.
        # This lever is not dead, it is downstream of the rewrite.
        self.nxt2 = None
        # How many of the next layer's predicted experts to issue. The full
        # top-8 measured 79.3% useful and read 28% more bytes than the
        # synchronous path -- the misses it removes are worth less than the
        # mispredictions cost, and it only wins because wasted bytes sit off the
        # critical path. Issuing fewer, higher-gate candidates trades coverage
        # for waste at a better rate.
        self.prefetch_k = None
        # Depth-2 defaults to one fewer candidate than depth-1. Its predictions
        # are 6 points worse and its bytes are the most speculative in flight,
        # so it should be the first thing to give up a slot.
        self.prefetch_k2 = None

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
        experts = [e for row in ii for e in row]
        if self.store.tier_overlap:
            placed, reads = self.store.submit(self.layer, experts)
        else:
            placed, reads = self.store.slots_for(self.layer, experts), None

        # Cross-layer prefetch. The next layer's router is 78.5% accurate on
        # THIS layer's MoE input (`crosslayer_probe.py`), against 33.0% for the
        # previous token's selection at the same layer -- the prefetch NOTES
        # records as failed. Issued after this layer's own reads so those own
        # the queue first, and before the gather below, which is the window the
        # reads get to land in.
        # Issued nearest-first: L+1 before L+2, so the reads most likely to be
        # needed soonest own the queue ahead of the more speculative ones. A
        # slot claimed here goes to the fresh end of the LRU, so the L+2 pass
        # cannot evict what the L+1 pass just claimed.
        for blk, k in ((self.nxt, self.prefetch_k),
                       (self.nxt2, self.prefetch_k2)):
            if blk is None:
                continue
            pi, pg = blk.route(x)
            pi = pi.reshape(-1, self.top_k)
            if k and k < self.top_k:
                # route() returns the top-k unordered (argpartition), so the
                # highest-gate candidates have to be selected explicitly.
                order = mx.argsort(-pg.reshape(-1, self.top_k), axis=-1)
                pi = mx.take_along_axis(pi, order[:, :k], axis=-1)
            self.store.submit(
                blk.layer,
                [e for row in pi.tolist() for e in row],
                prefetch=True,
            )

        sc = gates.reshape(T, self.top_k).astype(x.dtype)
        out = self._apply(
            flat, ii, placed, sc, shape[-1], x.dtype, reads=reads
        )
        return out.reshape(shape).astype(x.dtype)

    def _apply(self, flat, ii, placed, sc, d, dtype, reads=None):
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
            if reads is not None:
                self.store.wait(reads[t])
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
        gflat = sc.reshape(-1).astype(mx.float32)
        parts = []
        for i, t in enumerate(present):
            if reads is not None:
                self.store.wait(reads[t])
            ent = np.argwhere(mask[t] > 0)                       # [n, 2]
            rhs = slot[t][ent[:, 0], ent[:, 1]]

            # Keep equal experts adjacent while gather_qmm runs.  At a wide
            # batch the same expert serves many rows; token-major order scatters
            # those uses across the grid and repeatedly pushes its weights
            # through the GPU caches.  Stable slot order measured 1.35x faster
            # at B=512 on M4.  Undo it before the scatter so route/reduction
            # order, and therefore the arithmetic contract, stays unchanged.
            order = np.argsort(rhs, kind="stable")
            sent = ent[order]
            rows = mx.array(sent[:, 0])
            idx = mx.array(rhs[order])[:, None]
            xin = mx.expand_dims(flat[rows], (-2, -3))           # [n, 1, 1, d]
            y = self._gather(t, xin, idx).reshape(len(ent), d)

            undo = np.empty_like(order)
            undo[order] = np.arange(len(order))
            y = y[mx.array(undo)]
            rows = mx.array(ent[:, 0])
            g = gflat[mx.array(ent[:, 0] * self.top_k + ent[:, 1])]
            val = y.astype(mx.float32) * g[:, None]
            parts.append((rows, val))
            if reads is not None and i + 1 < len(present):
                mx.async_eval(val)

        # Reads may have overlapped the independent tier gathers, but reduction
        # remains in the same tier and route order as the serial implementation.
        out = mx.zeros((T, d), dtype=mx.float32)
        for rows, val in parts:
            out = out.at[rows].add(val)
        return out.astype(dtype)

    def _gather(self, t, xin, idx):
        A = lambda p, k: self.store.arena[(t, p, k)]
        qm = lambda v, p: mx.gather_qmm(
            v, A(p, "weight"), A(p, "scales"), A(p, "biases"),
            rhs_indices=idx, transpose=True,
            group_size=GROUP, bits=BITS[t])
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
