"""Benchmark MiniMax-M2.5 expert dispatch strategies on Apple Silicon.

This is deliberately a one-layer probe.  It uses the real indexed expert bytes,
the real mixed 4/2-bit arena, and the production matrix shapes, but does not load
the other 61 layers or the dense backbone.  That makes dispatch experiments fast
enough to repeat while preserving the kernel and memory-access pattern that
matters at large decode batches.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from m25_arena import ArenaMoE, ArenaStore, BITS, GROUP, PROJS, SWIGLU


def routes(batch: int, experts: int, top_k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    inds = np.stack(
        [rng.choice(experts, size=top_k, replace=False) for _ in range(batch)]
    ).astype(np.uint32)
    scores = rng.random((batch, top_k), dtype=np.float32)
    scores /= scores.sum(axis=-1, keepdims=True)
    return inds, scores


def token_gather(moe, flat, ii, placed, sc, d, dtype, sort_slots):
    """Mixed-tier gather, optionally grouping equal slots for locality."""
    ii_np = np.asarray(ii, dtype=np.uint32)
    tiers = moe.store.tiers
    tier_of = {t: i for i, t in enumerate(tiers)}
    tier = np.empty_like(ii_np, dtype=np.uint8)
    slot = np.empty_like(ii_np, dtype=np.uint32)
    for e, (t, s) in placed.items():
        match = ii_np == e
        tier[match] = tier_of[t]
        slot[match] = s

    out = mx.zeros((len(ii), d), dtype=mx.float32)
    gates = sc.reshape(-1).astype(mx.float32)
    for ti, t in enumerate(tiers):
        ent = np.argwhere(tier == ti)
        if not len(ent):
            continue
        rhs = slot[ent[:, 0], ent[:, 1]]
        order = (
            np.argsort(rhs, kind="stable")
            if sort_slots
            else np.arange(len(rhs))
        )
        sorted_ent = ent[order]
        rows = mx.array(sorted_ent[:, 0])
        idx = mx.array(rhs[order])[:, None]
        xin = mx.expand_dims(flat[rows], (-2, -3))
        val = moe._gather(t, xin, idx).reshape(len(ent), d)

        # Restore token/route order before the scatter.  Sorting is solely an
        # execution-order change for the gather and must not change reduction
        # order or output.
        undo = np.empty_like(order)
        undo[order] = np.arange(len(order))
        rows = mx.array(ent[:, 0])
        val = val[mx.array(undo)]
        gidx = mx.array(ent[:, 0] * moe.top_k + ent[:, 1])
        out = out.at[rows].add(val.astype(mx.float32) * gates[gidx, None])
    return out.astype(dtype)


def expert_major(moe, flat, ii, placed, sc, d, dtype):
    """Reuse an expert matrix across every batch row routed to it."""
    ii_np = np.asarray(ii, dtype=np.uint32)
    values, positions = [], []
    for e in np.unique(ii_np):
        ent = np.argwhere(ii_np == e)
        rows = mx.array(ent[:, 0])
        tier, slot = placed[int(e)]
        array = lambda p, k: moe.store.arena[(tier, p, k)][slot]
        qmm = lambda x, p: mx.quantized_matmul(
            x,
            array(p, "weight"),
            array(p, "scales"),
            array(p, "biases"),
            transpose=True,
            group_size=GROUP,
            bits=BITS[tier],
        )
        x = flat[rows]
        hidden = SWIGLU(qmm(x, "up_proj"), qmm(x, "gate_proj"))
        values.append(qmm(hidden, "down_proj").astype(mx.float32))
        positions.extend((ent[:, 0] * moe.top_k + ent[:, 1]).tolist())

    # Expert-major execution changes only the order in which independent terms
    # are produced.  Put them back in router order before weighting/reduction.
    order = np.argsort(np.asarray(positions), kind="stable")
    y = mx.concatenate(values, axis=0)[mx.array(order)]
    return (y.reshape(len(ii), moe.top_k, d) * sc[..., None]).sum(
        axis=-2
    ).astype(dtype)


def timed(fn, repeats):
    samples = []
    out = first = None
    drift = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append(time.perf_counter() - t0)
        if first is None:
            first = out
        else:
            delta = mx.max(
                mx.abs(out.astype(mx.float32) - first.astype(mx.float32))
            )
            mx.eval(delta)
            drift = max(drift, float(delta))
    return out, samples, drift


def pipeline_once(args, flat, inds, scores, overlap):
    store = ArenaStore(
        args.index,
        ceiling_gb=args.ceiling_gb,
        hot_share=args.hot_share,
    )
    moe = ArenaMoE(store, args.layer, gate=None, bias=None, top_k=store.top_k)
    ii = inds.tolist()
    sc = mx.array(scores).astype(mx.bfloat16)
    mx.eval(sc)

    t0 = time.perf_counter()
    placed, reads = store.submit(args.layer, inds.reshape(-1).tolist())
    if overlap:
        out = moe._apply(
            flat, ii, placed, sc, flat.shape[-1], mx.bfloat16, reads=reads
        )
    else:
        store.wait([f for futs in reads.values() for f in futs])
        out = moe._apply(
            flat, ii, placed, sc, flat.shape[-1], mx.bfloat16
        )
    mx.eval(out)
    elapsed = time.perf_counter() - t0
    stats = store.stats()
    store.close()
    del moe, store
    mx.clear_cache()
    return out, elapsed, stats


def pipeline_probe(args, batch):
    idx = json.loads(Path(args.index).read_text())
    first = idx["experts"][f"L{args.layer}.E0"]
    d = (
        first["gate_proj"]["weight"][4][1]
        * (32 // BITS[first["tier"]])
    )
    inds, scores = routes(batch, idx["num_experts"], idx["top_k"])
    flat = mx.random.normal((batch, d)).astype(mx.bfloat16)
    mx.eval(flat)

    samples = {"serial": [], "tier-overlap": []}
    outputs = {}
    for _ in range(args.repeats):
        for name, overlap in (("serial", False), ("tier-overlap", True)):
            out, elapsed, stats = pipeline_once(
                args, flat, inds, scores, overlap
            )
            samples[name].append(elapsed)
            outputs[name] = out
            print(
                f"  {name:12s} {elapsed * 1000:7.1f} ms  "
                f"read {stats['bytes_read'] / 1e9:.2f} GB  "
                f"wait {stats['t_stall'] * 1000:.1f} ms",
                flush=True,
            )

    ref = outputs["serial"].astype(mx.float32)
    got = outputs["tier-overlap"].astype(mx.float32)
    diff = mx.max(mx.abs(got - ref))
    rel = mx.linalg.norm(got - ref) / (mx.linalg.norm(ref) + 1e-20)
    mx.eval(diff, rel)
    serial = np.median(samples["serial"])
    overlap = np.median(samples["tier-overlap"])
    print(
        f"batch {batch}: serial {serial * 1000:.1f} ms, "
        f"tier-overlap {overlap * 1000:.1f} ms ({serial / overlap:.2f}x); "
        f"max {float(diff):.3e} rel {float(rel):.3e}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="models/m25.idx")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--batches", default="32,128,512")
    ap.add_argument("--ceiling-gb", type=float, default=1.6)
    ap.add_argument("--hot-share", type=float, default=0.34)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--pipeline",
        action="store_true",
        help="compare cold serial I/O+compute with tier-level overlap",
    )
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    largest = max(batches)
    if args.pipeline:
        if len(batches) != 1:
            raise ValueError("--pipeline takes exactly one batch size")
        pipeline_probe(args, largest)
        return

    store = ArenaStore(
        args.index,
        ceiling_gb=args.ceiling_gb,
        hot_share=args.hot_share,
    )
    moe = ArenaMoE(store, args.layer, gate=None, bias=None, top_k=store.top_k)

    all_inds, all_scores = routes(largest, store.E, store.top_k)
    t0 = time.perf_counter()
    placed = store.slots_for(args.layer, all_inds.reshape(-1).tolist())
    load_time = time.perf_counter() - t0
    print(
        f"Apple Metal {mx.device_info()['device_name']}; layer {args.layer}; "
        f"arena {store.resident / 1e9:.2f} GB; loaded "
        f"{store.bytes_read / 1e9:.2f} GB in {load_time:.3f}s "
        f"({store.bytes_read / load_time / 1e9:.2f} GB/s)"
    )

    first = store.meta[f"L{args.layer}.E0"]
    d = (
        first["gate_proj"]["weight"][4][1]
        * (32 // BITS[first["tier"]])
    )
    for batch in batches:
        inds = all_inds[:batch]
        ii = inds.tolist()
        sc = mx.array(all_scores[:batch]).astype(mx.bfloat16)
        flat = mx.random.normal((batch, d)).astype(mx.bfloat16)
        mx.eval(flat, sc)

        variants = {
            "old-token": lambda: token_gather(
                moe, flat, ii, placed, sc, d, mx.bfloat16, False
            ),
            "production": lambda: moe._apply(
                flat, ii, placed, sc, d, mx.bfloat16
            ),
            "sorted-probe": lambda: token_gather(
                moe, flat, ii, placed, sc, d, mx.bfloat16, True
            ),
            "expert-major": lambda: expert_major(
                moe, flat, ii, placed, sc, d, mx.bfloat16
            ),
        }
        results = {}
        for name, fn in variants.items():
            # First execution compiles shape-specific Metal kernels.
            warm = fn()
            mx.eval(warm)
            out, samples, drift = timed(fn, args.repeats)
            results[name] = (out, samples, drift)

        ref = results["old-token"][0].astype(mx.float32)
        print(f"\nbatch {batch}")
        for name, (out, samples, drift) in results.items():
            diff = mx.max(mx.abs(out.astype(mx.float32) - ref))
            rel = mx.linalg.norm(out.astype(mx.float32) - ref) / (
                mx.linalg.norm(ref) + 1e-20
            )
            mx.eval(diff, rel)
            print(
                f"  {name:14s} median {np.median(samples) * 1000:8.1f} ms  "
                f"range {min(samples) * 1000:7.1f}-{max(samples) * 1000:7.1f}  "
                f"max {float(diff):.3e} rel {float(rel):.3e} "
                f"run-drift {drift:.3e}"
            )

    store.close()


if __name__ == "__main__":
    main()
