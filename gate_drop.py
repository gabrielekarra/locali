"""What does it cost to simply NOT compute an expert the cache does not have?

Every offloading system treats the router's top-k as a requirement: the expert
was selected, therefore it must be fetched, therefore the token waits. That is
the assumption that makes disk bandwidth set SPEED.

But an expert contributes `gate_e * f_e(x)`, the gates are normalised and
skewed, and the cost of an expert is not uniform -- a resident one is free, a
missing one is 4.42 MB and a stall. So the decision rule that matches the actual
cost structure is not "drop the smallest gates", it is

    skip e  iff  e is NOT resident  AND  gate_e < tau

which puts the quality loss exactly where the speed gain is, and leaves a warm
cache computing the full top-8. HOBBIT picks precision by gate magnitude and
DALI picks placement by workload; neither uses cache residency to decide whether
to evaluate the expert at all.

This script measures the two things that decide whether that works:

  1. the gate mass distribution -- how much of the output the tail carries
  2. block error against the true top-8, as a function of how many experts are
     dropped, both for the oracle order (drop smallest gates) and for the
     cache-aware rule (drop smallest gates AMONG MISSES)

Run before building anything. If the tail carries real mass, the idea is dead
and PLAN.md's ceiling stands.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from neuron_tail_live import build_n_layers

GROUP, BITS = 64, 4


def block_out(sm, x, inds, gates):
    """Reference MoE block on resident weights, same math path as m25_stream."""
    flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    ii = inds.reshape(-1, inds.shape[-1]).tolist()
    gg = gates.reshape(-1, inds.shape[-1]).astype(mx.float32)
    routed = {}
    for t, row in enumerate(ii):
        for slot, e in enumerate(row):
            routed.setdefault(e, []).append((t, slot))
    out = mx.zeros_like(flat)
    for e, uses in routed.items():
        qm = lambda v, m: mx.quantized_matmul(
            v, m.weight[e], m.scales[e], m.biases[e],
            transpose=True, group_size=GROUP, bits=BITS)
        rows = mx.array([t for t, _ in uses])
        xb = flat[rows]
        h = nn.silu(qm(xb, sm.gate_proj)) * qm(xb, sm.up_proj)
        g = mx.array([float(gg[t, slot]) for t, slot in uses])[:, None]
        out = out.at[rows].add(qm(h, sm.down_proj) * g)
    return out.reshape(x.shape)


def keep_mask_oracle(g, keep_k):
    """Drop the smallest gates outright: the standard adaptive-k rule."""
    order = np.argsort(-g, axis=-1)
    m = np.zeros_like(g, dtype=bool)
    np.put_along_axis(m, order[:, :keep_k], True, axis=-1)
    return m


def keep_mask_cache_aware(g, resident, keep_k):
    """Keep every RESIDENT expert regardless of gate, and spend the remaining
    budget on the largest gates among the misses. A hit costs nothing, so there
    is never a reason to drop one."""
    E = g.shape[-1]
    m = resident.copy()
    for t in range(g.shape[0]):
        budget = keep_k - int(m[t].sum())
        if budget <= 0:
            # More resident than the budget: keep them all, they are free.
            continue
        cand = [e for e in range(E) if not m[t, e]]
        cand.sort(key=lambda e: -g[t, e])
        for e in cand[:budget]:
            m[t, e] = True
    return m


def renorm(gates_np, mask):
    kept = gates_np * mask
    s = kept.sum(axis=-1, keepdims=True)
    return np.where(s > 0, kept / np.maximum(s, 1e-20), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--hit", type=float, default=0.53,
                    help="cache hit rate to simulate for the cache-aware rule")
    a = ap.parse_args()

    print(__doc__.strip().split("\n\n")[0])
    snap = Path(a.snap)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(Path("eval/pride_prejudice.txt").read_text()[:8000])
                   ["input_ids"][:a.tokens])

    model, cfg = build_n_layers(snap, a.layer + 1)
    layer = model.model.layers[a.layer]
    blk = layer.block_sparse_moe
    h = model.model.embed_tokens(ids[None])
    for i in range(a.layer):
        h = model.model.layers[i](h, mask=None, cache=None)
    x = layer.post_attention_layernorm(
        h + layer.self_attn(layer.input_layernorm(h), None, None))

    top_k = blk.num_experts_per_tok
    scores = mx.sigmoid(blk.gate(x.astype(mx.float32)))
    inds = mx.argpartition(-(scores + blk.e_score_correction_bias),
                           kth=top_k - 1, axis=-1)[..., :top_k]
    g = mx.take_along_axis(scores, inds, axis=-1)
    gates = g / (mx.sum(g, axis=-1, keepdims=True) + 1e-20)
    mx.eval(inds, gates)

    gn = np.array(gates.reshape(-1, top_k).astype(mx.float32))
    order = np.sort(gn, axis=-1)[:, ::-1]
    print(f"\n--- gate mass, layer {a.layer}, {gn.shape[0]} tokens, top-{top_k}")
    cum = np.cumsum(order, axis=-1).mean(axis=0)
    for i in range(top_k):
        print(f"  top-{i+1}: slot mean {order[:, i].mean():.4f}   "
              f"cumulative {cum[i]*100:5.1f}%")

    ref = block_out(blk.switch_mlp, x, inds, gates)
    mx.eval(ref)
    rel = lambda y: float(mx.linalg.norm(y - ref) / mx.linalg.norm(ref))

    rng = np.random.default_rng(0)
    resident = rng.random(gn.shape) < a.hit      # which of the top-k are cached

    print(f"\n--- block error when experts are dropped "
          f"(cache-aware assumes {a.hit*100:.0f}% resident)")
    print(f"  {'kept':>4}  {'bytes':>6}  {'drop-smallest':>14}  {'cache-aware':>12}")
    out = {}
    for keep_k in range(top_k, 0, -1):
        mo = keep_mask_oracle(gn, keep_k)
        mc = keep_mask_cache_aware(gn, resident, keep_k)
        errs = []
        for m in (mo, mc):
            gm = mx.array(renorm(gn, m).reshape(gates.shape).astype(np.float32))
            y = block_out(blk.switch_mlp, x, inds, gm)
            mx.eval(y)
            errs.append(rel(y))
        # Bytes are only paid for kept experts that MISS.
        miss_o = (mo & ~resident).sum() / mo.shape[0]
        miss_c = (mc & ~resident).sum() / mc.shape[0]
        print(f"  {keep_k:>4}  {miss_c/ (top_k*(1-a.hit)):>5.2f}x  "
              f"{errs[0]:>14.4f}  {errs[1]:>12.4f}")
        out[keep_k] = {"err_drop_smallest": errs[0], "err_cache_aware": errs[1],
                       "miss_per_token_oracle": float(miss_o),
                       "miss_per_token_cache": float(miss_c)}

    Path("results/gate_drop.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/gate_drop.json")


if __name__ == "__main__":
    main()
