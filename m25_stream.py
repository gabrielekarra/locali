"""Streaming MoE block for MiniMax-M2.5: experts come off disk, nothing else changes.

The dense part of M2.5 is 4.04B params -- 2.3 GB at 4-bit -- so it stays
resident and only the routed experts stream. That ratio is the entire reason
this model was chosen over Kimi K3, whose dense part alone does not fit.

Two things force a per-expert path rather than mlx_lm's stacked `gather_qmm`:

  1. Mixed precision. The index serves hot experts at 4-bit and cold ones at
     2-bit, and tensors of different bit widths cannot be stacked into one
     [E, out, in] array -- different packed widths, different group counts.
  2. Only top-k experts are resident at any moment, so there is no [E, ...]
     tensor to gather from in the first place.

So each selected expert is dequantized on its own and applied directly. That is
slow and deliberately so: this is the reference implementation whose job is to
be CORRECT, and `verify` below holds it to bit-identity against the resident
model. Speed comes after, and only against something known to be right.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from m25_store import M25Store, PROJS
from neuron_tail_live import build_n_layers

BITS = {"hot": 4, "cold": 2}       # tier -> bits; both checkpoints use group 64
GROUP = 64


class StreamingMoE:
    """Drop-in for MiniMaxSparseMoeBlock.__call__, expert weights from disk."""

    def __init__(self, store: M25Store, layer: int, gate, bias, top_k: int):
        self.store, self.layer, self.gate = store, layer, gate
        self.bias, self.top_k = bias, top_k

    def route(self, x):
        """Exactly mlx_lm.models.minimax: the correction bias steers SELECTION,
        the combination weights come from the UNBIASED sigmoid scores."""
        scores = mx.sigmoid(self.gate(x.astype(mx.float32)))
        inds = mx.argpartition(-(scores + self.bias), kth=self.top_k - 1,
                               axis=-1)[..., :self.top_k]
        g = mx.take_along_axis(scores, inds, axis=-1)
        return inds, g / (mx.sum(g, axis=-1, keepdims=True) + 1e-20)

    def __call__(self, x):
        inds, gates = self.route(x)
        flat = x.reshape(-1, x.shape[-1])
        rows = []
        ii = inds.reshape(-1, self.top_k)
        gg = gates.reshape(-1, self.top_k).astype(mx.float32)
        for t in range(flat.shape[0]):
            xt = flat[t].astype(mx.float32)
            acc = mx.zeros_like(xt)
            for slot, e in enumerate(ii[t].tolist()):
                w = self.store.get(self.layer, e)
                bits = BITS[self.store.tier(self.layer, e)]
                dq = lambda p: mx.dequantize(
                    w[p]["weight"], w[p]["scales"], w[p]["biases"],
                    group_size=GROUP, bits=bits).astype(mx.float32)
                a = nn.silu(dq("gate_proj") @ xt) * (dq("up_proj") @ xt)
                acc = acc + float(gg[t, slot]) * (dq("down_proj") @ a)
            rows.append(acc)
        return mx.stack(rows).reshape(x.shape).astype(x.dtype)


def attach(model, store, layer):
    blk = model.model.layers[layer].block_sparse_moe
    return StreamingMoE(store, layer, blk.gate, blk.e_score_correction_bias,
                        blk.num_experts_per_tok)


def resident_same_path(blk, x, inds, gates, bits=4):
    """The identical arithmetic StreamingMoE does, but reading the weights from
    the resident stacked tensors instead of from disk.

    This is the reference that makes bit-identity meaningful. Comparing against
    mlx_lm's blk(x) would not: that path is gather_qmm on packed weights, a
    genuinely different computation from dequantize-then-matmul, so it differs
    by ~1e-2 no matter how correct the fetch is -- and a tolerance wide enough
    to pass it would hide exactly the indexing bugs this check exists to catch
    (the lesson recorded in 8c58570).
    """
    sm = blk.switch_mlp
    dq = lambda m, i: mx.dequantize(m.weight[i], m.scales[i], m.biases[i],
                                    group_size=GROUP, bits=bits).astype(mx.float32)
    flat = x.reshape(-1, x.shape[-1])
    ii = inds.reshape(-1, inds.shape[-1])
    gg = gates.reshape(-1, inds.shape[-1]).astype(mx.float32)
    rows = []
    for t in range(flat.shape[0]):
        xt = flat[t].astype(mx.float32)
        acc = mx.zeros_like(xt)
        for slot, e in enumerate(ii[t].tolist()):
            a = nn.silu(dq(sm.gate_proj, e) @ xt) * (dq(sm.up_proj, e) @ xt)
            acc = acc + float(gg[t, slot]) * (dq(sm.down_proj, e) @ a)
        rows.append(acc)
    return mx.stack(rows).reshape(x.shape).astype(x.dtype)


def verify(snap, index_path, layer, tokens, ceiling_gb):
    """Streaming must reproduce the resident weights EXACTLY, same math path.

    Bit-identity is only claimed against an index whose experts are the same
    bytes the resident model holds (--hot-frac 1.0). Against the mixed index the
    outputs differ by construction -- the cold experts really are different
    weights -- so that case reports the gap instead of asserting on it.
    """
    from transformers import AutoTokenizer
    snap = Path(snap)
    idx = json.loads(Path(index_path).read_text())
    tiers = {v["tier"] for v in idx["experts"].values()}
    exact = tiers == {"hot"}

    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(Path("eval/pride_prejudice.txt").read_text()[:8000])
                   ["input_ids"][:tokens])
    model, cfg = build_n_layers(snap, layer + 1)
    blk = model.model.layers[layer].block_sparse_moe

    h = model.model.embed_tokens(ids[None])
    for i in range(layer):
        h = model.model.layers[i](h, mask=None, cache=None)
    x = model.model.layers[layer].post_attention_layernorm(
        h + model.model.layers[layer].self_attn(
            model.model.layers[layer].input_layernorm(h), None, None))

    store = M25Store(index_path, ceiling_gb=ceiling_gb)
    sm = attach(model, store, layer)
    inds, gates = sm.route(x)
    got = sm(x)
    ref = resident_same_path(blk, x, inds, gates)
    kern = blk(x)
    mx.eval(ref, got, kern)

    f32 = lambda a: a.astype(mx.float32)
    d = float(mx.max(mx.abs(f32(got) - f32(ref))))
    rel = float(mx.linalg.norm(f32(got) - f32(ref)) /
                (mx.linalg.norm(f32(ref)) + 1e-20))
    dk = float(mx.max(mx.abs(f32(ref) - f32(kern))))
    s = store.stats()
    print(f"layer {layer}, {tokens} tokens, ceiling {ceiling_gb} GB, "
          f"index tiers {sorted(tiers)}")
    print(f"  vs resident, same math path: max abs {d:.3e}  relative {rel:.3e}")
    print(f"  dequant+matmul vs gather_qmm kernel: max abs {dk:.3e} "
          "(different arithmetic, not a defect)")
    print(f"  hit {s['hit_rate']*100:.1f}%  reads {s['bytes_read']/1e6:.0f} MB  "
          f"evictions {s['evictions']}  peak {s['peak']/1e6:.0f} MB")
    if exact:
        assert d == 0.0, f"streaming diverged from resident by {d:.3e}"
        print("  MATCH: bit-identical to the resident block")
    else:
        print(f"  mixed index: {rel:.1%} relative error is the 2-bit cold tier, "
              "expected -- see quant_delta.py")
    store.close()
    return d, rel, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25.idx")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--ceiling-gb", type=float, default=1.0)
    a = ap.parse_args()
    verify(a.snap, a.index, a.layer, a.tokens, a.ceiling_gb)


if __name__ == "__main__":
    main()
