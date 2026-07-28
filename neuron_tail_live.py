"""The half of the GNP test that the weight-side screen could not settle.

neuron_tail.py ranked neurons by ||W2[:,j]||*||w1_j||*||w3_j||, which is the
right expectation only if x is isotropic. It is not: hidden states live on a
low-dimensional manifold, so how well w1_j aligns with that manifold varies per
neuron and is invisible to weight norms. That alignment is the whole reason MoNE
can report "most neuron activations are near zero". So: real activations.

Measures, on real hidden states entering a real MoE block, for the experts the
router actually picked:

    eps_2(m)  static  -- neurons ordered once, by mean contribution over tokens
              oracle  -- neurons ordered per token, by that token's contribution

static is what a pack layout can bake in for free. oracle is the ceiling any
runtime predictor could reach. The GAP between them prices the 0.5 GB sketch:
no gap means the sketch is pointless, a large gap means static paging is the
wrong design and only the sketch can work.

SAFETY: M2.5 is 128.7 GB and this machine has 24. Loading the whole model would
OOM the Mac. This builds a TWO-LAYER model and loads only those tensors, so the
cap is structural rather than a promise about laziness. Layers 0-1 of a 2-layer
build compute exactly what layers 0-1 of the full model compute, so the layer-1
MoE input is the real one.
"""

import argparse
import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx_lm.models.minimax import Model, ModelArgs
from transformers import AutoTokenizer

NEED_GB = 8


def build_n_layers(snap: Path, n: int = 2):
    cfg = json.loads((snap / "config.json").read_text())
    cfg["num_hidden_layers"] = n
    if isinstance(cfg.get("attn_type_list"), list):
        cfg["attn_type_list"] = cfg["attn_type_list"][:n]
    q = cfg.pop("quantization")
    model = Model(ModelArgs.from_dict(cfg))

    keep = {}
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    # norm/lm_head live in the last shard and are never reached: the capture
    # happens inside layer 1, so the forward stops there.
    want = {k for k in index["weight_map"]
            if k.startswith("model.embed_tokens")
            or any(k.startswith(f"model.layers.{i}.") for i in range(n))}
    for shard in sorted({index["weight_map"][k] for k in want}):
        arrs = mx.load(str(snap / shard))
        keep.update({k: v for k, v in arrs.items() if k in want})

    # This checkpoint is mixed precision: 4-bit everywhere except the 62 router
    # gates at 8-bit, carried as per-module overrides in the quantization dict.
    # Same predicate mlx_lm.utils uses, so the shapes line up.
    def class_predicate(path, m):
        if path in q:
            return q[path]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{path}.scales" in keep

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=class_predicate)
    model.load_weights(list(keep.items()), strict=False)
    mx.eval(model.parameters())
    return model, cfg


def capture(model, ids, layer=1):
    """Run the 2-layer stack, intercept the MoE block of `layer`.

    __call__ resolves on the type, not the instance, so the patch goes on the
    class with an identity guard. Routing is recomputed exactly as
    mlx_lm.models.minimax does it: the correction bias steers SELECTION, while
    the combination weights come from the UNBIASED sigmoid scores.
    """
    blk = model.model.layers[layer].block_sparse_moe
    grab = {}
    cls = type(blk)
    orig = cls.__call__

    def spy(self, x):
        if self is blk:
            scores = mx.sigmoid(self.gate(x.astype(mx.float32)))
            k = self.num_experts_per_tok
            inds = mx.argpartition(-(scores + self.e_score_correction_bias),
                                   kth=k - 1, axis=-1)[..., :k]
            g = mx.take_along_axis(scores, inds, axis=-1)
            grab["x"], grab["inds"] = x, inds
            grab["gates"] = g / (mx.sum(g, axis=-1, keepdims=True) + 1e-20)
        return orig(self, x)

    cls.__call__ = spy
    try:
        h = model.model.embed_tokens(ids[None])
        for i in range(layer + 1):      # stop at the captured layer, no lm_head
            h = model.model.layers[i](h, mask=None, cache=None)
        mx.eval(h)
    finally:
        cls.__call__ = orig
    assert "x" in grab, "layer never reached the MoE block"
    return grab, blk


def eps2(sorted_sq):
    """||discarded|| / ||total|| for every prefix, from squared contributions."""
    cs = mx.cumsum(sorted_sq, axis=-1)
    return mx.sqrt(mx.maximum(1.0 - cs / cs[..., -1:], 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--text", default="eval/pride_prejudice.txt")
    ap.add_argument("--sketch-rank", type=int, default=0,
                    help="rank of the resident W1/W3 sketch used to PREDICT the order")
    ap.add_argument("--sketch-experts", type=int, default=24)
    a = ap.parse_args()

    avail = psutil.virtual_memory().available / 1e9
    assert avail > NEED_GB, f"only {avail:.1f} GB free, need {NEED_GB}; close things first"

    snap = Path(a.snap)
    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(Path(a.text).read_text()[:20000])["input_ids"][:a.tokens])
    model, cfg = build_n_layers(snap, a.layer + 1)
    grab, blk = capture(model, ids, a.layer)

    x, inds, gates = grab["x"][0], grab["inds"][0], grab["gates"][0]
    d_ff = cfg["intermediate_size"]
    T, K = inds.shape

    # per-token, per-routed-expert neuron contribution ||W2[:,j]|| * |a_j|
    W1, W3, W2 = blk.switch_mlp.gate_proj, blk.switch_mlp.up_proj, blk.switch_mlp.down_proj
    dq = lambda m, i: mx.dequantize(m.weight[i], m.scales[i], m.biases[i],
                                    group_size=m.group_size, bits=m.bits).astype(mx.float32)
    contrib = []
    for t in range(T):
        xt = x[t].astype(mx.float32)
        for e in inds[t].tolist():
            act = nn.silu(dq(W1, e) @ xt) * (dq(W3, e) @ xt)
            contrib.append(mx.abs(act) * mx.linalg.norm(dq(W2, e), axis=-2))
        if t % 64 == 0:
            mx.eval(contrib[-1])
    C = mx.stack(contrib)                       # [T*K, d_ff]
    mx.eval(C)

    # The catch the oracle row hides: computing a_j needs W1 and W3, which are
    # 2/3 of the expert's bytes. Fetching them to decide what to fetch saves
    # nothing. A predictor must rank neurons from something already RESIDENT --
    # a rank-r sketch of W1,W3 (r=16 is 31 KB/expert, 0.5 GB for all 15,872).
    # This measures the true error incurred when the PREDICTED order is used.
    # Cheaper predictors than a sketch: the ORDER may be dominated by one factor.
    # Ranking by |w3_j.x| needs only W3 (1/3 of the expert) instead of W1+W3
    # (2/3), so bytes become 1/3 + m*2/3 rather than 2/3 + m/3.
    cheap = {}
    for name in ("w3only", "w1only", "silu_w1"):
        rows = []
        for t in range(T):
            xt = x[t].astype(mx.float32)
            for e in inds[t].tolist():
                z1, z3 = dq(W1, e) @ xt, dq(W3, e) @ xt
                pred = {"w3only": mx.abs(z3), "w1only": mx.abs(z1),
                        "silu_w1": mx.abs(nn.silu(z1))}[name]
                true = mx.abs(nn.silu(z1) * z3) * mx.linalg.norm(dq(W2, e), axis=-2)
                rows.append(mx.take(true, mx.argsort(-pred)))
            if t % 32 == 0:
                mx.eval(rows[-1])
        cheap[name] = eps2(mx.stack(rows) ** 2)
        mx.eval(cheap[name])

    sketch_row = None
    if a.sketch_rank:
        rr, ne = a.sketch_rank, a.sketch_experts
        pairs = [(t, e) for t in range(T) for e in inds[t].tolist()][:ne * 8]
        seen, sk = {}, []
        for t, e in pairs:
            if e not in seen:
                u1, s1, v1 = mx.linalg.svd(dq(W1, e), stream=mx.cpu)
                u3, s3, v3 = mx.linalg.svd(dq(W3, e), stream=mx.cpu)
                seen[e] = ((u1[:, :rr] * s1[:rr]) @ v1[:rr],
                           (u3[:, :rr] * s3[:rr]) @ v3[:rr])
            A1, A3 = seen[e]
            xt = x[t].astype(mx.float32)
            pred = mx.abs(nn.silu(A1 @ xt) * (A3 @ xt))
            true = mx.abs(nn.silu(dq(W1, e) @ xt) * (dq(W3, e) @ xt)) * \
                mx.linalg.norm(dq(W2, e), axis=-2)
            sk.append(mx.take(true, mx.argsort(-pred)))
            mx.eval(sk[-1])
        sketch_row = eps2(mx.stack(sk) ** 2)
        print(f"sketch rank {rr}: {len(seen)} experts, {len(sk)} (token,expert) pairs")

    order = mx.argsort(-mx.mean(C, axis=0))     # static: one order for all tokens
    st = eps2(mx.take(C, order, axis=-1) ** 2)
    orc = eps2(mx.sort(C, axis=-1)[:, ::-1] ** 2)

    fr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.8, 0.9]
    idx = [int(f * d_ff) - 1 for f in fr]
    print(f"\nlayer {a.layer}, {T} tokens x top-{K}, d_ff {d_ff}\n")
    print(f"{'m/d_ff':>8} " + " ".join(f"{f:>7.0%}" for f in fr))
    print(f"{'null':>8} " + " ".join(f"{(1-f)**.5:>7.2f}" for f in fr))
    print(f"{'static':>8} " + " ".join(f"{float(mx.mean(st[:, i])):>7.2f}" for i in idx))
    if sketch_row is not None:
        print(f"{'sketch':>8} " + " ".join(
            f"{float(mx.mean(sketch_row[:, i])):>7.2f}" for i in idx))
    for name, row in cheap.items():
        print(f"{name:>8} " + " ".join(f"{float(mx.mean(row[:, i])):>7.2f}" for i in idx))
    print(f"{'oracle':>8} " + " ".join(f"{float(mx.mean(orc[:, i])):>7.2f}" for i in idx))
    print("\nrelative output error of one expert when only the top-m neurons are")
    print("fetched. static = free (baked into the pack); oracle = ceiling for any")
    print("runtime predictor. Their gap is what the 0.5 GB sketch would have to buy.")

    out = Path("results") / f"neuron_tail_live_L{a.layer}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "layer": a.layer, "tokens": T, "top_k": K, "d_ff": d_ff,
        "static": {str(f): float(mx.mean(st[:, i])) for f, i in zip(fr, idx)},
        "oracle": {str(f): float(mx.mean(orc[:, i])) for f, i in zip(fr, idx)},
    }, indent=2))
    print(f"\nwrote {out}")


def _self_check():
    c = mx.array([[4.0, 3.0, 0.0, 0.0]])
    e = eps2(c ** 2)
    assert abs(float(e[0, 0]) - 0.6) < 1e-5, float(e[0, 0])   # ||3||/||5||
    assert float(e[0, 1]) < 1e-6 and float(e[0, -1]) < 1e-6
    flat = mx.ones((1, 100))
    assert abs(float(eps2(flat ** 2)[0, 49]) - 0.5 ** 0.5) < 0.01
    print("self-check ok")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
