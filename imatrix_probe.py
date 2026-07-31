"""Does an activation-weighted 2-bit quantization beat the RTN 2-bit we ship?

`requantize_m25.py` builds the cold tier with `mx.quantize(deq, group_size=64,
bits=2)`: plain affine round-to-nearest, scale and bias taken from each group's
min/max. Every input channel counts the same. They do not: the MoE block
computes W @ x, so an error in column i of W is scaled by x_i, and the hidden
states entering layer 1 are anything but isotropic.

DS4 collects exactly this statistic (`gguf-tools/imatrix/`) and reports the
resulting Q2 models as "verified to be actually high quality". This measures
whether the same idea pays here, on OUR checkpoint, before anything is rebuilt.

The weighted problem, per group of `group` input channels and per output row:

    minimise  sum_i  imp_i * (W_i - (s * q_i + b))^2      q_i integer in [0, 2^b-1]

which is the same storage as RTN -- one scale, one bias, the same packed q --
so the comparison is bit-for-bit fair on size. Solved by scanning a grid of
range shrinks, then refitting (s, b) by weighted least squares with q fixed,
twice. `--self-check` asserts that uniform importance reproduces RTN.

WHAT IS AND IS NOT MEASURED
  - Calibration tokens and evaluation tokens are DISJOINT. `shadow_moe_probe`
    is the cautionary tale here: train 0.18 against test 0.64.
  - gate_proj and up_proj share one importance vector, estimated from every
    calibration token rather than per expert. An expert only ever sees the
    tokens routed to it, so this is biased; at a few tokens per expert the
    per-expert estimate is worse, and the bias is the better trade. down_proj
    IS per expert -- its input is that expert's own SwiGLU hidden state, and
    nothing else can stand in for it.
  - The error reported is the relative error of the MoE BLOCK output at layer
    1, against the 4-bit block, on real routed activations. Same quantity
    `quant_delta.py` reports, so the RTN column is comparable to the 5.8% in
    the README.

SAFETY: builds a TWO-LAYER model, as `neuron_tail_live.py` does. The full
checkpoint is 128.7 GB against 24 GB of RAM and loading it takes the machine
down.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil

from neuron_tail_live import build_n_layers, capture

# Range shrinks tried before the least-squares refit. RTN is alpha = 1.0; the
# weighted optimum is almost always tighter, because the min/max of a group are
# usually low-importance outliers that RTN spends its whole range on.
ALPHAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)


def _wls(q, w, p):
    """Weighted least squares for (s, b) in w ~= s*q + b, per group.

    q, w: [..., group]   p: broadcastable importance, same trailing dim.
    Returns s, b with a trailing singleton so they broadcast back over `group`.
    """
    Sp = (p * np.ones_like(q)).sum(-1, keepdims=True)
    Spq = (p * q).sum(-1, keepdims=True)
    Spqq = (p * q * q).sum(-1, keepdims=True)
    Spw = (p * w).sum(-1, keepdims=True)
    Spqw = (p * q * w).sum(-1, keepdims=True)
    det = Spqq * Sp - Spq * Spq
    ok = np.abs(det) > 1e-30
    det = np.where(ok, det, 1.0)
    s = np.where(ok, (Sp * Spqw - Spq * Spw) / det, 0.0)
    b = np.where(ok, (Spqq * Spw - Spq * Spqw) / det, Spw / np.maximum(Sp, 1e-30))
    return s, b


def quantize_weighted(W, imp, bits=2, group=64):
    """Affine quantize [out, in] with per-input-channel importance `imp` [in].

    Returns the DEQUANTIZED matrix. Storage is identical to RTN -- one scale and
    one bias per (row, group) -- so only the reconstruction differs.
    """
    O, I = W.shape
    assert I % group == 0, (I, group)
    G = I // group
    w = W.reshape(O, G, group).astype(np.float32)
    p = imp.reshape(1, G, group).astype(np.float32)
    p = p / (p.mean() + 1e-30)                  # scale-free; keeps det conditioned
    n = float(2 ** bits - 1)

    lo = w.min(-1, keepdims=True)
    hi = w.max(-1, keepdims=True)
    mid = 0.5 * (lo + hi)
    span = hi - lo

    best_err = None
    best = None
    for alpha in ALPHAS:
        rng = np.maximum(span * alpha, 1e-12)
        s = rng / n
        b = mid - 0.5 * rng
        for _ in range(2):
            q = np.clip(np.rint((w - b) / np.maximum(s, 1e-30)), 0.0, n)
            s, b = _wls(q, w, p)
            s = np.where(np.abs(s) < 1e-30, 1e-30, s)
        q = np.clip(np.rint((w - b) / s), 0.0, n)
        deq = s * q + b
        err = (p * (w - deq) ** 2).sum(-1, keepdims=True)
        if best_err is None:
            best_err, best = err, deq
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best = np.where(take, deq, best)
    return best.reshape(O, I)


def quantize_rtn(W, bits=2, group=64):
    """The shipped path: mx.quantize / mx.dequantize, exactly as requantize_m25."""
    a = mx.array(W)
    w, s, b = mx.quantize(a, group_size=group, bits=bits)
    return np.asarray(mx.dequantize(w, s, b, group_size=group, bits=bits)
                      .astype(mx.float32))


def block_out(dq, inds_t, g, x_t):
    """y = sum_e g_e * down_e(silu(gate_e x) * up_e x), all float32 numpy."""
    y = None
    for pos, e in enumerate(inds_t):
        W1, W3, W2 = dq(e)
        z1 = W1 @ x_t
        a = (z1 / (1.0 + np.exp(-z1))) * (W3 @ x_t)
        term = float(g[pos]) * (W2 @ a)
        y = term if y is None else y + term
    return y


def relerr(y, ref):
    return float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--cal-tokens", type=int, default=256)
    ap.add_argument("--eval-tokens", type=int, default=8)
    ap.add_argument("--bits-list", default="2,3",
                    help="bit widths to sweep; the cold tier ships at 2")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--text", default="eval/pride_prejudice.txt")
    ap.add_argument("--need-gb", type=float, default=9.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    vm = psutil.virtual_memory()
    # macOS reports inactive pages as used; they are reclaimable and the model
    # build will take them. `available` alone is far too pessimistic here.
    headroom = (vm.available + getattr(vm, "inactive", 0)) / 1e9
    assert headroom > a.need_gb, (
        f"only {headroom:.1f} GB reclaimable (available {vm.available/1e9:.1f}), "
        f"need {a.need_gb}; close things first")

    from transformers import AutoTokenizer
    snap = Path(a.snap)
    tok = AutoTokenizer.from_pretrained(str(snap))
    text = Path(a.text).read_text()
    n_tok = a.cal_tokens + a.eval_tokens
    ids = mx.array(tok(text[:200 * n_tok])["input_ids"][:n_tok])
    assert ids.size >= n_tok, f"only {ids.size} tokens in {a.text}"

    print(f"building {a.layer + 1} layers ({headroom:.1f} GB reclaimable)", flush=True)
    model, cfg = build_n_layers(snap, a.layer + 1)
    grab, blk = capture(model, ids, a.layer)
    x = np.asarray(grab["x"][0].astype(mx.float32))          # [T, d_model]
    inds = np.asarray(grab["inds"][0])                        # [T, k]
    gates = np.asarray(grab["gates"][0].astype(mx.float32))   # [T, k]
    T, K = inds.shape
    cal = slice(0, a.cal_tokens)
    ev = slice(a.cal_tokens, T)
    print(f"layer {a.layer}: {T} tokens, top-{K}, d_model {x.shape[1]}, "
          f"{a.cal_tokens} calibration / {T - a.cal_tokens} evaluation", flush=True)

    sm = blk.switch_mlp
    mods = (sm.gate_proj, sm.up_proj, sm.down_proj)

    def deq4(e):
        """The 4-bit reference triple for expert e, as float32 numpy."""
        out = []
        for m in mods:
            d = mx.dequantize(m.weight[e], m.scales[e], m.biases[e],
                              group_size=m.group_size, bits=m.bits).astype(mx.float32)
            mx.eval(d)
            out.append(np.asarray(d))
        return out

    # ---- importance ---------------------------------------------------------
    # gate/up see x. One vector for the layer, from every calibration token.
    imp_in = (x[cal] ** 2).mean(0)

    eval_experts = sorted({int(e) for e in inds[ev].reshape(-1)})
    print(f"{len(eval_experts)} distinct experts routed by the evaluation tokens",
          flush=True)

    # down sees that expert's own SwiGLU hidden state, so it must be per expert.
    imp_ff = {}
    for e in eval_experts:
        rows = np.where((inds[cal] == e).any(axis=1))[0]
        if rows.size == 0:
            imp_ff[e] = None                 # never calibrated -> stays RTN
            continue
        W1, W3, _ = deq4(e)
        z1 = x[cal][rows] @ W1.T
        h = (z1 / (1.0 + np.exp(-z1))) * (x[cal][rows] @ W3.T)
        imp_ff[e] = (h ** 2).mean(0)

    cov = sum(1 for e in eval_experts if imp_ff[e] is not None)
    seen = [int((inds[cal] == e).any(axis=1).sum()) for e in eval_experts]
    print(f"down_proj calibrated for {cov}/{len(eval_experts)} experts, "
          f"median {int(np.median(seen))} calibration tokens each", flush=True)

    # ---- quantize each evaluation expert three ways --------------------------
    # The `uniform` control -- same solver, importance switched off -- was run
    # once and settled the question it existed for: at 2 bits on layer 1 it
    # measured 0.2938 against RTN's 0.2696, i.e. the alpha grid plus the
    # least-squares refit HURTS on real weights even though it helps on random
    # ones. So none of the imatrix gain is the solver, and the control is not
    # rerun here. `results/imatrix_L1.json` keeps that measurement.
    bits_list = [int(b) for b in a.bits_list.split(",")]
    base = min(bits_list)
    cache4 = {}
    q_rtn = {b: {} for b in bits_list}
    q_im = {b: {} for b in bits_list}
    cache_mix = {}
    for n, e in enumerate(eval_experts, 1):
        W1, W3, W2 = deq4(e)
        cache4[e] = (W1, W3, W2)
        ff = imp_ff[e]
        for b in bits_list:
            q_rtn[b][e] = tuple(quantize_rtn(W, b, a.group) for W in (W1, W3, W2))
            q_im[b][e] = (
                quantize_weighted(W1, imp_in, b, a.group),
                quantize_weighted(W3, imp_in, b, a.group),
                quantize_weighted(W2, ff, b, a.group) if ff is not None
                else q_rtn[b][e][2],
            )
        # DS4's asymmetry: aggressive on gate/up, one tier better on down.
        if base + 1 in q_im:
            cache_mix[e] = (q_im[base][e][0], q_im[base][e][1], q_im[base + 1][e][2])
        if n % 8 == 0 or n == len(eval_experts):
            print(f"  quantized {n}/{len(eval_experts)}", flush=True)

    # ---- block error on the held-out tokens ---------------------------------
    schemes = {}
    for b in bits_list:
        schemes[f"rtn{b}"] = (float(b), q_rtn[b])
        schemes[f"imatrix{b}"] = (float(b), q_im[b])
    if cache_mix:
        schemes[f"imatrix{base}_down{base + 1}"] = (
            (2.0 * base + (base + 1)) / 3.0, cache_mix)

    rows = {k: [] for k in schemes}
    for t in range(a.cal_tokens, T):
        xt, it, gt = x[t], [int(v) for v in inds[t]], gates[t]
        ref = block_out(lambda e: cache4[e], it, gt, xt)
        for name, (_, cache) in schemes.items():
            rows[name].append(
                relerr(block_out(lambda e: cache[e], it, gt, xt), ref))

    res = {k: float(np.mean(v)) for k, v in rows.items()}

    print(f"\nlayer {a.layer} MoE block, relative error vs the 4-bit block, "
          f"{T - a.cal_tokens} held-out tokens, {len(eval_experts)} experts\n")
    print(f"  {'scheme':<28} {'bits/w':>7} {'rel err':>9}  {'vs RTN':>7}")
    for name, (bw, _) in schemes.items():
        b = int(name.replace("imatrix", "").replace("rtn", "").split("_")[0])
        ref_rtn = res.get(f"rtn{b}")
        gain = f"{ref_rtn / (res[name] + 1e-20):.2f}x" if ref_rtn else ""
        print(f"  {name:<28} {bw:>7.2f} {res[name]:>9.4f}  {gain:>7}")

    out = Path(a.out or f"results/imatrix_L{a.layer}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "layer": a.layer, "bits_list": bits_list, "group": a.group,
        "cal_tokens": a.cal_tokens, "eval_tokens": T - a.cal_tokens,
        "eval_experts": len(eval_experts), "down_calibrated": cov,
        "bits_per_weight": {k: v[0] for k, v in schemes.items()},
        "block_err": res, "per_token": rows,
    }, indent=2))
    print(f"wrote {out}")


def _self_check():
    """Uniform importance must reproduce RTN, and real importance must beat it
    on the weighted objective it optimises."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 256)).astype(np.float32)
    imp = np.ones(256, dtype=np.float32)

    d_rtn = quantize_rtn(W, bits=2, group=64)
    d_uni = quantize_weighted(W, imp, bits=2, group=64)
    e_rtn = np.linalg.norm(d_rtn - W) / np.linalg.norm(W)
    e_uni = np.linalg.norm(d_uni - W) / np.linalg.norm(W)
    assert e_uni <= e_rtn * 1.02, (e_uni, e_rtn)
    print(f"self-check: uniform-weight fit {e_uni:.4f} vs mx RTN {e_rtn:.4f}")

    # A skewed importance vector: the fit must win on ITS objective.
    skew = (rng.random(256).astype(np.float32) ** 4) + 1e-3
    d_skew = quantize_weighted(W, skew, bits=2, group=64)
    obj = lambda d: float((skew * (W - d) ** 2).sum())
    assert obj(d_skew) < obj(d_rtn), (obj(d_skew), obj(d_rtn))
    print(f"self-check: weighted objective {obj(d_skew):.1f} vs RTN {obj(d_rtn):.1f}")

    # Bit depth must move the unweighted error monotonically.
    prev = None
    for b in (2, 3, 4):
        e = np.linalg.norm(quantize_weighted(W, imp, b, 64) - W) / np.linalg.norm(W)
        assert prev is None or e < prev, (b, e, prev)
        prev = e
    print("self-check ok")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
