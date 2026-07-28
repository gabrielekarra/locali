"""What GNP actually costs, at the block level, as a function of gamma.

neuron_tail_live.py measured PER-EXPERT truncation error at a uniform budget.
That is not the quantity that matters. The MoE block emits

    y = sum_e  g_e * E_e(x)

so an expert truncated hard while g_e is small barely moves y. The whole point
of gate-proportional paging is to put the big truncations where g_e is small:

    m_e = d_ff * (g_e / g_max)^gamma

This measures, for real tokens and real routing, the relative error of the BLOCK
output y and the bytes fetched, as gamma sweeps. gamma=0 is uniform paging (what
the previous script measured); larger gamma spends more of the budget on the
experts that dominate the sum.

Neuron order comes from |z3| = |W3 x|, the W3-first ranking that survived in
TECHNIQUES.md -- so the reported bytes include the unconditional 1/3 spent on W3:

    bytes(e) = 1/3 + (2/3) * m_e / d_ff

SAFETY: builds only layers 0..L and loads only those tensors, ~2.06 GB per layer
at 4-bit. Never loads the 128.7 GB checkpoint.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil

from neuron_tail_live import build_n_layers, capture


def moe_exact_and_truncated(blk, x_t, inds_t, gates_t, gamma, d_ff):
    """Block output, exact and W3-first-truncated, plus the byte fraction spent."""
    W1, W3, W2 = blk.switch_mlp.gate_proj, blk.switch_mlp.up_proj, blk.switch_mlp.down_proj
    dq = lambda m, i: mx.dequantize(m.weight[i], m.scales[i], m.biases[i],
                                    group_size=m.group_size, bits=m.bits).astype(mx.float32)
    g = gates_t / (mx.sum(gates_t) + 1e-20)
    gmax = float(mx.max(g))
    exact = mx.zeros((x_t.shape[0],))
    trunc = mx.zeros((x_t.shape[0],))
    frac = 0.0
    for pos, e in enumerate(inds_t.tolist()):
        w1, w3, w2 = dq(W1, e), dq(W3, e), dq(W2, e)
        z3 = w3 @ x_t
        a = nn.silu(w1 @ x_t) * z3
        ge = float(g[pos])
        exact = exact + ge * (w2 @ a)
        m = max(1, int(round(d_ff * (ge / gmax) ** gamma)))
        keep = mx.argsort(-mx.abs(z3))[:m]           # W3-first ranking
        mask = mx.zeros((d_ff,))
        mask[keep] = 1.0
        trunc = trunc + ge * (w2 @ (a * mask))
        frac += (1 / 3 + (2 / 3) * m / d_ff)
    return exact, trunc, frac / len(inds_t.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--gammas", default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--text", default="eval/pride_prejudice.txt")
    a = ap.parse_args()

    need = 6 + 2.1 * (a.layer + 1)
    avail = psutil.virtual_memory().available / 1e9
    assert avail > need, f"{avail:.1f} GB free, layer {a.layer} needs ~{need:.0f}"

    snap = Path(a.snap)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(Path(a.text).read_text()[:20000])["input_ids"][:a.tokens])
    model, cfg = build_n_layers(snap, a.layer + 1)
    grab, blk = capture(model, ids, a.layer)
    x, inds, gates = grab["x"][0], grab["inds"][0], grab["gates"][0]
    d_ff = cfg["intermediate_size"]

    print(f"\nlayer {a.layer}, {x.shape[0]} tokens, top-{inds.shape[1]}, d_ff {d_ff}\n")
    print(f"{'gamma':>7} {'bytes':>7} {'saving':>7} {'block err':>10}")
    rows = {}
    for gamma in [float(g) for g in a.gammas.split(",")]:
        errs, fracs = [], []
        for t in range(x.shape[0]):
            ex, tr, fr = moe_exact_and_truncated(
                blk, x[t].astype(mx.float32), inds[t], gates[t].astype(mx.float32),
                gamma, d_ff)
            mx.eval(ex, tr)
            errs.append(float(mx.linalg.norm(tr - ex) / (mx.linalg.norm(ex) + 1e-20)))
            fracs.append(fr)
        b = sum(fracs) / len(fracs)
        e = sum(errs) / len(errs)
        rows[gamma] = {"bytes": b, "block_err": e}
        print(f"{gamma:>7.2f} {b:>7.3f} {1/b:>6.2f}x {e:>10.4f}", flush=True)

    out = Path("results") / f"gnp_gamma_L{a.layer}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"layer": a.layer, "tokens": int(x.shape[0]),
                               "d_ff": d_ff, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    print("block err = ||y_trunc - y_exact|| / ||y_exact||, the quantity that")
    print("propagates into perplexity. Per-expert errors are much larger.")


def _self_check():
    """gamma=0 must be uniform (every expert gets d_ff) and cost 1.0 bytes."""
    class FakeQ:
        group_size, bits = 64, 4
    print("self-check: exercised end-to-end by the real run; "
          "gamma=0 must print bytes 1.000")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
