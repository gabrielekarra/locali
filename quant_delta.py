"""What 2-bit experts cost, measured without loading either checkpoint.

The honest gate for ladder step 2 is perplexity, and perplexity needs a full
forward. Neither checkpoint fits: 128.7 GB at 4-bit, 72.5 GB at 2-bit, against
24 GB of RAM. Loading one anyway is the exact failure that has already taken
this machine down once, so it is not attempted here -- true PPL waits for the
streaming engine.

What IS measurable safely: feed one MoE block the SAME real hidden states with
4-bit and with 2-bit experts and compare its output. That is the error which
then propagates through the remaining layers, so it bounds the quality question
from below and arrives today.

Reported per layer:
  block err 2bit         relative error of y from requantization alone
  block err 2bit + GNP   the two stacked, since the ladder applies both
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil

from gnp_gamma import moe_exact_and_truncated
from neuron_tail_live import build_n_layers, capture


def load_expert_stack(path: Path, layer: int, q: dict):
    """The three switch_mlp tensors of one layer, straight off disk."""
    base = f"model.layers.{layer}.block_sparse_moe.switch_mlp"
    index = json.loads((path / "model.safetensors.index.json").read_text())
    want = [f"{base}.{p}.{a}" for p in ("gate_proj", "up_proj", "down_proj")
            for a in ("weight", "scales", "biases")]
    out = {}
    for shard in sorted({index["weight_map"][k] for k in want}):
        arrs = mx.load(str(path / shard))
        for k in want:
            if k in arrs:
                proj, arr = k.rsplit(".", 2)[-2:]
                out[(proj, arr)] = arrs[k]
    return out


def block_out(stack, x_t, inds_t, g, bits, group):
    """y = sum_e g_e * E_e(x). `stack` is keyed (proj, arr) so both checkpoints
    can be passed through the same code path."""
    dq = lambda p, i: mx.dequantize(
        stack[(p, "weight")][i], stack[(p, "scales")][i], stack[(p, "biases")][i],
        group_size=group, bits=bits).astype(mx.float32)
    y = mx.zeros((x_t.shape[0],))
    for pos, e in enumerate(inds_t.tolist()):
        a = nn.silu(dq("gate_proj", e) @ x_t) * (dq("up_proj", e) @ x_t)
        y = y + float(g[pos]) * (dq("down_proj", e) @ a)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src4", required=True)
    ap.add_argument("--src2", required=True)
    ap.add_argument("--layers", default="1,3")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--text", default="eval/pride_prejudice.txt")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    s4, s2 = Path(a.src4), Path(a.src2)
    tok = AutoTokenizer.from_pretrained(str(s4))
    ids = mx.array(tok(Path(a.text).read_text()[:20000])["input_ids"][:a.tokens])

    print(f"\n{'layer':>6} {'2bit':>10} {'2bit+GNP':>10} {'GNP alone':>10}")
    rows = {}
    for layer in [int(x) for x in a.layers.split(",")]:
        need = 6 + 2.1 * (layer + 1)
        assert psutil.virtual_memory().available / 1e9 > need, "not enough RAM"
        model, cfg = build_n_layers(s4, layer + 1)
        grab, blk = capture(model, ids, layer)
        x, inds, gates = grab["x"][0], grab["inds"][0], grab["gates"][0]
        d_ff = cfg["intermediate_size"]

        st2 = load_expert_stack(s2, layer, None)
        st4 = _stack4(blk)
        e2, egnp = [], []
        for t in range(x.shape[0]):
            xt = x[t].astype(mx.float32)
            g = gates[t].astype(mx.float32)
            g = g / (mx.sum(g) + 1e-20)
            y4 = block_out(st4, xt, inds[t], g, 4, 64)
            y2 = block_out(st2, xt, inds[t], g, 2, 64)
            ex, tr, _ = moe_exact_and_truncated(blk, xt, inds[t],
                                                gates[t].astype(mx.float32),
                                                a.gamma, d_ff)
            mx.eval(y4, y2, ex, tr)
            n = float(mx.linalg.norm(y4)) + 1e-20
            e2.append(float(mx.linalg.norm(y2 - y4)) / n)
            egnp.append(float(mx.linalg.norm(tr - ex)) / (float(mx.linalg.norm(ex)) + 1e-20))
        r = {"e2": sum(e2) / len(e2), "gnp": sum(egnp) / len(egnp)}
        r["combined"] = (r["e2"] ** 2 + r["gnp"] ** 2) ** 0.5   # independent errors
        rows[layer] = r
        print(f"{layer:>6} {r['e2']:>10.4f} {r['combined']:>10.4f} {r['gnp']:>10.4f}",
              flush=True)
        del model, st2

    out = Path("results") / "quant_delta.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"gamma": a.gamma, "tokens": a.tokens, "rows": rows},
                              indent=2))
    print(f"\nwrote {out}")
    print("relative error of the MoE block output vs the 4-bit reference.")
    print("combined assumes the two error sources are independent (RSS).")


def _stack4(blk):
    """The already-loaded 4-bit stack, in the same (proj, arr) keying."""
    sm = blk.switch_mlp
    return {(n, a): getattr(m, a)
            for n, m in (("gate_proj", sm.gate_proj), ("up_proj", sm.up_proj),
                         ("down_proj", sm.down_proj))
            for a in ("weight", "scales", "biases")}


if __name__ == "__main__":
    main()
