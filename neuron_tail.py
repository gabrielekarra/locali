"""Does gate-proportional neuron paging have anything to work with? Weight-side screen.

TECHNIQUES.md A rests on one claim: inside a routed expert, the per-neuron
contributions are concentrated enough that fetching the top-m of 1536 keeps most
of the mass. This is the cheap half of that test -- it needs one shard and no
forward pass, so it can run while the checkpoint is still downloading.

An expert is a sum of rank-1 neuron terms:
    E(x) = sum_j  W2[:,j] * silu(w1_j . x) * (w3_j . x)
Under isotropic x the expected magnitude of term j scales with
    s_j = ||W2[:,j]|| * ||w1_j|| * ||w3_j||
which is exactly the STATIC ranking technique A would bake into the pack layout.
So the tail-mass curve of s_j is the free variant's ceiling.

    eps(m) = sum_{j>m} s_j / sum_j s_j

Read it against the null: for a flat spectrum eps(m/d) = 1 - m/d, so eps at half
the neurons is 0.50. Meaningfully below that is structure; at or above it there
is nothing to page and technique A is dead in its cheap form.

What this CANNOT settle: how much extra concentration real per-token activations
add, and whether a per-token oracle beats the static order enough to justify the
0.5 GB sketch. That needs hidden states and comes after the download.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

CHUNK = 16


def neuron_scores(shard, layer, q, n_experts=None):
    """s_j per expert for one MoE layer -> [E, d_ff]."""
    base = f"model.layers.{layer}.block_sparse_moe.switch_mlp"
    arrs = mx.load(str(shard))
    out = []
    E = arrs[f"{base}.gate_proj.weight"].shape[0]
    for i in range(0, n_experts or E, CHUNK):
        hi = min(i + CHUNK, n_experts or E)
        deq = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            deq[proj] = mx.dequantize(
                arrs[f"{base}.{proj}.weight"][i:hi],
                arrs[f"{base}.{proj}.scales"][i:hi],
                arrs[f"{base}.{proj}.biases"][i:hi],
                group_size=q["group_size"], bits=q["bits"]).astype(mx.float32)
        # gate/up are [chunk, d_ff, d_model] -> neuron j is a ROW
        # down is     [chunk, d_model, d_ff] -> neuron j is a COLUMN (hence the
        # transposed layout technique A requires on disk)
        n1 = mx.linalg.norm(deq["gate_proj"], axis=-1)
        n3 = mx.linalg.norm(deq["up_proj"], axis=-1)
        n2 = mx.linalg.norm(deq["down_proj"], axis=-2)
        s = n1 * n3 * n2
        mx.eval(s)
        out.append(s)
        del deq
    return mx.concatenate(out, axis=0)


def tail_curve(s, fracs, l2=False):
    """Discarded mass when the top m neurons are kept, averaged over experts.

    l2=False: sum of |contributions| -- an UPPER bound on the error, since it
              assumes every discarded term points the same way.
    l2=True:  ||discarded|| / ||total|| assuming the discarded directions are
              uncorrelated, which is the realistic error for an expert whose
              W2 columns are near-orthogonal. This is the number that matters.
    """
    srt = mx.sort(s, axis=-1)[:, ::-1]
    if l2:
        srt = srt ** 2
    csum = mx.cumsum(srt, axis=-1)
    tot = csum[:, -1:]
    d = s.shape[-1]
    out = {}
    for f in fracs:
        frac = 1.0 - csum[:, int(f * d) - 1] / tot[:, 0]
        out[f] = float(mx.mean(mx.sqrt(frac) if l2 else frac))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", default="1")
    ap.add_argument("--experts", type=int, default=None, help="subset, for a fast look")
    a = ap.parse_args()

    q = json.loads(Path(a.config).read_text())["quantization"]
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.8, 0.9]
    rows = {}
    for layer in [int(x) for x in a.layers.split(",")]:
        s = neuron_scores(Path(a.shard), layer, q, a.experts)
        rows[layer] = (tail_curve(s, fracs), tail_curve(s, fracs, l2=True))
        # concentration summary: how many neurons hold 90% of the mass
        srt = mx.sort(s, axis=-1)[:, ::-1]
        c = mx.cumsum(srt, axis=-1) / mx.cumsum(srt, axis=-1)[:, -1:]
        n90 = float(mx.mean(mx.sum((c < 0.9).astype(mx.float32), axis=-1))) + 1
        print(f"L{layer}: experts {s.shape[0]}, d_ff {s.shape[1]}, "
              f"neurons holding 90% of mass: {n90:.0f} ({n90/s.shape[1]*100:.0f}%)")

    print(f"\n{'m/d_ff':>8} " + " ".join(f"{f:>7.0%}" for f in fracs))
    print(f"{'null':>8} " + " ".join(f"{1-f:>7.2f}" for f in fracs) + "   <- flat spectrum")
    for layer, (r1, r2) in rows.items():
        print(f"{'L'+str(layer)+' L1':>8} " + " ".join(f"{r1[f]:>7.2f}" for f in fracs))
    print(f"{'null L2':>8} " + " ".join(f"{(1-f)**0.5:>7.2f}" for f in fracs))
    for layer, (r1, r2) in rows.items():
        print(f"{'L'+str(layer)+' L2':>8} " + " ".join(f"{r2[f]:>7.2f}" for f in fracs))
    print("\nL1 rows bound the error above; L2 rows are the realistic relative")
    print("output error. Both under the STATIC weight-norm ranking.")


def _self_check():
    """The curve must read the null correctly, and detect a planted heavy tail."""
    flat = mx.ones((4, 1000))
    c = tail_curve(flat, [0.25, 0.5])
    cl2 = tail_curve(flat, [0.5], l2=True)
    assert abs(cl2[0.5] - 0.5 ** 0.5) < 0.01, cl2
    assert abs(c[0.5] - 0.5) < 0.01 and abs(c[0.25] - 0.75) < 0.01, c
    # power-law s_j ~ j^-2: the top 10% should hold the overwhelming majority
    j = mx.arange(1, 1001, dtype=mx.float32)
    heavy = mx.broadcast_to((j ** -2.0)[None], (4, 1000))
    ch = tail_curve(heavy, [0.1, 0.5])
    # eps falls as m grows: keeping more neurons discards less.
    assert ch[0.1] < 0.05 and ch[0.1] > ch[0.5], ch
    assert c[0.25] > c[0.5], c
    print(f"self-check ok: flat eps(0.5)={c[0.5]:.3f}, power-law eps(0.1)={ch[0.1]:.4f}")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
