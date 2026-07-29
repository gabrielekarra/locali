"""Do the 256 experts of a layer share structure worth factoring out?

Every offloading system in the literature -- MoE-Infinity, HOBBIT, DALI,
SwapMoE, Pre-gated MoE -- treats an expert as atomic. They change WHEN it moves,
WHICH one moves, and at what precision. None of them changes what an expert IS
on disk.

But `bytes_per_expert` is the term the whole cost model hangs off:

    bytes/token = layers x top_k x (1 - hit) x bytes_per_expert

and it is only irreducible if the experts are mutually incompressible. If they
are not -- if a layer's experts are a shared component plus small deviations --
then the shared component is 8 MB per layer, fits in RAM forever, and only the
deviation has to come off the disk.

    W_e = B + D_e         B resident (62 layers x 8 MB = 0.5 GB)
                          D_e streamed

The question is entirely empirical and the answer is in the checkpoint. Three
things get measured here, cheapest first:

  1. mean subtraction: how big is ||W_e - Wbar|| against ||W_e||
  2. rank: the singular spectrum ACROSS experts, which says how many shared
     directions carry the layer
  3. what either buys in bits, at matched reconstruction error

A negative result is worth as much as a positive one: it would say the 4.42 MB
is real and the ceiling in PLAN.md stands.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

GROUP, BITS = 64, 4


def load_experts(snap, layer, proj, rows):
    """Dequantize a slice of one layer's stacked expert tensor.

    A slice, because [256, 1536, 3072] in fp32 is 4.8 GB and the point is a
    statistic, not the tensor. Output rows are sampled rather than the input
    dimension because quantization groups run along the input axis and slicing
    across a group would change what is being measured.
    """
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    base = f"model.layers.{layer}.block_sparse_moe.switch_mlp.{proj}"
    shard = idx["weight_map"][f"{base}.weight"]
    arrs = mx.load(str(snap / shard))
    w, s, b = (arrs[f"{base}.{k}"] for k in ("weight", "scales", "biases"))
    w, s, b = w[:, :rows], s[:, :rows], b[:, :rows]
    deq = mx.dequantize(w, s, b, group_size=GROUP, bits=BITS)
    mx.eval(deq)
    return np.array(deq.astype(mx.float32))          # [E, rows, in]


def report(name, M):
    """M is [E, d]: one row per expert, flattened."""
    E, d = M.shape
    nrm = np.linalg.norm(M, axis=1)

    mean = M.mean(axis=0, keepdims=True)
    resid = M - mean
    rel_mean = np.linalg.norm(resid, axis=1) / nrm

    # Spectrum across experts. E is small (256) so this is exact and cheap:
    # the Gram matrix is [E, E] however wide the experts are.
    G = M @ M.T
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    energy = np.cumsum(ev) / ev.sum()

    # Pairwise cosine between experts, as a sanity read on "shared structure".
    Mn = M / nrm[:, None]
    cos = Mn @ Mn.T
    off = cos[~np.eye(E, dtype=bool)]

    print(f"\n--- {name}   [{E} experts x {d} weights]")
    print(f"  pairwise cosine: mean {off.mean():+.4f}  p95 {np.percentile(off, 95):+.4f}")
    print(f"  after mean subtraction: ||D_e||/||W_e|| = {rel_mean.mean():.4f} "
          f"(min {rel_mean.min():.4f})")
    for r in (1, 2, 4, 8, 16, 32, 64, 128):
        if r < E:
            print(f"  rank {r:3d}: {energy[r-1]*100:6.2f}% of energy captured, "
                  f"residual norm {np.sqrt(max(1-energy[r-1], 0)):.4f}")
    return rel_mean.mean(), energy


def bits_equivalent(rel_resid):
    """What the residual being smaller is worth, in bits per weight.

    Affine quantization error scales with the range being encoded. If the
    residual has k times smaller norm than the original, encoding it to the same
    ABSOLUTE error takes log2(k) fewer bits per weight. This is the whole claim,
    and it is why a small residual translates into fetched bytes rather than
    just into a nice-looking norm.
    """
    if rel_resid <= 0:
        return float("inf")
    return np.log2(1.0 / rel_resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layers", default="1,30,60")
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--rows", type=int, default=48)
    a = ap.parse_args()

    print(__doc__.strip().split("\n\n")[0])
    snap = Path(a.snap)
    out = {}
    for layer in [int(x) for x in a.layers.split(",")]:
        W = load_experts(snap, layer, a.proj, a.rows)
        E = W.shape[0]
        rel, energy = report(f"layer {layer}  {a.proj}", W.reshape(E, -1))
        saved = bits_equivalent(rel)
        print(f"  => mean-subtraction is worth {saved:.2f} bits/weight "
              f"at matched absolute error")
        out[layer] = {"rel_resid": float(rel), "bits_saved": float(saved),
                      "energy_r8": float(energy[7]),
                      "energy_r32": float(energy[31])}
        del W

    Path("results/basis_probe.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/basis_probe.json")


if __name__ == "__main__":
    main()
