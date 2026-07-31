"""The half of the basis question that `basis_probe.py` could not answer.

`basis_probe.py` flattens each expert to a row of `[E, out*in]` and reports
pairwise cosine +0.096, rank 128 of 256 for 83% of the energy. That kills
`W_e = B + D_e`. But it is a statement about the MATRICES, and the matrices are
not what the cost model pays for. What a token pays for is the MAP restricted to
the manifold the hidden states actually occupy:

    E_e|_M  :  x  ->  down_e( swiglu(gate_e x, up_e x) )     x in M

Two near-orthogonal matrices can still be strongly correlated as functions on a
low-dimensional M. `neuron_tail_live.py` already relies on M being low
dimensional -- that is the whole reason weight-norm ranking failed there. So the
functional question is open, and it is the one the cost model hangs off.

If a resident set R can synthesise a missing expert,

    E_e(x) ~= sum_{j in R} a_ej E_j(x)                (a: 256 x |R| floats/layer)

then a miss stops being a read. The term (1 - hit) in

    bytes/token = layers x top_k x (1 - hit) x bytes_per_expert

goes to zero by construction rather than by caching, and the problem moves from
the disk onto the GPU -- on the one machine where the experts the GPU needs are
already in the memory it gathers from.

Measures, on real hidden states entering a real MoE block, all 256 experts
evaluated (not just the routed ones):

    1. functional cosine + spectrum across experts -- the direct analogue of
       basis_probe, in function space instead of weight space.
    2. held-out synthesis error of a missing expert from a resident set.
    3. BLOCK error when every non-resident expert in the real top-8 is replaced
       by its synthesis. This is the number comparable to NOTES: 7.4% for the
       4b/2b tier, 11.4% for dropping one expert.

Baselines are reported alongside, because a synthesis that only beats "drop it"
proves nothing: the bar is the tier table, not zero.

SAFETY: M2.5 is 128.7 GB and this machine has 24. Builds a TWO-LAYER model and
loads only those tensors, exactly as `neuron_tail_live.py` does.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil
from transformers import AutoTokenizer

from neuron_tail_live import build_n_layers, capture

NEED_GB = 8


def all_expert_outputs(blk, x, n_exp, chunk=8):
    """Y[t, e, :] = expert e applied to token t, for every expert.

    SwitchGLU takes indices, so passing arange(E) broadcast over tokens
    evaluates the whole layer. Chunked over tokens because the result is
    T x E x d_model and that is the largest array in the probe.
    """
    T = x.shape[0]
    out = np.empty((T, n_exp, x.shape[-1]), dtype=np.float32)
    idx = mx.broadcast_to(mx.arange(n_exp)[None], (chunk, n_exp))
    for s in range(0, T, chunk):
        xs = x[s:s + chunk]
        ii = idx[: xs.shape[0]]
        y = blk.switch_mlp(xs, ii)
        mx.eval(y)
        out[s:s + xs.shape[0]] = np.asarray(y.astype(mx.float32))
        del y
    return out


def functional_cosine(F):
    """F is [E, T*d]: each expert as one vector of its responses on the trace."""
    n = np.linalg.norm(F, axis=1, keepdims=True)
    Fn = F / np.maximum(n, 1e-20)
    C = Fn @ Fn.T
    off = C[~np.eye(F.shape[0], dtype=bool)]
    sv = np.linalg.svd(F, compute_uv=False)
    energy = np.cumsum(sv ** 2) / (sv ** 2).sum()
    return off, energy


def synth(Ytr, Yte, resident, targets, ridge=1e-3):
    """Least-squares synthesis of each target expert from the resident set.

    Fit on train tokens, score on held-out. One coefficient vector per expert,
    shared across tokens -- a per-token fit would need the missing expert to
    know its own answer, which is the thing being avoided.
    """
    Tt, d = Ytr.shape[0], Ytr.shape[2]
    A = Ytr[:, resident, :].transpose(0, 2, 1).reshape(Tt * d, len(resident))
    B = Yte[:, resident, :].transpose(0, 2, 1).reshape(Yte.shape[0] * d, len(resident))
    G = A.T @ A
    G[np.diag_indices_from(G)] += ridge * np.trace(G) / len(resident)
    coef, errs, best1 = {}, [], []
    for e in targets:
        a = np.linalg.solve(G, A.T @ Ytr[:, e, :].reshape(-1))
        t = Yte[:, e, :].reshape(-1)
        r = np.linalg.norm(B @ a - t) / max(np.linalg.norm(t), 1e-20)
        # nearest single resident expert, rescaled: the cheap baseline
        cs = (B.T @ t) / np.maximum(np.linalg.norm(B, axis=0) * np.linalg.norm(t), 1e-20)
        best1.append(1.0 - float(np.max(np.abs(cs))) ** 2)
        coef[e], _ = a, errs.append(float(r))
    return coef, np.array(errs), np.sqrt(np.maximum(np.array(best1), 0))


def block_error(Yte, inds_te, gates_te, resident, coef):
    """Real router, real gates. Non-resident experts replaced by synthesis."""
    rset = set(int(i) for i in resident)
    num, den, drop_num = 0.0, 0.0, 0.0
    for t in range(Yte.shape[0]):
        ex = np.zeros(Yte.shape[2], dtype=np.float32)
        ap = np.zeros_like(ex)
        dp = np.zeros_like(ex)
        for e, g in zip(inds_te[t], gates_te[t]):
            e = int(e)
            ex += g * Yte[t, e]
            if e in rset:
                ap += g * Yte[t, e]
                dp += g * Yte[t, e]
            else:
                ap += g * (Yte[t, resident, :].T @ coef[e])
        num += float(np.sum((ap - ex) ** 2))
        drop_num += float(np.sum((dp - ex) ** 2))
        den += float(np.sum(ex ** 2))
    return np.sqrt(num / den), np.sqrt(drop_num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--train-frac", type=float, default=0.67)
    ap.add_argument("--resident", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--text", default="eval/pride_prejudice.txt")
    ap.add_argument("--out", default="results/func_basis.json")
    a = ap.parse_args()

    avail = psutil.virtual_memory().available / 1e9
    assert avail > NEED_GB, f"only {avail:.1f} GB free, need {NEED_GB}; close things first"

    snap = Path(a.snap)
    tok = AutoTokenizer.from_pretrained(str(snap))
    ids = mx.array(tok(Path(a.text).read_text()[:40000])["input_ids"][: a.tokens])
    model, cfg = build_n_layers(snap, a.layer + 1)
    grab, blk = capture(model, ids, a.layer)
    E = cfg["num_local_experts"]

    x = grab["x"][0]
    inds = np.asarray(grab["inds"][0])
    gates = np.asarray(grab["gates"][0].astype(mx.float32))
    T = x.shape[0]
    print(f"layer {a.layer}: {T} real hidden states, {E} experts, d={x.shape[-1]}")

    # the manifold itself, for context on whatever the functional numbers say
    xn = np.asarray(x.astype(mx.float32))
    sv = np.linalg.svd(xn - xn.mean(0), compute_uv=False)
    en = np.cumsum(sv ** 2) / (sv ** 2).sum()
    r90 = int(np.searchsorted(en, 0.90) + 1)
    print(f"activation manifold: rank {r90} of {min(xn.shape)} for 90% of energy")

    Y = all_expert_outputs(blk, x, E)
    del model, blk
    mx.clear_cache()

    F = Y.transpose(1, 0, 2).reshape(E, -1)
    off, energy = functional_cosine(F)
    print(f"\nFUNCTIONAL cosine across experts: mean {off.mean():+.4f}  "
          f"p95 {np.percentile(off, 95):+.4f}   (weight-space was +0.0960)")
    for r in (8, 16, 32, 64, 128):
        if r <= len(energy):
            print(f"  rank {r:3d}: {energy[r - 1] * 100:6.2f}% of functional energy")

    ntr = int(T * a.train_frac)
    Ytr, Yte = Y[:ntr], Y[ntr:]
    freq = np.bincount(inds[:ntr].ravel(), minlength=E)
    order = np.argsort(-freq)

    rec = {"layer": a.layer, "tokens": T, "train": ntr,
           "manifold_rank90": r90,
           "func_cos_mean": float(off.mean()),
           "func_energy": {str(r): float(energy[r - 1]) for r in (8, 16, 32, 64, 128)},
           "resident": {}}

    for R in a.resident:
        resident = np.sort(order[:R])
        targets = [e for e in range(E) if e not in set(resident.tolist())]
        coef, errs, base1 = synth(Ytr, Yte, resident, targets)
        blk_err, drop_err = block_error(Yte, inds[ntr:], gates[ntr:], resident, coef)
        cov = float(np.isin(inds[ntr:], resident).mean())
        print(f"\nresident {R} (by trace frequency), {len(targets)} targets, "
              f"top-8 coverage {cov * 100:.1f}%")
        print(f"  synthesis rel err  median {np.median(errs):.4f}  "
              f"mean {errs.mean():.4f}  p90 {np.percentile(errs, 90):.4f}")
        print(f"  best single resident expert (baseline)  median {np.median(base1):.4f}")
        print(f"  BLOCK err  synthesis {blk_err:.4f}   drop-missing {drop_err:.4f}")
        rec["resident"][str(R)] = {
            "coverage": cov, "n_targets": len(targets),
            "synth_median": float(np.median(errs)), "synth_mean": float(errs.mean()),
            "base_single_median": float(np.median(base1)),
            "block_err": float(blk_err), "block_err_drop": float(drop_err)}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
