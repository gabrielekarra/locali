# Three techniques the offloading literature does not have

All expert-offloading work surveyed in `K3-DESIGN.md` §3 shares one assumption,
stated plainly in the survey literature:

> "Expert weights are paged at expert granularity, and an entire expert's
> parameters must be resident in GPU memory before its feed-forward computation
> can execute."

That assumption is false, and dropping it is worth more than any cache policy.

---

## The structural fact everything below rests on

A SwiGLU expert is not an atom. It is a **sum of independent rank-1 terms**, one
per intermediate neuron:

```
E(x) = W2 · ( silu(W1 x) ⊙ W3 x )
     = Σ_{j=1..d_ff}  W2[:,j] · a_j ,      a_j = silu(w1_j·x) · (w3_j·x)
```

For MiniMax-M2.5, `d_ff = 1536`. Each neuron `j` owns exactly three vectors —
row `j` of W1, row `j` of W3, column `j` of W2 — totalling `3 × 3072` params,
**2.88 KB at 2-bit**. So an expert is 1536 independently fetchable pieces, and
any subset of them yields a valid partial sum.

Two published results say the tail of that sum is cheap to lose. MoNE
(arXiv 2510.05781) finds "most neuron activations are near zero" and that
pruning up to **60%** of the parameters in that subset costs negligible task
performance. DualSparse-MoE (arXiv 2508.18376) ranks neurons by accumulated
absolute gate value and skips the unimportant ones. Both spend that slack on
**compute**. Nobody spends it on **I/O**, which is the only currency that
matters when the model is on disk.

---

## A. Gate-proportional neuron paging (GNP)

**The idea.** The MoE output is `y = Σ_e g_e E_e(x)`. An expert with gate weight
`g_e = 0.03` contributes 3% of the block output, yet expert-granular paging
spends the same 4.42 MB on it as on the `g_e = 0.30` expert. Allocate *bytes* in
proportion to *contribution* instead.

**The allocation.** Truncating expert `e` to its top `m_e` neurons costs
`g_e · ε_e(m_e)`, where `ε_e` is the discarded tail mass. Minimising
`Σ_e g_e ε_e(m_e)` subject to a byte budget `Σ_e m_e = B` is water-filling; with
a power-law tail `ε ≈ C·m^(-β)` the optimum is `m_e ∝ (g_e C_e)^(1/(1+β))`.
In practice one tunable exponent:

```
m_e = d_ff · (g_e / g_max)^γ        γ = 0  -> today's uniform paging
                                    γ -> ∞ -> top-1 expert only
```

`γ` is a continuous speed/quality dial where today there is only the coarse
integer knob of adaptive top-k (which is just `m_e ∈ {0, d_ff}`).

**Which neurons.** Picking the top `|a_j|` needs `W1, W3` — circular. Two ways out:

1. *Static* — rank neurons once on calibration data by `E[‖W2[:,j]‖·|a_j|]` and
   store each expert with its neurons in that order. A budget of `m` is then
   **one contiguous read of the first m**. Free at runtime.
2. *Sketched* — keep a rank-16 sketch of `W1,W3` resident (16×3072×2 params per
   expert ≈ 31 KB at 2-bit, **0.5 GB for all 15,872 experts**), predict `a_j`,
   fetch the predicted top-m. Costs 0.5 GB of RAM, adapts per token.

**Layout requirement.** Neurons must be contiguous on disk. Rows of W1 and W3
already are; `W2` is `[d_model, d_ff]`, so neuron `j` is a **column** and a
partial fetch would be strided. **Store W2 transposed.** This also makes reads
larger and contiguous, which measurably matters — see C.

### MEASURED 2026-07-28 -- the design above is wrong, and the corrected one works

`neuron_tail.py` and `neuron_tail_live.py`, MiniMax-M2.5 layer 1, real hidden
states, relative L2 output error of one expert when only its top-m neurons are
fetched (`results/neuron_tail_live_L1.json`):

```
  m/d_ff     10%     20%     30%     40%     50%     66%     80%     90%
    null    0.95    0.89    0.84    0.77    0.71    0.58    0.45    0.32
  static    0.86    0.78    0.71    0.65    0.57    0.45    0.33    0.22
  sketch    0.78    0.70    0.64    0.57    0.50    0.40    0.31    0.21
  w3only    0.57    0.44    0.34    0.26    0.19    0.11    0.05    0.02
  oracle    0.41    0.27    0.19    0.13    0.09    0.04    0.02    0.00
```

Three findings, in the order they arrived:

1. **The concentration is real and large.** The oracle keeps 40% of neurons at
   13% error. MoNE's claim transfers to a model not trained for it.
2. **Both proposed rankings failed.** Static weight-norm order lands at 0.65
   where flat is 0.77 -- almost nothing. The rank-16 resident sketch closes only
   ~15% of the static-to-oracle gap; higher ranks do not rescue it, because
   expert matrices are near-full-rank (this repo's `svd_expert_basis.py`) and
   the ranking depends on a *product* of two noisy inner products.
3. **The fix is structural, not statistical.** The oracle needs
   `a_j = silu(w1_j·x)·(w3_j·x)`, and that looked like it required W1 and W3 --
   2/3 of the expert, i.e. fetching what you were trying to avoid. But the
   ORDER is carried almost entirely by the linear factor: ranking on `|w3_j·x|`
   alone reaches 0.26 at m=40% against the oracle's 0.13, closing 75% of the gap
   while reading **only W3, one third of the expert**.

### The corrected technique: W3-first two-stage fetch

Per layer, not per expert, so it costs two round trips per layer total:

1. Fetch `W3` for all top-k experts -- 1/3 of the bytes, unconditional.
2. Compute `z3 = W3 x`, rank neurons by `|z3_j|`.
3. Fetch the top `m_e` rows of `W1` and columns of `W2`, with the per-expert
   budget still set by the router: `m_e = d_ff · (g_e/g_max)^γ`.

Bytes become `1/3 + (2/3)·mean(m_e)` instead of `1`. Measured frontier:

| m | bytes | saving | per-expert error |
|---|---|---|---|
| 30% | 0.533 | **1.88x** | 34% |
| 50% | 0.667 | **1.50x** | 19% |
| 66% | 0.773 | **1.29x** | 11% |

The alternative that needs no ranking at all -- fetch W1+W3, compute the exact
`a_j`, fetch only the needed W2 columns -- is strictly worse: 1.30x at 19%
error where W3-first gives 1.50x at the same error.

These are *per-expert* errors. The block error is `Σ g_e · err_e`, so allocating
m by gate weight puts the large truncations exactly where they are cheapest.

### gamma, measured at the BLOCK level (`gnp_gamma.py`, `results/gnp_gamma_L*.json`)

Per-expert error is the wrong metric: the block emits `y = Σ g_e E_e(x)`, so a
hard truncation on a small-`g_e` expert barely moves `y`. Relative error of the
actual block output, real tokens, real routing:

```
             layer 1                     layer 3
gamma   bytes  saving  block err    bytes  saving  block err
 0.00   1.000   1.00x    0.0000                              <- self-check: exact
 0.50   0.871   1.15x    0.0129     0.902   1.11x    0.0163
 1.00   0.775   1.29x    0.0320     0.825   1.21x    0.0404
 1.50   0.702   1.42x    0.0523
 2.00   0.647   1.55x    0.0741     0.715   1.40x    0.0907
 3.00   0.571   1.75x    0.1148     0.643   1.56x    0.1387
 4.00   0.525   1.91x    0.1523
```

Gate weighting is worth about **3.4x**: at comparable bytes the block error is
3.2% where the uniform per-expert error was 11%. That is the technique working
exactly as designed -- the budget goes where the sum is.

No sharp knee; error grows roughly linearly with saving. **gamma 1.5-2.0**
(1.4-1.55x for 5-9% block error) is the defensible operating range. For scale:
requantizing experts 4-bit -> 2-bit costs ~41% relative weight RMS error, so
GNP's few percent is cheap next to a step already in the plan.

**Open, and it matters:** layer 3 is consistently worse than layer 1 (1.40x vs
1.55x at gamma=2). Two shallow layers is not a model. Measuring layer 30+ needs
all preceding layers resident (~2.06 GB each), which does not fit in 24 GB --
so the deep-layer answer has to wait for the streaming engine itself. If the
degradation with depth continues, the whole-model number will be below 1.4x.

---

## B. Fractional residency

Expert-granular caching is binary, and that produces the hard floor measured in
this repo: below `top-k` slots per layer the hit rate is **zero** (Qwen3-Next,
top-10: 4.1% at 8 slots, 41.9% at 15). Kimi K3 sits far below that floor, which
is why every cache policy in the literature evaluates to a no-op there.

Neuron granularity dissolves the floor: instead of *some* experts fully resident,
keep the **top-p neurons of every expert** resident and stream the tail. Residency
becomes continuous rather than granular.

Honest accounting, because this is weaker than it first looks:

- **MiniMax-M2.5**: LRU already reaches ~76% hit. Fractional residency alone is
  *worse* (p=256 → 11.7 GB resident, 1.83 GB/token cold vs LRU's 0.53). Temporal
  locality beats uniform spreading whenever locality exists.
- **Kimi K3**: 19.3 GB buys only 70 of 3072 neurons per expert — a 2.3% byte
  saving. Also not a win on bytes.

So B does **not** pay as a bandwidth trick. What it buys is a **graceful
degradation dial where none exists**: with LRU at 0% hit the only options are
"fetch everything, exact, slow" or "drop the expert, large error". Fractional
residency lets a model that cannot be cached at all still answer at reduced
fidelity without touching the disk. That is a capability, not a speedup, and it
should be presented as one.

The right composition for M2.5 is **LRU on whole experts, GNP on the misses**:
`0.53 GB × (B/k·d_ff) ≈ 0.27 GB/token`.

---

## C. Co-activation packing

Measured on this machine today (`bw.py`, F_NOCACHE, 40 GB file):

```
block    4.42MB   7.96MB   17.5MB
GB/s       4.00     4.46     4.66
```

Random reads at M2.5's 2-bit expert size run **17% slower** than at 4× that
size, and adding threads does not help (saturated at 8). So `k` scattered
expert-sized reads per layer leave bandwidth on the table.

**Fix:** order experts in the pack file so that experts which co-activate land
contiguously, turning several small reads into one large one. The ordering comes
from the co-activation matrix of a calibration trace (spectral ordering / TSP on
`P(e_i, e_j` co-selected`)`). Purely a layout decision — bit-exact, no quality
cost, ~17% of `t_io`.

This is *LLM in a flash*'s row-column bundling moved up a level: Apple bundles
neurons of a dense FFN, this bundles experts of a MoE. It matters **more** once
GNP is on, because GNP makes individual reads smaller.

---

## What must be measured before any of this is believed

GNP's entire premise is one empirical claim: **the per-token distribution of
`|a_j|` inside a routed expert is heavy-tailed enough that the top-m carry the
mass.** MoNE says yes for models trained that way; M2.5 was not.

The test needs one layer, not the whole checkpoint:

1. Capture real hidden states `x` entering layer 1's MoE block on ~1k tokens.
2. For each routed expert, compute `a_j` and the tail mass curve
   `ε(m) = Σ_{j>m} ‖W2[:,j]‖·|a_j| / Σ_j ‖W2[:,j]‖·|a_j|`.
3. Report `ε(m)` at `m/d_ff ∈ {0.1 … 1.0}`, and — the number that decides it —
   how much of `ε` is predicted by the *static* ranking versus the per-token one.

If static ranking captures most of it, GNP is nearly free. If it does not, GNP
needs the 0.5 GB sketch. If `ε(0.5)` is large in both, GNP is dead and this
document should say so.
