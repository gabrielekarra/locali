# Getting to 20 tok/s

The target is 20 tok/s. This is the analysis of whether it is reachable on this
machine, what is, and in what order to build it. Numbers come from `cost_model.py`
and the measurements in `NOTES.md`; nothing here is estimated from spec sheets.

Current: **1.05 tok/s** decode, 8 GB ceiling, 53.4% hit, 64 tokens.

## The bound

A token touches `62 layers x top-8 = 496` experts. Every one that misses the
cache is a read. That gives a hard relation between target speed and hit rate,
independent of every implementation detail — no amount of overlap, batching or
kernel work changes it, because it is bytes against bandwidth:

| target | byte budget/token | hit rate required |
|---|---|---|
| 2 tok/s | 2330 MB | — (free) |
| 5 tok/s | 932 MB | 57.5% |
| 10 tok/s | 466 MB | 78.7% |
| **20 tok/s** | **233 MB** | **89.4%** |

at 4.66 GB/s and 4.42 MB per 2-bit expert. Measured hit rate is 49.1% at 8 GB
and 61.7% at 10 GB. The cache buys roughly 6 points per GB in that range and the
curve is concave — the earlier study in NOTES puts ~16 GB at 71%. The machine has
24 GB total and must hold a 2.3 GB core.

**89.4% is not reachable.** Not by a better policy either: `cost_model.py`
computes a fractional-knapsack upper bound on *any* static trace-derived pinning
and gets 45.0% at 9 GB, where LRU already measures 53.0%. The cache is not the
thing leaving performance on the table.

## Three floors, and only one of them is the disk

**1. Bytes.** Above. At full quality this caps the machine near 5-7 tok/s.

**2. Kernel launches.** Expert matmul is 14.04 GFLOP/token measured at 0.25
s/token — **56 GFLOP/s on hardware rated in TFLOPs**. It is not FLOP-bound, it is
launch-bound: `62 x 8 x 3 = 1488` matvec kernels per token, each on a
[1,3072]x[3072,1536]. This is why `t_cmp` is 0.36 s/token, and 0.36 alone caps
everything at 2.8 tok/s regardless of the disk.

**3. Host copy.** `numpy->mx` runs at 4-5 GB/s under memory pressure (26 GB/s
idle — it is pressure, not the API; three separate fixes were measured and none
moved it). At 71% hit that is ~0.15 s/token, serial with compute on the main
thread.

Batching does not rescue the first floor. Kernel launches amortise perfectly,
bytes do not, because a batch large enough to matter routes to most of the layer:

| batch | experts/layer | GB/token |
|---|---|---|
| 1 | 8.0 | 2.19 |
| 8 | 57.4 | 1.97 (0.90x) |
| 32 | 163.3 | 1.40 (0.64x) |
| 64 | 222.4 | 0.95 (0.43x) |

and by batch 32 the working set per layer exceeds anything the cache can hold, so
the hit rate collapses at the same time. Batching is worth ~2x on bytes at best,
and it buys latency nothing.

## What is actually reachable, in build order

Each row assumes the ones above it. Predicted from the cost model.

| # | change | mechanism | predicted |
|---|---|---|---|
| 0 | *(today)* | | 1.05 |
| 1 | **stack same-tier experts** | one `gather_qmm` per tier per projection: 1488 launches -> ~370 | ~1.8 |
| 2 | **cross-layer prefetch** | run layer L+1's router on layer L's residual; fetch during L's compute | ~2.6 |
| 3 | **16 GB ceiling** | 71% hit, needs the machine mostly idle | ~3.5 |
| 4 | **expert-major cold pack** | one 4.42 MB read per expert instead of nine scattered | ~4.2 |

**~4 tok/s at full quality.** Four times faster than today, five times short of
the target.

### On #1, the one with the largest single factor

`m25_stream.py` rejects mlx_lm's stacked `gather_qmm` because hot and cold
experts have different bit widths and cannot share an `[E, out, in]` tensor. That
is true, but it forbids one stacked call, not two: group the routed experts by
tier and issue one `gather_qmm` per tier per projection. Worst case 6 kernels per
layer instead of 24, typical case fewer. The cost is a concatenation of the
fetched experts into a stacked buffer — which must be measured before building,
because at 8 x 7.96 MB per layer it could cost more in copies than it saves in
launches. That measurement is the first thing to do.

### On #2, and why it is not the prefetch that already failed

NOTES records a prefetch attempt that gave +10% at 37% accuracy and did not
survive at realistic length. That predicted the *next token's* experts from the
current token's. The proposal here is different and is what DALI does: predict
layer **L+1** from layer **L**'s residual, within the same token. The residual
stream changes slowly across adjacent layers, so layer L's hidden state is a good
approximation of L+1's router input — a cheap 3072x256 matmul gives a candidate
top-8 one layer early, which is exactly the window needed to hide the read behind
the previous layer's compute. Mispredictions cost wasted bandwidth, never
correctness: the true experts are still fetched, just late.

This is the only idea here that attacks the *serialisation* rather than the
bytes, and it composes with everything else.

## If 20 tok/s is mandatory

It requires giving up quality, and the repo already has the instruments to price
each concession (`quant_delta.py`, `gnp_gamma.py`, `neuron_tail_live.py`):

| concession | bytes | measured error |
|---|---|---|
| top-8 -> top-4 | 2.0x | unmeasured — measure first |
| 2-bit -> 1.58-bit cold | ~1.4x | unmeasured, extrapolates poorly from 0.2568 at 2-bit |
| GNP gamma=2 | 1.55x | 7.4% block error, and scattered reads cost bandwidth |

top-4 plus 1.58-bit gets the byte budget to ~0.22 GB/token at 71% hit — 20 tok/s
*on the disk*. But then floors 2 and 3 bind, so it still needs #1 and #2 above,
and lands near 10 tok/s with a model degraded enough that the output is a
different question from the one this project set out to answer.

**The honest framing: 20 tok/s is a hardware statement, not a software one.** A
128 GB Mac holds all 128.7 GB resident and runs at 30+ tok/s with none of this
machinery — NOTES already says the published numbers come from exactly that and
are not comparable. On 24 GB against a 4.66 GB/s disk, the ceiling at full
quality is 5-7 tok/s.

## Four experiments against the bound

The bound is `bytes/token = layers x top_k x (1-hit) x bytes_per_expert`. The
first draft of this document treated every term as given. Three of them were
attacked directly and one assumption turned out to be wrong — but not the one
expected.

### 1. Factor the experts: shared basis + residual — DEAD

Every system in the literature treats an expert as atomic; they change when it
moves, which one, at what precision. None changes what an expert *is* on disk.
If `W_e = B + D_e` with `B` resident (62 layers x 8 MB = 0.5 GB), only `D_e`
streams and `bytes_per_expert` collapses.

`basis_probe.py`, layer 1, gate_proj, 256 experts:

| | |
|---|---|
| mean pairwise cosine | +0.096 |
| after mean subtraction | `\|\|D_e\|\|/\|\|W_e\|\| = 0.9859` |
| rank 1 / 32 / 128 energy | 24.3% / 56.1% / 83.1% |

**The experts are near-orthogonal.** Mean subtraction is worth 0.02 bits per
weight. Reaching a 0.41 residual takes rank 128 of 256, a basis costing more
than the experts it factors. The 4.42 MB is real.

### 2. Skip experts the cache does not have — DEAD

An expert contributes `gate_e * f_e(x)`, and a resident expert is free while a
missing one is 4.42 MB and a stall. So the rule matching the cost structure is
*skip iff not resident AND gate below threshold* — putting the quality loss
exactly where the speed gain is. HOBBIT picks precision by gate magnitude, DALI
picks placement by workload; neither lets residency decide whether to evaluate.

`gate_drop.py` measures the gate distribution first, and it ends the idea:

| slot | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| mean gate | 0.192 | 0.146 | 0.120 | 0.107 | 0.093 |
| cumulative | 19.2% | 33.7% | 58.6% | 80.6% | 100% |

**The gates are nearly flat** — 2x from first to eighth. M2.5 routes sigmoid
top-8, not the skewed softmax top-2 that the adaptive-k literature assumes.
Dropping one expert costs 11.4% block error; keeping four costs 65%, worse than
2-bit quantization. Every one of the eight genuinely matters.

### 3. The cache as its own draft model — DEAD

The resident 8 GB is a complete model at zero disk cost: skip non-resident
experts, renormalise, and a full forward pass reads nothing. Draft with the
cache, verify a batch against the disk. No second model, no extra memory, and it
improves as the cache warms.

Measured at a 5 GB ceiling, 42.9% hit, 24 steps: draft **189 ms/token against
1053** — 5.6x — at **16.7% acceptance**. Speculation needs roughly α > 0.5 to
pay; at 0.167 with k=4 a verify plus four drafts buys 1.2 accepted tokens and is
slower than not drafting. Consequence of #2: skipping 57% of a flat-gated top-8
destroys the output.

### 4. Routing correlation — the assumption that WAS wrong

The batching table above assumed tokens route independently. They do not, and
the 53% hit rate was already saying so: with an 8 GB cache holding ~1800 of
15872 experts, independent routing would hit ~11%.

Measured over 32 decode tokens (`results/seq_trace.json`), union of experts per
layer over B consecutive tokens:

| B | measured | if independent | GB/token |
|---|---|---|---|
| 1 | 8.0 | 8.0 | 2.192 |
| 4 | 22.5 | 30.5 | 1.541 |
| 8 | 35.0 | 57.4 | 1.200 |
| 16 | 54.3 | 102.0 | 0.930 |
| 32 | 81.8 | 163.3 | **0.701** |

33.4% of a token's top-8 is shared with the previous token, and the union grows
at **half** the independent rate. **Batching is worth 3.1x on bytes, not the
1.6x claimed above.** At B=32 with a 53% hit that is ~0.33 GB/token — about 14
tok/s — and B=64 crosses 20.

## Revised conclusion

The three ideas that would have bought single-stream latency all died on the
same fact: **M2.5's router is flat and its experts are orthogonal.** There is no
redundancy to exploit inside a token. For latency, PLAN's ceiling of 4-6 tok/s
stands.

Across tokens there is redundancy, it is large, and it was under-measured.
**20 tok/s is reachable as batch throughput** — B=32-64, which needs the stacked
`gather_qmm` (launch cost amortises perfectly over the batch) and nothing else
new. That is a different product than a fast chat loop: 32 completions at once,
for generation, evaluation, or serving.

## What I would do

Build #1 and #2. They are the two with real factors in them, they are
quality-neutral, and together they take this from 1.05 to roughly 2.6 tok/s —
after which the machine, not the code, is the constraint. Measure the stacking
copy cost before writing #1; it is the assumption the whole gain rests on.
