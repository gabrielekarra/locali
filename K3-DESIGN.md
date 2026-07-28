# Kimi K3 on 24 GB — research, verdict, and a design from scratch

Target: `moonshotai/Kimi-K3`, 2.78T params, on a MacBook Air M4, 24 GiB RAM,
internal SSD + external NVMe. Written 2026-07-28. Every number below is either
read off the real checkpoint or measured on this machine; nothing is assumed
that could be measured.

---

## 0. Verdict first

**One decoded token requires reading 25.8 GB of expert weights. The machine has
25.8 GB of RAM.** The per-token working set is the size of the whole machine.

That single fact deletes the entire premise this repo was built on. `moe-stream`
is an LRU cache with a hard ceiling, and its deliverable is a hit-rate-vs-ceiling
curve. At K3 scale **that curve is flat at zero** — usable RAM (19.3 GB) holds
2.4 experts per layer against a top-16 router, i.e. it cannot hold even the
current token's own working set for the current layer plus anything reusable.
Every caching, eviction, and prefetching idea in the literature is inert here.

What is left is arithmetic: bytes fetched per token, divided by bytes per second.

Reachable, in order of confidence:

| | GB/token | tok/s | quality |
|---|---|---|---|
| stock MXFP4, internal SSD only | 25.8 | **0.18** | native |
| + 3 drives, bandwidth-proportional placement | 25.8 | **0.42** | native, lossless |
| + 3-bit experts, adaptive top-12, spec. decode γ=4 | 10.9 | **0.94** | degraded |
| + 2.5-bit experts, top-8, 2-bit backbone | 6.2 | **1.64** | badly degraded |

Prefill is ~141 s **regardless of prompt length** (any prompt of a few hundred
tokens touches all 896 experts per layer, so prefill = one full read of the
checkpoint). So: ~2.5 min to ingest, then ~1 tok/s.

**K3 on this machine is an async oracle, not a chatbot.** A 500-token answer is
~10 minutes. If that is acceptable the project is feasible; if you wanted to
type at it, it is not, and no amount of engineering changes that.

Reproduce: `python k3_budget.py`. Its self-check reproduces the only independent
measurement in existence (PipeNetwork `kimi-k3-mlx`, 0.14–0.20 tok/s) from first
principles, which is why I trust the rest of its output.

---

## 1. The checkpoint, measured

From `config.json` and the HF param census
(`BF16 57,179,884,544 / U8 2,722,740,830,208 / total 2,779,931,837,184`):

| | value | note |
|---|---|---|
| layers | 93 (1 dense + **92 MoE**) | `first_k_dense_replace: 1` |
| routed experts | **896**, top-**16** | 1.8% sparsity |
| shared experts | 2 per layer | dense, every token |
| expert shape | 3584 → 3072 → 3584 | `routed_expert_hidden_size: 3584` |
| **params per expert** | **33,030,144** | w1,w3 (3584×3072), w2 (3072×3584) |
| **bytes per expert** | **17,547,264** (17.55 MB) | MXFP4 = 4 bits + uint8 scale/32 = 4.25 bpp |
| routed total | 2,722.7B = **1,446 GB** | 97.9% of the model |
| **non-routed total** | **57.2B, bf16, 114 GB** | *not quantized* — see the `ignore` list |
| repo on disk | **1,561 GB** | 96 shards |

Two things here are not in any write-up I found and both are load-bearing.

**(a) The 594 GB figure circulating in blogs is wrong.** The repo is 1,561 GB.
`1446 GB / 2722.7B = 0.5313 bytes/param` confirms MXFP4-with-scales exactly.

**(b) MXFP4 applies *only* to routed experts.** The quant config's `ignore` list
is `self_attn`, `shared_experts`, `mlp.*_proj`, `lm_head`, `vision_tower`. So the
57.2B always-on params ship in **bf16**. Everyone quotes "K3 is natively 4-bit";
the part that must live in your RAM is not.

Always-on breakdown (text-only, vision dropped): KDA attention ×69 = 30.6B,
shared experts = 12.2B, MLA ×24 = 5.6B, latent up/down = 4.7B, embed+lm_head =
2.4B, routers/norms = 1.35B. **Total 56.9B.**

---

## 2. The three walls

**Wall 1 — RAM, and it is the dense part, not the experts.** 56.9B always-on
params into 19.3 GB usable is **2.71 bits/param, with zero left for an expert
cache**. A 4-bit backbone needs 32 GB and does not fit. This inverts my own
earlier note in memory ("disk- and bandwidth-blocked long before RAM matters") —
RAM binds first, on the *dense* weights.

**Wall 2 — bandwidth.** 92 layers × 16 experts × 17.55 MB = **25.83 GB/token**.

**Wall 3 — capacity.** 1,446 GB of experts vs 183 GB free internally. External
storage is mandatory, and at 3-bit it is still 1.02 TB.

A fourth, quieter one: **KDA state is 217 MB per sequence** (69 layers × 96 heads
× 128×128 × 2B). Linear attention is O(1) in context but with a fat constant, so
concurrency caps at ~26 sequences even handing it 30% of RAM. That kills the
"just batch it" escape.

---

## 3. What the literature offers, and why almost none of it applies

I read the offloading line end to end. It is a healthy field and it is aimed at
a different regime: 8-of-64-style MoEs where a cache of 20–30% residency is
affordable and the bottleneck is PCIe latency, not raw bytes.

| system | mechanism | why it does not transfer to 16/896 |
|---|---|---|
| **Mixtral-offloading** | LRU + speculative expert prefetch | LRU needs residency > top-k; we have 2.4 slots vs top-16 |
| **MoE-Infinity** | sequence-level expert activation tracing → prefetch | prefetch hides *latency*; we are bandwidth-saturated, nothing to hide behind |
| **Pre-gated MoE / SP-MoE** | gate one layer early, prefetch | same — no spare bandwidth to prefetch into |
| **Fiddler** | compute experts on CPU instead of moving weights | Mac is unified memory; there is no CPU↔GPU copy to avoid. Inapplicable by construction |
| **HOBBIT** | mixed-precision: low-bit for cold experts | ✅ *survives* — this is real bytes saved, see §5 |
| **FlashMoE** | learned (recency+frequency) eviction, +21% hit over LRU | +21% of ~0 is 0 |
| **fMoE / MoEpic / ADEPT** | finer granularity, domain-aware prefetch | ditto; all are cache-quality plays |
| **BuddyMoE / SMoE** | substitute a *similar* cached expert on miss | we have no cached experts to substitute *from* |
| **EcoSpec** (2607.12696) | cost-aware draft-tree: prefer draft tokens reusing already-loaded experts | ✅ *partly* — but they measure only 0.6% expert reduction on DeepSeek-V3.1, whose aux-loss-free router is deliberately balanced. K3 uses `noaux_tc` + Quantile Balancing, i.e. the same balanced regime. Expect the DeepSeek number, not the Qwen one |
| **MoE-SpeQ** | draft model predicts future experts, overlap I/O | overlap ≠ fewer bytes |
| **REAP pruning** (PipeNetwork) | drop 654 of 896 experts by saliency | cuts *disk* 1446→390 GB; per-token bytes unchanged, still top-16 |
| **LLM in a flash** (Apple) | windowing + row-column bundling for ReLU sparsity | assumes predictable per-neuron sparsity in a *dense* FFN; MoE routing is coarse and data-dependent |

The pattern: **the field optimizes hit rate, and hit rate is not a variable we
have.** Every technique whose benefit routes through the cache is worth exactly
zero here.

A blunt cross-check on the sparsity itself: with 16/896 the union of experts over
*n* tokens grows as `16·(1 + (n−1)(1−r))` — linear in n, slope `1−r`. Batching
and speculative decoding buy *only* r, the cross-token repeat rate. Our own
measured r on the closest analogue (Qwen3-Next-80B, 512 experts, top-10, 2.0%
sparsity vs K3's 1.8%) is **0.42**, so the whole family of amortization tricks is
capped at ~1.7× no matter how clever the scheduler.

---

## 4. Ideas killed before proposing them — by this repo's own data

The best idea I had was **shared-basis expert compression**: all 896 experts of a
K3 layer are functions on the *same* 3584-dim latent (`routed_expert_down_proj`
is shared across experts), which is the ideal setting for a joint factorization
`W_e ≈ U S_e Vᵀ`. At rank 256 that is 179× smaller per expert and the entire
model's experts would fit in ~8 GB of RAM. It would have dissolved the problem.

`results/svd_expert_basis_*.json` already refutes it. On Qwen3-30B layer 0:
rank 64 of 128 experts captures only **59.8%** of the energy, and
`components_above_noise = 128 of 128` — every component is real structure, above
the explicitly computed quantization noise floor. The spectrum is nearly flat
(top singular values 825, 818, 814, 808, 794…). Experts are close to orthogonal.
Compression ratio available: ~1.1×.

This should have been obvious in hindsight: MoE training *enforces* expert
decorrelation via load balancing, and K3 pushes harder than most (Quantile
Balancing). The measurement must be repeated on K3 before it is fully closed, but
the prior is strongly negative and I am not building on it.

Two more, killed by the architecture:

- **Hot/cold storage tiering by global expert frequency.** `topk_method: noaux_tc`
  plus Quantile Balancing engineers routing to be *uniform over the training
  corpus*. There is no globally hot 10%. (Per-*domain* skew is a different claim
  and does survive — see §5.)
- **Self-speculation via an MTP head.** `num_nextn_predict_layers: 0`. K3 ships
  no multi-token-prediction head, so speculative decoding needs an external
  drafter from another family, and cross-family acceptance is poor. The γ=4 row
  in the table is optimistic; treat it as an upper bound.

---

## 5. The design: `stride`

Given the above, the system that fits is embarrassingly simpler than the one this
repo already has. Four ideas, in descending order of how much they buy.

### 5.1 Delete the cache. RAM goes to the dense backbone.

The allocation rule is a one-line argument and it is decisive:

> A byte of RAM holding a **dense** weight saves exactly **1 byte/token**,
> deterministically. A byte of RAM holding an **expert** saves
> `dh/dC × 25.8 GB`, and at 2.4 slots/layer against top-16, `dh/dC ≈ 0`.

So dense wins by a landslide and the expert cache should be **exactly a
double-buffer** — enough to keep reads in flight, nothing more. Keep from
`expert_store.py`: the packed file, `os.pread`, the thread pool, `F_NOCACHE`.
Delete: the LRU, the eviction policy, the ceiling invariant, the hit-rate
metrics, `reuse_distance.py`, `cache_negotiation_gate.py`. That is most of the
repo, and it is dead weight at this scale.

Corollary that contradicts `CLAUDE.md`'s phase 5: **do not build a prefetcher.**
Prefetching converts latency into throughput, and we have no latency problem —
we have a bandwidth problem. A mispredicted prefetch spends bandwidth we cannot
spare. Just keep 16 reads in flight.

### 5.2 Bandwidth-proportional expert placement — the only lossless 1.7×

This is the piece I have not seen anywhere, and it costs nothing in quality.

With several drives, reads run in parallel and wall time is set by the **busiest**
drive: `t = maxd (access_share[d] / bw[d])`. The obvious placement — spread the
pack evenly — sets `access_share ∝ capacity`. On this machine that is pathological,
because **the internal SSD is the fastest device *and* the smallest**:

```
internal   183 GB @ 4.64 GB/s   (measured)
ext NVMe  1000 GB @ 2.80 GB/s   (estimate — measure before trusting)
ext NVMe  1000 GB @ 2.80 GB/s

capacity-proportional placement :  6.11 GB/s   (internal idles 88% of the time)
bandwidth-proportional placement: 10.24 GB/s   (= sum of all drives)
```

**1.68×, for free, by deciding which expert lives on which drive.** The condition
is that the internal SSD's 8% of the pack must absorb 45% of the reads — i.e. the
workload's hot expert set must be that skewed. Global routing is uniform by
design (§4), but *per-domain* routing is not: PipeNetwork measured 57% expert
overlap within code, and only 17.8% between code and Chinese. So placement is
computed per-workload from a calibration trace, and the model stays bit-exact.

This generalises the REAP insight — **don't prune the cold experts, demote them.**

### 5.3 Then, and only then, the lossy levers

Each has a measurable quality cost and a knob. Order them by cost per byte saved:

1. **Expert requantization 4.25 → 3 bit** (1.42×). One-time streaming rewrite of
   the pack. `score_perplexity.py` already exists to price it; on GLM-4.5-Air the
   repo measured 3-bit experts at +20.7% PPL (1.585 → 1.913).
2. **Adaptive top-k by gate mass** (~1.33×): fetch experts in gate order until
   cumulative mass ≥ τ. Expect modest gains — `noaux_tc` balancing flattens the
   gate distribution, which is exactly why EcoSpec got 0.6% on DeepSeek.
3. **Speculative decoding, γ≈4** (~1.4×, optimistic): capped at 1/(1−r) ≈ 1.7×
   by the union law, and needs an external drafter (§4).
4. **HOBBIT-style mixed precision**: 3-bit for the pack, 4.25-bit copies of the
   domain-hot set on the internal drive. Composes with 5.2 for free.

### 5.4 Dense backbone: hold or stream, per block

At 2.71 bits/param the backbone is the quality bottleneck. But streaming a dense
block is *cheap and perfectly predictable* — 1 byte/token, zero stalls, full
sequential bandwidth. So it is a per-block decision, not all-or-nothing:

- **hold** KDA + MLA + latent up/down + routers at the highest bits that fit
  (attention is the most quantization-sensitive part),
- **stream** the 12.2B of shared experts if that buys the attention another bit.

`k3_budget.py` prices both sides; `plan(dense_bits={"shared_experts": None, …})`.

---

## 6. Plan

Phased, each ending in a commit, cheapest-decisive-experiment first.

**P0 — decide before downloading 1.5 TB.** Everything so far cost zero bytes of
K3. Two measurements gate the whole project:
  - attach the external NVMe, run `bw.py` against it (and both at once). If
    aggregate < 8 GB/s, the honest answer is 0.4 tok/s and stop.
  - download **layers 0–3 only** (~30 GB). Enough to run real routing and
    measure the two numbers the design hinges on: cross-token repeat rate `r`
    (is 0.42 transferable?) and gate-mass concentration (does adaptive top-k
    buy anything under `noaux_tc`?). Also re-run `svd_expert_basis.py` on a K3
    layer to formally close §4.

**P1 — placement.** Calibration trace on the target domain → per-expert access
histogram → assignment solving `access_share[d] ∝ bw[d]`. Verify the predicted
effective bandwidth with a replay of the trace against the real drives, no model.

**P2 — the engine.** Strip `expert_store.py` to pack + pread + double-buffer.
Port the K3 MoE block to MLX (the `KimiSparseMoeBlock` latent structure, shared
`routed_expert_down/up_proj`). Backbone quantized per §5.4. Correctness gate is
unchanged and non-negotiable: bit-identical logits vs a reference forward on a
few layers.

**P3 — the lossy dial.** Requantize the pack, adaptive-k, and a PPL curve for
each setting. Deliverable is the table in §0 with measured rather than modelled
numbers.

**P4 — speculative decoding**, only if P0 says `r` is real.

---

## 7. What would change the verdict

- **A 64 GB machine.** 19.3 → ~55 GB usable makes the backbone 4-bit with room
  to spare, and moves everything from "degraded" to "native". This is by far the
  cheapest fix and it is a purchase, not an engineering project.
- **`r` measuring much higher than 0.42 on K3.** Would make speculation and
  batching worth 3–4× instead of 1.4×.
- **A K3 MTP head appearing** in the promised technical report.
- **`svd_expert_basis.py` behaving differently on K3** than on Qwen. Unlikely,
  but it is the only result that would dissolve the problem rather than shave it.

---

## Sources

- [moonshotai/Kimi-K3 config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) — architecture, quant config
- [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3) — Stable LatentMoE, KDA, MXFP4
- [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) — param census, REAP pruning, the only measured tok/s
- [EcoSpec: Cost-Aware Speculative Decoding for MoE](https://arxiv.org/html/2607.12696)
- [MoE-SpeQ](https://arxiv.org/abs/2511.14102) · [SP-MoE](https://arxiv.org/html/2510.10302)
- [FlashMoE](https://arxiv.org/html/2601.17063v1) · [HOBBIT](https://arxiv.org/html/2411.01433v1) · [fMoE](https://arxiv.org/html/2502.05370v1)
- [BuddyMoE](https://arxiv.org/pdf/2511.10054) · [SMoE expert substitution](https://arxiv.org/pdf/2508.18983)
- [In-depth Analysis on Caching and Pre-fetching in MoE Offloading](https://arxiv.org/pdf/2511.05814)
- [LLM in a flash (Apple)](https://arxiv.org/abs/2312.11514)
