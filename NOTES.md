# Notes

Measurement log for GLM-4.5-Air streaming. Numbers only, with the method that
produced them and the mistakes that produced the wrong ones first.

Machine: M4, 24 GB unified memory (25.8 reported), internal SSD, macOS.
Model: `mlx-community/GLM-4.5-Air-4bit`, 106B-A12B, 60.1 GB, 12 shards.
Protocol unless stated: 3 prompts x 40 tokens, temperature 0, `--reference`.

## The model, from its index

| quantity | value |
|---|---|
| MoE layers | 45 (of 46; layer 0 is dense) |
| routed experts / layer | 128, top-8 |
| bytes per expert | 9.73 MB (4-bit affine, group size 64) |
| routed total | 56.1 GB |
| core (embeddings, attention, shared experts, norms, gates) | 4.0 GB |
| **floor: one token's working set** | 45 x 8 x 9.73 MB = **3.50 GB** |

The floor law from the index predicted the measured minimum ceiling exactly:
`--ceiling-gb 3.6` gives 8 slots/layer, the smallest that holds one decode
step. Below it `translate()` raises rather than silently thrashing.

## Residency curve

| mode | pool | peak | tok/s | hit | cold read |
|---|---|---|---|---|---|
| wave, 16 slots | 0.156 GB | 4.45 GB | 0.754 | 0% | 535.9 GB |
| LRU 3.6 GB | 3.504 GB | 7.72 GB | 0.763 | 29.8% | 376.0 GB |
| LRU 7.5 GB | 7.445 GB | 11.75 GB | 0.818 | 40.3% | 320.1 GB |
| LRU 8.5 GB | 8.321 GB | 12.64 GB | 0.844 | 41.9% | 311.2 GB |
| wave + 3-bit | ~0.28 GB | 4.56 GB | 0.915 | 0% | 457.1 GB |

Every 4-bit row byte-identical to every other. **2.8x the RAM buys 12% speed**
(4.45 → 12.64 GB, 0.754 → 0.844): cold bytes fall 42% but wall time only 11%,
because the disk is fast and the non-disk cost is fixed.

At 6.25% residency GLM-4.5-Air hits 29.8%. Qwen3-30B measured 40.5% at the
same residency, Qwen3-Next-80B 52.9% at 6.6%. Third model, third curve — the
hit-rate curve does not transfer between models, so it has to be measured on
the one you intend to run.

## Optimization, in the order it happened

Profiled first. The store's own counters, decode with `store.profile = True`:

| bucket | s (of 43.7 wall, 20 tokens) | share |
|---|---|---|
| `t_fetch` | 27.71 | 63% |
| `t_gather` | 5.03 | 12% |
| `t_eval` | 3.92 | 9% |
| `t_write` | 3.87 | 9% |
| `t_sync` | 2.81 | 6% |

Fetch was already at hardware speed: 112.6 GB / 27.71 s = 4.06 GB/s, against
a benchmarked 3.95 GB/s for the same access pattern. Nothing to win there.

| change | tok/s | note |
|---|---|---|
| baseline | 0.583 | |
| deferred pool eval | 0.662 | +13.5% |
| `as_completed` writes | 0.754 | +14% |

**Deferred eval.** `_translate_disabled` ran `mx.eval()` on the pool every
layer — 45 GPU pipeline drains per token. The LRU path had already replaced
that with a periodic drain; now both use `EVAL_EVERY`. Honest accounting:
`t_eval` fell 3.92 → 0.59 s but `t_gather` rose 5.03 → 7.74. The work was not
waste, it just moved; the gain came from removing the barrier, not the work.

**`as_completed` writes.** `.map()` joins the entire batch before the first
numpy→MLX copy begins, so all 8 reader threads idled at every layer boundary.
Writing each expert as its own bytes land returned 6.2 s of a 43 s run — more
than the copies cost, because the threads stopped draining. Applied to the LRU
path too, but there slots are assigned *before* the reads and in request
order: which expert lands in which slot must not depend on disk timing, or the
hit rate stops being reproducible. Verified — hits, misses and evictions came
back bit-identical (16430 / 38639 / 38279).

**Where it stopped.** Three measurements, each stripping a layer:

| | s/token |
|---|---|
| raw preads, real batched pattern | 0.98 |
| `translate()` alone, no model, no GPU | 1.05 |
| full generation | 1.33 |

Disk 74%, store 5%, model 21%. Ceiling at 4-bit with no cache: 3.5 GB ÷
3.95 GB/s = 1.13 tok/s; 0.754 is 67% of it. The remainder is 45 router syncs
per token plus a gather that cannot start before the pool writes materialize.
Removing that means prefetching layer L+1 during layer L, and L+1's routing
does not exist yet.

Note the gather is *already* overlapped in production. The profile makes it
look like 21% of wall because `store.profile = True` forces an eval after it
to attribute the time — which serialises the thing being measured. The
profiled run is 1.83 s/token, the real one 1.33.

## Page cache

`F_NOCACHE` (fcntl 48) on the expert files:

| | wave (0.16 GB) | LRU 7.5 GB |
|---|---|---|
| page cache on | 0.754 | 0.818 |
| `F_NOCACHE` | 0.680 | 0.801 |
| OS contribution | +10.9% | +2.1% |

Substitutes, as expected: once our cache holds 40% of the working set the OS
has nothing left to give. On this machine the model is 60 GB and RAM 25.8, so
without the flag every number is flattered by a tier that will not exist for a
400 GB model.

## Quality

Perplexity, `eval/pride_prejudice.txt`, 512-token windows, 10 windows,
5110 tokens. Perplexity rather than token agreement, which conflates
trajectory divergence with degradation.

| pack | PPL | relative |
|---|---|---|
| uniform 4-bit | 1.5851 | — |
| hot16@4b + cold@3b | 1.9125 | +20.7% |

Cost 20.7% perplexity, bought 21% speed and 2.6x less RAM than the cache that
matches it. For reference, the same recipe on Qwen3-30B measured 6.73 → 8.54
(+26.9%).

## Mistakes worth keeping

**Two result files overwritten.** `--log-routing` writes
`results/routing_trace.json` and `score_perplexity.py` writes
`results/ppl_*.json`, both fixed paths. Running them for GLM destroyed the
Qwen equivalents, and `results/` is gitignored so git could not help. Fixed
paths plus an ignored directory means "no backup". The surviving Qwen values:
routing trace gone, PPL 6.73 uniform / 8.54 mixed3b.

**Model-specific constants in tests.** The ceiling tests hardcoded ceilings
like 0.2 GB, sized for Qwen's 2.65 MB experts. On 9.73 MB experts the same
number buys 1 slot instead of 9, and the tests either failed or silently
tested nothing. They now derive ceilings from the indexed model's geometry.

**Requantizer cleanup crashed twice** after writing 42 GB correctly — `mem`
and `mmap` were bound only on the packed branch. Cost two full rewrites of the
pack before `--verify-only` was added, which checks an existing pack instead
of rebuilding it to re-run a check.

## Open

- No comparison against unpatched `mlx_lm`: 60 GB cannot be held resident here.
  Correctness rests on identity across residency modes, which is a regression
  check, not a proof.
- Speculative prefetch (predict layer L+1's experts from layer L's hidden
  state) is the only remaining path past ~1.1 tok/s at 4-bit.
- 10 PPL windows is thin. 40 would take ~30 minutes per pack.

## Second session: prefetch, and a measurement protocol that was lying

### A full answer, finally

Every number above is from 40-token runs. GLM-4.5-Air is a reasoning model:
at 40 tokens it has not left the `<think>` block, so the whole curve was
measured on three truncated openings of a reasoning trace. First complete
run: 600 tokens, `--ceiling-gb 6.0`, one prompt.

| | |
|---|---|
| 600 tokens | 575.6 s = **1.0 tok/s** |
| peak | 10.00 GB, ceiling held (5.693 GB resident) |
| hit rate | 41.2% |
| cold read | 1260.4 GB |

It answers the "17 sheep, all but 9 die" trick correctly (9), checks itself
against a smaller case, and explicitly names the misreading it is avoiding.
At 600 tokens it is still inside `<think>` and never emits a final answer
outside it; that needs ~1000-1500 tokens.

### Generation length changes the answer, in both directions

| config | 40 tokens | 600 tokens |
|---|---|---|
| wave 16 | 0.754 | **0.691** |
| LRU 6.0 GB | ~0.80 (interpolated) | **1.0** |

Opposite signs. Expert-major prefill reads each expert once per layer, so it
is *cheaper per token* than decode; at 40 tokens it is a large share of the
average and flatters wave mode. The LRU instead gets better with length as
the cache warms. **A 40-token protocol is not a constant bias, it distorts
per mode**, which is worse. The tables above are 40-token numbers and should
be read as such until the curve is re-run at 600.

### Profile of the regime that matters

Warm cache, 100 tokens, 40% hit, `store.profile = True`:

| bucket | s of 113 | share |
|---|---|---|
| `t_fetch` | 81.9 | 72% |
| `t_gather` | 18.2 | 16% |
| `t_sync` | 10.8 | 9.5% |
| `t_eval` | 1.4 | 1.3% |

234.3 GB in 81.9 s = **2.86 GB/s**, against 4.06 GB/s measured in the 0%-hit
wave regime and a 3.95 GB/s benchmark. The disk did not get slower: the
batches did. At 0% hit every layer asks for 8 experts and fills 8 reader
threads; at 40% it asks for ~4.8 and three threads idle. **The cache sabotages
its own read throughput** -- 22.6 s of the 113, purely to shallow queues.

### Prefetch: not validated

While a layer computes, start reading what the next N layers used at the
previous decode step. Two measurements, same config, different length:

| | 120 tokens | 600 tokens |
|---|---|---|
| wave 16 | 0.793 | 0.691 |
| wave 16 + prefetch 4 | **0.874** (+10.2%) | **0.587** (−15%) |

Accuracy 37%: 15798 guesses used, 26843 discarded. Wrong guesses still cost a
read, so real traffic is ~721 GB against the 460.6 GB `bytes_read_cold`
reports -- 57% more bytes for, at best, 10% more speed. At 600 tokens that
extra traffic appears to cost more than it buys.

Confound not excluded: the 600-token pair ran back to back in one sweep, the
prefetch run second, on a fanless machine. Thermal accumulation would produce
the same sign. **The feature stays off by default and the +10.2% claim in its
commit message is not reproduced at realistic length.** Settling it needs both
runs cold, in randomised order.

### Where prefetch cannot work, proven

With per-layer LRU pools nothing evicts layer L's entries between two visits
to layer L -- its pool is touched only by layer L. So R_t is still resident at
token t+1, misses are R_{t+1} \ R_t, and prefetching R_t reads bytes that are
never wanted. The constructor now raises instead. Found by a test asserting
`prefetch_used > 0`, which failed for three different reasons in a row (cache
too large to evict, working set larger than the pool, random access order
instead of decode order) before the real one surfaced.

### Out of reach, with numbers

Kimi K3 shipped 2026-07-27: 1560.9 GB, 93 layers (92 MoE), 896 experts,
top-16, native mxfp4, `kimi_linear` attention with 24 of 93 layers full.

| | per expert | floor/token | core |
|---|---|---|---|
| GLM-4.5-Air (runs) | 9.73 MB | 3.50 GB | 4.0 GB |
| GLM-5.2 | 21.2 MB | 12.7 GB | 10.6 GB |
| Kimi K3 | 17.55 MB | **25.8 GB** | ~114 GB (by subtraction, unverified) |

At 3.95 GB/s K3 is 0.15 tok/s, but the floor is not what kills it -- the core
is. Streaming never touches the core, and it grows with the model. Going
bigger makes the one number that decides feasibility worse.

There is no "K3 Air": Air is Zhipu's naming, not Moonshot's. "K3 Fast" is a
serving tier on the same weights, not a checkpoint. The streamable member of
that family already exists: `Kimi-Linear-48B-A3B-Instruct`, 27.6 GB, 1.14 GB
core.

### The honest position on what this is for

On 16 GB every model this engine runs, `llama.cpp` with mmap also runs. What
it does that mmap cannot: a hard, known-in-advance RAM ceiling, and
mixed-precision per-expert packs. The page-cache measurement sizes the gap --
the OS contributed +10.9% with no cache of ours and +2.1% with one. Automatic
residency is a decent uncontrolled cache, not a substitute for a bound.

The experiment that would settle whether this is necessary has not been run:
this engine at a fixed ceiling against llama.cpp mmap, same machine, other
apps open. Until then "does this need to exist" has no data behind it.
