# locali — expert streaming for MoE models that don't fit in RAM

## What this is

Run a MoE model whose routed-expert weights exceed RAM. Experts stay on disk and
are read on demand into a cache with a hard byte ceiling; only the dense
backbone stays resident. Target is **MiniMax-M2.5** (229B, 128.7 GB at 4-bit,
runs with a 2.74 GB resident core on a 24 GB Mac). `README.md` has the design,
`TECHNIQUES.md` the techniques and the measurements behind them.

## Hard rules

- **The ceiling is a hard invariant.** Evict *before* inserting, never after —
  trimming afterwards leaves a window where residency exceeds the ceiling, which
  is the whole thing. `m25_store.py`'s self-check asserts it under an access
  pattern designed to thrash.
- **Never mmap the weights and let the OS manage residency.** We manage it, with
  explicit `os.pread` into buffers we own, and `F_NOCACHE` so the unified buffer
  cache cannot become a second unaccounted tier and make the metrics lie.
- **Never call `mlx_lm.load` on these checkpoints.** 128.7 GB into 24 GB of RAM
  takes the machine down; it has happened. Build only the layers you need, or
  tear the expert modules out before anything materialises them, then assert on
  process RSS rather than trusting that the arrays stayed lazy.
- **One model-loading process at a time.** Sequential, foreground preferred.
- **No silent fallbacks.** A short read, a missing tensor, a working set larger
  than the ceiling: crash with a message naming the fix.
- **Correctness before speed, and compare through the same math path.** Checking
  a streamed block against `gather_qmm` sits at ~1e-2 no matter how correct the
  fetch is, because packed-weight matmul and dequantize-then-matmul are
  different computations — and a tolerance wide enough to pass that hides
  exactly the indexing bugs the check exists to catch. Hold the arithmetic
  fixed so the only variable is the indirection, then demand `0.000e+00`.
- **Measure before optimizing.** Three costs, not one: `t_io` (cold bytes /
  disk), `t_ram` (active bytes / memory bus — a floor no cache moves), `t_over`
  (implementation). Leaving out `t_ram` once predicted 43 tok/s for a run that
  measured 5.2.
- **Index, don't copy.** Experts are addressed as `(shard, offset)` into the
  original safetensors. A copied mixed-precision pack would be 84 GB.
- Dependencies: mlx, mlx-lm, numpy, psutil. Ask before adding.
- Results go to `results/`; never overwrite a previous run. Fixed-path writers
  have already destroyed data once.

## macOS / Apple Silicon

- Reader threads cap at 8; measured throughput is flat from 8 upward.
- Disk bandwidth depends on block size: 4.00 GB/s at 4.42 MB, 4.66 at 17.5 MB.
  Measure with `bw.py` against a file **larger than RAM** — a 8 GB file reported
  a fictional 14.7 GB/s because the page cache served it.
- The memory bus is 120 GB/s on M4 base. Published 30+ tok/s results on 128 GB
  Macs run at 400 GB/s with the model fully resident; not comparable.
- Weights and indexes live under `models/`, gitignored.

## What decides feasibility, before downloading anything

Two numbers: the **dense core** (total params minus routed-expert params — it
can never be streamed) and **bytes per expert** (which sets how many slots a GB
of cache buys). The per-token floor is `moe_layers × top_k × bytes_per_expert`.

The threshold that matters is **top-k**: if the cache holds fewer slots per layer
than the router selects, the hit rate is zero by construction and no policy
rescues it. M2.5 works here because it has no shared experts, so the dense core
is 2.26 GB and the cache lands at 58 slots against top-8. `budget.py` computes
all of this from a config.
