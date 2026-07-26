# locali — expert streaming for MoE models that don't fit in RAM

## What this is

Run a MoE model whose routed-expert weights exceed RAM. Experts stay on disk
and are read on demand into a small resident pool. Validated on
GLM-4.5-Air (106B, 60.1 GB checkpoint, runs in 4.45 GB). See `README.md` for
the design and `NOTES.md` for every measured number.

## Hard rules

- **The ceiling is a hard invariant.** Pools are fixed-size at construction
  and every write is an in-place slot update, so resident bytes cannot grow.
  A property test asserts it under random access. Keep it that way.
- **Never mmap the expert file and let the OS manage residency.** We manage
  it, with explicit `os.pread` into buffers we own. `--no-page-cache` exists
  so the OS buffer cache cannot silently become a second, unaccounted tier —
  use it for any number meant to predict a model larger than RAM.
- **No silent fallbacks.** A short read, a missing tensor, a call whose
  working set exceeds the pool: crash with a message naming the fix.
- **Correctness before speed.** Any change to the residency path must leave
  token streams byte-identical across residency modes. Check it before
  reporting a speedup.
- **Measure before optimizing, and re-measure after.** The store carries
  `t_fetch/t_write/t_eval/t_sync/t_gather` counters for this. Note that
  `store.profile = True` forces evals that serialise the pipeline — a
  profiled run is not the run you are shipping.
- **Nothing model-specific in tests.** Derive sizes from the indexed model's
  geometry. Hardcoded ceilings tuned to one model silently stop testing
  anything on another.
- Dependencies: mlx, mlx-lm, numpy, psutil. Ask before adding.
- Results go to `results/` with a timestamp; never overwrite a previous run.
  Two fixed-path writers (`--log-routing`, `score_perplexity.py`) violate this
  and have already destroyed data — fix them before reusing them on a new model.

## macOS / Apple Silicon

- Reader threads cap at 8; measured throughput is flat from 8 to 24.
- Watch memory pressure. `run_experiment.py` refuses to start without headroom
  and caps MLX's allocation, because a failed run once took the machine down.
- Model files and indexes live under `models/`, gitignored.

## Adding a model

Works on any checkpoint whose routed experts are stacked per layer under
`layer.mlp.switch_mlp` with affine quantization — GLM, Qwen3-MoE, Qwen3-Next.
`index_inplace.py` will refuse loudly otherwise.

Two numbers decide feasibility before downloading anything: the **resident
core** (repo bytes minus routed-expert bytes — it can never be streamed) and
**bytes per expert** (which sets how many slots a GB of cache buys). The
per-token floor is `moe_layers x top_k x bytes_per_expert`; if the cache is
smaller than that, hit rate is ~0 by construction and only `--wave-slots`
will run at all.
