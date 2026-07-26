# locali

Run a 106B mixture-of-experts model on a Mac in **4.45 GB of RAM**.

GLM-4.5-Air is a 60.1 GB checkpoint. It does not fit in a 24 GB machine, let
alone a 16 GB one. But at any instant a MoE model only needs 8 of its 128
experts per layer — the other 120 are dead weight sitting in memory. This
repo leaves them on disk and reads the ones the router actually asks for.

```
$ uv run python index_inplace.py --model mlx-community/GLM-4.5-Air-4bit
45 MoE layers x 128 experts = 5760 entries
9.73 MB/expert, 56.1 GB routed -- referenced in place, not copied

$ uv run python run_experiment.py --verify --reference --wave-slots 16
Model loaded in 2.8s, active memory 4.24 GB
Patched 45 MoE layers (wave pool, 16 slots shared across all layers,
                       resident=0.156 GB, no cache)
[0] 40 tokens in 52.0s (0.8 tok/s)
MLX peak memory: 4.45 GB
```

It is slow. That is the trade: 0.75 tokens/s instead of not running at all.

## The numbers

M4, 24 GB, internal SSD. 3 prompts x 40 tokens, temperature 0.

| residency mode | pool | peak RAM | tok/s | hit rate |
|---|---|---|---|---|
| wave, 16 slots | 0.16 GB | **4.45 GB** | 0.754 | 0% |
| LRU, 3.6 GB ceiling | 3.50 GB | 7.72 GB | 0.763 | 29.8% |
| LRU, 7.5 GB ceiling | 7.45 GB | 11.75 GB | 0.818 | 40.3% |
| LRU, 8.5 GB ceiling | 8.32 GB | 12.64 GB | 0.844 | 41.9% |
| wave + 3-bit experts | 0.28 GB | 4.56 GB | **0.915** | 0% |

All five produce **byte-identical output** at 4-bit. The 3-bit pack is a
quality change and is measured separately below.

The striking row is the last one against the third: **shrinking the experts
beats caching them.** 3-bit at 4.56 GB is faster than 4-bit with a 7.5 GB
cache at 11.75 GB — 2.6x less RAM for 12% more speed. On a machine where the
cache cannot fit at all, that is the only lever left.

## How it works

Three ideas, in order of how much they matter.

**1. Index the experts in place.** The obvious design copies every routed
expert into one packed file laid out for fast reads. It also doubles the disk
requirement — 60 GB of checkpoint becomes 116 GB of checkpoint plus pack, and
for a 418 GB model on a 512 GB SSD that is the difference between running and
not. Unnecessary: MLX already stores a layer's experts as one stacked tensor
per projection, so expert *e* is a contiguous byte range at a computable
offset. `index_inplace.py` reads the safetensors *headers* — never the data —
and writes a 5.8 MB index of offsets into the original shards. It takes
seconds and copies nothing.

The cost is 9 preads per expert instead of 1, because the 9 arrays
(gate/up/down x weight/scales/biases) live in 9 different tensors. Measured:
3.95 GB/s for 9 preads vs 4.05 GB/s for 3. Six percent, for half the disk.

**2. One shared wave pool instead of per-layer pools.** The cache wants
per-layer LRU pools, but those cost `n_layers x slots x bytes_per_expert`
whether or not the cache can do anything useful. On GLM-4.5-Air the minimum
useful cache is 45 x 8 x 9.73 MB = 3.5 GB. Only the *active* layer's 8
experts must actually be resident: `--wave-slots 16` allocates one 16-slot
pool shared by all 45 layers, 0.156 GB, refilled per call.

This matters most where the cache is hopeless. GLM-5.2 at 4-bit has a 10.6 GB
resident core and a 12.7 GB/token working set; on a 16 GB machine there is no
cache worth having, and per-layer pools alone would want 9.9 GB that the
machine does not have.

**3. Expert-major prefill.** Prompt processing iterates experts, not tokens,
so each expert is read at most once per layer per prefill regardless of prompt
length.

## Install

```sh
uv venv --python 3.12
uv sync
export HF_HOME=./models/hf
uv run hf download mlx-community/GLM-4.5-Air-4bit
uv run python index_inplace.py --model mlx-community/GLM-4.5-Air-4bit
```

Needs the checkpoint on disk (60 GB) and nothing else — no export step, no
second copy.

## Use

```sh
# Lowest RAM. Works on a 16 GB machine.
uv run python run_experiment.py --verify --reference --wave-slots 16

# Trade RAM for speed: per-layer LRU under a hard byte ceiling.
uv run python run_experiment.py --verify --reference --ceiling-gb 7.5

# Honest measurement: stop the OS buffer cache from acting as a second,
# unaccounted cache tier. Required for any number meant to predict how a
# model much larger than RAM will behave.
uv run python run_experiment.py --verify --reference --wave-slots 16 --no-page-cache
```

Fewer bytes per token, at a quality cost — the popular experts stay 4-bit,
the rest go to 3-bit:

```sh
uv run python run_experiment.py --verify --reference --wave-slots 16 --log-routing
uv run python requantize_experts.py --hot-n 16 --cold-bits 3 --out-suffix _mixed3b
uv run python run_experiment.py --verify --reference --wave-slots 16 --pack _mixed3b
```

Results land in `results/` with a timestamp; nothing is overwritten.

## Where the time goes

Disk is not the thing to optimize — it is already at hardware speed. Three
measurements, each removing one layer:

| measurement | s/token | includes |
|---|---|---|
| raw preads, real access pattern | 0.98 | disk only |
| `translate()`, no model, no GPU | 1.05 | + Python, index lookups, numpy→MLX copies |
| full generation | 1.33 | + router sync, gather, MLX graph |

So: **disk 74%, store 5%, model 21%.** In-run read throughput measures
4.06 GB/s against a 3.95 GB/s benchmark ceiling. The absolute limit at 4-bit
with no cache is 3.5 GB/token ÷ 3.95 GB/s = **1.13 tok/s**, and 0.754 is 67%
of it.

What got it there, in order (all byte-identical, wave mode):

| change | tok/s |
|---|---|
| starting point | 0.583 |
| deferred pool eval instead of a drain per layer | 0.662 |
| write each expert as its bytes land (`as_completed`) | 0.754 |

The second one is the interesting one: `.map()` joins the whole batch before
the first numpy→MLX copy starts, so all 8 reader threads idle at every layer
boundary. Interleaving the copies with the reads still in flight returned
6.2 s of a 43 s run — more than the copies themselves cost, because the
threads stopped draining.

What is left is structural: 45 router syncs per token, and a gather that
cannot start until the pool writes materialize. Removing it means prefetching
layer L+1's experts during layer L — and layer L+1's routing does not exist
until layer L finishes. That is speculation, not optimization.

## Quality

Perplexity on held-out text (`eval/pride_prejudice.txt`, 512-token windows).
Token agreement is not used as the metric: it conflates trajectory divergence
with degradation.

| pack | perplexity | vs 4-bit | bytes/expert |
|---|---|---|---|
| uniform 4-bit | 1.5851 | — | 9.73 MB |
| hot16 @ 4-bit + cold @ 3-bit | 1.9125 | **+20.7%** | 7.57 MB cold |

Ten windows, 5110 tokens — enough to rank the packs, not enough to publish.

So the 3-bit row in the headline table costs 20.7% more perplexity for 21%
more speed and 2.6x less RAM than the equivalent cache. Whether that is worth
it depends on the model: it is a bad trade at 106B on a 24 GB machine, where
you can just use the cache, and the only trade available at 744B on a 16 GB
one, where you cannot.

## The page cache is not your friend

Reads go through the OS unified buffer cache unless you say otherwise, which
on a 24 GB Mac reading a 60 GB model quietly provides a second cache tier you
never sized. `--no-page-cache` sets `F_NOCACHE` and turns it off:

| | wave (0.16 GB pool) | LRU 7.5 GB |
|---|---|---|
| page cache on | 0.754 | 0.818 |
| `F_NOCACHE` | 0.680 | 0.801 |
| what the OS was giving | +10.9% | +2.1% |

The two caches are substitutes: once ours holds 40% of the working set, the
OS has nothing left to contribute. Any number meant to predict a 400 GB model
— where the page cache will never help — must be measured with the flag on.

## What is proven and what is not

**Proven.** Resident expert bytes never exceed the configured ceiling — a
property test hammers it with random access patterns, and pools are fixed-size
by construction, so every write is an in-place slot update. The wave pool and
the LRU cache produce byte-identical token streams across every ceiling tested.
Reads through `ExpertStore` are byte-equal to an independent read path.

**Not proven.** There is no comparison against unpatched `mlx_lm`, because a
60 GB model cannot be held resident on this machine to compare against. The
correctness claim here is "identical across residency modes", which is a
regression check, not a proof that the streaming math equals the stock math.
Establishing that needs layer-by-layer equivalence against a resident model on
a machine large enough to hold one.

## Hardware notes

- Apple Silicon, macOS, MLX. Unified memory means reads land where both CPU
  and GPU can see them.
- Reader threads cap at 8. Measured 3.95 GB/s at 8 threads, 3.79 at 16, 3.85
  at 24 — the queue is full at 8 and more threads buy nothing.
- `run_experiment.py` refuses to start without headroom and caps MLX's total
  allocation, because a failed run once took the whole machine down.
- Model files and indexes live under `models/`, gitignored.

## Layout

```
expert_store.py        residency: pools, LRU, wave mode, the ceiling invariant
index_inplace.py       offsets into the original shards, no copy
patched_model.py       the MoE patch; identical math, streamed weights
run_experiment.py      CLI
requantize_experts.py  mixed-precision packs
score_perplexity.py    quality metric
metrics.py             counters
tests/                 ceiling invariant, slot mapping, wave partitioning
results/               measured runs
```
