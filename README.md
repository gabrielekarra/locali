# Locali — DeepSeek V4 Flash on consumer hardware

Locali runs `DeepSeek-V4-Flash-0731` on Apple Silicon even when the routed
experts do not fit in unified memory. The dense MLX backbone stays resident;
the 256 routed experts are read from SSD into a fixed-size unified-memory arena
and consumed in place by Metal kernels.

The current target is a 24 GB Apple M4 MacBook Air. With the published
2.44-bit mixed MLX checkpoint, Locali keeps about 6.49 GB of dense weights and a
6.99 GB expert arena resident. On the bundled benchmark corpus, two final
64-token prefill + 128-token decode replicas measured 2.32-2.34 generated
tokens/s and 2.37-2.38 steady tokens/s.

## Run the chat

The optimized local layout expects:

- `mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed` under `.runtime/models/`;
- the DeepSeek V4 MLX implementation from oMLX under `.runtime/omlx/`;
- a Locali expert index under `models/`.

One-time setup:

```sh
uv sync
git clone --depth 1 https://github.com/jundot/omlx.git .runtime/omlx
.venv/bin/hf download mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed \
  --local-dir .runtime/models/DeepSeek-V4-Flash-0731-2.4bit-mixed

.venv/bin/python dsv4_index.py \
  --snapshot .runtime/models/DeepSeek-V4-Flash-0731-2.4bit-mixed \
  --out models/dsv4-2.4bit.idx
```

Start the text-only TUI:

```sh
python deepseek_v4.py --nothink
```

The fixed BOS/system prefix is evaluated before `locali>` appears, so only the
new user suffix is processed after submission. Output streams token by token
through one persistent multi-turn KV cache.

The aggressive 2.44-bit checkpoint defaults to deterministic greedy decoding
and a direct-answer system prompt. Stochastic sampling remains explicit:

```sh
python deepseek_v4.py --nothink --temp 1 --top-p 1 --min-p 0.05
```

`--system` overrides the interactive prompt. `--seed` makes sampling
reproducible; without it, sampling uses OS entropy.

## Packed expert layout

The base checkpoint runs directly, but the expert-major pack collapses nine
sparse tensor reads per expert into one vectored read:

```sh
.venv/bin/python pack_experts.py \
  --index models/dsv4-2.4bit.idx \
  --data .runtime/models/dsv4-experts.pack \
  --out-index models/dsv4-2.4bit-packed.idx \
  --layers all --tiers cold
```

The checkpoint is about 92.8 GB decimal. The optional pack adds about 77.9 GB,
so the fully optimized installation needs roughly 171 GB before filesystem
overhead. The unpacked layout needs about 100 GB and remains supported.

## Engine

- `dsv4_engine.py` loads only the resident backbone and installs streamed MoE
  modules in all 43 transformer layers.
- `arena.py` owns the bounded SSD-to-unified-memory expert cache.
- `dsv4_arena.py` implements V4 top-6 routing, clamped SwiGLU and affine
  `gather_qmm` over arena slots.
- `native/locali_core.c` provides the allocation-free SLRU/LFU scheduler used
  in the decode loop.
- `dsv4_kernels.py` contains optional fused Metal/CUDA expert kernels. They are
  benchmark flags and stay off when they lose to stock MLX kernels.
- `dsv4_index.py` and `pack_experts.py` build the zero-copy layouts.

DSpark/MTP stages embedded in the checkpoint can be indexed and benchmarked,
but are disabled for chat because their measured draft acceptance does not
repay the extra expert traffic on this SSD-streamed path.

## Benchmark and quality checks

Canonical performance run:

```sh
.venv/bin/python dsv4_bench.py \
  --index models/dsv4-2.4bit-packed.idx \
  --ceiling-gb 7 --os-cache \
  --out results/deepseek_v4_flash_final.json
```

The final two replicas measured:

| Run | Prefill | Decode | Steady | Expert hit | Expert bytes read |
|---|---:|---:|---:|---:|---:|
| 1 | 5.61 t/s | 2.32 t/s | 2.37 t/s | 60.6% | 92.14 GB |
| 2 | 5.71 t/s | 2.34 t/s | 2.38 t/s | 60.6% | 92.14 GB |

Decode speed is routing-sensitive: this corpus produces a lower expert hit rate
than earlier traces, so the JSON includes hit rate, bytes read and I/O stall
alongside throughput.

Teacher-forced continuation check:

```sh
.venv/bin/python dsv4_quality.py \
  --index models/dsv4-2.4bit-packed.idx --ceiling-gb 7
```

The 20-case fixture is self-contained in
`eval/deepseek_v4_flash_20.jsonl`. The measured 2.44-bit checkpoint reached
average NLL 0.73796, perplexity 2.09 and 81.9% token top-1 on its 480 target
tokens. These values measure the checkpoint; the streaming implementation
executes its expert bytes exactly.

Run the test suite with:

```sh
pytest -q
```

The repository cleanup baseline is `19 passed`; the native C target also builds
with `-Werror` and reports `locali_core: all tests passed`.

Detailed measurements are recorded in
`results/deepseek_v4_flash_locali_m4_24gb_20260805.md`.
