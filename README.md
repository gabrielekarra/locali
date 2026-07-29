# locali

**A 229B model generating text on a 24 GB Mac.** 128.7 GB of weights, 2.28 GB
resident, a memory ceiling you set in gigabytes and that the process cannot
exceed, and output bit-identical to the framework's own kernel.

```
$ python m25_engine.py --snap <4-bit snapshot> --arena --ceiling-gb 6 --tokens 32
dense core resident: 2.28 GB  (loaded in 1s)

decode 32 tokens in 23.7s (1.35 tok/s)
  hit 40.3%  read 55.09 GB (1.72 GB/token)
  evictions 10256  peak 6.00 GB
  mlx active 8.29 GB

The capital of France is Paris. The official language is French. The
currency is the Euro. The population is about 67 million.
```

`peak 6.00 GB` against a 6.00 GB ceiling is not luck and not a trim-after-the-
fact. Residency *cannot* exceed the ceiling, because the arena that holds the
experts is allocated once at that size and is the only memory there is.

It is slow. 1.35 tokens/second single-stream, 8.31 aggregate at batch 512. The
interesting part is not the speed — it is that a model 5x larger than the
machine runs at all, under a hard bound, without lying about what it is doing.

---

## The idea

MiniMax-M2.5 is 229B parameters. 224.7B of them are routed experts that a given
token never touches: 62 layers, 256 experts each, top-8 selected. The dense
backbone — everything that runs for every token — is **4.02B parameters, 2.26 GB
at 4-bit**.

So the backbone stays in RAM and the experts stay on disk, read on demand into a
cache with a hard byte ceiling.

That works here for one structural reason: **M2.5 has no shared experts.** The
threshold that decides feasibility is top-k. If the cache holds fewer slots per
layer than the router selects, the hit rate is zero by construction and no
policy rescues it. A 2.26 GB backbone leaves room for hundreds of slots against
top-8. A model with a fat always-on core leaves none, and then none of this
works. `budget.py` computes that from a config before you download anything.

## What is actually new here

Every published MoE offloading system — MoE-Infinity, HOBBIT, DALI, SwapMoE,
Pre-gated MoE — moves expert weights across a bus into device memory. Their
designs are about hiding that transfer.

**Apple Silicon has no transfer.** The GPU reads the same LPDDR the CPU writes.
And it turns out MLX hands out a *writable* pointer into its own buffers, which
land 16 KB page-aligned — exactly what `F_NOCACHE` needs to DMA into user memory
instead of bouncing through the kernel.

So the bytes go from the SSD controller **straight into the memory the GPU will
gather from**. Nobody copies them. Measured, F_NOCACHE, 1.18 MB blocks:

| | |
|---|---|
| `pread` → `bytes` → `mx.array` | 1.89 GB/s |
| `preadv` → freshly allocated MLX array | 1.49 GB/s |
| **`preadv` → preallocated arena slot** | **2.62 GB/s** |

+39%, and the `numpy → mx` conversion — which was 29% of the runtime — stops
existing rather than getting faster. Allocating per read is *worse* than the copy
it saves: `mx.zeros` writes the bytes the read is about to overwrite.

The arena is laid out exactly as MLX's `SwitchGLU` expects, so `gather_qmm`
selects experts by slot index with no restacking, and 24 matvec launches per
layer become 6.

## Correctness

The streamed block is **bit-identical to `mlx_lm`'s own MoE block**:

```
vs mlx_lm blk(x), SAME kernel: max abs 0.000e+00  relative 0.000e+00
MATCH: bit-identical to mlx_lm's own block
```

Not "within tolerance". Zero. That check is the reason to trust anything else in
this repo, and getting there required copying the framework exactly — the
normalised scores cast to `x.dtype` *before* weighting, and `swiglu` as one fused
op rather than a separate `silu(g) * u`, which rounds differently on its own.

An earlier version of this project compared against a hand-written reference and
sat at 7.8e-3, because dequantize-then-matmul and `gather_qmm` are genuinely
different computations. A tolerance wide enough to pass that would hide exactly
the indexing bugs the check exists to catch.

## The measurements that killed good ideas

This is the part worth reading. Four attacks on the fundamental bound
(`bytes/token = layers × top_k × (1 − hit) × bytes_per_expert`). Three died.

**Factor the experts into a shared basis + residual.** If `W_e = B + Δ_e` with
`B` resident at 8 MB per layer, only the residual streams and `bytes_per_expert`
collapses. Dead: mean pairwise cosine between a layer's experts is **+0.096**,
mean-subtraction leaves 98.6% of the norm, and capturing 83% of the energy takes
rank 128 of 256 — a basis costing more than what it factors. The experts are
near-orthogonal. (`basis_probe.py`)

**Skip experts the cache doesn't have.** A resident expert is free; a missing one
is 4.42 MB and a stall. So skip-if-missing-and-low-gate puts the quality loss
exactly where the speed gain is. Dead on the gate distribution: **0.192 at slot 1
against 0.093 at slot 8**. M2.5 routes sigmoid top-8, not the skewed softmax
top-2 the adaptive-k literature assumes. Dropping *one* expert costs 11.4% block
error. (`gate_drop.py`)

**Use the cache as its own draft model.** Whatever is resident is a complete
model at zero disk cost — skip the experts you don't have, renormalise, draft,
then verify the batch against the disk. No second model, no extra memory, and it
improves as the cache warms. It drafts at **189 ms/token against 1053, 5.6x** —
at **16.7% acceptance** (5 GB ceiling, 42.9% hit, 24 steps). Speculation needs
roughly 50% to pay, so it loses. Downstream of the flat gates above.

**A smarter cache policy.** Dead before it was written: the fractional-knapsack
upper bound on *any* static trace-derived pinning is 45.0% at 9 GB, and plain LRU
already measures 53.0%. (`cost_model.py`)

Also measured and discarded: sorting reads by disk offset (1.95 vs 2.07 tok/s),
more than 8 reader threads (flat), and chunking the token dimension to overlap
fetch with compute — which hid 24s of disk and cost more than it hid.

**What did move:** the assumption that tokens route independently. They don't.
33.4% of a token's top-8 is shared with the previous token, and the union over B
consecutive tokens grows at *half* the independent rate. That is the whole case
for batching, and it is measured, not modelled.

| B | ceiling | GB/token | tok/s aggregate |
|---|---|---|---|
| 1 | 6 GB | 1.72 | 1.35 |
| 32 | 6 GB | 1.077 | 2.07 |
| 64 | 4 GB | 0.686 | 2.98 |
| 512 | 2.5 GB | **0.143** | **8.31** |

The ceiling *shrinks* down that table — the machine had to give the KV cache
room — so the gain is not the cache getting bigger. Past B=8 the hit rate is 0%
anyway: a pass at B=32 touches ~82 experts per layer, 28 GB against a 6 GB arena,
so nothing survives to be reused. All of the win is the union, none of it reuse.

## Is 20 tok/s possible?

Not single-stream, and the arithmetic is short. A token touches 62 × 8 = 496
experts; every miss is a read. At 4.66 GB/s the byte budget for 20 tok/s is 233
MB/token against a 2.19 GB working set — an **89.4% hit rate**, on a 24 GB
machine where 10 GB of cache measures 61.7%.

Three independent floors, and only one is the disk: bytes, kernel launches
(expert matmul measured at **56 GFLOP/s on hardware rated in TFLOPs** — it is
launch-bound, not FLOP-bound), and the host copy. `PLAN.md` prices all of it.

As *aggregate throughput* the batch curve above does reach it, around B≈1000, if
compute holds. As latency, on this machine, no. That is a hardware statement: a
128 GB Mac holds all 128.7 GB resident and does 30+ tok/s with none of this
machinery.

## Setup

```sh
uv venv --python 3.12 && uv sync
hf download mlx-community/MiniMax-M2.5-4bit          # 128.7 GB
python requantize_m25.py --src <snap> --dst models/m25-2bit --bits 2
python build_index.py --src4 <snap> --src2 models/m25-2bit --out models/m25.idx
python m25_engine.py --snap <snap> --arena --ceiling-gb 6
```

Requires Apple Silicon. Tested on an M4 with 24 GB.

**A warning the hard way:** nothing here may call `mlx_lm.load` on these
checkpoints. 128.7 GB into 24 GB of RAM takes the machine down; it has happened.
Every script builds only the layers it needs, or tears the expert modules out
before anything materialises them.

**And a subtler one:** process RSS cannot see these weights. MLX allocates
through Metal, and a run holding 10.28 GB of arrays reports 3.82 GB resident. The
load-time guard was reading RSS specifically to catch the failure above — a
number blind to exactly the allocations that cause it. It reads
`mx.get_active_memory()` now.

## Layout

```
m25_arena.py       the arena: DMA into unified memory, gather_qmm in place
m25_engine.py      full 62-layer model, dense core resident, experts streamed
m25_store.py       the earlier per-expert LRU store, kept as the reference path
m25_stream.py      per-expert streaming block and its bit-identity check
build_index.py     addresses experts as (shard, offset) -- zero bytes copied
requantize_m25.py  builds the 2-bit cold tier
cost_model.py      what a target costs before you build for it
budget.py          feasibility from a config, before downloading anything
hw_probe.py        page alignment, DMA paths, memcpy scaling
basis_probe.py     do experts share structure worth factoring out (no)
gate_drop.py       what dropping an expert actually costs (a lot)
quant_delta.py     what each quantization actually costs
gnp_gamma.py       block error vs the gate-proportional exponent
bw.py              disk bandwidth at expert-sized blocks
```

`NOTES.md` is the measurement log — every number above with the method that
produced it, and the mistakes that produced the wrong ones first. `TECHNIQUES.md`
covers the techniques. `PLAN.md` is the cost model and what is left.

## Honest status

- **Single-stream is 1.35 tok/s.** Useful for watching a 229B model think on a
  laptop; not useful for chatting with it.
- **The 2-bit cold tier costs quality.** 5.8% relative block error at layer 1,
  and it gets worse with depth. `quant_delta.py` prices the alternatives.
- **Prefill measurements have 1.5x run-to-run variance** on identical work. Cause
  unknown. Single-run prefill numbers in `NOTES.md` should be read with that in
  mind; the decode numbers repeat to three digits.
- **One prompt, short outputs.** Length is covered to 64 tokens, variety is not,
  and greedy decode that falls into a loop flatters the hit rate.
- **`gather_qmm` computes each tier's entries separately**, which is right at
  batch, but no attempt has been made to overlap fetch with compute per tier —
  the one remaining idea with a real factor in it.

## License

Not chosen yet — there is no LICENSE file in this repository. Without one the
default is "all rights reserved", which is probably not what you want if this is
going public. Add one before announcing it.
