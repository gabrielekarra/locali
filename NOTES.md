# Notes

Measurement log for MiniMax-M2.5 streaming. Numbers only, with the method that
produced them and the mistakes that produced the wrong ones first.

Machine: M4, 24 GB unified memory (25.77 reported), internal SSD, macOS.
Model: `mlx-community/MiniMax-M2.5-4bit`, 229B, 128.7 GB, 27 shards.

## The model, from its config and param census

| quantity | value |
|---|---|
| MoE layers | 62 |
| routed experts / layer | 256, top-8 |
| shared experts | **none** — this is why it fits |
| params per expert | 14,155,776 (3 x 3072 x 1536) |
| bytes per expert | 7.96 MB at 4-bit, 4.42 MB at 2-bit |
| routed total | 224.7B, 126 GB at 4-bit |
| dense core | 4.02B, **2.26 GB at 4-bit** |
| cache that leaves | ~16 GB = 58 slots/layer of 256, against top-8 |

58 > 8 is the fact the project rests on. Below top-k slots the hit rate is zero
by construction — measured, on an earlier trace: 4.1% at 8 slots against top-10,
41.9% at 15. A model whose dense core eats the cache never gets above that floor
and no policy rescues it.

## Disk bandwidth depends on the block

| block | 8t | 16t | 24t | 32t | 48t | 64t |
|---|---|---|---|---|---|---|
| 4.42 MB | 4.00 | 3.98 | 3.93 | 3.87 | 4.17 | 4.00 |
| 7.96 MB | 4.33 | 4.43 | 4.46 | 4.54 | 4.44 | 4.68 |
| 17.5 MB | 4.57 | 4.66 | 4.24 | 4.76 | 4.52 | 4.46 |

Random `pread` with `F_NOCACHE`, GB/s. Reads at an expert's size run ~17% slower
than at 4x that size, and threads stop helping at 8.

**The mistake first:** the same benchmark against an 8 GB file reported 14.7 GB/s
at 16 threads. `F_NOCACHE` does not evict what the page cache already holds, and
the file had just been written. Any bandwidth number has to come from a file
larger than RAM.

## What quantization actually costs

Relative error of the MoE block output against the 4-bit reference, real hidden
states, `quant_delta.py`:

| setting | bits/weight | MB/expert | block error |
|---|---|---|---|
| 2-bit g64 | 2.50 | 4.42 | 0.2568 |
| 2-bit g32 | 3.00 | 5.31 | 0.2377 |
| 3-bit g64 | 3.50 | 6.19 | 0.1278 |
| 3-bit g32 | 4.00 | 7.08 | 0.1129 |

Layer 3 is worse than layer 1 for the same setting: 0.3316 against 0.2568 at
2-bit. Depth costs.

2-bit g32 is dominated — more bytes than 2-bit g64 for barely less error, and
far worse than 3-bit. **Mixed hot/cold beats every uniform setting**, because
cached experts set quality and cost no disk while streamed experts set speed and
are rare:

| hot/cold | slots | hit | error | tok/s |
|---|---|---|---|---|
| 2b / 2b | 58.1 | 83.8% | 25.7% | 52.4 |
| 3b / 2b | 41.5 | 76.8% | 15.8% | 36.7 |
| 3b / 3b | 41.5 | 76.8% | 12.8% | 26.2 |
| **4b / 2b** | 32.3 | 71.1% | **7.4%** | **29.4** |
| 4b / 3b | 32.3 | 71.1% | 3.7% | 21.0 |

4b/2b dominates 3b/3b on both axes. The plan had been uniform 2-bit; measuring
it first is what stopped that.

## Gate-proportional neuron paging

Relative L2 error of one expert when only its top-m neurons are fetched, by how
the neurons are ranked (`neuron_tail_live.py`, layer 1, real activations):

| m/d_ff | 10% | 20% | 30% | 40% | 50% | 66% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|
| flat spectrum | 0.95 | 0.89 | 0.84 | 0.77 | 0.71 | 0.58 | 0.45 | 0.32 |
| static weight-norm | 0.86 | 0.78 | 0.71 | 0.65 | 0.57 | 0.45 | 0.33 | 0.22 |
| rank-16 sketch | 0.78 | 0.70 | 0.64 | 0.57 | 0.50 | 0.40 | 0.31 | 0.21 |
| **\|W3 x\| only** | 0.57 | 0.44 | 0.34 | 0.26 | 0.19 | 0.11 | 0.05 | 0.02 |
| oracle | 0.41 | 0.27 | 0.19 | 0.13 | 0.09 | 0.04 | 0.02 | 0.00 |

The concentration is real — the oracle keeps 40% of neurons at 13% error — but
the two cheap ways of finding it both failed. Static ranking lands at 0.65 where
flat is 0.77; a resident rank-16 sketch closes 15% of the gap, and higher ranks
do not rescue it because expert matrices are near-full-rank and the ranking
depends on a product of two noisy inner products.

What works is structural rather than statistical: the order is carried by the
linear factor, so ranking on `|W3 x|` alone — one third of the expert — closes
75% of the gap.

Block-level, with the budget set by the router (`gnp_gamma.py`):

| gamma | bytes | saving | block err (L1) | block err (L3) |
|---|---|---|---|---|
| 0.00 | 1.000 | 1.00x | 0.0000 | — |
| 0.50 | 0.871 | 1.15x | 0.0129 | 0.0163 |
| 1.00 | 0.775 | 1.29x | 0.0320 | 0.0404 |
| 2.00 | 0.647 | 1.55x | 0.0741 | 0.0907 |
| 3.00 | 0.571 | 1.75x | 0.1148 | 0.1387 |

Gate weighting is worth 3.4x: 3.2% block error where the uniform per-expert
error at comparable bytes was 11%. `gamma=0` printing exactly 1.000 bytes and
0.0000 error is the self-check — uniform budget means every expert whole.

## First end-to-end run

```
dense core resident: 2.74 GB  (loaded in 1s)
prefill 5 tokens in 13.4s
  hit 22.5%  read 10.22 GB  evictions 791  peak 6.00 GB  rss 6.29 GB
  next token: ' Paris'
```

Peak equal to the ceiling to the byte, over 791 evictions. Read the 22.5% as a
cold cache over five tokens, not an operating point.

**Correctness, and the trap in it.** The streamed block first came out 7.8e-3
from the resident one, which looks like a bug and is not: `gather_qmm` on packed
weights and dequantize-then-matmul are different computations. Comparing against
the framework kernel can only ever be a tolerance check, and a tolerance wide
enough to pass 7.8e-3 would hide any indexing bug worth catching. Holding the
arithmetic fixed so the only variable is the fetch gives **0.000e+00**.

## Generation, with a KV cache and a trace-derived index

`m25_engine.py`, 6 GB ceiling — the same peak the run above hit, so the cache
size is held fixed and the only variables are the batching, the KV cache and the
index built from the routing trace:

```
dense core resident: 2.73 GB  (loaded in 1s)
prefill 5 tokens in 9.7s (0.52 tok/s)
  hit 0.0%  read 10.79 GB
decode 16 tokens in 32.9s (0.49 tok/s)
  hit 40.7%  read 26.46 GB (1.65 GB/token)
  evictions 5592  peak 6.00 GB  rss 2.96 GB
  where the time went: pread 24.2s  numpy->mx 9.2s  eval 0.3s = 33.8s of 42.5s (79%)

The capital of France is Paris. The official language is French. The currency
is the Euro. The population
```

Prefill and decode are reported separately because they are different regimes:
prefill touches most experts of every layer, decode touches top-k, and one
averaged tok/s hides both.

**Prefill's 0.0% against the earlier 22.5% is the batching, not a regression.**
The old path looped over tokens and re-fetched the same expert once per token,
so a five-token prefill scored hits on its own repeats. Grouping tokens by
expert fetches each one once per layer call and leaves no repeat to hit. Same
bytes, honest denominator.

**Where it actually goes: 24.2s of pread is 37.25 GB at 1.54 GB/s**, against the
4.00-4.66 GB/s `bw.py` measures on this disk. The reads are serial and
single-threaded, one array at a time, at a block size well under an expert.
Nothing about the cache or the policy is the bottleneck — the fetch is. The
`numpy->mx` 9.2s is 4.0 GB/s of memcpy, close to `t_ram` and not going away
without reading into a buffer mlx already owns.

**Correctness after the rewrite.** Both paths moved to `quantized_matmul` on
grouped tokens; bit-identity against the resident block still prints
`0.000e+00` (`--index models/m25-allhot.idx --layer 1`), with the gather_qmm
kernel gap unchanged at 7.812e-03.

## Threading the fetch

The router hands over the whole routed set before any of it is needed, so the
misses do not have to be discovered one at a time. `get_many` takes the batch,
evicts once for the total, and issues every read through a pool of 8.

Same ceiling, same prompt, same index — only the fetch changed:

| | serial | 8 threads |
|---|---|---|
| prefill | 0.52 tok/s | **0.73** |
| decode | 0.49 tok/s | **0.56** |
| pread | 24.2s | **12.7s** |
| numpy->mx | 9.2s | 12.8s |
| eval | 0.3s | 0.2s |
| total | 42.5s | 35.5s |
| rss | 2.96 GB | 4.66 GB |

Hit rate, bytes read and evictions are unchanged to three digits (40.7%, 26.45
GB, 5591 against 5592), which is the check that this changed only the fetch:
same reads, issued differently.

**2.93 GB/s, not the 4.00-4.66 `bw.py` reports.** That benchmark reads uniform
4.42 MB blocks. An expert is nine reads — three weights of a few MB each and six
scale/bias arrays of a few KB — so most of the queue is small reads that never
reach streaming bandwidth. The remaining factor is in the block size, not in
more threads.

**The cost moved rather than vanished: `numpy->mx` is now the largest single
line, and it went up.** 37 GB at 2.9 GB/s against a 120 GB/s bus is not
bandwidth — it is ~50k separate `mx.array` allocations at ~256 us each. It got
worse because a batch holds every raw buffer alive at once where the serial path
reused one hot page, which is also where the 1.7 GB of extra RSS comes from.

## Open

- `numpy->mx` at 12.8s, ~256 us per array, is the next factor. It is allocator
  pressure, not the memory bus: one buffer per expert, sliced, or a free-list
  fed by evictions.
- The per-batch raw buffers are a real second tier that the ceiling does not
  account for: 278 MB transient at the widest prefill layer. Small, but it is
  exactly the kind of unaccounted residency the ceiling exists to forbid.
- `rss 2.96 GB` under a 6.00 GB store-accounted peak (serial run): mlx's Metal
  buffers are not landing in RSS, so the two numbers are measuring different
  things and process RSS is no longer the independent check on the ceiling it
  was meant to be.
- Decode's 40.7% is 21 slots/layer at a 6 GB ceiling. The 58-slot design point
  needs ~16 GB free and has not been run.
- GNP measured on layers 1 and 3 only, 24 tokens. Layers past ~5 need every
  preceding layer resident, which does not fit — so the deep-layer answer waits
  on the engine. Both GNP and quantization get worse with depth, so the
  whole-model numbers should be expected below the shallow ones.
