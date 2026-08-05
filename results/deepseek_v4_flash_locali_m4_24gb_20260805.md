# DeepSeek V4 Flash through Locali on M4/24 GB

Measured on 2026-08-05 on a fanless 10-core Apple M4 MacBook Air with 24 GB
unified memory and macOS 26.6.

## Runtime boundary

Locali keeps the mixed-precision MLX backbone resident and streams only the
routed experts through a bounded unified-memory arena. The architecture and
chat encoding come from the DeepSeek V4 implementation vendored by oMLX;
Locali owns loading, expert placement, SSD reads, scheduling and streamed MoE
execution.

At the tested 7 GB ceiling:

| Allocation | Size |
|---|---:|
| Resident dense backbone | 6.49 GB |
| Fixed expert arena, 988 slots | 6.99 GB |
| Total active MLX memory | 13.50 GB |

The checkpoint occupies 92.8 GB decimal. The expert-major pack adds 77.91 GB
and reduces each expert from nine sparse source reads to one vectored read.

## Performance

Every final row uses 64 tokenizer tokens from the bundled corpus and forces 128
greedy decode steps.

| Configuration | Prefill tok/s | Decode tok/s | Steady tok/s | Expert hit | Bytes read |
|---|---:|---:|---:|---:|---:|
| Packed + macOS cache run 1 | 5.61 | 2.32 | 2.37 | 60.6% | 92.14 GB |
| Packed + macOS cache run 2 | 5.71 | 2.34 | 2.38 | 60.6% | 92.14 GB |

Both replicas used the final repository tree and produced identical routing
statistics. The corpus routes less locally than earlier traces, which is why
throughput is reported with hit rate and bytes read instead of treated as a
model-independent constant.

The native allocation-free C planner measured 2.690 tok/s against 2.666 for the
equivalent Python planner on the same 32-token trace. Optional fused expert
math, hit/read overlap, LFU and embedded speculative decoding all remained off
by default after losing their controlled A/B runs on this machine.

## Interactive latency and output quality

Priming the immutable chat prefix before accepting input reduced the measured
submit-to-first-visible latency from 6.545 s to 4.152 s in the original short
probe. A consuming detokenizer property was also fixed so the first printable
segment reaches the terminal immediately.

For `Chi è spiderman?`, the final deterministic profile produced one coherent
152-token answer beginning `Spider-Man (Spiderman) è un supereroe dei fumetti`
and terminated at EOS. In the clean reproduction, first visible text arrived
3.728 s after submission and decode measured 2.95 tok/s.

The self-contained 20-case continuation fixture contains 480 target tokens:

| Avg target NLL | Perplexity | First token | Token top-1 |
|---:|---:|---:|---:|
| 0.73796 | 2.09 | 11/20 | 393/480 (81.9%) |

The streamed MoE is tested against resident MLX modules using the same
quantized bytes. Expert output is bit-identical; reduced end-to-end models also
verify dense loading and final logits.

## Reproduce

```sh
python deepseek_v4.py --nothink

.venv/bin/python dsv4_bench.py \
  --index models/dsv4-2.4bit-packed.idx --ceiling-gb 7 --os-cache \
  --out results/deepseek_v4_flash_final.json

.venv/bin/python dsv4_quality.py \
  --index models/dsv4-2.4bit-packed.idx --ceiling-gb 7
```
