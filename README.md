# Locali — local inference on consumer hardware

Locali runs models locally on machines that should not be able to run them,
and reports what that actually costs. Every number here was measured on a
fanless 10-core Apple M4 MacBook Air with 24 GB of unified memory; the JSON
behind each table is in `results/`.

The repository holds three paths that share one idea: the cost of local
inference is `work per forward pass x number of forward passes`, and most of
the available speed is in the second term, not the first.

## 1. Typed decisions

Generating a sentence to answer a question that has four possible answers
costs one forward pass per token. Answering it as a *typed decision* costs
one, total.

A question is a `Choice`, a `Score` or a `Bool`. The engine prefills the state
once, then reads logits at a single position, masked to the tokens that can
begin a valid option. Nothing off-schema can be returned, and the returned
probability is meant to be trusted rather than displayed.

```python
from resident_mlx import ResidentMLX
from schema import Choice
from decide import decide, prime

engine = ResidentMLX("mlx-community/Qwen3-4B-4bit")
primed = prime(engine, "You route support tickets. Answer with a single letter.")

question = Choice(name="dept", question="Which department?",
                  options=("billing", "technical", "sales"))
d = decide(engine, "Ticket: 'Charged twice, need a refund.'", question, primed=primed)
# d.value 'billing'   d.confidence 1.000   d.schema_mass 1.000
```

`prime()` prefills the immutable system prefix once and every later call forks
it, which is worth 2.5x (477 ms to 190.6 ms). `decide_many()` answers K
questions from one prefill; against re-fusing the state each time it wins
1.83x at K=8, with fork itself costing 0.04 ms.

`schema_mass` is the share of the model's probability that landed on any valid
option. It is reported, not hidden inside the renormalization, because it is
how a silent failure becomes visible. It has caught three in this repository:
a readout aimed at a token holding 0.000000 of the mass, an instruct model
being fed a bare completion prompt, and a reasoning model answering `<think>`
instead of an answer.

### Calibration

`calibration.py` fits temperature, vector or isotonic scaling on a labelled
split and reports ECE, MCE, Brier and NLL. Calibration reshapes confidence; it
never overturns which option was chosen.

Six resident 4-bit checkpoints over the 181-case fixture in `eval/`:

| model | accuracy | separation | ECE after | schema mass | primed decision |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.653 | 0.117 | 0.152 | 1.000 | 194.8 ms |
| Llama-3.2-3B | 0.569 | 0.306 | 0.123 | 0.994 | 149.8 ms |
| Qwen3-1.7B | 0.486 | 0.133 | 0.147 | 0.997 | 85.6 ms |
| Llama-3.2-1B | 0.472 | 0.146 | 0.137 | 0.963 | 57.7 ms |
| Qwen3-0.6B | 0.389 | 0.125 | 0.150 | 0.264 | 34.6 ms |
| gemma-3-1b | 0.347 | 0.160 | 0.144 | 1.000 | 49.6 ms |

`separation` is mean confidence when correct minus mean confidence when wrong.
It decides whether a confidence threshold can gate anything, and at 0.12 to
0.31 it is currently too weak to build an escalation cascade on. That is the
open problem, stated rather than smoothed over.

Accuracy is reported against a bag-of-words Naive Bayes floor, computed per
family and shipped in `eval_calibration.py` as a standing guard. On the
entailment family that floor is 0.62 and no model clears it. On sentiment it
is 0.28 and every model clears it by a wide margin. These checkpoints do
surface perception well and inference poorly, and a fixture that cannot show
the difference is not measuring anything — an earlier templated fixture was
solved outright by that Naive Bayes, at 1.00 on three families of four.

## 2. Per-frame decisions

A monitoring stream asks the same small question of every frame. It does not
need prose, so the typed-decision path above applies directly: one constrained
readout per frame.

What dominates that cost is the image. Measured back to back on one machine,
answering the same question about the same frame at 448x336:

| model | image tokens | ms/frame | resident |
|---|---:|---:|---:|
| MiniCPM-V-4.6-4bit | 63 | 308 | 2.16 GB |
| Qwen3-VL-4B-Instruct-4bit | ~160 | 772 | 3.10 GB |
| Qwen3-VL-30B-A3B-Instruct-4bit | ~160 | 1367 | 18.25 GB |
| Qwen3.8-27B-4bit (dense) | ~165 | 6283 | 16.06 GB |

Cost tracks image tokens, and image tokens are an architectural choice rather
than a resolution one. Qwen-VL emits tokens in proportion to pixels, so its
per-frame cost rises with resolution: 113 tokens and 493 ms at 320x240, 1233
tokens and 5850 ms at 1280x960. MiniCPM-V resamples to a fixed budget and
spends 63 tokens on any frame at 448x336 or below, which is why lowering the
resolution further buys it nothing at all.

That matters for how to spend effort. Degrading resolution is the obvious lever
and it costs detail. Choosing an architecture that resamples keeps the
resolution and the speed together.

Sparsity did not help here. The 30B MoE has roughly 3B active parameters and
is still slower than the dense 4B, because 30B of weights have to move through
a 120 GB/s memory system whether or not they are all active, and it needs six
times the resident memory to do it.

### Skipping frames

A fixed camera produces long runs of near-identical frames. `gate.py` compares
a 16x16 luminance signature against the last frame that was actually inferred —
not against the previous frame, or a slow drift walks past the threshold one
imperceptible step at a time — and skips when it has not moved.

On a 300-frame synthetic sequence with four events of 4 to 12 frames, sensor
noise and a lighting drift larger than any single event:

| threshold | inferred | skip | event recall | effective ms/frame |
|---:|---:|---:|---:|---:|
| 0.002 | 59 | 80.3% | 100% | 61.1 |
| 0.005 | 27 | 91.0% | 100% | 28.2 |
| 0.010 | 10 | 96.7% | 0% | 33.9 |

Skip rate alone is a vanity number: 0.010 skips 96.7% and sees nothing. The
pair is the result. At 0.005, 91% of frames are skipped, no event is missed,
detection lags by at most 3 frames, and 308 ms per frame becomes 28.2 ms.

A periodic heartbeat turns out to be a staleness bound rather than a detector.
To be relied on to land inside the shortest event it must be no longer than
that event, which caps skip at 1 - 1/length on its own and costs more than it
saves. The threshold does the work.

## 3. Streamed experts

The third path runs a model whose routed experts do not fit in unified memory.
`DeepSeek-V4-Flash-0731` at 2.44-bit keeps a 6.49 GB dense backbone resident
and streams 256 routed experts from SSD through a fixed 6.99 GB arena.

| Run | Prefill | Decode | Steady | Expert hit | Bytes read |
|---|---:|---:|---:|---:|---:|
| 1 | 5.61 t/s | 2.32 t/s | 2.37 t/s | 60.6% | 92.14 GB |
| 2 | 5.71 t/s | 2.34 t/s | 2.38 t/s | 60.6% | 92.14 GB |

That is 719.8 MB read per decoded token at an effective 3.11 GB/s, so 231.5 ms
of the 430 ms per token is I/O stall. With perfect overlap the ceiling is
4.32 tok/s, which leaves 1.86x for all remaining engineering and no more. The
way past a wall like that is to stop paying per token, which is what the first
two paths do.

- `dsv4_engine.py` loads the resident backbone and installs streamed MoE
- `arena.py` owns the bounded SSD-to-memory expert cache
- `native/locali_core.c` is the allocation-free SLRU/LFU scheduler
- `dsv4_index.py` and `pack_experts.py` build the zero-copy layouts

Setup, the packed expert layout and the chat TUI are documented in
`results/deepseek_v4_flash_locali_m4_24gb_20260805.md`.

## Running things

```sh
uv sync
pytest -q                                   # 121 passed, 1 skipped

python bench_resident.py --model mlx-community/Qwen3-4B-4bit \
    --out results/resident_qwen3_4b_4bit.json
python eval_calibration.py --model mlx-community/Qwen3-4B-4bit
```

Checkpoints download into `.runtime/models/` on first use.

## Measuring

Two rules this repository learned the expensive way.

Audit the instrument, not only the code. A labelled fixture that a
bag-of-words classifier can solve measures template matching. A readout can be
aimed at a token that carries none of the model's probability and still return
the right answer often enough to look correct.

Time with the variants interleaved inside each repetition, at least five
repetitions, report the min/max spread next to the median, and check the
machine is idle first. On a fanless part under concurrent load this repository
produced a table in which K=4 was slower than K=8.
