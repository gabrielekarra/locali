# locali — MiniMax-M2.5, 229B, on a 24 GB Mac

The whole model runs. Expert weights live on disk and are read on demand into an
LRU cache with a hard byte ceiling; only the dense backbone stays resident.

```
$ python m25_engine.py --snap <4-bit snapshot> --index models/m25.idx --ceiling-gb 6
dense core resident: 2.74 GB  (loaded in 1s)
prefill 5 tokens in 13.4s
  hit 22.5%  read 10.22 GB  evictions 791  peak 6.00 GB  rss 6.29 GB
  next token: ' Paris'
```

`peak 6.00 GB` against a 6.00 GB ceiling is the invariant working, not luck: the
store evicts *before* inserting, so the window in which residency could exceed
the ceiling does not exist.

## Why this model

A 229B MoE fits here for one structural reason: **no shared experts**. The dense
backbone is 4.02B params — 2.26 GB at 4-bit — which leaves ~16 GB of cache, or
58 slots per layer of 256 against top-8. That is *above* top-k, and above top-k
is where caching starts working at all. Below it the hit rate is zero and no
policy rescues it.

## What is measured

`budget.py` prices three costs per token — disk, memory bus, implementation —
and prints the path from where the engine is today to where it can get:

```
step                                          hit    t_io   t_ram   tok/s
0  today: 4-bit, Python, no overlap         59.9%    396m     49m     1.7
2  + 2-bit cold experts                     75.8%    133m     34m     3.2
3  + C/Metal engine (kills t_over)          75.8%    133m     34m     6.0
4  + prefetch: overlap I/O w/ compute       75.8%    133m     34m     7.5
7  + MTP speculation (gamma=2.1)            83.8%     52m     18m    19.2
8  + GNP W3-first (gamma=1.5)               83.8%     37m     18m    27.3
```

Rungs 0–2 exist. Everything from 3 on is designed and priced, not built.

`t_ram` is a floor, not a cost to optimise away: every weight a token touches
crosses the memory bus wherever it lives. On this machine that is 120 GB/s, and
it is what caps the ladder.

## Techniques

See `TECHNIQUES.md`. The new one is **gate-proportional neuron paging**: an
expert is a sum of `d_ff` rank-1 neuron terms, so it is divisible, while the
literature pages at expert granularity by assumption. Two ways of exploiting
that died on measurement; the one that works reads `W3` alone — a third of the
expert — ranks neurons by `|W3 x|`, and fetches the top `m_e` of `W1`/`W2` with
the budget set by the router's gate weight. Block error 3.2% at 1.29x.

Also measured, and it overturned the plan: uniform 2-bit experts cost 25.7%
block error, 3-bit costs 12.8%, but **mixed hot-4bit/cold-2bit beats both** —
7.4% error at 29.4 tok/s where uniform 3-bit gives 12.8% at 26.2. Cached experts
set quality and cost no disk; streamed experts set speed and are rare.

## Layout

```
m25_engine.py      full 62-layer model, dense core resident, experts streamed
m25_store.py       LRU + hard ceiling over positional reads (os.pread, no mmap)
m25_stream.py      the streaming MoE block, and its bit-identity check
build_index.py     addresses experts as (shard, offset) -- zero bytes copied
requantize_m25.py  builds the 2-bit cold tier
budget.py          the cost model and the ladder
neuron_tail*.py    the measurements that decided GNP
gnp_gamma.py       block-level error vs the gate-proportional exponent
quant_delta.py     what each quantization actually costs
bw.py              disk bandwidth at expert-sized blocks
```

## Setup

```sh
uv venv --python 3.12 && uv sync
hf download mlx-community/MiniMax-M2.5-4bit        # 128.7 GB
python requantize_m25.py --src <snap> --dst models/m25-2bit --bits 2
python build_index.py --src4 <snap> --src2 models/m25-2bit --out models/m25.idx
```

Every script that touches weights builds only the layers it needs, or tears the
expert modules out before anything materialises them. Nothing here may call
`mlx_lm.load` on these checkpoints — 128.7 GB into 24 GB of RAM takes the
machine down.

## Honest status

- The engine runs a forward pass. **There is no generation loop yet** — no KV
  cache, so no real tokens/s and no routing trace.
- 2.7 s/token is the reference path: hot and cold tiers have different bit
  widths and cannot share a stacked `gather_qmm`, so each expert is dequantized
  on its own. Slow on purpose, and pinned to `0.000e+00` against the resident
  weights before any of it gets optimised.
- The hot/cold split in the index is **an arbitrary placeholder**. It needs a
  real routing trace, which needs the generation loop.
- GNP is measured on two shallow layers over 24 tokens, and deeper layers are
  consistently worse. The whole-model number could land below 1.4x.
