"""What speculative decoding would actually cost here, before porting a draft.

On a GPU holding all its weights, verifying gamma draft tokens in one pass is
nearly free: the weights are read once regardless of how many tokens ride along,
so the speedup tracks the acceptance rate. This engine does not hold its weights.
A verify pass over gamma tokens must read the UNION of their routed experts, and
that union grows with gamma -- sublinearly, but it grows. Rejected drafts pay for
their experts too.

So the ceiling here is set by bytes, not by acceptance, and it is measurable
without a draft model at all. This measures it:

  experts(gamma) = mean over windows of gamma consecutive decode tokens of
                   |union of their top-k sets|, per layer

and turns it into the speedup an ideal draft could reach:

  speedup(gamma, alpha) = accepted(gamma, alpha) / (experts(gamma) / top_k)

with accepted = sum_{i=0..gamma} alpha^i, the standard chain-acceptance
expectation. A draft that is right every time still cannot beat
top_k / experts(gamma->inf).

The README's batching table already implies the shape -- "the union over B
consecutive tokens grows at HALF the independent rate" -- but that was measured
across a wide batch to justify serving throughput, and the numbers here are the
single-sequence consecutive-token case that speculation actually runs in.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from m25_arena import ArenaMoE
from m25_engine import load_streaming, make_sized_prompt_cache

ALPHAS = (0.6, 0.7, 0.8, 0.9, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25-hotpack.idx")
    ap.add_argument("--ceiling-gb", type=float, default=7.0)
    ap.add_argument("--hot-share", type=float, default=0.50)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max-gamma", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="results/spec_union.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    snap = Path(a.snap)
    tok = AutoTokenizer.from_pretrained(str(snap))
    model, store, cfg, dense = load_streaming(
        snap, a.index, a.ceiling_gb, arena=True, hot_share=a.hot_share,
        nocache=False)
    print(f"dense core resident: {dense:.2f} GB")

    per_layer = defaultdict(list)          # layer -> [set per decode token]
    orig = ArenaMoE.__call__

    def spy(self, x):
        inds, _ = self.route(x)
        per_layer[self.layer].append(
            set(np.asarray(inds).reshape(-1, self.top_k)[0].tolist()))
        return orig(self, x)

    ArenaMoE.__call__ = spy
    try:
        ids = mx.array(tok(a.prompt)["input_ids"])
        cache = make_sized_prompt_cache(model, ids.size + a.tokens)
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        per_layer.clear()                  # prefill is not the regime
        y = int(mx.argmax(logits[0, -1]))
        for _ in range(a.tokens):
            logits = model(mx.array([[y]]), cache=cache)
            mx.eval(logits)
            y = int(mx.argmax(logits[0, -1]))
    finally:
        ArenaMoE.__call__ = orig

    top_k = model.model.layers[0].block_sparse_moe.__dict__["_stream"].top_k
    n_tok = min(len(v) for v in per_layer.values())
    print(f"{n_tok} decode tokens x {len(per_layer)} layers, top-{top_k}\n")

    # experts(gamma): distinct experts a verify pass over gamma consecutive
    # tokens must have resident, averaged over every window and every layer.
    experts = {}
    for g in range(1, a.max_gamma + 1):
        vals = []
        for sets in per_layer.values():
            for i in range(0, n_tok - g + 1):
                u = set()
                for s in sets[i:i + g]:
                    u |= s
                vals.append(len(u))
        experts[g] = float(np.mean(vals))

    print(f"  {'gamma':>5} {'experts':>8} {'per token':>10} {'bytes vs g=1':>13}")
    for g in range(1, a.max_gamma + 1):
        print(f"  {g:>5} {experts[g]:>8.2f} {experts[g] / g:>10.2f} "
              f"{(experts[g] / g) / experts[1]:>12.2f}x")

    # A verify pass over gamma DRAFT tokens covers gamma+1 positions -- the
    # drafts plus the one the target would have produced anyway -- so its bytes
    # are experts(gamma+1), not experts(gamma). Pairing the acceptance of
    # gamma+1 tokens with the byte cost of gamma overstates every entry.
    print(f"\n  speedup = accepted(gamma, alpha) / (experts(gamma+1)/{top_k})")
    print(f"  {'gamma':>5} " + " ".join(f"a={al:<4}" for al in ALPHAS))
    best = (0.0, None, None)
    table = {}
    for g in range(1, a.max_gamma):
        row = []
        for al in ALPHAS:
            accepted = min(sum(al ** i for i in range(g + 1)), g + 1)
            sp = accepted / (experts[g + 1] / top_k)
            row.append(sp)
            if sp > best[0]:
                best = (sp, g, al)
        table[g] = dict(zip((str(x) for x in ALPHAS), row))
        print(f"  {g:>5} " + " ".join(f"{v:>5.2f} " for v in row))

    gmax = a.max_gamma - 1
    print(f"\n  best {best[0]:.2f}x at gamma={best[1]}, acceptance {best[2]:.0%}")
    print(f"  ceiling with a PERFECT draft, gamma={gmax}: "
          f"{(gmax + 1) / (experts[gmax + 1] / top_k):.2f}x")
    print("\n  Optimistic on top of that: the union is measured along the trace "
          "the model\n  actually generated. A rejected draft routes somewhere "
          "else, so real unions\n  grow faster than these.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "tokens": n_tok, "layers": len(per_layer), "top_k": top_k,
        "experts_per_gamma": experts, "speedup": table,
        "best": {"speedup": best[0], "gamma": best[1], "alpha": best[2]},
    }, indent=2))
    print(f"\nwrote {a.out}")
    store.close()


if __name__ == "__main__":
    main()
