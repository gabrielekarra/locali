"""Is layer L+1's router predictable from layer L's MoE input?

`depth_bw.py` established that the engine stalls 9.5 ms per layer against a
synthetic 9.65 ms at the same queue depth, and that the depth is capped at ~4.4
by the algorithm rather than by any tunable: a layer misses top_k x (1 - hit)
experts, issues those reads, and barriers on them. Depth 4 measures 3.96 GB/s
where depth 16 measures 5.91, so raising depth is worth ~1.5x at no cost in
quality. The only way to raise it is to have layer L+1's reads already in flight
while layer L computes.

That needs L+1's expert set one layer early. NOTES records a prefetch that
predicted the NEXT TOKEN's experts from the current token's and reached 37%,
which did not survive at length. This is the other axis, and the one DALI uses:
same token, next layer.

The prediction is not free of hazard. The MoE input at L+1 is

    RMSNorm_{L+1}( r_L + moe_L(x_L) + attn_{L+1}(...) )

so predicting from x_L skips an entire attention block and two norms with
different scales. What is being measured is whether the RANKING survives that,
not whether the logits do.

Reported against the two baselines that matter: the same-layer top-8 of the
PREVIOUS token, which is the prefetch that already failed, and chance at
8/256 = 3.1%.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from m25_arena import ArenaMoE
from m25_engine import load_streaming, make_sized_prompt_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25-hotpack.idx")
    ap.add_argument("--ceiling-gb", type=float, default=7.0)
    ap.add_argument("--hot-share", type=float, default=0.50)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--os-cache", action="store_true", default=True)
    ap.add_argument("--distances", default="1",
                    help="predict layer L+d from layer L's MoE input, for each "
                         "d. Depth is what the queue needs: d=1 puts one layer "
                         "of reads in flight, d=2 puts two")
    ap.add_argument("--out", default="results/crosslayer.json")
    a = ap.parse_args()
    dists = sorted(int(d) for d in a.distances.split(","))

    snap = Path(a.snap)
    tok = AutoTokenizer.from_pretrained(str(snap))
    model, store, cfg, dense = load_streaming(
        snap, a.index, a.ceiling_gb, arena=True, hot_share=a.hot_share,
        nocache=not a.os_cache)
    print(f"dense core resident: {dense:.2f} GB")

    streamers = [l.block_sparse_moe.__dict__["_stream"]
                 for l in model.model.layers]
    top_k = streamers[0].top_k
    nlayers = len(streamers)

    # (distance, layer) -> predicted top-k set, made `distance` layers early
    pred_for = {d: {} for d in dists}
    prev_true = {}         # layer -> that layer's top-k on the PREVIOUS token
    hits_next = {d: defaultdict(list) for d in dists}
    hits_prevtok = defaultdict(list)
    orig = ArenaMoE.__call__

    def spy(self, x):
        inds, _ = self.route(x)
        true = set(np.asarray(inds).reshape(-1, top_k)[0].tolist())

        for d in dists:
            if self.layer in pred_for[d]:
                hits_next[d][self.layer].append(
                    len(true & pred_for[d][self.layer]) / top_k)
        if self.layer in prev_true:
            hits_prevtok[self.layer].append(
                len(true & prev_true[self.layer]) / top_k)
        prev_true[self.layer] = true

        for d in dists:
            nxt = self.layer + d
            if nxt < nlayers:
                s = streamers[nxt]
                # s.route on THIS layer's input: the whole question is whether
                # the ranking survives the missing attention blocks and the norm
                # changes. Each extra layer of distance skips another one.
                pi, _ = s.route(x)
                pred_for[d][nxt] = set(
                    np.asarray(pi).reshape(-1, top_k)[0].tolist())

        return orig(self, x)

    ArenaMoE.__call__ = spy
    try:
        ids = mx.array(tok(a.prompt)["input_ids"])
        cache = make_sized_prompt_cache(model, ids.size + a.tokens)
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        # prefill routes many tokens at once; only decode is the regime of interest
        for d in dists:
            hits_next[d].clear(); pred_for[d].clear()
        hits_prevtok.clear(); prev_true.clear()
        y = int(mx.argmax(logits[0, -1]))
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            logits = model(mx.array([[y]]), cache=cache)
            mx.eval(logits)
            y = int(mx.argmax(logits[0, -1]))
        dt = time.perf_counter() - t0
    finally:
        ArenaMoE.__call__ = orig

    per_layer = {d: {L: float(np.mean(v)) for L, v in sorted(hits_next[d].items())}
                 for d in dists}
    prevtok = {L: float(np.mean(v)) for L, v in sorted(hits_prevtok.items())}
    print(f"\n{a.tokens} decode tokens in {dt:.1f}s\n")
    print(f"router at layer L+d predicted from layer L's MoE input, "
          f"fraction of top-{top_k} recovered\n")
    print(f"  {'d':>3} {'overall':>8} {'experts':>9}   by depth")
    for d in dists:
        allv = np.array(list(per_layer[d].values()))
        bands = []
        for lo in range(0, nlayers, 16):
            sel = [v for L, v in per_layer[d].items() if lo <= L < lo + 16]
            if sel:
                bands.append(f"{lo:2d}-{min(lo + 15, nlayers - 1):2d} "
                             f"{np.mean(sel) * 100:4.1f}%")
        print(f"  {d:>3} {allv.mean() * 100:>7.1f}% "
              f"{allv.mean() * top_k:>6.2f}/{top_k}   {'  '.join(bands)}")
    pv = np.array(list(prevtok.values()))
    print(f"\n  previous token, same layer (the prefetch that failed): "
          f"{pv.mean() * 100:.1f}%")
    print(f"  chance: {top_k}/{cfg['num_local_experts']} = "
          f"{top_k / cfg['num_local_experts'] * 100:.1f}%")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"tokens": a.tokens, "top_k": top_k, "distances": dists,
               "by_distance": {str(d): per_layer[d] for d in dists},
               "overall": {str(d): float(np.mean(list(per_layer[d].values())))
                           for d in dists},
               "prev_token": prevtok}
    if dists == [1]:                      # keep the original schema readable
        payload["next_layer"] = per_layer[1]
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {a.out}")
    store.close()


if __name__ == "__main__":
    main()
