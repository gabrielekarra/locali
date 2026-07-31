"""Full 62-layer MiniMax-M2.5 with only the dense core resident.

The whole point: `mlx_lm.load` on this checkpoint would pull 128.7 GB into a
24 GB machine. Instead the model is constructed, every switch_mlp is torn out
BEFORE anything is evaluated, and only the dense tensors are loaded -- 4.04B
params, 2.3 GB at 4-bit. Experts arrive from disk through M25Store.

MLX arrays are lazy until eval, so constructing the 62 SwitchGLU modules costs
nothing as long as their parameters are dropped before any mx.eval touches
them. `load_streaming` asserts on measured residency after the swap rather than
trusting that, because getting it wrong once took this machine down.

That assert used to read process RSS, which cannot see these allocations at
all: a run holding 10.28 GB of mlx arrays reported 3.67 GB resident, because
Metal buffers do not land in RSS. It now reads mlx's own accounting, which
does agree with the store's -- 8.00 GB of ceiling plus a 2.73 GB core came to
10.28 GB active.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx_lm.models.minimax import Model, ModelArgs

from m25_store import M25Store
from m25_stream import StreamingMoE
from m25_arena import ArenaStore, ArenaMoE

DENSE_PREFIXES = ("model.embed_tokens", "model.norm", "lm_head")
# Ceiling on the DENSE core at load, checked against mlx's accounting
# rather than RSS -- see load_streaming.
CORE_LIMIT_GB = 12.0
# Measured dense allocation after the routed experts are removed. The wide-
# batch preflight adds this to the hard arena, exact KV capacity, live output
# tensors, and explicit transient headroom.
DENSE_CORE_GB = 2.30


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def make_sized_prompt_cache(model, capacity):
    """Allocate only the KV positions this run can consume.

    MLX's default KVCache grows in blocks of 256 positions. At serving batches
    in the thousands that unused tail is tens of gigabytes. The generation
    loops know their prompt and decode ceilings, so the per-instance step can be
    exact without dropping a key or value.
    """
    from mlx_lm.models.cache import make_prompt_cache

    if capacity < 1:
        raise ValueError("KV cache capacity must be positive")
    cache = make_prompt_cache(model)
    for layer_cache in cache:
        if type(layer_cache).__name__ == "KVCache":
            layer_cache.step = capacity
    return cache


def wide_batch_memory_gb(cfg, ceiling_gb, batch, capacity):
    """Conservative resident-memory estimate for the exact-capacity batch path."""
    head_dim = cfg.get("head_dim") or (
        cfg["hidden_size"] // cfg["num_attention_heads"]
    )
    kv = (
        2  # keys and values
        * cfg["num_hidden_layers"]
        * batch
        * cfg["num_key_value_heads"]
        * capacity
        * head_dim
        * 2  # bfloat16
        / 1e9
    )
    logits = batch * cfg["vocab_size"] * 2 / 1e9
    moe_terms = (
        batch
        * cfg["num_experts_per_tok"]
        * cfg["hidden_size"]
        * 4  # float32 mixed-tier accumulator
        / 1e9
    )
    transient_headroom = 0.75
    return (
        ceiling_gb
        + DENSE_CORE_GB
        + kv
        + logits
        + moe_terms
        + transient_headroom
    )


def load_streaming(snap: Path, index_path: str, ceiling_gb: float,
                   trace: bool = False, threads: int = 8,
                   arena: bool = False, hot_share: float = 0.33,
                   nocache: bool = True, cache_policy: str = "lru"):
    cfg = json.loads((snap / "config.json").read_text())
    q = cfg.pop("quantization")
    model = Model(ModelArgs.from_dict(cfg))

    # Tear the experts out before anything can materialise them. After this the
    # only large arrays the model can hold are dense ones.
    if arena:
        store = ArenaStore(index_path, ceiling_gb=ceiling_gb,
                           hot_share=hot_share, threads=threads,
                           nocache=nocache, cache_policy=cache_policy)
        Block = ArenaMoE
    else:
        store = M25Store(index_path, ceiling_gb=ceiling_gb, trace=trace,
                         threads=threads, nocache=nocache)
        Block = StreamingMoE
    streamers = []
    for li, layer in enumerate(model.model.layers):
        blk = layer.block_sparse_moe
        sm = Block(store, li, blk.gate, blk.e_score_correction_bias,
                   blk.num_experts_per_tok)
        blk.switch_mlp = nn.Module()          # drop the [256, ...] parameters
        blk.__dict__["_stream"] = sm
        streamers.append(sm)

    index = json.loads((snap / "model.safetensors.index.json").read_text())
    want = {k for k in index["weight_map"]
            if "switch_mlp" not in k and
            (k.startswith(DENSE_PREFIXES) or k.startswith("model.layers."))}
    keep = {}
    for shard in sorted({index["weight_map"][k] for k in want}):
        arrs = mx.load(str(snap / shard))
        keep.update({k: v for k, v in arrs.items() if k in want})

    def class_predicate(path, m):
        if path in q:
            return q[path]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{path}.scales" in keep

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=class_predicate)
    # Re-attach after quantize, which rebuilds submodules.
    for li, layer in enumerate(model.model.layers):
        blk = layer.block_sparse_moe
        streamers[li].gate = blk.gate
        streamers[li].bias = blk.e_score_correction_bias
        blk.switch_mlp = nn.Module()
        blk.__dict__["_stream"] = streamers[li]

    model.load_weights(list(keep.items()), strict=False)
    mx.eval(model.parameters())
    del keep

    # Route every MoE block through its streamer. Patching the class once is
    # enough because each block carries its own _stream.
    cls = type(model.model.layers[0].block_sparse_moe)
    cls.__call__ = lambda self, x: self.__dict__["_stream"](x)

    # RSS cannot see this. mlx allocates through Metal, and a run holding 10.28
    # GB of arrays reported 3.67 GB resident -- so the guard that exists to
    # catch 128.7 GB landing in a 24 GB machine was reading a number blind to
    # exactly the allocations that would cause it. mlx's own accounting is the
    # real residency; RSS is kept alongside because they disagree and the gap
    # is worth seeing.
    got, rss = mx.get_active_memory() / 1e9, rss_gb()
    dense = got - store.resident / 1e9
    assert dense < CORE_LIMIT_GB, (
        f"dense core alone took {dense:.1f} GB of mlx memory "
        f"({got:.1f} GB including experts; rss says {rss:.1f}) -- aborting"
    )
    return model, store, cfg, dense


def generate(model, store, tok, prompt, max_tokens):
    """Greedy decode with a KV cache. Reports prefill and decode separately
    because they are different regimes: prefill touches most experts of every
    layer, decode touches top-k, and averaging them hides both."""
    ids = mx.array(tok(prompt)["input_ids"])
    cache = make_sized_prompt_cache(model, ids.size + max_tokens)

    t0 = time.perf_counter()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    pre = store.stats()

    out, y = [], int(mx.argmax(logits[0, -1]))
    # Sampled in chunks, because one aggregate cannot tell a cache still warming
    # from a steady state -- and this repo has already shipped a technique that
    # looked good over a few tokens and died at length.
    chunk, marks, last = 16, [], (store.stats(), time.perf_counter())
    t0 = time.perf_counter()
    for i in range(max_tokens):
        out.append(y)
        logits = model(mx.array([[y]]), cache=cache)
        mx.eval(logits)
        y = int(mx.argmax(logits[0, -1]))
        if (i + 1) % chunk == 0:
            s, now = store.stats(), time.perf_counter()
            d = {k: s[k] - last[0][k] for k in ("hits", "misses", "bytes_read")}
            marks.append((i + 1, d["hits"] / max(d["hits"] + d["misses"], 1),
                          d["bytes_read"] / 1e9 / chunk, chunk / (now - last[1])))
            last = (s, now)
    t_decode = time.perf_counter() - t0

    post = store.stats()
    dec = {k: post[k] - pre[k] for k in ("hits", "misses", "bytes_read")}
    dec["hit_rate"] = dec["hits"] / max(dec["hits"] + dec["misses"], 1)
    return tok.decode(out), t_prefill, t_decode, ids.size, pre, dec, marks


def generate_batch(model, store, tok, n_seq, prompt_len, max_tokens):
    """Decode n_seq independent sequences in lockstep.

    The point is not parallelism for its own sake. A decode step fetches the
    UNION of what its tokens route to, and that union was measured growing at
    half the independent rate -- 81.8 experts per layer for 32 tokens where
    independence predicts 163.3. So every sequence added amortises the fetch
    over more output, and the 1488 matvec launches per pass amortise perfectly.

    Prompts are distinct slices of the same book: equal length so no padding is
    needed, and different content so the routing genuinely diverges. Feeding the
    same prompt n times would share every expert and report a fiction.
    """
    text = Path("eval/pride_prejudice.txt").read_text()
    all_ids = tok(text)["input_ids"]
    need = n_seq * prompt_len
    assert len(all_ids) >= need, f"need {need} tokens, book gives {len(all_ids)}"
    ids = mx.array([all_ids[i * prompt_len:(i + 1) * prompt_len]
                    for i in range(n_seq)])

    cache = make_sized_prompt_cache(model, prompt_len + max_tokens)
    t0 = time.perf_counter()
    logits = model(ids, cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    pre = store.stats()

    y = mx.argmax(logits[:, -1], axis=-1)
    chunk, marks, last = 8, [], (store.stats(), time.perf_counter())
    out = []
    t0 = time.perf_counter()
    for i in range(max_tokens):
        out.append(y)
        logits = model(y[:, None], cache=cache)
        mx.eval(logits)
        y = mx.argmax(logits[:, -1], axis=-1)
        if (i + 1) % chunk == 0:
            s, now = store.stats(), time.perf_counter()
            d = {k: s[k] - last[0][k] for k in ("hits", "misses", "bytes_read")}
            marks.append((i + 1, d["hits"] / max(d["hits"] + d["misses"], 1),
                          d["bytes_read"] / 1e9 / (chunk * n_seq),
                          chunk * n_seq / (now - last[1])))
            last = (s, now)
    t_decode = time.perf_counter() - t0

    post = store.stats()
    dec = {k: post[k] - pre[k] for k in ("hits", "misses", "bytes_read")}
    dec["hit_rate"] = dec["hits"] / max(dec["hits"] + dec["misses"], 1)
    texts = [tok.decode([int(o[j]) for o in out]) for j in range(min(n_seq, 2))]
    return texts, t_prefill, t_decode, ids.size, pre, dec, marks


def measure_draft(model, store, tok, prompt, n):
    """Acceptance rate of the cache as its own draft model.

    At each step the SAME state is decoded twice: once for real, and once with
    every non-resident expert skipped and the gates renormalised over what is
    left -- a full forward pass that reads nothing. Generation advances on the
    true token, so this is the acceptance rate a speculative decoder would see,
    not a compounding-divergence measurement.

    If the cache drafts well, a verify pass over k drafted tokens costs about
    what one token costs today (measured: the union of k consecutive tokens'
    experts grows at half the independent rate), and single-stream latency stops
    being one disk round-trip per token.
    """
    ids = mx.array(tok(prompt)["input_ids"])
    cache = make_sized_prompt_cache(model, ids.size + n)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    y = int(mx.argmax(logits[0, -1]))

    from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
    assert can_trim_prompt_cache(cache), "cache cannot be rewound"

    agree, drafted, t_draft, t_true = 0, [], 0.0, 0.0
    for _ in range(n):
        step = mx.array([[y]])
        # Draft, then REWIND. Both passes consume the same position, so without
        # the trim the second one would append a duplicate entry and every
        # measurement after it would be against a corrupted context.
        StreamingMoE.draft = True
        t0 = time.perf_counter()
        dl = model(step, cache=cache)
        mx.eval(dl)
        t_draft += time.perf_counter() - t0
        d = int(mx.argmax(dl[0, -1]))
        StreamingMoE.draft = False
        trim_prompt_cache(cache, 1)

        t0 = time.perf_counter()
        logits = model(step, cache=cache)
        mx.eval(logits)
        t_true += time.perf_counter() - t0
        y = int(mx.argmax(logits[0, -1]))
        agree += int(d == y)
        drafted.append((d, y))
    return agree / n, t_draft / n, t_true / n, drafted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--index", default="models/m25.idx")
    ap.add_argument("--ceiling-gb", type=float, default=8.0)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--trace-out", help="write the routing histogram here")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1,
                    help="decode this many independent sequences at once")
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--arena", action="store_true",
                    help="preallocated unified-memory arena + gather_qmm")
    ap.add_argument("--hot-share", type=float, default=0.60,
                    help="fraction of the ceiling given to the 4-bit arena; "
                         "0.60 is tuned for single-stream at a 9 GB ceiling")
    ap.add_argument("--os-cache", action="store_true",
                    help="also let macOS cache expert files; may speed repeated "
                         "interactive reads but uses reclaimable system RAM")
    ap.add_argument(
        "--cache-policy",
        choices=("lru", "slru-cold", "slru-all"),
        default="slru-cold",
        help="expert eviction policy. slru-cold (default) reserves half the "
             "2-bit tier for repeat entries; slru-all also protects 4-bit "
             "entries and is useful under tighter ceilings",
    )
    ap.add_argument("--measure-draft", type=int, default=0,
                    help="acceptance rate of the resident cache as a draft model")
    ap.add_argument("--prefetch", action="store_true",
                    help="issue layer L+1's predicted experts during layer L "
                         "(arena only); raises queue depth, costs mispredicted "
                         "reads")
    ap.add_argument("--prefetch-k", type=int, default=5,
                    help="prefetch only the N highest-gate predicted experts "
                         "(default: 5, tuned at a 9 GB ceiling; 0 means all)")
    ap.add_argument("--prefetch-depth", type=int, default=1, choices=(1, 2),
                    help="layers of lookahead. 2 also issues L+2, whose router "
                         "is 72.5%% accurate on layer L's input (78.5%% at L+1)")
    ap.add_argument("--prefetch-k2", type=int, default=4,
                    help="candidates for the L+2 pass, when --prefetch-depth 2")
    ap.add_argument("--profile", action="store_true",
                    help="attribute time inside the MoE block; forces an\n                          eval at each boundary, so the run is slower")
    a = ap.parse_args()

    snap = Path(a.snap)
    raw_cfg = json.loads((snap / "config.json").read_text())
    if a.batch > 1:
        # The batch loop knows its maximum cache capacity, unlike an open-ended
        # interactive prompt, so account for each material resident explicitly.
        need = wide_batch_memory_gb(
            raw_cfg,
            a.ceiling_gb,
            a.batch,
            a.prompt_len + a.tokens,
        )
    else:
        # Preserve the established conservative guard for arbitrary prompts.
        need = a.ceiling_gb + 6
    avail = psutil.virtual_memory().available / 1e9
    assert avail > need, (f"only {avail:.1f} GB free, need {need:.1f} for a "
                          f"{a.ceiling_gb:.1f} GB ceiling; close things or "
                          f"lower --ceiling-gb")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap))
    t0 = time.perf_counter()
    model, store, cfg, core = load_streaming(snap, a.index, a.ceiling_gb,
                                             trace=bool(a.trace_out),
                                             threads=a.threads,
                                             arena=a.arena,
                                             hot_share=a.hot_share,
                                             nocache=not a.os_cache,
                                             cache_policy=a.cache_policy)
    print(f"dense core resident: {core:.2f} GB  (loaded in {time.perf_counter()-t0:.0f}s)")

    blocks = [l.block_sparse_moe.__dict__["_stream"]
              for l in model.model.layers] if a.arena else []
    if a.prefetch:
        assert a.arena, "--prefetch needs --arena"
        for cur, nxt in zip(blocks, blocks[1:]):
            cur.nxt = nxt
            cur.prefetch_k = a.prefetch_k or None
        if a.prefetch_depth == 2:
            for cur, nxt2 in zip(blocks, blocks[2:]):
                cur.nxt2 = nxt2
                cur.prefetch_k2 = a.prefetch_k2 or None
        print("prefetch router: original quantized GPU path")

    if a.measure_draft:
        acc, td, tt, pairs = measure_draft(model, store, tok, a.prompt,
                                           a.measure_draft)
        print(f"\ncache-as-draft over {a.measure_draft} steps")
        print(f"  acceptance {acc*100:.1f}%   draft {td*1000:.0f} ms/token  "
              f"true {tt*1000:.0f} ms/token   ({tt/td:.1f}x)")
        s = store.stats()
        print(f"  hit {s['hit_rate']*100:.1f}%  peak {s['peak']/1e9:.2f} GB")
        agree = [tok.decode([d]) for d, y in pairs if d == y][:12]
        print(f"  accepted: {agree}")
        store.close()
        return

    if a.batch > 1:
        texts, t_pre, t_dec, n_pre, pre, dec, marks = generate_batch(
            model, store, tok, a.batch, a.prompt_len, a.tokens)
        n_out = a.batch * a.tokens
        print(f"\nbatch {a.batch}, prefill {n_pre} tokens in {t_pre:.1f}s "
              f"({n_pre/t_pre:.1f} tok/s)")
        print(f"  hit {pre['hit_rate']*100:.1f}%  read {pre['bytes_read']/1e9:.2f} GB")
        print(f"decode {n_out} tokens ({a.tokens} x {a.batch}) in {t_dec:.1f}s "
              f"-> {n_out/t_dec:.2f} tok/s aggregate, "
              f"{a.tokens/t_dec:.2f} per sequence")
        print(f"  hit {dec['hit_rate']*100:.1f}%  read {dec['bytes_read']/1e9:.2f} GB "
              f"({dec['bytes_read']/1e9/n_out:.3f} GB/token)")
        if len(marks) > 1:
            print("  by chunk:  " + "   ".join(
                f"@{n} {hr*100:.0f}% {gb:.2f}GB/t {tps:.1f}tok/s"
                for n, hr, gb, tps in marks))
        s = store.stats()
        print(f"  evictions {s['evictions']}  arena {s['peak']/1e9:.2f} GB  "
              f"blocked on disk {s['t_stall']:.1f}s of {t_pre+t_dec:.1f}s")
        print(f"  mlx active {mx.get_active_memory()/1e9:.2f} GB")
        for i, t in enumerate(texts):
            print(f"  seq{i}: {t[:70]!r}")
        store.close()
        return

    if a.profile:
        StreamingMoE.prof = {}
    text, t_pre, t_dec, n_pre, pre, dec, marks = generate(model, store, tok, a.prompt,
                                                   a.tokens)
    print(f"\nprefill {n_pre} tokens in {t_pre:.1f}s ({n_pre/t_pre:.2f} tok/s)")
    print(f"  hit {pre['hit_rate']*100:.1f}%  read {pre['bytes_read']/1e9:.2f} GB")
    print(f"decode {a.tokens} tokens in {t_dec:.1f}s ({a.tokens/t_dec:.2f} tok/s)")
    print(f"  hit {dec['hit_rate']*100:.1f}%  read {dec['bytes_read']/1e9:.2f} GB "
          f"({dec['bytes_read']/1e9/a.tokens:.2f} GB/token)")
    if len(marks) > 1:
        print("  by chunk:  " + "   ".join(
            f"@{n} {hr*100:.0f}% {gb:.2f}GB/t {tps:.2f}tok/s" for n, hr, gb, tps in marks))
    s = store.stats()
    print(f"  evictions {s['evictions']}  peak {s['peak']/1e9:.2f} GB  "
          f"rss {rss_gb():.2f} GB")
    print(f"  mlx itself: active {mx.get_active_memory()/1e9:.2f} GB  "
          f"buffer cache {mx.get_cache_memory()/1e9:.2f} GB  "
          f"peak {mx.get_peak_memory()/1e9:.2f} GB")
    # t_pread is thread-seconds now that reads overlap compute; t_stall is
    # the part that could not be hidden and is the only one that is runtime.
    if s.get("prefetched"):
        pf = s["prefetched"]
        print(f"  prefetch: {pf} experts issued a layer early, "
              f"{s['prefetch_used']} used ({s['prefetch_used']/pf*100:.1f}%), "
              f"{s['prefetch_wasted']} evicted unused")
    acc = s['t_stall'] + s['t_convert'] + s['t_eval']
    print(f"  where the time went: blocked on disk {s['t_stall']:.1f}s  "
          f"numpy->mx {s['t_convert']:.1f}s  eval {s['t_eval']:.1f}s  "
          f"= {acc:.1f}s of {t_pre + t_dec:.1f}s ({acc/(t_pre+t_dec)*100:.0f}%)")
    if a.profile:
        P = StreamingMoE.prof
        # 'gates' is measured INSIDE 'expert', not beside it.
        print("  inside the MoE block: " + "  ".join(
            f"{k} {v:.1f}s" for k, v in sorted(P.items(), key=lambda kv: -kv[1])))
    print(f"\n{a.prompt}{text}")

    if a.trace_out:
        n = store.dump_trace(a.trace_out)
        print(f"\nwrote {a.trace_out} ({n} routed accesses)")
    store.close()


if __name__ == "__main__":
    main()
