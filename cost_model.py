"""What limits tok/s, and what each proposed change is worth before building it.

Three costs, per CLAUDE.md, and the point of writing them down together is that
they do not compose the way intuition says: a change that halves the bytes read
does not halve the runtime, because the fetch is only part of the token, and a
change that doubles disk bandwidth is worth nothing on bytes that were hits.

  t_io    bytes that miss / effective disk bandwidth
  t_ram   bytes converted into mlx / host copy speed -- a floor no cache moves
  t_cmp   router + expert matmuls + the dense backbone

Everything here is fitted to measured runs in NOTES.md, not to peak specs. In
particular `DISK_NOW` is the 3.0 GB/s the engine actually gets at 144 KB and
1.18 MB blocks, not the 4.66 GB/s bw.py reports at 4.42 MB -- the gap between
those two numbers is itself one of the proposals below.
"""

import json
from pathlib import Path

LAYERS, TOP_K = 62, 8
HOT_MB, COLD_MB = 7.96, 4.42          # bytes per expert by tier
DISK_NOW = 3.0                        # GB/s, measured in situ, mixed block sizes
DISK_BLOCK = 4.66                     # GB/s, bw.py at 4.42 MB blocks
RAM_COPY = 4.3                        # GB/s, numpy->mx under memory pressure
# Measured at 9 GB / 64 tokens: 67.8s decode, of which pread 27.4 and convert
# 21.4 (scaled to the decode share). What is left is compute, and it is the one
# term no fetch change can touch.
T_CMP = 0.36                          # s/token


def per_token_gb(hit, cold_frac=0.75, mb=None):
    """Bytes a token pulls off disk. The router picks top_k in every layer; a
    hit costs nothing, a miss costs one expert at its tier's size."""
    if mb is None:
        mb = cold_frac * COLD_MB + (1 - cold_frac) * HOT_MB
    return LAYERS * TOP_K * (1 - hit) * mb / 1000


def tok_s(hit, disk=DISK_NOW, ram=RAM_COPY, cmp_=T_CMP, mb=None, overlap=False):
    gb = per_token_gb(hit, mb=mb)
    t_io, t_ram = gb / disk, gb / ram
    # Fetch and convert are serial today: the batch is read, then converted.
    fetch = t_io + t_ram
    return 1.0 / (max(fetch, cmp_) if overlap else fetch + cmp_)


def static_pin_bound(trace_path, index_path, budget_gb):
    """Upper bound on ANY static policy: knapsack the trace by hits per byte.

    This is not a policy proposal, it is a ceiling. If LRU is already close to
    it, no smarter eviction rule is worth writing; if it is far below, the gap
    is real and recoverable. Fractional relaxation, so it is a true upper bound
    rather than an achievable number.
    """
    tr = json.loads(Path(trace_path).read_text())
    idx = json.loads(Path(index_path).read_text())["experts"]
    items = []
    for l, d in tr["counts"].items():
        for e, n in d.items():
            tier = idx[f"L{l}.E{e}"]["tier"]
            mb = HOT_MB if tier == "hot" else COLD_MB
            items.append((n / mb, n, mb))
    items.sort(reverse=True)
    total = sum(n for _, n, _ in items)
    spent, hits = 0.0, 0
    for _, n, mb in items:
        if spent + mb / 1000 > budget_gb:
            break
        spent += mb / 1000
        hits += n
    return hits / total, spent, total


def main():
    print(__doc__.strip().split("\n\n")[0])
    print()

    print("=== where a token goes now (9 GB ceiling, 56.3% hit, measured 0.94)")
    gb = per_token_gb(0.563)
    print(f"  bytes/token {gb:.2f} GB   measured 1.19")
    for name, v in [("t_io", gb / DISK_NOW), ("t_ram", gb / RAM_COPY),
                    ("t_cmp", T_CMP)]:
        print(f"  {name:6s} {v:.3f}s")
    print(f"  predicted {tok_s(0.563):.2f} tok/s")
    print()

    print("=== what each lever is worth, alone")
    base = tok_s(0.563)
    rows = [
        ("baseline", dict()),
        ("expert-major cold pack (one 4.42 MB read)", dict(disk=DISK_BLOCK)),
        ("overlap fetch with compute", dict(overlap=True)),
        ("hit 56% -> 71% (16 GB ceiling)", dict(hit=0.71)),
        ("top-8 -> top-6 (25% fewer experts)", dict(mb=0.75 * 4.87)),
        ("cold tier 2-bit -> 1.58-bit", dict(mb=0.75 * 3.1 + 0.25 * HOT_MB)),
    ]
    for name, kw in rows:
        kw = {"hit": 0.563, **kw}
        v = tok_s(**kw)
        print(f"  {name:44s} {v:5.2f} tok/s  {v/base:4.2f}x")
    print()

    print("=== stacked, in the order they are cheap to build")
    stack, kw = [], {"hit": 0.563}
    for name, add in [("+ expert-major pack", dict(disk=DISK_BLOCK)),
                      ("+ overlap", dict(overlap=True)),
                      ("+ 16 GB ceiling", dict(hit=0.71)),
                      ("+ top-6", dict(mb=0.75 * 4.87))]:
        kw.update(add)
        v = tok_s(**kw)
        stack.append((name, v))
        print(f"  {name:44s} {v:5.2f} tok/s  {v/base:4.2f}x")
    print()

    if Path("results/routing_trace.json").exists():
        print("=== is LRU leaving anything on the table")
        for budget in (6, 9, 16):
            hr, spent, n = static_pin_bound(
                "results/routing_trace.json", "models/m25.idx", budget)
            print(f"  {budget:2d} GB: static-pin upper bound {hr*100:5.1f}%  "
                  f"({spent:.1f} GB placed, {n} accesses)")
        print("  measured LRU: 49.1% at 8 GB, 53.0% at 9, 61.7% at 10")


if __name__ == "__main__":
    main()
