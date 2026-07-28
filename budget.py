"""What MiniMax-M2.5 costs per token on this Mac, and where the speed can come from.

Three costs, not one. Leaving out the second predicted 43 tok/s for a streaming
run that measured 5.2, which is how it earned its place here:

  t_io    cold expert bytes / SSD bandwidth
  t_ram   ACTIVE weight bytes / unified-memory bandwidth -- irreducible. Every
          weight the token touches crosses the bus wherever it lives, so this is
          a floor no amount of caching moves.
  t_over  everything else: Python, per-layer syncs, staging copies

Constants are measured on this machine (`bw.py` for bandwidth, an earlier
streaming run for t_over and the reuse-distance curve) except RAM_BW, which is
the M4 published figure. Hit rate is the only MODELLED quantity here; treat it
as +/-10 points.
"""

import argparse
from math import log

GB = 1e9

# Measured by bw.py with F_NOCACHE against a 40 GB file, at the 4.42 MB block an
# expert occupies at 2-bit. Bandwidth is block-size dependent -- 4.00 GB/s here
# against 4.66 at 17.5 MB -- and saturates at 8 threads.
BW = 4.00 * GB
RAM_BW = 120 * GB        # M4 base (MacBook Air). M4 Pro 273, M4 Max 546.
USABLE = 19.3 * GB       # 25.77 GB RAM - OS - runtime - activations

# Calibrated from an earlier streaming run on this machine by subtracting the
# two physical terms from its end-to-end throughput: 5.2 tok/s over 48 layers
# with 0.208 GB/token of cold reads and 3.11 GB/token of active weights leaves
# 121 ms/token that is neither disk nor memory bus.
T_OVER = (1 / 5.2 - 0.208 * GB / BW - 3.11 * GB / RAM_BW) / 48

# Geometry from config.json; total from the safetensors census.
L_MOE, E, TOPK = 62, 256, 8
HIDDEN, MOE_INTER = 3072, 1536
EXPERT_PARAMS = 3 * HIDDEN * MOE_INTER           # 14,155,776
TOTAL_PARAMS = 228_703_644_928
ROUTED = EXPERT_PARAMS * E * L_MOE
DENSE = TOTAL_PARAMS - ROUTED                    # 4.04B -- no shared experts
KV_PER_TOKEN = L_MOE * 2 * (8 * 128) * 2         # 254 KB/token, 8 KV heads

# Slots/layer needed for a target hit rate, measured by reuse distance on an
# earlier trace, carried to a different expert count by the exponent that same
# analysis produced (the hot set scales as N^alpha, alpha rising along the curve).
REF_N, REF = 128, {0.50: 15, 0.70: 29, 0.90: 58}
ALPHA = {0.50: 0.28, 0.70: 0.59, 0.90: 1.02}


def eff_bpp(bits):
    """Bytes per param for MLX group quant: group 64, fp16 scale and bias."""
    return (bits + 0.5) / 8


def hit_rate(slots, k=TOPK, n_experts=E):
    """Hard floor: below top-k slots the cache cannot hold even the current
    token's own working set for a layer, LRU thrashes, and the rate is ~0.
    Both measured curves show exactly that shape."""
    if slots <= k:
        return 0.0
    pts = sorted((REF[t] * (n_experts / REF_N) ** ALPHA[t], t) for t in REF)
    if slots <= pts[0][0]:
        return pts[0][1] * (slots - k) / max(pts[0][0] - k, 1e-9)
    for (s0, h0), (s1, h1) in zip(pts, pts[1:]):
        if slots <= s1:
            return h0 + (h1 - h0) * log(slots / s0) / log(s1 / s0)
    return min(0.97, pts[-1][1] + 0.07 * log(slots / pts[-1][0]))


def step(expert_bits=4, dense_bits=4, ctx=16384, kv_bits=16, hit_bonus=0.0,
         k=TOPK, overlap=False, t_over=T_OVER, gamma=1.0, gnp=1.0, bw=BW):
    """One rung of the ladder.

    gamma = accepted tokens per forward pass (MTP speculation); a verification
    pass reads the dense weights once for all of them, and the expert union
    grows at slope (1-r) with r=0.42 measured by reuse distance.
    gnp = byte multiplier from gate-proportional neuron paging (TECHNIQUES.md).
    """
    kv = ctx * KV_PER_TOKEN * kv_bits / 16
    dense_b = DENSE * eff_bpp(dense_bits)
    embed = 200064 * HIDDEN * eff_bpp(dense_bits)     # a lookup, not a matmul
    cache = 25.77 * GB - (3.5 + 1.0 + 1.0) * GB - kv - dense_b
    eb = EXPERT_PARAMS * eff_bpp(expert_bits)
    slots = cache / eb / L_MOE
    h = min(0.97, hit_rate(slots, k) + hit_bonus)
    union = k * (1 + (gamma - 1) * 0.58) if gamma > 1 else k
    cold = L_MOE * k * eb * (1 - h) * gnp * (union / k / gamma)
    ram = (L_MOE * union * eb / gamma + (dense_b - embed) / gamma)
    t_io, t_ram = cold / bw, ram / RAM_BW
    t = (max(t_io, t_ram) if overlap else t_io + t_ram) + L_MOE * t_over / gamma
    return dict(slots=slots, resid=slots / E, hit=h, cache=cache / GB,
                disk=(ROUTED * eff_bpp(expert_bits) + dense_b) / GB,
                gb_tok=cold / GB, t_io=t_io * 1e3, t_ram=t_ram * 1e3, tok_s=1 / t)


LADDER = [
    ("0  today: 4-bit, Python, no overlap", dict()),
    ("1  + KV at 8-bit (frees cache)", dict(kv_bits=8)),
    ("2  + 2-bit cold experts", dict(kv_bits=8, expert_bits=2)),
    ("3  + C/Metal engine (kills t_over)", dict(kv_bits=8, expert_bits=2, t_over=0)),
    ("4  + prefetch: overlap I/O w/ compute", dict(kv_bits=8, expert_bits=2,
                                                   t_over=0, overlap=True)),
    ("5  + profile-guided residency (+8pt)", dict(kv_bits=8, expert_bits=2,
                                                  t_over=0, overlap=True, hit_bonus=0.08)),
    ("6  + adaptive top-6", dict(kv_bits=8, expert_bits=2, t_over=0, overlap=True,
                                 hit_bonus=0.08, k=6)),
    ("7  + MTP speculation (gamma=2.1)", dict(kv_bits=8, expert_bits=2, t_over=0,
                                              overlap=True, hit_bonus=0.08, k=6,
                                              gamma=2.1)),
    ("8  + GNP W3-first (gamma=1.5)", dict(kv_bits=8, expert_bits=2, t_over=0,
                                           overlap=True, hit_bonus=0.08, k=6,
                                           gamma=2.1, gnp=0.702)),
]


def _check():
    assert EXPERT_PARAMS == 14_155_776
    assert abs(DENSE / 1e9 - 4.04) < 0.02, DENSE / 1e9
    # No shared experts is the whole reason this model fits: the dense core has
    # to leave room for a cache bigger than one layer's top-k.
    assert DENSE * eff_bpp(4) / GB < 3.0

    # Hit rate: zero at or below top-k, monotone above it, never saturating.
    assert hit_rate(TOPK) == 0 and hit_rate(TOPK - 1) == 0
    xs = [hit_rate(s) for s in range(TOPK + 1, TOPK + 400)]
    assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])) and max(xs) < 1.0

    # The cache must sit ABOVE top-k, which is what makes caching work at all.
    assert step(kv_bits=8, expert_bits=2)["slots"] > TOPK

    # Each rung must help, and none may beat the memory-bus floor.
    tok = [step(**kw)["tok_s"] for _, kw in LADDER]
    assert all(b > a for a, b in zip(tok, tok[1:])), tok
    r = step(kv_bits=8, expert_bits=2, t_over=0, overlap=True)
    assert r["tok_s"] <= 1000 / r["t_ram"] + 1e-9

    # t_ram is a floor: dropping bits lowers it, and it never vanishes.
    assert step(expert_bits=2)["t_ram"] < step(expert_bits=4)["t_ram"]
    assert step(expert_bits=2)["t_ram"] > 0
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    _check()
    if a.check:
        return
    print(f"\nMiniMax-M2.5: {TOTAL_PARAMS/1e9:.0f}B total, {ROUTED/1e9:.0f}B routed, "
          f"{DENSE/1e9:.2f}B dense ({DENSE*eff_bpp(4)/GB:.2f} GB at 4-bit)")
    print(f"24 GB Mac, {BW/GB:.2f} GB/s measured disk, {RAM_BW/GB:.0f} GB/s memory bus\n")
    print(f"{'step':<42} {'hit':>6} {'t_io':>7} {'t_ram':>7} {'tok/s':>7}")
    for name, kw in LADDER:
        r = step(**kw)
        print(f"{name:<42} {r['hit']*100:>5.1f}% {r['t_io']:>6.0f}m "
              f"{r['t_ram']:>6.0f}m {r['tok_s']:>7.1f}")
    r = step(kv_bits=8, expert_bits=2)
    print(f"\ndisk {r['disk']:.0f} GB, cache {r['cache']:.1f} GB, "
          f"{r['slots']:.0f} slots/layer of {E} against top-{TOPK}")


if __name__ == "__main__":
    main()
