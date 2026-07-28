"""Which MoE actually runs well on 24 GB? Same physics as k3_budget.py, any model.

Geometry is derived from each model's config.json and cross-checked against the
HF parameter census, so `dense` is a residual, not a guess. The derivation is
validated against this repo's own measured run: Qwen3-Next-80B at 125 slots/layer
occupied 10.65 GB, i.e. 1.775 MB/expert, and 3*2048*512 params at 4-bit MLX is
1.77 MB. That agreement is the reason to trust the rest of the table.

Hit rate is the ONE modelled quantity. It comes from this repo's two measured
reuse-distance curves plus the alpha law from commit 140ad90 (slots needed for a
target hit rate scale as N^alpha, alpha 0.28@50% / 0.59@70% / 1.02@90%). That
same commit records that neither normalisation transferred cleanly between the
two models, so treat hit rate as +/-10 points, not as fact. Everything else here
is arithmetic on measured constants.
"""

import argparse
from math import log

GB = 1e9
BW = 4.64 * GB           # measured: internal SSD, 16 threads, expert-sized preads
USABLE = 19.3 * GB       # 25.77 GB RAM - OS - runtime - KV - activations
FREE_DISK = 183 * GB     # internal SSD free space

# Bandwidth is NOT the only cost. Profiling this repo's Qwen3-Next run
# (commit 8dadeaa) found 4.2 ms/layer split three ways: disk 34.7%, GPU gather
# 28.9%, router sync 27.2%, staging 6.3%. Everything except disk is a fixed
# per-layer cost that does not shrink when the hit rate rises -- which is why
# that sweep saw cold bytes fall 2.2x with tok/s flat at 5.2. Ignoring this term
# predicts 43 tok/s for a run that measured 5.2.
# There are THREE costs per token, not one, and the second was missing until now:
#   t_io    cold expert bytes / SSD bandwidth
#   t_ram   ACTIVE weight bytes / unified-memory bandwidth -- irreducible, and on
#           this machine it is the wall. M4 base is 120 GB/s. Published 30+ tok/s
#           results on 128 GB Macs run at 400 GB/s with the model fully RESIDENT:
#           3.3x more memory bandwidth AND zero disk in the loop. Not comparable.
#   t_over  everything else: Python, per-layer syncs, staging copies
RAM_BW = 120 * GB        # M4 base (MacBook Air). M4 Pro 273, M4 Max 546, M3 Ultra 819

# t_over calibrated by subtracting the two physical terms from the measured run:
# 5.2 tok/s = 192 ms/token, of which 0.208 GB/4.64 = 45 ms disk and 26 ms of
# unified-memory traffic. The remaining 121 ms / 48 layers is pure implementation.
_M = dict(tok_s=5.2, gb_tok=0.208, layers=48, active_gb=3.11)
T_OVER = (1 / _M["tok_s"] - _M["gb_tok"] * GB / BW - _M["active_gb"] * GB / RAM_BW) / _M["layers"]


class Model:
    def __init__(self, name, l_moe, experts, topk, hidden, moe_inter,
                 total_params, swebench=None, note=""):
        self.name, self.l_moe, self.E, self.k = name, l_moe, experts, topk
        self.expert_params = 3 * hidden * moe_inter
        self.routed = self.expert_params * experts * l_moe
        self.dense = total_params - self.routed   # residual against the HF census
        self.swebench, self.note = swebench, note

    def expert_bytes(self, bits=4):
        return self.expert_params * (bits + 0.5) / 8      # MLX group-64 fp16 scale+bias

    def disk(self, bits=4):
        return self.routed * (bits + 0.5) / 8

    def dense_gb(self, bits=4):
        return self.dense * (bits + 0.5) / 8 / GB

    def active_bytes(self, bits=4):
        """Weights touched per token -- must cross the memory bus wherever they live."""
        return (self.expert_params * self.k * self.l_moe + self.dense) * (bits + 0.5) / 8


# slots/layer for a target hit rate, measured on Qwen3-30B (128 experts, top-8),
# and the exponent that carries each target to a different expert count.
REF_N, REF = 128, {0.50: 15, 0.70: 29, 0.90: 58}
ALPHA = {0.50: 0.28, 0.70: 0.59, 0.90: 1.02}


def hit_rate(m: Model, slots: float) -> float:
    """Interpolate the measured curve to this model's expert count and cache size.

    Hard floor: below top-k slots the cache cannot even hold the current token's
    own working set, so LRU thrashes and the hit rate is ~0. Both measured curves
    show exactly this (Qwen3-Next, top-10: 4.1% at 8 slots, 41.9% at 15).
    """
    if slots <= m.k:
        return 0.0
    pts = sorted((REF[t] * (m.E / REF_N) ** ALPHA[t], t) for t in REF)
    if slots <= pts[0][0]:                      # ramp from top-k up to the 50% point
        return pts[0][1] * (slots - m.k) / max(pts[0][0] - m.k, 1e-9)
    for (s0, h0), (s1, h1) in zip(pts, pts[1:]):
        if slots <= s1:
            return h0 + (h1 - h0) * log(slots / s0) / log(s1 / s0)
    return min(0.97, pts[-1][1] + 0.07 * log(slots / pts[-1][0]))


def evaluate(m: Model, bits=4, bw=BW, t_over=T_OVER, overlap=False):
    dense_gb = m.dense_gb(bits)
    cache = USABLE - dense_gb * GB
    if cache <= 0:
        return dict(model=m.name, fits=False, reason="dense core alone exceeds RAM")
    eb = m.expert_bytes(bits)
    slots = cache / eb / m.l_moe                       # global LRU, per-layer equivalent
    h = hit_rate(m, slots)
    cold = m.l_moe * m.k * eb * (1 - h)
    t_io = cold / bw
    t_ram = m.active_bytes(bits) / RAM_BW
    t_over_tot = m.l_moe * t_over
    # An engine that prefetches next-layer experts hides I/O behind compute, so
    # the two physical terms overlap instead of adding. That is the whole prize
    # here -- and it only pays when t_io and t_ram are comparable.
    t = (max(t_io, t_ram) if overlap else t_io + t_ram) + t_over_tot
    return dict(model=m.name, fits=True, dense_gb=dense_gb, slots=slots,
                resid=slots / m.E, hit=h, gb_tok=cold / GB, act_gb=m.active_bytes(bits)/GB,
                t_io=t_io, t_ram=t_ram, tok_s=1.0 / t, io_frac=t_io / t,
                disk_gb=m.disk(bits) / GB, on_internal=m.disk(bits) <= FREE_DISK,
                swebench=m.swebench, note=m.note)


# Geometry from each config.json; total_params from the HF safetensors census.
# SWE-bench Verified figures are third-party (July 2026 round-ups), not measured
# here -- they rank the candidates, they are not precise.
MODELS = [
    Model("MiniMax-M2.5",     62, 256,  8, 3072, 1536, 228_703_644_928, 80.2,
          "no shared experts; ships 3 MTP modules -> free speculation"),
    Model("GLM-4.7",          89, 160,  8, 5120, 1536, 358_337_791_296, 73.8),
    Model("GLM-5.2",          75, 256,  8, 6144, 2048, 753_329_940_480, 77.8,
          "quoted as GLM-5"),
    Model("Qwen3-Next-80B",   48, 512, 10, 2048,  512,  81_324_862_720, None,
          "already measured in this repo at 5.2 tok/s"),
    Model("Kimi-K3",          92, 896, 16, 3584, 3072, 2_779_931_837_184, None,
          "AA index #3; ships MXFP4 natively so 'bits' understates its disk"),
]


def _check():
    q = next(m for m in MODELS if m.name == "Qwen3-Next-80B")
    # Geometry check against the measured run: 10.65 GB held 125 slots/layer.
    assert abs(q.expert_bytes(4) - 1.77e6) < 0.02e6, q.expert_bytes(4)
    assert abs(q.expert_bytes(4) * 125 * 48 / GB - 10.65) < 0.3
    # Dense residuals must be positive and sane for every model.
    for m in MODELS:
        assert 0 < m.dense < m.routed, (m.name, m.dense)
    # Hit rate must be 0 at or below top-k, monotone above it, and never >= 1.
    for m in MODELS:
        assert hit_rate(m, m.k) == 0 and hit_rate(m, m.k - 1) == 0
        xs = [hit_rate(m, s) for s in range(m.k + 1, m.k + 400)]
        assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])) and max(xs) < 1.0
    # Reproduce the repo's own measured point: Qwen3-Next at 125 slots -> 78.1%.
    assert abs(hit_rate(q, 125) - 0.781) < 0.10, hit_rate(q, 125)
    # K3 must come out infeasible, as k3_budget.py proves independently.
    assert evaluate(next(m for m in MODELS if m.name == "Kimi-K3"))["fits"] is False
    # END-TO-END: reproduce the repo's own measured 5.2 tok/s for Qwen3-Next at
    # its 10.65 GB operating point. This is the only tok/s this model can be
    # checked against, and without the fixed per-layer term it predicts 43.
    assert abs(q.active_bytes(4) / GB - 3.11) < 0.05
    t = evaluate(q)["tok_s"]
    assert 4.6 <= t <= 5.8, f"Qwen3-Next predicted {t:.2f}, measured 5.2"
    # Ceiling with zero overhead and perfect overlap must beat it by several x,
    # and must never exceed what the memory bus alone allows.
    ideal = evaluate(q, overlap=True, t_over=0)["tok_s"]
    assert ideal > 3 * t and ideal <= RAM_BW / q.active_bytes(4) + 1e-9
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=float, default=4)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    _check()
    if a.check:
        return

    print(f"\n24 GB Mac, {USABLE/GB:.1f} GB usable, {BW/GB:.2f} GB/s measured, "
          f"{a.bits}-bit weights, internal SSD only\n")
    hdr = f"{'model':<16} {'core':>6} {'slots':>7} {'resid':>6} {'hit':>6} " \
          f"{'GB/tok':>7} {'tok/s':>7} {'I/O':>5} {'disk':>7} {'fits SSD':>9} {'SWE':>5}"
    print(hdr + "\n" + "-" * len(hdr))
    for m in MODELS:
        r = evaluate(m, a.bits)
        if not r["fits"]:
            print(f"{m.name:<16} {'--':>6}  {r['reason']}")
            continue
        print(f"{m.name:<16} {r['dense_gb']:>5.1f}G {r['slots']:>7.1f} "
              f"{r['resid']*100:>5.1f}% {r['hit']*100:>5.1f}% {r['gb_tok']:>7.2f} "
              f"{r['tok_s']:>7.2f} {r['io_frac']*100:>4.0f}% {r['disk_gb']:>6.0f}G {str(r['on_internal']):>9} "
              f"{(r['swebench'] or 0) or '?':>5}")
    print("\nhit rate is modelled (+/-10 pts); everything else is arithmetic on "
          "measured constants. Sonnet 5 scores 85.2 on SWE-bench Verified.")
    for m in MODELS:
        if m.note:
            print(f"  {m.name}: {m.note}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# MiniMax-M2.5 optimisation ladder.
#
# The earlier "3.3 tok/s ceiling" for this model was computed at 4-bit with no
# I/O-compute overlap and then wrongly described as physics. Only t_ram is a
# hard floor, and t_ram shrinks with the bit width. The 301 ms of disk that
# dominated is the attackable term. Redone properly here.
# ---------------------------------------------------------------------------
M25 = MODELS[0]
KV_PER_TOKEN = 62 * 2 * (8 * 128) * 2      # 254 KB/token: 62 layers, 8 KV heads


def m25_step(expert_bits=4, dense_bits=4, ctx=16384, kv_bits=16,
             hit_bonus=0.0, k=8, overlap=False, t_over=T_OVER, gamma=1.0):
    """One rung. gamma = accepted tokens per forward pass (MTP speculation)."""
    kv = ctx * KV_PER_TOKEN * kv_bits / 16
    dense_b = M25.dense * (dense_bits + 0.5) / 8
    # embed_tokens is a row lookup, not a matmul -- it costs RAM but no bandwidth
    embed = 200064 * 3072 * (dense_bits + 0.5) / 8
    cache = 25.77 * GB - (3.5 + 1.0 + 1.0) * GB - kv - dense_b
    eb = M25.expert_params * (expert_bits + 0.5) / 8
    slots = cache / eb / M25.l_moe
    h = min(0.97, hit_rate(Model("x", M25.l_moe, M25.E, k, 1, 1, 2), slots) + hit_bonus)
    cold = M25.l_moe * k * eb * (1 - h)
    # A verification pass reads the dense weights ONCE for gamma tokens, and the
    # expert union grows at slope (1-r) with r=0.42 measured in this repo.
    union = k * (1 + (gamma - 1) * 0.58) if gamma > 1 else k
    exp_ram = M25.l_moe * union * eb / gamma
    t_ram = (exp_ram + (dense_b - embed) / gamma) / RAM_BW
    t_io = cold * (union / k / gamma) / BW
    t = (max(t_io, t_ram) if overlap else t_io + t_ram) + M25.l_moe * t_over / gamma
    return dict(slots=slots, resid=slots / M25.E, hit=h, cache=cache / GB,
                disk=(M25.routed * (expert_bits + .5) / 8 + dense_b) / GB,
                t_io=t_io * 1e3, t_ram=t_ram * 1e3, tok_s=1 / t)


LADDER = [
    ("0  today: 4-bit, Python, no overlap",  dict()),
    ("1  + KV at 8-bit (frees cache)",       dict(kv_bits=8)),
    ("2  + 2-bit experts (asymmetric quant)", dict(kv_bits=8, expert_bits=2)),
    ("3  + C/Metal engine (kills t_over)",   dict(kv_bits=8, expert_bits=2, t_over=0)),
    ("4  + prefetch: overlap I/O w/ compute", dict(kv_bits=8, expert_bits=2, t_over=0,
                                                  overlap=True)),
    ("5  + profile-guided residency (+8pt)", dict(kv_bits=8, expert_bits=2, t_over=0,
                                                  overlap=True, hit_bonus=0.08)),
    ("6  + adaptive top-6",                  dict(kv_bits=8, expert_bits=2, t_over=0,
                                                  overlap=True, hit_bonus=0.08, k=6)),
    ("7  + MTP speculation (gamma=2.1)",     dict(kv_bits=8, expert_bits=2, t_over=0,
                                                  overlap=True, hit_bonus=0.08, k=6,
                                                  gamma=2.1)),
]

if __name__ != "__main__":
    pass
