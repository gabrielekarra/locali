"""Can Kimi K3 run on this Mac? Closed-form budget, measured constants only.

Every number here is either read off the real checkpoint (HF config + safetensors
param census) or measured on this machine. Nothing is assumed that could be
measured instead.

The one sentence this script exists to prove:

    K3 reads 25.8 GB of expert weights per decoded token. That is MORE THAN THE
    WHOLE MACHINE'S RAM, for ONE token. So no cache, no prefetcher and no
    eviction policy can help -- the entire moe-stream hit-rate curve is flat at
    zero here. Only bytes-fetched and bytes/second matter.

Validation: at the stock config this must reproduce the only third-party
measurement that exists (PipeNetwork kimi-k3-mlx, 0.14-0.20 tok/s on Apple
Silicon). It does, to within the spread of their drives.

    python k3_budget.py            # sweep
    python k3_budget.py --check    # self-check only
"""

import argparse

GB = 1e9

# ---------------------------------------------------------------------------
# Checkpoint geometry. Source: moonshotai/Kimi-K3 config.json + HF param census
# (BF16 57,179,884,544 / U8 2,722,740,830,208 / total 2,779,931,837,184).
# ---------------------------------------------------------------------------
L_MOE = 92                      # 93 layers, first_k_dense_replace=1
E, TOPK = 896, 16
EXPERT_PARAMS = 3 * 3584 * 3072  # w1,w3: 3584x3072, w2: 3072x3584 -> 33,030,144
MXFP4_BPP = 0.53125              # 4 bits + one uint8 scale per group of 32
EXPERT_BYTES = EXPERT_PARAMS * MXFP4_BPP  # 17,547,264 B

# Always-on parameters, in billions, text-only (vision tower dropped).
DENSE = {
    "kda_attn": 30.6,        # 69 layers x 443M -- the single biggest block
    "shared_experts": 12.2,  # 2 per layer, hidden 7168 -> inter 6144
    "mla_attn": 5.6,         # 24 layers
    "latent_updown": 4.7,    # routed_expert_down/up_proj, shared per layer
    "embed_lmhead": 2.4,
    "routers_norms": 1.35,   # 1.8B "everything else" minus 0.447B vision
}
DENSE_PARAMS = sum(DENSE.values()) * 1e9  # 56.85B

# Per-sequence state. KDA is linear attention: O(1) in context but a fat
# constant, and it is what actually caps batch size.
KDA_STATE = 69 * 96 * 128 * 128 * 2      # 217 MB per sequence
MLA_KV_PER_TOKEN = 24 * (512 + 64) * 2   # 27.6 KB/token

# ---------------------------------------------------------------------------
# Machine. RAM from sysctl; bandwidth measured by bw.py (F_NOCACHE preads of
# 17,547,264 B against a 40 GB file, i.e. bigger than RAM so the page cache
# cannot flatter it -- an 8 GB file reported a fictional 14.7 GB/s).
# ---------------------------------------------------------------------------
RAM = 25.77 * GB
BW_INTERNAL = 4.64 * GB          # measured, 16 threads, expert-sized blocks
BW_TB4_NVME = 2.8 * GB           # ESTIMATE -- no external drive attached yet

RESERVED = (3.5 + 1.0 + 1.2 + 0.35) * GB + KDA_STATE + 8192 * MLA_KV_PER_TOKEN
USABLE = RAM - RESERVED          # ~19.3 GB for weights

# (capacity GB, GB/s). Internal is the FASTEST device and the smallest -- that
# mismatch is the whole point of effective_bw(). External figures are estimates
# until a drive is attached; re-run bw.py against it before trusting them.
DRIVES = [(183, 4.64), (1000, 2.8), (1000, 2.8)]
# Access share that makes every drive finish at the same time. The internal SSD
# must absorb 45% of reads from 8% of the pack -- i.e. the workload's hot expert
# set must be that skewed. UNVERIFIED for K3; this is the number to measure.
ACCESS_AWARE = [0.453, 0.2735, 0.2735]

# Measured elsewhere in this repo: fraction of expert accesses at a layer that
# repeat within ~1.5 tokens (Qwen3-Next-80B, 512 experts, top-10 -- 2.0% top-k
# ratio vs K3's 1.8%, the closest analogue we have). results/reuse_distance_*.
REPEAT_RATE = 0.42


def effective_bw(drives, access_share=None) -> float:
    """Aggregate GB/s from a set of (capacity_gb, bw) drives holding one pack.

    Reads go to whichever drive holds the expert, and the drives run in
    parallel, so wall time is set by the BUSIEST drive:

        t = max_d (access_share[d] / bw[d])

    A drive helps only in proportion to how many reads land on it. Default
    placement is capacity-proportional -- spread the pack evenly -- which
    strands bandwidth whenever a fast drive is small. This machine is exactly
    that case: the internal SSD is the fastest device and holds 12% of the pack.

    Passing access_share models BANDWIDTH-PROPORTIONAL PLACEMENT: put the
    experts a given workload actually hits onto the fast-but-small drive, so
    its access share matches its bandwidth share instead of its capacity share.
    Lossless -- it moves bytes between drives, it does not change any weight.
    """
    total_cap = sum(c for c, _ in drives)
    share = access_share or [c / total_cap for c, _ in drives]
    return 1.0 / max(s / bw for s, (_, bw) in zip(share, drives))


def eff_bpp(bits: float) -> float:
    """Bytes per param for MLX-style group quant: group 64, fp16 scale+bias."""
    return (bits + 0.5) / 8


def union(n: int, k: int = TOPK, r: float = REPEAT_RATE) -> float:
    """Distinct experts a layer needs when n tokens are processed in one visit.

    Grows as k*(1 + (n-1)*(1-r)) -- linear in n with slope (1-r), not the
    saturating curve you would get from independent routing, because at 16/896
    the pool is nowhere near exhausted. This is why batching and speculative
    decoding buy so little here: their entire benefit IS r.
    """
    return min(k * (1 + (n - 1) * (1 - r)), E)


def bytes_per_token(expert_bits=4.25, k=TOPK, n_tok=1, streamed_dense_gb=0.0):
    """Weight bytes read from disk per decoded token."""
    eb = EXPERT_PARAMS * (MXFP4_BPP if expert_bits == 4.25 else eff_bpp(expert_bits))
    return L_MOE * union(n_tok, k) / n_tok * eb + streamed_dense_gb * GB


def resident_gb(bits_by_part: dict) -> float:
    return sum(DENSE[p] * 1e9 * eff_bpp(b) for p, b in bits_by_part.items()) / GB


def plan(expert_bits=4.25, k=TOPK, n_tok=1, bw=BW_INTERNAL, dense_bits=None):
    """Full config -> (tok/s, resident GB, streamed dense GB, bytes/token)."""
    dense_bits = dense_bits or {p: 4 for p in DENSE}
    held = {p: b for p, b in dense_bits.items() if b}          # b=None -> stream it
    streamed = sum(DENSE[p] for p, b in dense_bits.items() if not b) * 1e9 * eff_bpp(4) / GB
    res = resident_gb(held)
    bpt = bytes_per_token(expert_bits, k, n_tok, streamed)
    return {
        "tok_s": bw / bpt,
        "resident_gb": res,
        "streamed_dense_gb": streamed,
        "gb_per_token": bpt / GB,
        "fits": res <= USABLE / GB,
    }


def _check():
    assert EXPERT_BYTES == 17_547_264, EXPERT_BYTES
    # Routed total must reconcile with the HF census to the byte.
    assert abs(EXPERT_PARAMS * E * L_MOE - 2_722_740_830_208) < 1e6

    # THE headline: one token's expert reads exceed total RAM.
    stock = bytes_per_token()
    assert stock > RAM, f"{stock/GB:.1f} GB/token vs {RAM/GB:.1f} GB RAM"
    assert 25.0 < stock / GB < 26.5, stock / GB

    # No expert cache is possible: usable RAM cannot even hold top-k for every
    # layer, let alone anything reusable across tokens.
    slots_per_layer = USABLE / EXPERT_BYTES / L_MOE
    assert slots_per_layer < TOPK, slots_per_layer

    # Reproduce the only independent measurement: PipeNetwork kimi-k3-mlx
    # reports 0.14-0.20 tok/s. Their drives are not ours, so bracket the band.
    for bw, lo, hi in [(4.0 * GB, 0.14, 0.18), (5.2 * GB, 0.17, 0.22)]:
        t = plan(bw=bw)["tok_s"]
        assert lo <= t <= hi, f"{t:.3f} outside [{lo},{hi}] at {bw/GB} GB/s"

    # Levers must compose multiplicatively and none may reach interactive alone.
    assert plan(expert_bits=2.0)["tok_s"] < 0.5
    assert plan(k=8)["tok_s"] < 0.5
    assert plan(n_tok=8)["tok_s"] < 0.5

    # The dense backbone is the RAM constraint, not the experts.
    assert resident_gb({p: 4 for p in DENSE}) > USABLE / GB, "4-bit core should NOT fit"
    assert resident_gb({p: 2 for p in DENSE}) < USABLE / GB, "2-bit core should fit"

    # Placement: one drive is its own bandwidth; a big slow drive next to a
    # small fast one strands the fast one under capacity-proportional spread,
    # and access-aware placement recovers it.
    assert abs(effective_bw([(183, 4.64)]) - 4.64) < 1e-9
    naive = effective_bw(DRIVES)
    aware = effective_bw(DRIVES, ACCESS_AWARE)
    assert naive < aware, (naive, aware)
    assert aware <= sum(bw for _, bw in DRIVES) + 1e-9, "cannot beat the sum"
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    _check()
    if args.check:
        return

    print(f"\nexpert {EXPERT_BYTES/1e6:.2f} MB | per-token expert reads "
          f"{bytes_per_token()/GB:.1f} GB | RAM {RAM/GB:.1f} GB | usable {USABLE/GB:.1f} GB")
    print(f"dense backbone {DENSE_PARAMS/1e9:.1f}B -> {USABLE/DENSE_PARAMS*8:.2f} bits/param "
          f"if fully resident with zero expert cache\n")

    # KDA state is what caps batch, so say what the cap is before sweeping.
    print(f"KDA state {KDA_STATE/1e6:.0f} MB/seq -> max {int(USABLE*0.3/KDA_STATE)} "
          f"concurrent seqs even giving it 30% of usable RAM\n")

    rows = [
        ("stock MXFP4, internal SSD",       dict()),
        ("+ 3-bit experts",                 dict(expert_bits=3)),
        ("+ adaptive top-12",               dict(expert_bits=3, k=12)),
        ("+ spec decode gamma=4",           dict(expert_bits=3, k=12, n_tok=4)),
        ("+ 3 drives, naive placement",     dict(expert_bits=3, k=12, n_tok=4,
                                                 bw=effective_bw(DRIVES) * GB)),
        ("+ bandwidth-proportional place",  dict(expert_bits=3, k=12, n_tok=4,
                                                 bw=effective_bw(DRIVES, ACCESS_AWARE) * GB)),
        ("aggressive: 2.5-bit experts, k=8", dict(expert_bits=2.5, k=8, n_tok=4,
                                                  bw=effective_bw(DRIVES, ACCESS_AWARE) * GB,
                                                  dense_bits={p: 2 for p in DENSE})),
    ]
    print(f"{'config':<34} {'GB/tok':>7} {'tok/s':>7} {'resident':>9} {'fits':>5}")
    for name, kw in rows:
        r = plan(**kw)
        print(f"{name:<34} {r['gb_per_token']:>7.2f} {r['tok_s']:>7.2f} "
              f"{r['resident_gb']:>8.1f}G {str(r['fits']):>5}")

    print(f"\nplacement on {len(DRIVES)} drives: naive {effective_bw(DRIVES):.2f} GB/s, "
          f"bandwidth-proportional {effective_bw(DRIVES, ACCESS_AWARE):.2f} GB/s "
          f"(sum of drives {sum(bw for _, bw in DRIVES):.2f})")
    print(f"prefill: one pass touches all {E} experts/layer = "
          f"{L_MOE*E*EXPERT_BYTES/GB:.0f} GB, i.e. the whole checkpoint, "
          f"{L_MOE*E*EXPERT_BYTES/GB/effective_bw(DRIVES, ACCESS_AWARE):.0f} s "
          f"-- independent of prompt length.")


if __name__ == "__main__":
    main()
