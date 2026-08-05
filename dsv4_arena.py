"""DeepSeek-V4 routed experts backed by Locali's unified-memory SSD arena."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from arena import ArenaMoE, ArenaStore
from locali_native import NativeCache, available as native_available
from dsv4_kernels import eligible as fused_eligible
from dsv4_kernels import fused_affine2_swiglu


class V4ArenaStore(ArenaStore):
    """The existing Locali arena with V4 quantization metadata validated."""

    def __init__(self, index_path, native_scheduler=True, **kwargs):
        config = json.loads(Path(index_path).read_text())
        self.expert_quant = config.get("quantization", {})
        for tier, quant in self.expert_quant.items():
            if quant.get("mode", "affine") != "affine":
                raise ValueError(
                    f"DeepSeek V4 streaming requires affine MLX weights, got "
                    f"{quant.get('mode')!r} for tier {tier}"
                )
        super().__init__(index_path, **kwargs)
        self.use_f16_moe = {}
        for tier in self.tiers:
            sample = next(
                value for value in self.meta.values() if value["tier"] == tier
            )
            metadata_dtypes = {
                sample[proj][kind][5]
                for proj in ("gate_proj", "up_proj", "down_proj")
                for kind in ("scales", "biases")
            }
            self.use_f16_moe[tier] = metadata_dtypes == {"F16"}
        self.native_caches = {}
        if native_scheduler and native_available():
            for tier in self.tiers:
                self.native_caches[tier] = NativeCache(
                    self.layers * self.E,
                    self.slots[tier],
                    self.protected_cap[tier],
                    policy=(
                        "lfu-decay"
                        if self.cache_policy == "lfu-decay"
                        else "slru"
                    ),
                )
        elif self.cache_policy == "lfu-decay":
            raise RuntimeError("lfu-decay requires the compiled Locali C core")

    def _key_id(self, key):
        layer, expert = key
        return layer * self.E + expert

    def _decode_key(self, key_id):
        return divmod(key_id, self.E)

    def quant(self, tier, field):
        defaults = {"bits": 2, "group_size": 128, "mode": "affine"}
        return self.expert_quant.get(tier, {}).get(field, defaults[field])

    def _place_keys(self, layer, want, prefetch):
        if not self.native_caches:
            return super()._place_keys(layer, want, prefetch)
        if not want:
            return {}, [], {tier: [] for tier in self.tiers}
        tiers = {self.tier(layer, expert) for expert in want}
        if len(tiers) != 1:
            raise ValueError(f"layer {layer} spans multiple expert tiers: {tiers}")
        tier = tiers.pop()
        native_cache = self.native_caches[tier]
        if layer == 0 and not prefetch and len(want) <= self.top_k:
            for cache in self.native_caches.values():
                cache.note_token()
        if not prefetch and len(want) > self.slots[tier] - 1:
            raise ValueError(
                f"layer {layer} needs {len(want)} {tier} experts but the arena "
                f"has {self.slots[tier]-1}; raise --ceiling-gb"
            )
        results = native_cache.plan(
            [layer * self.E + expert for expert in want],
            pinned_slots=tuple(self.pinned.get(tier, ())),
            touch_hits=not prefetch,
        )
        out, issued = {}, []
        waits = {name: [] for name in self.tiers}
        for expert, result in zip(want, results):
            key = (layer, expert)
            if result.hit:
                if not prefetch:
                    self.hits += 1
                    if key in self.pf_keys:
                        self.pf_keys.discard(key)
                        self.prefetch_used += 1
                futs = self.inflight.get(key)
                if futs:
                    if all(f.done() for f in futs):
                        del self.inflight[key]
                    elif not prefetch:
                        waits[tier].extend(futs)
            else:
                if not prefetch:
                    self.misses += 1
                if result.evicted_key >= 0:
                    victim = self._decode_key(result.evicted_key)
                    self.inflight.pop(victim, None)
                    if victim in self.pf_keys:
                        self.pf_keys.discard(victim)
                        self.prefetch_wasted += 1
                    self.evictions += 1
                if result.placed:
                    issued.append((expert, tier, result.slot))
            if result.placed:
                out[expert] = (tier, result.slot)
            elif not prefetch:
                raise RuntimeError(
                    f"native scheduler could not place demanded L{layer}.E{expert}"
                )
        if not prefetch:
            self.pinned = {tier: {slot for _, slot in out.values()}}
        return out, issued, waits

    def _protected_size(self, tier):
        if tier in self.native_caches:
            return self.native_caches[tier].protected_size
        return super()._protected_size(tier)

    def stats(self):
        stats = super().stats()
        stats["scheduler"] = "c" if self.native_caches else "python"
        return stats

    def close(self):
        for cache in self.native_caches.values():
            cache.close()
        self.native_caches = {}
        super().close()


class V4ArenaMoE(ArenaMoE):
    """V4 top-6 routing and clamped SwiGLU over streamed affine weights.

    Routing remains the checkpoint's resident ``MoEGate``.  In particular this
    preserves token-id hash routing in layers 0..2 and the selection-only bias
    in scored layers.  This class replaces only the 256 routed experts; the
    always-on shared expert stays resident and is added by the model block.
    """

    def __init__(self, store: V4ArenaStore, layer: int, gate, top_k: int,
                 swiglu_limit: float = 10.0, fused_decode: bool = False,
                 overlap_hits: bool = False, fp32_swiglu: bool = False):
        super().__init__(store, layer, gate, bias=None, top_k=top_k)
        self.swiglu_limit = swiglu_limit
        self.scores_follow_output = True
        self.fused_decode = fused_decode
        self.overlap_hits = overlap_hits
        self.fp32_swiglu = fp32_swiglu

    def route(self, x, input_ids=None):
        return self.gate(x, input_ids)

    def _gather(self, tier, inputs, indices):
        bits = int(self.store.quant(tier, "bits"))
        group_size = int(self.store.quant(tier, "group_size"))
        mode = self.store.quant(tier, "mode")
        if self.store.use_f16_moe[tier] and inputs.dtype == mx.bfloat16:
            inputs = inputs.astype(mx.float16)
        def array(proj, kind):
            return self.store.arena[(tier, proj, kind)]

        if self.fused_decode and fused_eligible(
            inputs,
            indices,
            array("up_proj", "weight"),
            array("up_proj", "scales"),
            array("up_proj", "biases"),
            group_size=group_size,
            bits=bits,
        ):
            hidden = fused_affine2_swiglu(
                inputs,
                indices,
                array("up_proj", "weight"),
                array("up_proj", "scales"),
                array("up_proj", "biases"),
                array("gate_proj", "weight"),
                array("gate_proj", "scales"),
                array("gate_proj", "biases"),
                group_size=group_size,
                limit=self.swiglu_limit,
            )
            return mx.gather_qmm(
                hidden,
                array("down_proj", "weight"),
                array("down_proj", "scales"),
                array("down_proj", "biases"),
                rhs_indices=indices,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )

        def qmm(value, proj):
            return mx.gather_qmm(
                value,
                array(proj, "weight"),
                array(proj, "scales"),
                array(proj, "biases"),
                rhs_indices=indices,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )

        up = qmm(inputs, "up_proj")
        gate = qmm(inputs, "gate_proj")
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        if self.fp32_swiglu:
            dtype = up.dtype
            hidden = (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(
                dtype
            )
        else:
            hidden = nn.silu(gate) * up
        return qmm(hidden, "down_proj")
