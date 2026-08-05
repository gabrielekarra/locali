"""DeepSeek-V4 with a resident MLX backbone and Locali-streamed experts."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import psutil

from dsv4_arena import V4ArenaMoE, V4ArenaStore


EXPERT_MARKER = ".ffn.switch_mlp."


def _install_architecture(omlx_source: Path):
    """Register the upstream MLX V4 architecture used by the checkpoint.

    Locali owns the execution policy and streamed MoE. The dense architecture
    comes from the Apache-2.0 DeepSeek V4 implementation vendored by oMLX.
    """
    source = omlx_source.expanduser().resolve()
    if not (source / "omlx" / "patches" / "deepseek_v4").is_dir():
        raise FileNotFoundError(
            f"DeepSeek V4 MLX architecture not found at {source}. "
            "Clone https://github.com/jundot/omlx there or pass --omlx-source."
        )
    path = str(source)
    if path not in sys.path:
        sys.path.insert(0, path)
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    from mlx_lm.models.deepseek_v4 import Model, ModelArgs

    return Model, ModelArgs


def _dense_weight_names(
    snapshot: Path,
    *,
    include_mtp: bool = False,
) -> tuple[dict, set[str]]:
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text()
    )
    weight_map = index["weight_map"]
    names = {
        name
        for name in weight_map
        if EXPERT_MARKER not in name
        and (include_mtp or not name.startswith("mtp."))
    }
    return weight_map, names


def _load_selected(snapshot: Path, weight_map: dict, names: set[str]) -> dict:
    selected = {}
    shards = sorted({weight_map[name] for name in names})
    for shard in shards:
        arrays = mx.load(str(snapshot / shard))
        selected.update({name: value for name, value in arrays.items() if name in names})
        del arrays
    return selected


def load_streaming(
    snapshot: Path,
    index_path: Path,
    omlx_source: Path,
    ceiling_gb: float = 7.0,
    threads: int = 8,
    nocache: bool = True,
    cache_policy: str = "slru-all",
    prefetch_depth: int = 0,
    prefetch_k: int = 5,
    prefetch_k2: int = 4,
    native_scheduler: bool = True,
    fused_decode: bool = False,
    overlap_hits: bool = False,
    mtp: bool = False,
    mtp_depth: int = 5,
    mtp_share: float = 0.08,
):
    """Load V4's dense core and route every expert through Locali.

    With ``mtp=True`` the checkpoint's three embedded DSpark blocks are also
    attached. Their dense parameters remain resident, while their much larger
    routed-expert stacks use a distinct streaming tier in the same arena.
    """
    snapshot = snapshot.expanduser().resolve()
    index_path = index_path.expanduser().resolve()
    config = json.loads((snapshot / "config.json").read_text())
    if native_scheduler:
        from locali_native import ensure_available

        native_scheduler = ensure_available()
    Model, ModelArgs = _install_architecture(omlx_source)
    if mtp:
        from omlx.patches.mlx_lm_mtp import (
            apply_mlx_lm_mtp_patch,
            set_mtp_active,
            set_mtp_depth,
        )

        set_mtp_depth(mtp_depth)
        set_mtp_active(True)
    try:
        if mtp and not apply_mlx_lm_mtp_patch():
            raise RuntimeError("oMLX refused the DeepSeek V4 DSpark/MTP patch")
        model = Model(ModelArgs.from_dict(config))
    finally:
        if mtp:
            # This flag controls construction only. The model instance keeps
            # its own decode-enabled marker after loading.
            set_mtp_active(False)

    if mtp and not getattr(model, "_omlx_dspark_decode_enabled", False):
        raise ValueError("checkpoint does not expose an embedded DSpark decoder")

    streamers = []
    for layer_idx, layer in enumerate(model.model.layers):
        block = layer.ffn
        streamer = None
        # Drop the enormous lazy SwitchGLU graph before any model-wide eval.
        block.switch_mlp = nn.Module()
        block.__dict__["_locali_stream"] = streamer
        streamers.append(streamer)

    if mtp:
        for stage in model.mtp:
            # As above, discard the lazy 256-expert graph before evaluating
            # any model-wide parameters. Locali restores a streamed module
            # after dense loading has completed.
            stage.ffn.switch_mlp = nn.Module()
            stage.ffn.__dict__["_locali_stream"] = None

    weight_map, dense_names = _dense_weight_names(snapshot, include_mtp=mtp)
    dense = _load_selected(snapshot, weight_map, dense_names)
    dense = model.sanitize(dense)

    quant = config.get("quantization_config")
    if not quant:
        raise ValueError("checkpoint has no MLX quantization_config")

    def quantize_dense(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        weight = dense.get(f"{path}.weight")
        scales = dense.get(f"{path}.scales")
        if weight is None or scales is None:
            return False
        # Mixed checkpoints are occasionally hand-tuned after the allocator
        # writes config.json. Infer the only geometry that can load the actual
        # packed tensor instead of trusting stale per-module overrides.
        logical_input = int(module.weight.shape[-1])
        packed_input = int(weight.shape[-1])
        scale_groups = int(scales.shape[-1])
        if packed_input * 32 % logical_input or logical_input % scale_groups:
            raise ValueError(
                f"cannot infer affine quantization for {path}: weight "
                f"{weight.shape}, scales {scales.shape}, input {logical_input}"
            )
        bits = packed_input * 32 // logical_input
        group_size = logical_input // scale_groups
        if not 1 <= bits <= 8:
            raise ValueError(f"invalid inferred bit width {bits} for {path}")
        return {
            "group_size": group_size,
            "bits": bits,
            "mode": "affine",
        }

    nn.quantize(
        model,
        group_size=int(quant["group_size"]),
        bits=int(quant["bits"]),
        mode=quant.get("mode", "affine"),
        class_predicate=quantize_dense,
    )

    store = V4ArenaStore(
        index_path,
        ceiling_gb=ceiling_gb,
        hot_share=mtp_share,
        threads=threads,
        nocache=nocache,
        cache_policy=cache_policy,
        native_scheduler=native_scheduler,
    )
    for layer_idx, layer in enumerate(model.model.layers):
        block = layer.ffn
        streamer = V4ArenaMoE(
            store,
            layer_idx,
            block.gate,
            config["num_experts_per_tok"],
            config.get("swiglu_limit", 10.0),
            fused_decode=fused_decode,
            overlap_hits=overlap_hits,
        )
        block.switch_mlp = nn.Module()
        block.__dict__["_locali_stream"] = streamer
        streamers[layer_idx] = streamer

    if mtp:
        expected_main = int(config["num_hidden_layers"])
        index_config = json.loads(index_path.read_text())
        indexed_mtp = int(index_config.get("mtp_layers", 0) or 0)
        if indexed_mtp != len(model.mtp):
            store.close()
            raise ValueError(
                f"DSpark needs {len(model.mtp)} indexed stages, found "
                f"{indexed_mtp}; rebuild the index with dsv4_index.py --mtp"
            )
        for stage_idx, stage in enumerate(model.mtp):
            block = stage.ffn
            streamer = V4ArenaMoE(
                store,
                expected_main + stage_idx,
                block.gate,
                config["num_experts_per_tok"],
                config.get("swiglu_limit", 10.0),
                fused_decode=fused_decode,
                overlap_hits=overlap_hits,
                fp32_swiglu=True,
            )
            block.switch_mlp = nn.Module()
            block.__dict__["_locali_stream"] = streamer

    if prefetch_depth:
        if prefetch_depth not in (1, 2):
            raise ValueError("prefetch_depth must be 0, 1, or 2")
        for layer_idx, streamer in enumerate(streamers):
            if layer_idx + 1 < len(streamers):
                streamer.nxt = streamers[layer_idx + 1]
                streamer.prefetch_k = prefetch_k
            if prefetch_depth == 2 and layer_idx + 2 < len(streamers):
                streamer.nxt2 = streamers[layer_idx + 2]
                streamer.prefetch_k2 = prefetch_k2

    model.load_weights(list(dense.items()), strict=False)
    mx.eval(model.parameters())
    del dense
    model.eval()

    block_class = type(model.model.layers[0].ffn)

    def locali_moe(block, x, input_ids):
        if block.sharding_group is not None:
            raise ValueError("Locali V4 streaming does not support distributed sharding")
        routed = block.__dict__["_locali_stream"](x, input_ids)
        return routed + block.shared_experts(x)

    block_class.__call__ = locali_moe

    active = mx.get_active_memory() / 1e9
    arena = store.resident / 1e9
    dense_gb = active - arena
    physical_gb = psutil.virtual_memory().total / 1e9
    if active > physical_gb * 0.90:
        store.close()
        raise MemoryError(
            f"V4 working set is {active:.2f} GB on {physical_gb:.2f} GB physical RAM; "
            "lower --ceiling-gb"
        )
    return model, store, config, dense_gb


def generate_greedy(model, store, tokenizer, prompt: str, max_tokens: int):
    token_ids = tokenizer(prompt, return_tensors=None)["input_ids"]
    inputs = mx.array(token_ids, dtype=mx.int32)[None]
    cache = model.make_cache()

    start = time.perf_counter()
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - start
    after_prefill = store.stats()

    generated = []
    next_token = int(mx.argmax(logits[0, -1]))
    start = time.perf_counter()
    for _ in range(max_tokens):
        if next_token in tokenizer.all_special_ids:
            break
        generated.append(next_token)
        logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[0, -1]))
    decode_seconds = time.perf_counter() - start

    final = store.stats()
    decode = {
        key: final[key] - after_prefill[key]
        for key in ("hits", "misses", "bytes_read")
    }
    count = decode["hits"] + decode["misses"]
    decode["hit_rate"] = decode["hits"] / count if count else 0.0
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=False),
        "prompt_tokens": len(token_ids),
        "generated_tokens": len(generated),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill": after_prefill,
        "decode": decode,
    }


def generate_speculative_greedy(
    model,
    store,
    token_ids: list[int],
    max_tokens: int,
    *,
    prefill_step_size: int = 2048,
):
    """Run oMLX's exact DSpark draft/verify loop and expose useful metrics."""
    if not getattr(model, "_omlx_dspark_decode_enabled", False):
        raise ValueError("model was not loaded with DSpark enabled")
    if not token_ids:
        raise ValueError("speculative generation requires at least one input token")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    from mlx_lm.generate import BatchGenerator
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp_runtime

    mtp_stats = {}
    original_log_stats = mtp_runtime._log_mtp_stats

    def capture_stats(uid, stats, finish_reason):
        drafted = list(stats.depth_drafted)
        accepted = list(stats.depth_accepted)
        total_drafted = sum(drafted)
        mtp_stats.update(
            {
                "cycles": stats.cycles,
                "accepted": stats.accepts,
                "drafted": total_drafted,
                "accept_rate": (
                    stats.accepts / total_drafted if total_drafted else 0.0
                ),
                "depth_drafted": drafted,
                "depth_accepted": accepted,
                "backbone_seconds": stats.backbone_ms / 1000.0,
                "draft_seconds": stats.mtp_head_ms / 1000.0,
                "sample_seconds": stats.sample_ms / 1000.0,
                "cache_seconds": stats.cache_ops_ms / 1000.0,
                "finish_reason": finish_reason,
            }
        )
        original_log_stats(uid, stats, finish_reason)

    mtp_runtime._log_mtp_stats = capture_stats
    generator = BatchGenerator(
        model,
        max_tokens=max_tokens,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=prefill_step_size,
    )
    before = store.stats()
    generated = []
    started = time.perf_counter()
    first_elapsed = None
    after_first = None
    try:
        generator.insert([token_ids], max_tokens=[max_tokens])
        while len(generated) < max_tokens:
            responses = generator.next_generated()
            if not responses:
                continue
            for response in responses:
                generated.append(int(response.token))
                if first_elapsed is None:
                    first_elapsed = time.perf_counter() - started
                    after_first = store.stats()
                if response.finish_reason is not None:
                    break
            if responses[-1].finish_reason is not None:
                break
    finally:
        elapsed = time.perf_counter() - started
        prefill_seconds = generator._prompt_time_counter
        generator.close()
        mtp_runtime._log_mtp_stats = original_log_stats

    final = store.stats()
    traffic = {
        key: final[key] - before[key]
        for key in ("hits", "misses", "bytes_read")
    }
    count = traffic["hits"] + traffic["misses"]
    traffic["hit_rate"] = traffic["hits"] / count if count else 0.0
    steady_traffic = {
        key: final[key] - (after_first or before)[key]
        for key in ("hits", "misses", "bytes_read")
    }
    steady_count = steady_traffic["hits"] + steady_traffic["misses"]
    steady_traffic["hit_rate"] = (
        steady_traffic["hits"] / steady_count if steady_count else 0.0
    )
    decode_seconds = max(0.0, elapsed - prefill_seconds)
    return {
        "tokens": generated,
        "prompt_tokens": len(token_ids),
        "generated_tokens": len(generated),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "first_token_seconds": max(
            0.0, (first_elapsed or elapsed) - prefill_seconds
        ),
        "traffic": traffic,
        "steady_traffic": steady_traffic,
        "mtp": mtp_stats,
    }


@dataclass
class ChatTurn:
    text: str
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    decode: dict
    finish_reason: str


class LocaliChatSession:
    """A live V4 chat transcript backed by one persistent MLX KV cache.

    The cache contains every prompt and generated token except the terminal EOS.
    The next user turn starts by feeding that EOS followed by the next V4
    user/assistant markers.
    """

    def __init__(self, model, store, tokenizer, *, context_size: int = 32768):
        self.model = model
        self.store = store
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.cache = model.make_cache()
        self.cached_tokens = 0
        self.transcript_ids: list[int] = []
        self.turns = 0

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

    def prime(self, prefix: str, *, prefill_step_size: int = 2048) -> float:
        """Evaluate the fixed chat prefix before accepting the first message."""
        if self.turns or self.transcript_ids or self.cached_tokens:
            raise ValueError("chat prefix can only be primed on a new session")
        token_ids = self._tokenize(prefix)
        if not token_ids:
            return 0.0
        started = time.perf_counter()
        processed = 0
        while processed < len(token_ids):
            end = min(processed + prefill_step_size, len(token_ids))
            logits = self.model(
                mx.array(token_ids[processed:end], dtype=mx.int32)[None],
                cache=self.cache,
            )
            mx.eval(logits)
            processed = end
        elapsed = time.perf_counter() - started
        self.transcript_ids.extend(token_ids)
        self.cached_tokens += len(token_ids)
        return elapsed

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampler: Callable,
        stop_token_ids: set[int] | None = None,
        on_token: Callable[[int], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_prefill: Callable[[int, int], None] | None = None,
        prefill_step_size: int = 2048,
    ) -> ChatTurn:
        suffix_ids = self._tokenize(prompt)
        if not suffix_ids:
            raise ValueError("empty encoded chat prompt")
        room = self.context_size - len(self.transcript_ids) - len(suffix_ids)
        if room <= 0:
            raise ValueError(
                f"chat would exceed the {self.context_size}-token context; "
                "restart Locali to begin a new conversation"
            )
        max_tokens = min(max_tokens, room)

        # After Ctrl+C an in-flight Metal graph may have advanced only some
        # cache types.  Keep the token transcript authoritative and rebuild the
        # entire known-good prefix on the next turn instead of trusting a
        # half-updated cache.
        rebuilding = self.cached_tokens == 0 and bool(self.transcript_ids)
        prompt_ids = (
            [*self.transcript_ids, *suffix_ids] if rebuilding else suffix_ids
        )
        total = len(prompt_ids)
        processed = 0
        generated: list[int] = []
        detokenizer = None
        try:
            if on_prefill:
                on_prefill(0, total)
            started = time.perf_counter()
            logits = None
            while processed < total:
                end = min(processed + prefill_step_size, total)
                inputs = mx.array(prompt_ids[processed:end], dtype=mx.int32)[None]
                logits = self.model(inputs, cache=self.cache)
                mx.eval(logits)
                processed = end
                if on_prefill:
                    on_prefill(processed, total)
            prefill_seconds = time.perf_counter() - started
            after_prefill = self.store.stats()
            self.cached_tokens += total

            from mlx_lm.tokenizer_utils import TokenizerWrapper

            detokenizer = TokenizerWrapper(self.tokenizer).detokenizer
            detokenizer.reset()
            eos = self.tokenizer.eos_token_id
            eos_ids = set(
                eos if isinstance(eos, (list, tuple, set)) else [eos]
            )
            eos_ids.discard(None)
            if stop_token_ids:
                eos_ids.update(stop_token_ids)
            finish_reason = "length"

            def sample(last_logits) -> int:
                logprobs = last_logits - mx.logsumexp(
                    last_logits, axis=-1, keepdims=True
                )
                token = sampler(logprobs)
                mx.eval(token)
                return int(token.item())

            next_token = sample(logits[0, -1])
            started = time.perf_counter()
            for _ in range(max_tokens):
                if next_token in eos_ids:
                    finish_reason = "stop"
                    break
                generated.append(next_token)
                if on_token:
                    on_token(next_token)
                detokenizer.add_token(next_token)
                if on_text:
                    segment = detokenizer.last_segment
                    if segment:
                        on_text(segment)

                logits = self.model(
                    mx.array([[next_token]], dtype=mx.int32),
                    cache=self.cache,
                )
                mx.eval(logits)
                self.cached_tokens += 1
                next_token = sample(logits[0, -1])

            detokenizer.finalize()
            if on_text:
                segment = detokenizer.last_segment
                if segment:
                    on_text(segment)
            decode_seconds = time.perf_counter() - started
        except KeyboardInterrupt:
            # Preserve a partially visible answer as a completed assistant
            # turn by inserting EOS after an interrupted generation.
            # The next call reconstructs its KV state from these exact IDs.
            if generated:
                self.transcript_ids.extend(suffix_ids)
                self.transcript_ids.extend(generated)
                self.turns += 1
            self.cache = self.model.make_cache()
            self.cached_tokens = 0
            raise

        final = self.store.stats()
        decode = {
            key: final[key] - after_prefill[key]
            for key in ("hits", "misses", "bytes_read")
        }
        count = decode["hits"] + decode["misses"]
        decode["hit_rate"] = decode["hits"] / count if count else 0.0
        self.transcript_ids.extend(suffix_ids)
        self.transcript_ids.extend(generated)
        self.turns += 1
        return ChatTurn(
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            prompt_tokens=total,
            generated_tokens=len(generated),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            decode=decode,
            finish_reason=finish_reason,
        )
