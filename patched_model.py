"""Patch an MLX MoE model to stream routed-expert weights from an
ExpertStore, and load it WITHOUT ever materializing the experts.

load_streaming_model() is the memory-critical entry point: it loads the
model LAZILY (mlx_lm's lazy=True memory-maps the safetensors; nothing is
materialized), patches the switch_mlp modules to alias the store's pool
tensors BEFORE any evaluation, then materializes only what remains --
core weights plus the pools. The original expert tensors are dropped as
unevaluated mmap views: never read, never in RAM.

Measured on GLM-4.5-Air (106B, 60.1 GB checkpoint): 56.1 GB of routed
experts stay on disk, 4.0 GB of core loads, and generation peaks at
4.45 GB. Load takes 2.8 s because almost nothing is read.

Works on any checkpoint whose routed experts are stacked per layer under
`layer.mlp.switch_mlp` with affine quantization -- the GLM, Qwen3-MoE and
Qwen3-Next families all are. Dense layers without switch_mlp are skipped.

Per MoE layer, two changes; the math is never touched:
1. switch_mlp's sub-projection weight/scales/biases alias the store's
   PER-LAYER pool tensors (per-layer pools -- see expert_store.py's
   header for why the v1 global pool caused copies and OOM).
2. switch_mlp.__call__ is overridden (via __class__ reassignment --
   Python resolves dunders through the type) to translate router expert
   ids -> pool slot ids, then delegate to stock SwitchGLU.__call__.

Multi-token calls (prefill) with cache_enabled are chunked one token at
a time, and each chunk's output is evaluated IMMEDIATELY. The eval is
load-bearing, not a nicety: leaving 27 tokens x 48 layers of gathers
pending pinned every intermediate pool version and OOMed the machine.
"""

import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from expert_store import ARR_NAMES, PROJ_NAMES, ExpertStore


def _make_class_glu(template_glu, store: ExpertStore, layer_idx: int, bits: int) -> SwitchGLU:
    """A stock SwitchGLU whose projections alias one bits-class's pool
    tensors. Built from cheap throwaway QuantizedSwitchLinear instances
    (tiny dims) whose weight/scales/biases are immediately re-aliased --
    input/output dims are @properties derived from the aliased shapes, so
    the shells behave exactly like the class-sized module. Stock math,
    stock __call__, per class."""
    q = store.quant_of(bits)
    glu = SwitchGLU.__new__(SwitchGLU)
    nn.Module.__init__(glu)
    for proj in PROJ_NAMES:
        sub = QuantizedSwitchLinear(
            input_dims=q["group_size"], output_dims=1, num_experts=1,
            bias=False, group_size=q["group_size"], bits=bits, mode=q["mode"],
        )
        for arrname in ARR_NAMES:
            setattr(sub, arrname, store.slot_tensor(layer_idx, proj, arrname, bits=bits))
        setattr(glu, proj, sub)
    glu.activation = template_glu.activation
    return glu


def _make_patched_switch_glu_class(layer_idx: int, store: ExpertStore, class_glus: dict | None):
    class _PatchedSwitchGLU(SwitchGLU):
        def _cell_gather(self, x_rows, slot_rows, bits):
            """One sub-64 gather for cells of a single bits class."""
            glu = class_glus[bits] if class_glus is not None else None
            target = glu if glu is not None else self
            out = SwitchGLU.__call__(target, x_rows, slot_rows)
            mx.eval(out)
            return out

        def __call__(self, x, indices):
            leading_shape = indices.shape[:-1]
            top_k = indices.shape[-1]
            hidden = x.shape[-1]
            num_tokens = 1
            for d in leading_shape:
                num_tokens *= d

            # Mixed-precision pack: hot and cold experts live in different
            # pools with different quantization -- one gather can't span
            # both, so EVERY call (decode included) goes through the
            # cell/wave path, with waves split by bits class.
            # The wave pool holds one call's working set, not a layer's, so it
            # needs the expert-major path for the same reason the cache does:
            # prefill must be chunked into waves that fit the pool.
            if (store.cache_enabled or store.wave_slots) and (num_tokens > 1 or store.mixed):
                # EXPERT-MAJOR prefill: iterate experts, not tokens. All
                # tokens are known upfront, so group the (token, k) cells
                # by routed expert and process experts in waves that fit
                # the layer's pool -- each unique expert is read from disk
                # AT MOST ONCE per layer per prefill, regardless of prompt
                # length or ceiling (the old per-token path re-fetched an
                # expert every time it was evicted between tokens that
                # shared it). Each cell's math is independent in SwitchGLU
                # (the score-weighted sum over k happens in the caller),
                # so computing cells expert-grouped and unpermuting at the
                # end changes execution order only, not results.
                t0 = time.perf_counter()
                rows = [
                    [int(v) for v in row]
                    for row in indices.reshape(num_tokens, top_k).tolist()
                ]  # ONE router sync for the whole prompt (old path: one per token)
                store.t_sync += time.perf_counter() - t0

                tg = time.perf_counter()
                x_flat = x.reshape(num_tokens, hidden)
                uniq = sorted({e for row in rows for e in row})
                cells_of: dict[int, list[int]] = {e: [] for e in uniq}
                for t, row in enumerate(rows):
                    for k, e in enumerate(row):
                        cells_of[e].append(t * top_k + k)

                # Group unique experts by bits class (uniform pack: one
                # class), then waves within each class sized to that
                # class's pool. Ascending expert id within a class keeps
                # pread offsets forward-sequential.
                by_class: dict[int, list[int]] = {}
                for e in uniq:
                    by_class.setdefault(store.bits_of(layer_idx, e), []).append(e)
                store.t_group += time.perf_counter() - tg

                outs = []
                positions: list[int] = []
                for bits, class_uniq in sorted(by_class.items(), reverse=True):
                    wave_size = store.slots_of[bits]
                    for w in range(0, len(class_uniq), wave_size):
                        wave = class_uniq[w : w + wave_size]
                        tt = time.perf_counter()
                        slot_of = dict(zip(wave, store.translate(layer_idx, wave)))
                        store.t_translate += time.perf_counter() - tt

                        cells = [c for e in wave for c in cells_of[e]]
                        # Sub-batch below 64 cells: SwitchGLU flips to its
                        # _gather_sort path at indices.size >= 64, and that
                        # kernel's accumulation order differs bitwise from
                        # the unsorted path (measured ~1e-7/op, amplified
                        # to ~1.7 logit diff through 48 bf16 layers).
                        # Staying under the threshold keeps every gather on
                        # the exact kernel the bit-verified decode path
                        # uses. Disk behavior is unchanged -- translate
                        # already ran once per wave.
                        tg2 = time.perf_counter()
                        for lo in range(0, len(cells), 63):
                            sub = cells[lo : lo + 63]
                            token_idx = [c // top_k for c in sub]
                            cell_slots = [slot_of[rows[c // top_k][c % top_k]] for c in sub]

                            xw = x_flat[mx.array(token_idx)]
                            sw = mx.array(cell_slots, dtype=mx.uint32).reshape(-1, 1)
                            # _cell_gather routes to the class's aliased
                            # stock SwitchGLU (mixed pack) or self
                            # (uniform), and evals: the eval is
                            # load-bearing -- it releases this batch's
                            # references to the pool versions so the next
                            # wave's writes donate instead of copying, and
                            # bounds pending graph versions.
                            outs.append(self._cell_gather(xw, sw, bits))
                            positions.extend(sub)
                        store.t_gather += time.perf_counter() - tg2

                stacked = mx.concatenate(outs, axis=0)  # (num_cells, 1, hidden)
                # positions is a permutation of range(num_cells);
                # stacked[argsort(positions)] puts cell j's output at row j.
                order = mx.argsort(mx.array(positions, dtype=mx.uint32))
                y = stacked[order].reshape(num_tokens, top_k, hidden)
                return y.reshape(*leading_shape, top_k, hidden)

            t0 = time.perf_counter()
            flat_ids = [int(v) for v in indices.reshape(-1).tolist()]
            store.t_sync += time.perf_counter() - t0
            tt = time.perf_counter()
            slot_ids_flat = store.translate(layer_idx, flat_ids)
            store.t_translate += time.perf_counter() - tt
            slot_indices = mx.array(slot_ids_flat, dtype=mx.uint32).reshape(indices.shape)
            if not store.profile:
                return super().__call__(x, slot_indices)
            tg3 = time.perf_counter()
            out = super().__call__(x, slot_indices)
            mx.eval(out)
            store.t_gather += time.perf_counter() - tg3
            return out

    return _PatchedSwitchGLU


def patch_model(model, store: ExpertStore) -> int:
    """Patch every MoE layer's switch_mlp in place. Returns layers patched.

    Safe on both a fully-loaded model and a lazy one; on a lazy model the
    replaced expert arrays are dropped before ever being materialized."""
    patched = 0
    for layer_idx, layer in enumerate(model.layers):
        switch_mlp = getattr(layer.mlp, "switch_mlp", None)
        if switch_mlp is None:
            continue  # dense (non-MoE) layer -- none in this checkpoint, but don't assume

        # Alias the module's own tensors to the default-class pool either
        # way: this is what drops the original (possibly lazy/mmap) expert
        # tensors from the parameter tree so they are never materialized.
        for proj in PROJ_NAMES:
            sub = getattr(switch_mlp, proj)
            for arrname in ARR_NAMES:
                setattr(sub, arrname, store.slot_tensor(layer_idx, proj, arrname))

        # Mixed pack: hot and cold classes gather through separate stock
        # SwitchGLU shells aliased to their class pools (one gather can't
        # span two quantizations).
        class_glus = None
        if store.mixed:
            class_glus = {
                bits: _make_class_glu(switch_mlp, store, layer_idx, bits)
                for bits in store.bits_classes
            }

        switch_mlp.__class__ = _make_patched_switch_glu_class(layer_idx, store, class_glus)
        patched += 1

    # On a fully-loaded model the dropped expert tensors land in MLX's
    # reusable-buffer cache (still wired!) instead of being released --
    # measured 16.3 GB sitting there, which pushed past the ~19 GB wired
    # limit and OOMed generation. Harmless no-op on a lazy load.
    mx.clear_cache()

    return patched


def load_streaming_model(model_name: str, store: ExpertStore):
    """Load model WITHOUT materializing routed experts. Returns (model,
    tokenizer, patched_layer_count). Peak memory: core (~0.9 GB) + the
    store's pools + tokenizer overhead."""
    from mlx_lm import load

    model, tokenizer = load(model_name, lazy=True)
    patched = patch_model(model, store)
    # Materializes core weights + pool aliases only; the original expert
    # arrays are no longer reachable from the parameter tree, so their
    # safetensors regions are never read.
    mx.eval(model.parameters())
    mx.clear_cache()
    return model, tokenizer, patched
