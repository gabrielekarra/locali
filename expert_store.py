"""ExpertStore: per-layer resident slot pools for streamed MoE experts.

Design v2, rewritten after a real OOM crash took down the whole machine.
v1 used ONE global slot-tensor set shared (aliased) across all 48 layers.
Two mechanically fatal consequences at real ceiling sizes:

1. MLX arrays are functional under the hood: `pool[i] = row` creates a new
   version of the tensor and can only reuse ("donate") the old buffer if
   nothing else references it. With one shared pool, every layer's pending
   lazy gather_qmm graph held references to some version of that one
   tensor chain -- so writes silently degraded to full-pool copies
   (~2 GB each at a 2 GB ceiling; measured ~24 ms/layer, which is exactly
   the M4's memcpy time for the pool, and transiently 2x memory).
2. Chunked prefill built 27 tokens x 48 layers of pending gathers before
   any eval, pinning thousands of pool versions at once -> genuine Metal
   OOM, twice, at two different eval placements.

Per-layer pools fix both: each layer's writes touch only its own 9 small
tensors (ceiling/48 bytes), other layers' pending gathers can't block
donation, and the worst-case copy domain shrinks 48x. The offline
policy simulation on a real routing trace (see SESSION-NOTES.md) showed
per-layer partitioned LRU matches global LRU hit rate within noise at
every ceiling swept, so this costs nothing in hit rate.

Modes:
- cache_enabled=True: per-layer pools, slots_per_layer = total_slots/48,
  LRU per layer. The ceiling has a hard floor: >= 1 slot per layer, and
  practically >= top_k slots per layer or a single decode step can't fit.
- cache_enabled=False (Phase 3 verify): one shared num_experts-slot pool
  aliased by all layers, identity slot==expert_id mapping, no LRU
  (~340 MB). Unchanged semantics from Phase 3.
- wave_slots=N (cache_enabled=False): ONE shared N-slot pool for the whole
  model, refilled per call. For models where the cache cannot help: GLM-5.2
  on 16 GB leaves ~2 GB against a 12.7 GB/token floor, i.e. 0.5% residency,
  where hit rate is ~0 by construction and the LRU is dead weight. Per-layer
  pools of top_k slots would cost 75 x 8 x 21 MB = 9.9 GB; one shared wave
  costs 0.17 GB, because only the ACTIVE layer's experts must be resident.

Sources:
- packed (experts.bin from requantize_experts.py): one fd, all 9 arrays of an
  expert contiguous -> one pread per expert.
- in place (experts.idx from index_inplace.py): offsets point into the
  original safetensors shards, one fd per shard, 9 preads per expert. Halves
  the disk requirement -- no packed copy of the model exists.

translate() stays index translation only; the unmodified
SwitchGLU/QuantizedSwitchLinear math runs against the pool tensors.
"""

import fcntl
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
ARR_NAMES = ("weight", "scales", "biases")
MAX_READER_THREADS = 8
EVAL_EVERY = 8      # staged write batches between forced pool drains


class ExpertStore:
    def __init__(
        self,
        idx_path,
        bin_path,
        ceiling_gb: float | None = None,
        cache_enabled: bool = True,
        num_threads: int = MAX_READER_THREADS,
        log_routing: bool = False,
        wave_slots: int | None = None,
        no_page_cache: bool = False,
    ):
        self.index = json.loads(Path(idx_path).read_text())
        self.layer_ids: list[int] = self.index["layer_ids"]
        self.num_experts: int = self.index["num_experts"]
        self.quant: dict = self.index["quant"]
        self.experts_meta: dict = self.index["experts"]

        # In-place index: offsets address the original shards, one fd each.
        # Packed index: one experts.bin, every expert a single contiguous span.
        self.files: list[str] | None = self.index.get("files")
        self.packed = self.files is None
        if self.packed:
            self.fds = [os.open(str(bin_path), os.O_RDONLY)]
        else:
            self.fds = [os.open(p, os.O_RDONLY) for p in self.files]
        self.fd = self.fds[0]
        # F_NOCACHE (macOS): keep these reads out of the unified buffer cache.
        # Without it the OS silently provides a second cache tier we neither
        # sized nor accounted for -- on a 24 GB Mac reading a 60 GB model that
        # flatters every hit-rate-vs-speed number, and it is exactly the tier
        # that will NOT exist for a 418 GB model. Measurement honesty, not speed.
        self.no_page_cache = no_page_cache
        if no_page_cache:
            for fd in self.fds:
                fcntl.fcntl(fd, 48, 1)   # 48 == F_NOCACHE, absent from Python's fcntl

        self.cache_enabled = cache_enabled
        self.wave_slots = wave_slots
        if wave_slots is not None and cache_enabled:
            raise ValueError("wave_slots is a no-cache mode -- pass cache_enabled=False")

        # Mixed-precision pack (requantize_experts.py): per-expert "bits"
        # field + a cold_quant block. Uniform pack: one class, bits from
        # the top-level quant dict. Pools/LRU state are keyed
        # (layer, bits) either way; uniform is the one-class special case.
        self.default_bits: int = self.quant["bits"]
        self.mixed: bool = "cold_quant" in self.index
        if self.mixed:
            self.cold_quant: dict = self.index["cold_quant"]
            self.bits_classes = [self.default_bits, self.cold_quant["bits"]]
        else:
            self.cold_quant = None
            self.bits_classes = [self.default_bits]

        # Per-class template (shapes differ across classes) + bytes/expert.
        def find_template(bits: int) -> dict:
            for e in range(self.num_experts):
                meta = self.experts_meta[f"L{self.layer_ids[0]}.E{e}"]
                if meta.get("bits", self.default_bits) == bits:
                    return meta
            raise ValueError(f"no expert of class {bits}b in layer {self.layer_ids[0]}")

        self._template = {b: find_template(b) for b in self.bits_classes}
        self._bpe = {
            b: sum(
                arr["nbytes"]
                for proj, arrs in self._template[b].items()
                if proj in PROJ_NAMES
                for arr in arrs.values()
            )
            for b in self.bits_classes
        }
        self.bytes_per_expert = self._bpe[self.default_bits]

        def make_pool(bits: int, nslots: int) -> dict:
            pool = {}
            for proj in PROJ_NAMES:
                for arrname in ARR_NAMES:
                    entry = self._template[bits][proj][arrname]
                    # weight stays packed uint32; scales/biases must be real
                    # bfloat16 for gather_qmm (storage is uint16 bit pattern,
                    # numpy has no bfloat16 -- writes .view() back).
                    dt = mx.uint32 if entry["dtype"] == "uint32" else mx.bfloat16
                    pool[(proj, arrname)] = mx.zeros((nslots, *entry["shape"]), dtype=dt)
            return pool

        # slots per (class): uniform -> all budget to the one class.
        # Mixed -> hot class gets up to hot_n slots capped at half the
        # layer budget (when the ceiling affords all hot_n, the hot pool's
        # capacity equals its working set and never evicts -- pinning
        # emerges without a special case); the remainder goes to cold.
        self.slots_of: dict[int, int] = {}
        if cache_enabled:
            if ceiling_gb is None:
                raise ValueError("ceiling_gb is required when cache_enabled=True")
            layer_budget = ceiling_gb * 1e9 / len(self.layer_ids)
            if self.mixed:
                hot_bits, cold_bits = self.bits_classes
                hot_n = self.index.get("hot_n", 16)
                hot_slots = min(hot_n, int((layer_budget / 2) // self._bpe[hot_bits]))
                cold_slots = int((layer_budget - hot_slots * self._bpe[hot_bits]) // self._bpe[cold_bits])
                self.slots_of = {hot_bits: hot_slots, cold_bits: cold_slots}
            else:
                self.slots_of = {self.default_bits: int(layer_budget // self.bytes_per_expert)}
            for b, n in self.slots_of.items():
                if n < 1:
                    raise ValueError(
                        f"ceiling {ceiling_gb} GB gives {n} slots/layer for the {b}-bit "
                        f"class -- below the hard floor of 1; raise --ceiling-gb"
                    )
            self._pools = {
                (l, b): make_pool(b, self.slots_of[b])
                for l in self.layer_ids
                for b in self.bits_classes
            }
        else:
            nslots = wave_slots if wave_slots is not None else self.num_experts
            self.slots_of = {b: nslots for b in self.bits_classes}
            shared = {b: make_pool(b, nslots) for b in self.bits_classes}
            self._pools = {
                (l, b): shared[b] for l in self.layer_ids for b in self.bits_classes
            }
        self.slots_per_layer = self.slots_of[self.default_bits]

        mx.eval(*self._unique_tensors())

        # Per-(layer, class) LRU state (meaningful only when cache_enabled).
        self._lru: dict[tuple[int, int], OrderedDict[int, int]] = {
            k: OrderedDict() for k in self._pools
        }
        self._free: dict[tuple[int, int], list[int]] = {
            (l, b): list(range(self.slots_of[b])) for (l, b) in self._pools
        }

        self._lock = threading.Lock()
        self._pool_exec = ThreadPoolExecutor(max_workers=min(num_threads, MAX_READER_THREADS))

        self.hits = 0
        self.misses = 0
        self.bytes_read_cold = 0
        self.evictions = 0
        self.layer_stats: dict[int, list[int]] = {l: [0, 0] for l in self.layer_ids}

        # Optional raw access log for offline cache-policy simulation.
        self.log_routing = log_routing
        self.routing_trace: list[dict] = [] if log_routing else None

        # Wall-clock breakdown of the miss path, for profiling (seconds).
        self.t_fetch = 0.0   # disk pread + numpy wrap (thread pool total wall)
        self.t_write = 0.0   # staging setitem ops
        self.t_eval = 0.0    # mx.eval of dirty pool tensors (periodic)
        self.t_sync = 0.0    # router-indices GPU sync (accumulated by patched_model)
        # Profiling forces an eval after the decode gather so its time can be
        # attributed. That serialises the pipeline, so it is OFF by default --
        # timing runs must not change the thing being timed.
        self.profile = False
        self.t_group = 0.0   # pure-Python dedup/cell-grouping (patched_model)
        self.t_gather = 0.0  # gather_qmm submit + eval (patched_model)
        self.t_translate = 0.0  # slot mapping incl. LRU bookkeeping (patched_model)

        # Deferred-eval bookkeeping: staged writes are materialized either
        # by the model's own downstream evals (the common case) or forced
        # here every EVAL_EVERY batches, so staged-but-unmaterialized
        # versions stay bounded even if the caller never evaluates.
        self._dirty_layers: set[int] = set()
        self._staged_batches = 0

    def _unique_tensors(self) -> list[mx.array]:
        seen: dict[int, mx.array] = {}
        for pool in self._pools.values():
            for t in pool.values():
                seen[id(t)] = t
        return list(seen.values())

    @property
    def resident_bytes(self) -> int:
        """Bytes actually held by pool tensors (deduped -- disabled mode
        shares one pool across layers). Static after __init__: pools are
        fixed-size and every write is an in-place slot update, so the
        ceiling is true by construction."""
        return sum(t.nbytes for t in self._unique_tensors())

    def bits_of(self, layer: int, expert: int) -> int:
        return self.experts_meta[f"L{layer}.E{expert}"].get("bits", self.default_bits)

    def quant_of(self, bits: int) -> dict:
        if bits == self.default_bits:
            return self.quant
        return self.cold_quant

    def slot_tensor(self, layer: int, proj: str, arrname: str, bits: int | None = None) -> mx.array:
        if bits is None:
            bits = self.default_bits
        return self._pools[(layer, bits)][(proj, arrname)]

    def _fetch_expert_bytes(self, layer: int, expert: int) -> dict:
        # One pread for the whole expert: the exporter packs all 9 arrays of
        # a (layer, expert) contiguously exactly so this is a single span
        # read (profiled: 9 separate preads made fetch the largest cost
        # bucket in decode). numpy views carve the buffer without copying.
        meta = self.experts_meta[f"L{layer}.E{expert}"]
        entries = [(proj, arrname, meta[proj][arrname]) for proj in PROJ_NAMES for arrname in ARR_NAMES]
        if not self.packed:
            return self._fetch_inplace(layer, expert, entries)
        start = min(e["offset"] for _, _, e in entries)
        end = max(e["offset"] + e["nbytes"] for _, _, e in entries)
        raw = os.pread(self.fd, end - start, start)
        if len(raw) != end - start:
            raise IOError(
                f"short read for L{layer}.E{expert}: wanted {end - start} bytes "
                f"at offset {start}, got {len(raw)}"
            )
        result: dict = {}
        for proj, arrname, e in entries:
            np_dtype = np.uint32 if e["dtype"] == "uint32" else np.uint16
            count = e["nbytes"] // np.dtype(np_dtype).itemsize
            arr = np.frombuffer(raw, dtype=np_dtype, count=count, offset=e["offset"] - start)
            result.setdefault(proj, {})[arrname] = arr.reshape(e["shape"])
        return result

    def _fetch_inplace(self, layer: int, expert: int, entries: list) -> dict:
        # In-place index: the 9 arrays are slices of 9 DIFFERENT stacked
        # tensors, so there is no contiguous span to read -- one pread each,
        # into the shard that holds it.
        result: dict = {}
        for proj, arrname, e in entries:
            raw = os.pread(self.fds[e["file"]], e["nbytes"], e["offset"])
            if len(raw) != e["nbytes"]:
                raise IOError(
                    f"short read for L{layer}.E{expert}.{proj}.{arrname}: wanted "
                    f"{e['nbytes']} bytes at offset {e['offset']} of "
                    f"{self.files[e['file']]}, got {len(raw)}"
                )
            np_dtype = np.uint32 if e["dtype"] == "uint32" else np.uint16
            arr = np.frombuffer(raw, dtype=np_dtype, count=e["nbytes"] // np.dtype(np_dtype).itemsize)
            result.setdefault(proj, {})[arrname] = arr.reshape(e["shape"])
        return result

    def _stage_write(self, layer: int, bits: int, slot: int, data: dict) -> None:
        pool = self._pools[(layer, bits)]
        for proj in PROJ_NAMES:
            for arrname in ARR_NAMES:
                arr = mx.array(data[proj][arrname])
                if arrname != "weight":
                    arr = arr.view(mx.bfloat16)
                pool[(proj, arrname)][slot] = arr

    def _cold_bytes(self, data: dict) -> int:
        return sum(a.nbytes for proj in data.values() for a in proj.values())

    def _translate_disabled(self, layer: int, bits: int, unique_ids: list[int]) -> dict[int, int]:
        # Full pool: identity mapping, slot == expert id. Wave pool: the call's
        # working set is packed into slots 0..n-1 and overwritten next call.
        if self.wave_slots is None:
            slot_of = {eid: eid for eid in unique_ids}
        else:
            if len(unique_ids) > self.wave_slots:
                raise ValueError(
                    f"requested {len(unique_ids)} unique experts for layer {layer} "
                    f"but the wave pool has {self.wave_slots} slots -- raise "
                    "--wave-slots to at least top_k"
                )
            slot_of = {eid: i for i, eid in enumerate(unique_ids)}
        # Write each expert the moment ITS bytes land, not after all of them:
        # .map() joins the whole batch first, so every numpy->MLX copy waited
        # for the slowest read in the wave. as_completed overlaps the copies
        # with the reads still in flight (preads release the GIL).
        t0 = time.perf_counter()
        futs = {self._pool_exec.submit(self._fetch_expert_bytes, layer, eid): eid
                for eid in unique_ids}
        for fut in as_completed(futs):
            eid = futs[fut]
            data = fut.result()
            self._stage_write(layer, bits, slot_of[eid], data)
            self.misses += 1
            self.layer_stats[layer][1] += 1
            self.bytes_read_cold += self._cold_bytes(data)
        t2 = time.perf_counter()
        # Read and copy are now interleaved, so they are one bucket.
        self.t_fetch += t2 - t0
        # Same deferred-eval reasoning as _translate_lru: the caller's gather
        # depends on the written versions, so ordering holds in-stream and an
        # eval here is a pure pipeline drain (profiled: 9% of decode wall).
        # The periodic drain still bounds staged versions unconditionally --
        # with ONE shared wave pool every layer stages against the same
        # tensors, so an unbounded chain would pin a pool copy per layer.
        if unique_ids:
            self._dirty_layers.add((layer, bits))
            self._staged_batches += 1
            if self._staged_batches >= EVAL_EVERY:
                mx.eval(*[t for k in self._dirty_layers for t in self._pools[k].values()])
                self._dirty_layers.clear()
                self._staged_batches = 0
                self.t_eval += time.perf_counter() - t2
        return slot_of

    def _translate_lru(self, layer: int, bits: int, unique_ids: list[int]) -> dict[int, int]:
        if len(unique_ids) > self.slots_of[bits]:
            raise ValueError(
                f"requested {len(unique_ids)} unique {bits}-bit experts for layer "
                f"{layer} but that class has {self.slots_of[bits]} slots/layer -- "
                "ceiling too small to hold one call's working set; raise --ceiling-gb"
            )

        lru = self._lru[(layer, bits)]
        slot_of: dict[int, int] = {}
        misses: list[int] = []

        with self._lock:
            for eid in unique_ids:
                if eid in lru:
                    lru.move_to_end(eid)
                    slot_of[eid] = lru[eid]
                    self.hits += 1
                    self.layer_stats[layer][0] += 1
                else:
                    misses.append(eid)

        if misses:
            t0 = time.perf_counter()
            # Slots are assigned BEFORE the reads and in `misses` order, not in
            # completion order: which expert lands in which slot -- and so which
            # victim each eviction picks -- must not depend on disk timing, or
            # the hit rate stops being reproducible run to run. The assignment
            # needs no data, only the LRU order, so it costs nothing to do first.
            with self._lock:
                free = self._free[(layer, bits)]
                for eid in misses:
                    if free:
                        slot = free.pop()
                    else:
                        _, slot = lru.popitem(last=False)
                        self.evictions += 1
                    lru[eid] = slot
                    slot_of[eid] = slot
                    self.misses += 1
                    self.layer_stats[layer][1] += 1
            # Write each expert as ITS bytes land (see _translate_disabled).
            futs = {self._pool_exec.submit(self._fetch_expert_bytes, layer, eid): eid
                    for eid in misses}
            for fut in as_completed(futs):
                eid = futs[fut]
                data = fut.result()
                self._stage_write(layer, bits, slot_of[eid], data)
                self.bytes_read_cold += self._cold_bytes(data)
            t1 = time.perf_counter()
            self.t_fetch += t1 - t0
            with self._lock:
                # No per-batch eval (profiled: it cost an extra GPU pipeline
                # drain per miss-layer). The caller's gather_qmm depends on
                # the written versions, so ordering is guaranteed in-stream
                # and the model's own evals materialize the writes. The
                # periodic eval below only bounds staged versions when the
                # caller ISN'T evaluating (e.g. store used in isolation) --
                # the ceiling must hold unconditionally, not just when the
                # model happens to drain the pipeline for us.
                self._dirty_layers.add((layer, bits))
                self._staged_batches += 1
                if self._staged_batches >= EVAL_EVERY:
                    t2 = time.perf_counter()
                    mx.eval(*[t for k in self._dirty_layers for t in self._pools[k].values()])
                    self.t_eval += time.perf_counter() - t2
                    self._dirty_layers.clear()
                    self._staged_batches = 0

        return slot_of

    def translate(self, layer: int, expert_ids) -> list[int]:
        # Dedup up front: repeats within one call must map to one slot
        # (prefill routes many tokens to the same popular expert; without
        # dedup the LRU path leaked a slot per repeat).
        # In a mixed pack, all ids in one call must be the same bits class
        # (slot ids index that class's pool) -- the patched model splits by
        # class before calling.
        expert_ids = [int(e) for e in expert_ids]
        unique_ids = sorted(set(expert_ids))

        bits = self.bits_of(layer, unique_ids[0]) if unique_ids else self.default_bits
        if self.mixed:
            for eid in unique_ids:
                if self.bits_of(layer, eid) != bits:
                    raise ValueError(
                        f"translate() got mixed bits classes in one call for layer "
                        f"{layer} -- split ids by store.bits_of() first"
                    )

        if self.log_routing:
            self.routing_trace.append({"layer": layer, "expert_ids": unique_ids})

        if self.cache_enabled:
            slot_of = self._translate_lru(layer, bits, unique_ids)
        else:
            slot_of = self._translate_disabled(layer, bits, unique_ids)
        return [slot_of[eid] for eid in expert_ids]

    def layer_hit_rates(self) -> dict[int, float]:
        return {
            layer: (h / (h + m) if (h + m) else 0.0)
            for layer, (h, m) in self.layer_stats.items()
        }

    def save_routing_trace(self, path) -> None:
        if not self.log_routing:
            raise ValueError("log_routing=False -- nothing recorded")
        Path(path).write_text(json.dumps({
            "layer_ids": self.layer_ids,
            "cache_enabled": self.cache_enabled,
            "slots_per_layer": self.slots_per_layer,
            "trace": self.routing_trace,
        }))

    def close(self) -> None:
        for fd in self.fds:
            os.close(fd)
        self._pool_exec.shutdown(wait=True)
