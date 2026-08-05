from collections import OrderedDict
from pathlib import Path
import subprocess
import sys

import pytest

from locali_native import NativeCache


class ReferenceCache:
    def __init__(self, slots, protected_capacity):
        self.slots = slots
        self.protected_capacity = protected_capacity
        self.probation_capacity = slots - 1 - protected_capacity
        self.free = list(range(1, slots))
        self.probation = OrderedDict()
        self.protected = OrderedDict()

    def plan(self, key, pinned=(), touch=True):
        if key in self.protected:
            slot = self.protected[key]
            if touch:
                self.protected.move_to_end(key)
            return slot, True, -1, True
        if key in self.probation:
            slot = self.probation[key]
            if touch and self.protected_capacity:
                del self.probation[key]
                self.protected[key] = slot
                if len(self.protected) > self.protected_capacity:
                    old, old_slot = self.protected.popitem(last=False)
                    self.probation[old] = old_slot
            elif touch:
                self.probation.move_to_end(key)
            return slot, True, -1, True

        probation_full = len(self.probation) >= self.probation_capacity
        victim = None
        source = None
        if probation_full or not self.free:
            victim = next(
                (k for k, slot in self.probation.items() if slot not in pinned),
                None,
            )
            source = self.probation
        if victim is None and not probation_full and self.free:
            slot = self.free.pop()
            evicted = -1
        else:
            if victim is None:
                victim = next(
                    (k for k, slot in self.protected.items() if slot not in pinned),
                    None,
                )
                source = self.protected
            if victim is None:
                return -1, False, -1, False
            slot = source.pop(victim)
            evicted = victim
        self.probation[key] = slot
        return slot, False, evicted, True


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = tmp_path_factory.mktemp("locali-native") / f"liblocali_core{suffix}"
    shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(
        [
            "cc", "-O2", "-std=c11", *shared,
            str(root / "native" / "locali_core.c"), "-o", str(out),
        ],
        check=True,
    )
    return out


@pytest.mark.parametrize("protected_capacity", [0, 3])
def test_native_cache_matches_reference(native_library, protected_capacity):
    native = NativeCache(64, 8, protected_capacity, library=native_library)
    reference = ReferenceCache(8, protected_capacity)
    sequence = [0, 1, 2, 3, 0, 4, 5, 1, 6, 7, 2, 8, 0, 9, 3, 10]
    for position, key in enumerate(sequence):
        resident = [slot for slot in range(1, 8) if any(
            cache.get(k) == slot
            for cache in (reference.probation, reference.protected)
            for k in cache
        )]
        pinned = resident[:1] if position % 4 == 3 else []
        expected = reference.plan(key, pinned=pinned, touch=True)
        actual = native.plan([key], pinned_slots=pinned, touch_hits=True)[0]
        assert (actual.slot, actual.hit, actual.evicted_key, actual.placed) == expected
        assert native.probation_size == len(reference.probation)
        assert native.protected_size == len(reference.protected)
        for candidate in range(11):
            expected_slot = reference.probation.get(
                candidate, reference.protected.get(candidate, -1)
            )
            assert native.peek(candidate) == expected_slot
    native.close()


def test_native_lfu_decay_prefers_frequency_then_recency(native_library):
    native = NativeCache(
        16, 3, 0, policy="lfu-decay", library=native_library
    )
    try:
        assert not native.plan([1])[0].hit
        assert native.plan([1])[0].hit
        assert not native.plan([2])[0].hit
        result = native.plan([3])[0]
        assert result.evicted_key == 2
        for _ in range(16):
            native.note_token()
        result = native.plan([4])[0]
        assert result.evicted_key == 3
        assert native.peek(1) >= 0
    finally:
        native.close()
