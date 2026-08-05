"""ctypes boundary for Locali's native cache and expert scheduler core."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / "native" / (
    "liblocali_core.dylib" if sys.platform == "darwin" else "liblocali_core.so"
)


class _Result(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_int32),
        ("slot", ctypes.c_int32),
        ("evicted_key", ctypes.c_int32),
        ("hit", ctypes.c_uint8),
        ("placed", ctypes.c_uint8),
    ]


@dataclass(frozen=True)
class CacheResult:
    key: int
    slot: int
    evicted_key: int
    hit: bool
    placed: bool


def available(path: Path = DEFAULT_LIBRARY) -> bool:
    return path.is_file()


def ensure_available(path: Path = DEFAULT_LIBRARY) -> bool:
    """Build the tiny C core on first use; fall back cleanly without a compiler."""
    if available(path):
        return True
    try:
        completed = subprocess.run(
            ["make", "-C", str(ROOT / "native")],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0 and available(path)


class NativeCache:
    def __init__(
        self,
        key_capacity: int,
        slots: int,
        protected_capacity: int,
        *,
        policy: str = "slru",
        library: Path = DEFAULT_LIBRARY,
    ):
        self._cache = None
        self._lib = ctypes.CDLL(str(library))
        self._lib.locali_cache_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._lib.locali_cache_create.restype = ctypes.c_void_p
        self._lib.locali_cache_create_policy.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._lib.locali_cache_create_policy.restype = ctypes.c_void_p
        self._lib.locali_cache_destroy.argtypes = [ctypes.c_void_p]
        self._lib.locali_cache_note_token.argtypes = [ctypes.c_void_p]
        self._lib.locali_cache_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_Result),
        ]
        self._lib.locali_cache_plan.restype = ctypes.c_int
        self._lib.locali_cache_peek.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self._lib.locali_cache_peek.restype = ctypes.c_int32
        self._lib.locali_cache_probation_size.argtypes = [ctypes.c_void_p]
        self._lib.locali_cache_probation_size.restype = ctypes.c_int32
        self._lib.locali_cache_protected_size.argtypes = [ctypes.c_void_p]
        self._lib.locali_cache_protected_size.restype = ctypes.c_int32
        policies = {"slru": 0, "lfu-decay": 1}
        if policy not in policies:
            raise ValueError(f"unknown native cache policy: {policy}")
        self._cache = self._lib.locali_cache_create_policy(
            key_capacity, slots, protected_capacity, policies[policy]
        )
        if not self._cache:
            raise MemoryError("failed to create Locali native cache")

    def note_token(self) -> None:
        self._lib.locali_cache_note_token(self._cache)

    def plan(
        self,
        keys: list[int],
        *,
        pinned_slots: list[int] | tuple[int, ...] = (),
        touch_hits: bool = True,
    ) -> list[CacheResult]:
        if not keys:
            return []
        key_array = (ctypes.c_int32 * len(keys))(*keys)
        pinned_array = (
            (ctypes.c_int32 * len(pinned_slots))(*pinned_slots)
            if pinned_slots else None
        )
        results = (_Result * len(keys))()
        rc = self._lib.locali_cache_plan(
            self._cache,
            key_array,
            len(keys),
            pinned_array,
            len(pinned_slots),
            int(touch_hits),
            results,
        )
        if rc:
            raise ValueError(f"native cache plan failed with error {rc}")
        return [
            CacheResult(
                key=result.key,
                slot=result.slot,
                evicted_key=result.evicted_key,
                hit=bool(result.hit),
                placed=bool(result.placed),
            )
            for result in results
        ]

    def peek(self, key: int) -> int:
        return int(self._lib.locali_cache_peek(self._cache, key))

    @property
    def probation_size(self) -> int:
        return int(self._lib.locali_cache_probation_size(self._cache))

    @property
    def protected_size(self) -> int:
        return int(self._lib.locali_cache_protected_size(self._cache))

    def close(self) -> None:
        if getattr(self, "_cache", None):
            self._lib.locali_cache_destroy(self._cache)
            self._cache = None

    def __del__(self):
        self.close()
