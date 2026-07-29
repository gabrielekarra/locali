"""Low-level probe: what the Mac's architecture allows that a discrete GPU does not.

Every MoE offloading system in the literature moves weights host -> device over
PCIe, so their whole design is about hiding that transfer. Apple Silicon has no
transfer: CPU and GPU address the same LPDDR. `mx.array(np.frombuffer(raw))` is
therefore not a device upload, it is a memcpy from one region of unified memory
to another -- and it was measured at 29% of runtime.

If mlx will hand out a writable pointer to its own buffer, that copy is not
necessary at all: `preadv` can put the bytes the SSD controller produces
directly into the memory the GPU will read. The kernel's DMA engine writes once,
nobody copies. On a discrete GPU this needs GPUDirect Storage and vendor
support; here it needs a memoryview.

Four things get measured, all against a REAL model shard so F_NOCACHE and the
block sizes are the ones the engine actually sees:

  1. does writing through mlx's buffer work, and is it page-aligned
  2. preadv-into-mlx against pread -> bytes -> mx.array
  3. whether 16 KB page alignment matters under F_NOCACHE
  4. how far a parallel memcpy scales -- one core sees ~10 GB/s of a 120 GB/s bus
"""

import ctypes
import fcntl
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

F_NOCACHE = 48
PAGE = 16384                      # Apple Silicon, not the 4096 of x86
W, S = 1179648, 147456            # the engine's two real block sizes
SHARD = ("models/hf/hub/models--mlx-community--MiniMax-M2.5-4bit/snapshots/"
         "36fb6facb4697ac2e6c4e88b600cd8601fb62f08/model-00003-of-00027.safetensors")


def open_nocache(path):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def align(p):
    return "aligned" if p % PAGE == 0 else f"off by {p % PAGE}"


def probe_alignment():
    print("=== 1. mlx buffers: writable, and where do they land")
    a = mx.zeros((W // 4,), dtype=mx.uint32)
    mx.eval(a)
    mv = memoryview(a)
    ptr = np.asarray(a).__array_interface__["data"][0]
    print(f"  memoryview readonly={mv.readonly}  nbytes={mv.nbytes}")
    print(f"  buffer address {ptr:#x} -> {align(ptr)} (page {PAGE})")

    # A fresh Python bytes has a header, so its payload can never be aligned.
    b = bytes(W)
    bptr = ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value
    print(f"  bytes payload  {bptr:#x} -> {align(bptr)}")

    # Slab slot: mlx slicing is a lazy op, but the BUFFER is contiguous, so a
    # slot is a byte range in the parent's memoryview. That is what makes a
    # per-layer slab addressable without any copy.
    slab = mx.zeros((4, 1024), dtype=mx.uint32)
    mx.eval(slab)
    smv = memoryview(slab).cast("B")
    slot = smv[2 * 4096:3 * 4096]
    slot[:4] = b"\xef\xbe\xad\xde"
    mx.eval(slab)
    got = int(np.asarray(slab)[2, 0])
    print(f"  wrote through slab slot 2 -> mlx reads {got:#x} "
          f"{'OK' if got == 0xdeadbeef else 'MISMATCH'}")
    return ptr % PAGE == 0


def probe_read_paths(fd, size, n=400):
    """The comparison that matters: bytes-then-convert against preadv-in-place."""
    fsz = os.path.getsize(SHARD)
    rng = np.random.default_rng(0)
    # Page-aligned offsets: an unaligned OFFSET forces the kernel to read the
    # containing pages and shift, which is a second way to lose the DMA.
    offs = [(int(rng.integers(0, (fsz - size) // PAGE)) * PAGE) for _ in range(n)]

    def current():
        out = []
        for o in offs:
            raw = os.pread(fd, size, o)
            out.append(mx.array(np.frombuffer(raw, dtype=np.uint32)))
        mx.eval(out)
        return out

    def direct_fresh():
        """Allocating per read is the WRONG way to do this and is measured only
        to show why: mx.zeros writes `size` bytes of zeros that the read is about
        to overwrite, and touches fresh pages that then fault."""
        out = []
        for o in offs:
            a = mx.zeros((size // 4,), dtype=mx.uint32)
            mx.eval(a)
            got = os.preadv(fd, [memoryview(a).cast("B")], o)
            assert got == size, got
            out.append(a)
        return out

    # The slab: allocated once, faulted in once, then written by DMA forever.
    # This is what the cache would hold -- a per-layer arena of expert slots --
    # so the steady state has no allocation and no zeroing at all.
    slots = 16
    slab = mx.zeros((slots, size // 4), dtype=mx.uint32)
    mx.eval(slab)
    smv = memoryview(slab).cast("B")
    np.asarray(slab)[:] = 1                  # fault every page in, once

    def direct_slab():
        for i, o in enumerate(offs):
            s = (i % slots) * size
            got = os.preadv(fd, [smv[s:s + size]], o)
            assert got == size, got
        return slab

    for name, fn in (("pread -> bytes -> mx.array", current),
                     ("preadv -> fresh mlx array", direct_fresh),
                     ("preadv -> preallocated slab", direct_slab)):
        fn()                                 # warm
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        print(f"  {name:32s} {dt:6.3f}s  {size*n/dt/1e9:5.2f} GB/s  "
              f"{dt/n*1e6:7.1f} us/read")


def probe_unaligned(fd, size, n=300):
    """Does the 16 KB alignment actually buy anything under F_NOCACHE?"""
    fsz = os.path.getsize(SHARD)
    rng = np.random.default_rng(1)
    base = [(int(rng.integers(0, (fsz - size - PAGE) // PAGE)) * PAGE)
            for _ in range(n)]
    for label, shift in (("offset page-aligned", 0), ("offset +37 bytes", 37)):
        buf = mx.zeros((size // 4 + PAGE // 4,), dtype=mx.uint32)
        mx.eval(buf)
        mv = memoryview(buf).cast("B")
        t0 = time.perf_counter()
        for o in base:
            os.preadv(fd, [mv[:size]], o + shift)
        dt = time.perf_counter() - t0
        print(f"  {label:24s} {dt:6.3f}s  {size*n/dt/1e9:5.2f} GB/s")


def probe_memcpy_scaling():
    """One core sees a fraction of the bus. How much does threading recover?"""
    n = 64 * 1024 * 1024
    src = np.ones(n // 4, dtype=np.uint32)
    for t in (1, 2, 4, 8):
        dst = [np.empty(n // 4 // t, dtype=np.uint32) for _ in range(t)]
        chunks = [src[i * (n // 4 // t):(i + 1) * (n // 4 // t)] for i in range(t)]
        with ThreadPoolExecutor(max_workers=t) as pool:
            copy = lambda i: np.copyto(dst[i], chunks[i])
            list(pool.map(copy, range(t)))          # warm
            t0 = time.perf_counter()
            for _ in range(8):
                list(pool.map(copy, range(t)))
            dt = time.perf_counter() - t0
        print(f"  {t} thread(s): {n*8/dt/1e9:6.2f} GB/s")


def main():
    if not Path(SHARD).exists():
        raise SystemExit(f"missing {SHARD}")
    aligned = probe_alignment()

    fd = open_nocache(SHARD)
    print(f"\n=== 2. read paths, F_NOCACHE, {W/1e6:.2f} MB blocks (the weights)")
    probe_read_paths(fd, W)
    print(f"\n=== 2b. same at {S/1e3:.0f} KB (the scales and biases)")
    probe_read_paths(fd, S, n=800)
    print(f"\n=== 3. does page alignment matter under F_NOCACHE")
    probe_unaligned(fd, W)
    os.close(fd)

    print(f"\n=== 4. memcpy against the 120 GB/s bus")
    probe_memcpy_scaling()

    print(f"\n  mlx buffer page-aligned: {aligned}")


if __name__ == "__main__":
    main()
