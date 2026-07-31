"""Read bandwidth on the real expert packs, idle, at the real block sizes.

`bw.py` answers this for a synthetic file, and it CREATES that file if the path
is smaller than 40 GB -- which would destroy a pack. This one only ever opens
O_RDONLY, and reads the packs the engine actually reads.

The question it settles: the engine's cold reads are now one contiguous 4.42 MB
preadv per expert instead of nine scattered slices, and that changed decode time
by nothing. Either 4.42 MB random reads do not go faster than the nine-slice
pattern on this disk, or they do when idle and the engine never sees it because
the GPU is saturating the same LPDDR the DMA lands in.
"""

import argparse
import fcntl
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
HOT = 7_962_624     # 4-bit expert, one contiguous run in the hot pack
COLD = 4_423_680    # 2-bit expert, one contiguous run in the cold pack
FRAG = (1_179_648, 147_456)   # the nine-slice pattern: 3 weights + 6 scale/bias


def bench(path, blk, nthreads, nreads, nocache):
    fd = os.open(path, os.O_RDONLY)
    if nocache:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    size = os.path.getsize(path)
    nslots = size // blk
    slots = [random.randrange(nslots) for _ in range(nreads)]

    def rd(s):
        b = os.pread(fd, blk, s * blk)
        return len(b)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        total = sum(ex.map(rd, slots))
    dt = time.perf_counter() - t0
    os.close(fd)
    return total / dt / 1e9


def bench_frag(path, nthreads, nreads, nocache):
    """Same bytes per expert, but as the nine reads the safetensors layout forces.

    The slices are placed at random independent offsets, which is what
    tensor-major storage gives: an expert's nine runs are never adjacent.
    """
    fd = os.open(path, os.O_RDONLY)
    if nocache:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    size = os.path.getsize(path)
    jobs = []
    for _ in range(nreads):
        for n, count in ((FRAG[0], 3), (FRAG[1], 6)):
            for _ in range(count):
                jobs.append((n, random.randrange(size // n) * n))

    def rd(job):
        n, off = job
        return len(os.pread(fd, n, off))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        total = sum(ex.map(rd, jobs))
    dt = time.perf_counter() - t0
    os.close(fd)
    return total / dt / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-pack", default="models/m25-hot-expert.pack")
    ap.add_argument("--cold-pack", default="models/m25-cold-expert.pack")
    ap.add_argument("--reads", type=int, default=512)
    a = ap.parse_args()

    print(f"{'file':>12} {'pattern':>22} {'cache':>9} {'thr':>4} {'GB/s':>7}")
    for label, path, blk in (("hot", a.hot_pack, HOT), ("cold", a.cold_pack, COLD)):
        if not os.path.exists(path):
            print(f"  {path} missing, skipped")
            continue
        for nocache in (True, False):
            tag = "F_NOCACHE" if nocache else "os-cache"
            for nt in (8, 16):
                g = bench(path, blk, nt, a.reads, nocache)
                print(f"{label:>12} {f'1 x {blk/1e6:.2f}MB':>22} {tag:>9} "
                      f"{nt:>4} {g:>7.2f}", flush=True)
            g = bench_frag(path, 8, a.reads, nocache)
            print(f"{label:>12} {'9 slices scattered':>22} {tag:>9} "
                  f"{8:>4} {g:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
