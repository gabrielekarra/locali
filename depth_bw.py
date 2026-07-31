"""Read bandwidth against QUEUE DEPTH, in the engine's barrier-batched pattern.

Four different I/O configurations measured today -- 8 threads, 16 threads, cold
tier packed into contiguous runs, cold tier left as nine scattered slices -- all
produce the same `blocked on disk 14.2s`, the same 2.56 GB/s. A bandwidth limit
yields to more threads or bigger blocks. This one yields to nothing, so it is
probably not bandwidth.

The engine's access pattern explains why it cannot be. A layer misses top_k x
(1 - hit) = 8 x 0.547 = 4.4 experts. Those 4.4 reads are issued, then a barrier:
the gather cannot start until the last one lands. So the pool never has more than
~4.4 reads to schedule no matter how many threads it owns, and the cost per layer
is the LATENCY of the slowest of 4.4 concurrent reads, paid 62 times per token.

`pack_bw.py` queued 512 independent reads and measured 4.71 GB/s. That is the
bandwidth of a deep queue and the engine never has one.

This measures the same file the same way except for depth: issue `depth` reads,
wait for all of them, repeat. If the curve passes through ~2.5 GB/s around depth
4-5 and climbs toward 4.7 by depth 16, the stall is a serialisation cost and the
fix is to raise depth -- which means issuing layer L+1's reads during layer L's
compute, not tuning the disk.

Read-only. Never writes, never allocates the file.
"""

import argparse
import fcntl
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
HOT = 7_962_624


def bench(path, blk, depth, rounds, nocache, threads=None):
    """`rounds` batches of `depth` reads, each batch joined before the next."""
    fd = os.open(path, os.O_RDONLY)
    if nocache:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    nslots = os.path.getsize(path) // blk
    batches = [[random.randrange(nslots) for _ in range(depth)]
               for _ in range(rounds)]

    def rd(s):
        return len(os.pread(fd, blk, s * blk))

    total = 0
    lat = []
    with ThreadPoolExecutor(max_workers=threads or depth) as ex:
        t0 = time.perf_counter()
        for b in batches:
            tb = time.perf_counter()
            total += sum(ex.map(rd, b))      # map joins: this is the barrier
            lat.append(time.perf_counter() - tb)
        dt = time.perf_counter() - t0
    os.close(fd)
    lat.sort()
    return total / dt / 1e9, lat[len(lat) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="models/m25-hot-expert.pack")
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[1, 2, 4, 5, 8, 12, 16, 24, 32])
    ap.add_argument("--rounds", type=int, default=64)
    ap.add_argument("--nocache", action="store_true",
                    help="F_NOCACHE; default matches the engine's --os-cache")
    a = ap.parse_args()

    assert os.path.exists(a.pack), f"{a.pack} missing"
    print(f"{a.pack}, {HOT/1e6:.2f} MB blocks, "
          f"{'F_NOCACHE' if a.nocache else 'os-cache'}, "
          f"{a.rounds} barrier-joined batches per depth\n")
    print(f"{'depth':>6} {'GB/s':>7} {'ms/batch':>9} {'ms/read':>8}   note")
    for d in a.depths:
        gbs, ms = bench(a.pack, HOT, d, a.rounds, a.nocache)
        note = "<- engine: 4.4 misses/layer" if d in (4, 5) else ""
        print(f"{d:>6} {gbs:>7.2f} {ms:>9.2f} {ms / d:>8.2f}   {note}",
              flush=True)


if __name__ == "__main__":
    main()
