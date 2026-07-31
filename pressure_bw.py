"""Why the engine reads at 2.56 GB/s where the idle disk gives 4.71.

`pack_bw.py` measures the hot pack at 4.71 GB/s (os-cache, 8 threads, one 7.96 MB
run per expert). A live decode over the same file, same flags, same thread count
measures 36.34 GB in 14.2 s of stall -- 2.56 GB/s. The missing 45% is the largest
unclaimed factor left in the system and it costs no quality to recover, so it is
worth knowing which of two mechanisms it is.

    (a) PRESSURE. Just holding the arena resident degrades the read. The SSD
        DMAs into the same LPDDR the arena occupies, and macOS starts working
        harder for pages as the resident set approaches physical memory. If this
        is it, the ceiling has an optimum: past some size a bigger cache buys
        hit rate and pays for it in bandwidth, and `--ceiling-gb` should be swept
        rather than maximised.

    (b) CONTENTION. Only concurrent GPU traffic degrades the read. Then the
        ceiling is free to grow and the target is the overlap: the engine is
        blocked on disk 75% of the time, so the GPU is mostly idle while
        fetching, and whatever contention exists is self-inflicted scheduling.

The two prescriptions are opposite, which is why guessing is not an option.

Read-only on the pack. Never writes, never allocates the file.
"""

import argparse
import fcntl
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

F_NOCACHE = 48
HOT = 7_962_624


def read_bw(path, blk, nthreads, nreads, nocache):
    fd = os.open(path, os.O_RDONLY)
    if nocache:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    nslots = os.path.getsize(path) // blk
    slots = [random.randrange(nslots) for _ in range(nreads)]

    def rd(s):
        return len(os.pread(fd, blk, s * blk))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        total = sum(ex.map(rd, slots))
    dt = time.perf_counter() - t0
    os.close(fd)
    return total / dt / 1e9


def hold(gb):
    """Materialise `gb` of MLX arrays and keep them alive. mx.zeros writes."""
    if gb <= 0:
        return []
    chunk = 1 << 28                       # 256 MB per array, uint8
    n = max(1, int(gb * 1e9) // chunk)
    arrs = [mx.zeros((chunk,), dtype=mx.uint8) for _ in range(n)]
    mx.eval(arrs)
    return arrs


class GpuLoad:
    """Memory-bound GPU work: elementwise over arrays too big for any cache.

    Matmul would be FLOP-bound and would not contend for LPDDR, which is the
    thing being tested.
    """

    def __init__(self, mb=384):
        n = int(mb * 1e6) // 4
        self.a = mx.random.normal((n,))
        self.b = mx.random.normal((n,))
        mx.eval(self.a, self.b)
        self.stop = threading.Event()
        self.iters = 0
        self.t = None

    def _run(self):
        while not self.stop.is_set():
            c = self.a * 1.0000001 + self.b
            mx.eval(c)
            self.iters += 1

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        time.sleep(0.5)                   # let it reach steady state
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(timeout=10)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="models/m25-hot-expert.pack")
    ap.add_argument("--hold-gb", type=float, nargs="+", default=[0, 3, 6, 9])
    ap.add_argument("--reads", type=int, default=384)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nocache", action="store_true",
                    help="F_NOCACHE; default matches the engine's --os-cache")
    a = ap.parse_args()

    assert os.path.exists(a.pack), f"{a.pack} missing"
    print(f"{a.pack}, {HOT/1e6:.2f} MB blocks, {a.threads} threads, "
          f"{'F_NOCACHE' if a.nocache else 'os-cache'}\n")
    print(f"{'held GB':>8} {'mlx active':>11} {'idle GB/s':>10} "
          f"{'gpu-busy GB/s':>14} {'loss':>7}")

    for gb in a.hold_gb:
        arrs = hold(gb)
        active = mx.get_active_memory() / 1e9
        idle = read_bw(a.pack, HOT, a.threads, a.reads, a.nocache)
        with GpuLoad():
            busy = read_bw(a.pack, HOT, a.threads, a.reads, a.nocache)
        print(f"{gb:>8.1f} {active:>11.2f} {idle:>10.2f} {busy:>14.2f} "
              f"{(1 - busy / idle) * 100:>6.0f}%", flush=True)
        del arrs
        mx.clear_cache()


if __name__ == "__main__":
    main()
