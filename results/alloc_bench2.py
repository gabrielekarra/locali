"""Is the in-situ slowdown allocation churn? Fresh buffers vs reused ones."""
import os
import time
import numpy as np
import mlx.core as mx

W, S = 1179648, 147456
EXPERT = [W] * 3 + [S] * 6
N = 300
total = sum(EXPERT) * N

path = "/tmp/allocbench.bin"
if not os.path.exists(path) or os.path.getsize(path) < sum(EXPERT) * 4:
    with open(path, "wb") as f:
        f.write(os.urandom(sum(EXPERT) * 4))
fd = os.open(path, os.O_RDONLY)


def run(name, fn):
    fn(0)
    t0 = time.perf_counter()
    for i in range(N):
        fn(i)
    dt = time.perf_counter() - t0
    print(f"{name:44s} {dt:6.3f}s  {total/dt/1e9:5.2f} GB/s  "
          f"{dt/(N*9)*1e6:6.1f} us/array")


def fresh_bytes(i):
    """What the store does now: os.pread allocates a new bytes every read."""
    out = []
    off = 0
    for n in EXPERT:
        raw = os.pread(fd, n, off)
        out.append(mx.array(np.frombuffer(raw, dtype=np.uint32)))
        off += n
    mx.eval(out)
    return out


scratch = [bytearray(n) for n in EXPERT]
views = [memoryview(b) for b in scratch]


def reused_buffer(i):
    """preadv into buffers we already own: no allocation, no fresh page faults."""
    out = []
    off = 0
    for n, v in zip(EXPERT, views):
        got = os.preadv(fd, [v], off)
        assert got == n, got
        out.append(mx.array(np.frombuffer(v, dtype=np.uint32)))
        off += n
    mx.eval(out)
    return out


run("pread -> fresh bytes -> mx (current)", fresh_bytes)
run("preadv -> reused bytearray -> mx", reused_buffer)
os.close(fd)
