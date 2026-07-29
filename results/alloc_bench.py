"""Where do the 121 us per mx.array go: per-call overhead, or per-byte copy?"""
import time
import numpy as np
import mlx.core as mx

W, S = 1179648, 147456          # weight and scale/bias block sizes, bytes
EXPERT = [W] * 3 + [S] * 6      # one expert = 9 reads
N = 200                         # experts per trial

raws = [[bytes(n) for n in EXPERT] for _ in range(N)]
total = sum(EXPERT) * N


def bench(name, fn):
    fn(raws[0])                                   # warm
    t0 = time.perf_counter()
    for r in raws:
        fn(r)
    mx.eval(mx.zeros(1))
    dt = time.perf_counter() - t0
    print(f"{name:38s} {dt:6.3f}s  {total/dt/1e9:5.2f} GB/s  "
          f"{dt/(N*9)*1e6:6.1f} us/array")


def per_array(r):
    out = [mx.array(np.frombuffer(b, dtype=np.uint32)) for b in r]
    mx.eval(out)
    return out


def per_array_noeval(r):
    return [mx.array(np.frombuffer(b, dtype=np.uint32)) for b in r]


def one_slab(r):
    """One mx.array for the whole expert, then slice it into nine."""
    buf = np.frombuffer(b"".join(r), dtype=np.uint32)
    a = mx.array(buf)
    off, out = 0, []
    for n in EXPERT:
        out.append(a[off:off + n // 4])
        off += n // 4
    mx.eval(out)
    return out


def slab_no_slice(r):
    """Upper bound on what a slab could give: the copy alone."""
    a = mx.array(np.frombuffer(b"".join(r), dtype=np.uint32))
    mx.eval(a)
    return a


def numpy_only(r):
    """The join cost on its own -- a slab pays this and per-array does not."""
    return np.frombuffer(b"".join(r), dtype=np.uint32).sum()


bench("per array + eval (current)", per_array)
bench("per array, eval deferred", per_array_noeval)
bench("one slab + slice", one_slab)
bench("one slab, no slice", slab_no_slice)
bench("numpy join only (no mx)", numpy_only)

# What a pure host memcpy of the same bytes costs, for scale.
src = np.zeros(total // 4, dtype=np.uint32)
t0 = time.perf_counter()
dst = src.copy()
dt = time.perf_counter() - t0
print(f"{'numpy memcpy, same bytes':38s} {dt:6.3f}s  {total/dt/1e9:5.2f} GB/s")
