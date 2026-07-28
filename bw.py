"""Random-read bandwidth at expert granularity, F_NOCACHE so the page cache can't lie.

A MiniMax-M2.5 expert is 14,155,776 params: 7.96 MB at 4-bit, 4.42 MB at
2-bit. Those are the read sizes that matter; sequential dd numbers are not.
Bandwidth turns out to depend on the block, so both are measured.
"""
import os, sys, time, fcntl, random
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bwtest.bin"
FILE_GB = 40
EXPERT = 4_423_680

if not os.path.exists(PATH) or os.path.getsize(PATH) < FILE_GB * 2**30:
    print(f"creating {FILE_GB} GB at {PATH} ...", flush=True)
    with open(PATH, "wb") as f:
        chunk = os.urandom(1 << 24)
        for _ in range((FILE_GB * 2**30) // len(chunk)):
            f.write(chunk)
    os.sync()

size = os.path.getsize(PATH)
nslots = size // EXPERT

def bench(nthreads, nreads=192, blk=EXPERT):
    fd = os.open(PATH, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    slots = [random.randrange(size // blk) for _ in range(nreads)]
    def rd(s):
        b = os.pread(fd, blk, s * blk)
        assert len(b) == blk
        return len(b)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        total = sum(ex.map(rd, slots))
    dt = time.perf_counter() - t0
    os.close(fd)
    return total / dt / 1e9, dt / nreads * 1e3

print(f"file {size/1e9:.1f} GB, block {blk_mb:.2f} MB" if False else
      f"file {size/1e9:.1f} GB, {nslots} expert-sized slots\n")
print(f"{'block':>10} {'threads':>8} {'GB/s':>8} {'ms/read':>9}")
for blk_name, blk in [("4.42MB", EXPERT), ("7.96MB", 7_962_624), ("17.5MB", 17_547_264)]:
    for nt in (1, 2, 4, 8, 16):
        gbs, ms = bench(nt, nreads=192 if blk >= EXPERT else 1024, blk=blk)
        print(f"{blk_name:>10} {nt:>8} {gbs:>8.2f} {ms:>9.2f}", flush=True)
