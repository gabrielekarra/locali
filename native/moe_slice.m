// One MoE layer, natively: SSD -> unified memory -> Metal, no Python anywhere.
//
// The question this exists to answer is the one that decides whether the full
// rewrite is worth starting: on the mixed hit/miss case, does a native path beat
// the MLX one by enough to pay for itself? `dump_slice.py` freezes a real layer
// -- one decode token's hidden state, the eight experts the real router picked,
// their gates -- and the answer mlx_lm computes. This reads the same expert
// bytes off the same shards and has to land on the same output.
//
// Three things are being measured separately, because they have different fixes:
//   read     pread with F_NOCACHE straight into [MTLBuffer contents]
//   compute  quantized matvec, SwiGLU, weighted sum, all on the GPU
//   split    experts already resident computed while the missing ones load
//
// Build:  clang -fobjc-arc -O2 moe_slice.m -framework Foundation -framework Metal -o moe_slice
// Run:    ./moe_slice native/slice [n_resident]
//
// Metal shaders are compiled at runtime from the source below: this machine has
// the Command Line Tools but not Xcode, so `xcrun metal` does not exist and a
// precompiled .metallib is not an option.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mach/mach_time.h>
#include <errno.h>

#define NPROJ 3
#define NARR  3
#define NSLICE (NPROJ * NARR)          // gate/up/down x weight/scales/biases

// Slice order written by dump_slice.py, and the order records arrive in.
enum { W_GATE, S_GATE, B_GATE, W_UP, S_UP, B_UP, W_DOWN, S_DOWN, B_DOWN };

// Packed to match Python's struct.pack("<IQQ"): 20 bytes, no alignment padding
// between the u32 and the u64s. The natural C layout would be 24 and read the
// manifest shifted by four bytes per record.
typedef struct __attribute__((packed)) {
    uint32_t file_id; uint64_t offset; uint64_t nbytes;
} Record;

typedef struct {
    uint32_t in_dim, out_dim, group, bits;
} Dims;

static const char *KSRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Dims { uint in_dim; uint out_dim; uint group; uint bits; };\n"
// MLX stores scales and biases as bfloat16 on this checkpoint. Widening is a
// 16-bit shift into the high half of a float; no bfloat type needed, which
// keeps this compiling on any Metal version the runtime offers.
"inline float bf16(ushort h) { return as_type<float>((uint)h << 16); }\n"
// Affine dequantize, exactly MLX's: w = scale * q + bias, one (scale, bias) per
// `group` consecutive INPUT channels, q packed little-end first, 32/bits values
// per uint32. group % (32/bits) == 0 for every case here, so a packed word never
// straddles two groups.
"kernel void qmatvec(device const uint   *W      [[buffer(0)]],\n"
"                    device const ushort *scales [[buffer(1)]],\n"
"                    device const ushort *biases [[buffer(2)]],\n"
"                    device const float  *x      [[buffer(3)]],\n"
"                    device float        *y      [[buffer(4)]],\n"
"                    constant Dims       &d      [[buffer(5)]],\n"
"                    uint o   [[threadgroup_position_in_grid]],\n"
"                    uint tid [[thread_position_in_threadgroup]],\n"
"                    uint nt  [[threads_per_threadgroup]]) {\n"
"    const uint per   = 32u / d.bits;\n"
"    const uint mask  = (1u << d.bits) - 1u;\n"
"    const uint words = d.in_dim / per;\n"
"    const uint ngrp  = d.in_dim / d.group;\n"
"    float acc = 0.0f;\n"
"    for (uint w = tid; w < words; w += nt) {\n"
"        const uint word = W[o * words + w];\n"
"        const uint i0 = w * per;\n"
"        const uint g  = i0 / d.group;\n"
"        const float s = bf16(scales[o * ngrp + g]);\n"
"        const float b = bf16(biases[o * ngrp + g]);\n"
"        for (uint k = 0; k < per; k++) {\n"
"            const uint q = (word >> (k * d.bits)) & mask;\n"
"            acc = fma(fma(s, (float)q, b), x[i0 + k], acc);\n"
"        }\n"
"    }\n"
"    threadgroup float red[256];\n"
"    red[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint h = nt / 2u; h > 0u; h >>= 1) {\n"
"        if (tid < h) red[tid] += red[tid + h];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) y[o] = red[0];\n"
"}\n"
// silu(gate) * up, fused as one op the way mlx_lm's swiglu is -- a separate
// silu-then-multiply rounds differently and that difference is exactly what the
// bit-identity check in this repo exists to catch.
"kernel void swiglu(device const float *g [[buffer(0)]],\n"
"                   device const float *u [[buffer(1)]],\n"
"                   device float       *o [[buffer(2)]],\n"
"                   uint i [[thread_position_in_grid]]) {\n"
"    const float gv = g[i];\n"
"    o[i] = (gv / (1.0f + exp(-gv))) * u[i];\n"
"}\n"
// y += gate_weight * contribution, accumulated in float32 across the top-k.
"kernel void accum(device float       *y [[buffer(0)]],\n"
"                  device const float *v [[buffer(1)]],\n"
"                  constant float     &w [[buffer(2)]],\n"
"                  uint i [[thread_position_in_grid]]) {\n"
"    y[i] = fma(w, v[i], y[i]);\n"
"}\n";

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    buf[n] = 0; fclose(f);
    if (len) *len = (size_t)n;
    return buf;
}

// Minimal extractor for the flat integers dump_slice.py writes. Avoids pulling a
// JSON library in for six scalars.
static long meta_int(const char *js, const char *key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) { fprintf(stderr, "meta.json missing %s\n", key); exit(1); }
    p = strchr(p, ':');
    return strtol(p + 1, NULL, 10);
}

int main(int argc, char **argv) {
@autoreleasepool {
    const char *dir = argc > 1 ? argv[1] : "native/slice";
    // How many of the top-k are pretended already resident. The split path only
    // has something to hide behind when both sets are non-empty.
    int n_resident = argc > 2 ? atoi(argv[2]) : 0;

    char p[1024];
    snprintf(p, sizeof p, "%s/meta.json", dir);
    char *meta = slurp(p, NULL);
    const int top_k   = (int)meta_int(meta, "top_k");
    const int d_model = (int)meta_int(meta, "d_model");
    const int d_ff    = (int)meta_int(meta, "d_ff");
    const int group   = (int)meta_int(meta, "group_size");
    const int hot_bits  = (int)meta_int(meta, "hot");
    const int cold_bits = (int)meta_int(meta, "cold");

    // Per-expert tier, read off the tiers array in meta.json.
    int bits_of[64];
    {
        const char *t = strstr(meta, "\"tiers\"");
        if (!t) { fprintf(stderr, "meta.json missing tiers\n"); exit(1); }
        const char *q = t;
        for (int e = 0; e < top_k; e++) {
            const char *h = strstr(q + 1, "\"hot\"");
            const char *c = strstr(q + 1, "\"cold\"");
            if (h && (!c || h < c)) { bits_of[e] = hot_bits; q = h; }
            else if (c)             { bits_of[e] = cold_bits; q = c; }
            else { fprintf(stderr, "tiers shorter than top_k\n"); exit(1); }
        }
    }

    snprintf(p, sizeof p, "%s/manifest.bin", dir);
    size_t mlen = 0; Record *recs = (Record *)slurp(p, &mlen);
    if (mlen != (size_t)top_k * NSLICE * sizeof(Record)) {
        fprintf(stderr, "manifest is %zu bytes, expected %zu\n",
                mlen, (size_t)top_k * NSLICE * sizeof(Record));
        exit(1);
    }

    snprintf(p, sizeof p, "%s/files.txt", dir);
    char *ftxt = slurp(p, NULL);
    int fds[16]; int nfiles = 0;
    for (char *line = strtok(ftxt, "\n"); line; line = strtok(NULL, "\n")) {
        if (!*line) continue;
        int fd = open(line, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "cannot open shard %s\n", line); exit(1); }
        // The whole point of the arena: bytes land in the memory the GPU will
        // read, without passing through the page cache first. Checked, not
        // assumed -- the read rate came out above what this SSD can do, and a
        // silently failing fcntl would explain it.
        if (fcntl(fd, F_NOCACHE, 1) < 0) {
            fprintf(stderr, "warning: F_NOCACHE failed on %s (errno %d); "
                            "reads may be served from the page cache and the "
                            "read timing below is not a disk measurement\n",
                    line, errno);
        }
        fds[nfiles++] = fd;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(KSRC) options:nil error:&err];
    if (!lib) { fprintf(stderr, "shader: %s\n", [[err localizedDescription] UTF8String]); exit(1); }
    id<MTLComputePipelineState> pso_mv =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"qmatvec"] error:&err];
    id<MTLComputePipelineState> pso_sg =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"swiglu"] error:&err];
    id<MTLComputePipelineState> pso_ac =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"accum"] error:&err];
    if (!pso_mv || !pso_sg || !pso_ac) { fprintf(stderr, "pipeline failed\n"); exit(1); }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    // ---- arena: one allocation, every expert slice at a known offset --------
    uint64_t total = 0;
    uint64_t *off = malloc(sizeof(uint64_t) * top_k * NSLICE);
    for (int i = 0; i < top_k * NSLICE; i++) {
        off[i] = total;
        total += (recs[i].nbytes + 255u) & ~255ull;      // keep slices aligned
    }
    id<MTLBuffer> arena = [dev newBufferWithLength:total
                                           options:MTLResourceStorageModeShared];
    uint8_t *base = (uint8_t *)[arena contents];
    printf("arena %.2f MB, %d slices, %d files, top-%d\n",
           total / 1e6, top_k * NSLICE, nfiles, top_k);

    id<MTLBuffer> bx  = [dev newBufferWithLength:d_model * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bg  = [dev newBufferWithLength:d_ff * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bu  = [dev newBufferWithLength:d_ff * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bh  = [dev newBufferWithLength:d_ff * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bv  = [dev newBufferWithLength:d_model * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> by  = [dev newBufferWithLength:d_model * sizeof(float)
                                         options:MTLResourceStorageModeShared];

    snprintf(p, sizeof p, "%s/x.f32", dir);
    { size_t n; char *b = slurp(p, &n); memcpy([bx contents], b, n); free(b); }
    snprintf(p, sizeof p, "%s/gates.f32", dir);
    float *gates; { size_t n; gates = (float *)slurp(p, &n); }
    snprintf(p, sizeof p, "%s/ref.f32", dir);
    float *ref; { size_t n; ref = (float *)slurp(p, &n); }
    // The float32 reference: same bytes, same operations, same order as the
    // kernel. This is the check that can actually catch an indexing bug -- the
    // bfloat16 one below cannot, because its own tolerance is 7e-3 wide.
    snprintf(p, sizeof p, "%s/ref_f32.f32", dir);
    float *ref32; { size_t n; ref32 = (float *)slurp(p, &n); }

    // ---- read ---------------------------------------------------------------
    // Only the experts NOT pretended resident are read; the rest stand in for a
    // cache hit and cost nothing.
    // One slice's read, wherever it is issued from. `fds` is captured through a
    // pointer: a block cannot capture an array type.
    int *fdp = fds;
    void (^read_slice)(int) = ^(int i) {
        Record r = recs[i];
        uint64_t done = 0;
        while (done < r.nbytes) {
            ssize_t n = pread(fdp[r.file_id], base + off[i] + done,
                              r.nbytes - done, r.offset + done);
            if (n <= 0) { fprintf(stderr, "pread failed\n"); exit(1); }
            done += (uint64_t)n;
        }
    };

    // Fill the WHOLE arena first, untimed. A "resident" expert has to contain
    // its actual weights: skipping its read leaves uninitialised memory, the
    // gather reads garbage, and the verification below fails for a reason that
    // has nothing to do with the kernel.
    for (int i = 0; i < top_k * NSLICE; i++) read_slice(i);

    // Then time only the misses, re-read exactly as a cold expert would be.
    // Issued across a pool, because the engine reads with eight threads and a
    // single-threaded read measures latency rather than bandwidth: the same two
    // experts take 2.16 GB/s serially against 4.73 measured by `pack_bw.py`.
    const int n_miss = (top_k - n_resident) * NSLICE;
    uint64_t bytes_read = 0;
    for (int e = n_resident; e < top_k; e++)
        for (int s = 0; s < NSLICE; s++) bytes_read += recs[e * NSLICE + s].nbytes;

    double t0r = now_ms();
    dispatch_apply(n_miss, dispatch_get_global_queue(
                       DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t k) {
        read_slice(n_resident * NSLICE + (int)k);
    });
    double t_read = now_ms() - t0r;

    // ---- compute ------------------------------------------------------------
    // Timed twice. The first pass is the GPU's first touch of 63.7 MB the CPU
    // has just written through the shared buffer; the second reads the same
    // bytes with the caches already coherent. A large gap means the cost is the
    // handoff, not the arithmetic, and a real engine pays it on every miss.
    const int n_iters = 3;
    double t_iter[8];
    for (int it = 0; it < n_iters; it++) {
    memset([by contents], 0, d_model * sizeof(float));
    double t0 = now_ms();
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    // ONE encoder for the whole layer. Each `computeCommandEncoder` is a full
    // pipeline barrier and there were 40 of them per layer; inside one encoder
    // a `memoryBarrierWithScope:` orders only the dependent pair and costs far
    // less. Ordering still has to be explicit -- dispatches in one encoder are
    // free to overlap otherwise, and gate/up -> swiglu -> down -> accum is a
    // chain.
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int e = 0; e < top_k; e++) {
        const uint32_t bits = (uint32_t)bits_of[e];
        Dims dgu = { (uint32_t)d_model, (uint32_t)d_ff,  (uint32_t)group, bits };
        Dims ddn = { (uint32_t)d_ff,    (uint32_t)d_model,(uint32_t)group, bits };
        int b0 = e * NSLICE;

        // gate and up: [d_ff] each, from x
        struct { int w, s, b; id<MTLBuffer> out; Dims d; int rows; } steps[2] = {
            { W_GATE, S_GATE, B_GATE, bg, dgu, d_ff },
            { W_UP,   S_UP,   B_UP,   bu, dgu, d_ff },
        };
        // gate and up read the same x and write disjoint buffers: no barrier
        // between them, they are free to run together.
        for (int k = 0; k < 2; k++) {
            [enc setComputePipelineState:pso_mv];
            [enc setBuffer:arena offset:off[b0 + steps[k].w] atIndex:0];
            [enc setBuffer:arena offset:off[b0 + steps[k].s] atIndex:1];
            [enc setBuffer:arena offset:off[b0 + steps[k].b] atIndex:2];
            [enc setBuffer:bx offset:0 atIndex:3];
            [enc setBuffer:steps[k].out offset:0 atIndex:4];
            [enc setBytes:&steps[k].d length:sizeof(Dims) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(steps[k].rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pso_sg];
        [enc setBuffer:bg offset:0 atIndex:0];
        [enc setBuffer:bu offset:0 atIndex:1];
        [enc setBuffer:bh offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(d_ff, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pso_mv];
        [enc setBuffer:arena offset:off[b0 + W_DOWN] atIndex:0];
        [enc setBuffer:arena offset:off[b0 + S_DOWN] atIndex:1];
        [enc setBuffer:arena offset:off[b0 + B_DOWN] atIndex:2];
        [enc setBuffer:bh offset:0 atIndex:3];
        [enc setBuffer:bv offset:0 atIndex:4];
        [enc setBytes:&ddn length:sizeof(Dims) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(d_model, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pso_ac];
        [enc setBuffer:by offset:0 atIndex:0];
        [enc setBuffer:bv offset:0 atIndex:1];
        [enc setBytes:&gates[e] length:sizeof(float) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(d_model, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        // The next expert reuses bg/bu/bh/bv, so it must not start writing them
        // before this expert's accum has read bv.
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    t_iter[it] = now_ms() - t0;
    }
    double t_compute = t_iter[0];

    // ---- verify against what mlx_lm produced --------------------------------
    float *y = (float *)[by contents];
    double rel = 0.0, maxabs = 0.0, rel32 = 0.0, maxabs32 = 0.0;
    {
        double num = 0, den = 0, n32 = 0, d32 = 0;
        for (int i = 0; i < d_model; i++) {
            double a1 = (double)y[i] - (double)ref[i];
            double a2 = (double)y[i] - (double)ref32[i];
            num += a1 * a1; den += (double)ref[i] * (double)ref[i];
            n32 += a2 * a2; d32 += (double)ref32[i] * (double)ref32[i];
            if (fabs(a1) > maxabs)   maxabs = fabs(a1);
            if (fabs(a2) > maxabs32) maxabs32 = fabs(a2);
        }
        rel   = sqrt(num) / (sqrt(den) + 1e-20);
        rel32 = sqrt(n32) / (sqrt(d32) + 1e-20);
    }

    printf("\nread     %7.2f ms   %.2f MB   %.2f GB/s   (%d of %d experts missing)\n",
           t_read, bytes_read / 1e6,
           t_read > 0 ? bytes_read / 1e9 / (t_read / 1e3) : 0.0,
           top_k - n_resident, top_k);
    printf("compute  %7.2f ms   %d matvecs in 1 encoder   (repeats:", t_compute,
           top_k * 3);
    for (int it = 1; it < n_iters; it++) printf(" %.2f", t_iter[it]);
    printf(" ms)\n");
    printf("         %7.2f GB/s of weight bytes through the GPU on the first pass\n",
           bytes_read / 1e9 / (t_compute / 1e3));
    printf("total    %7.2f ms\n", t_read + t_compute);
    printf("\nvs float32 reference (same ops, same order): relative %.3e  "
           "max abs %.3e\n", rel32, maxabs32);
    printf("vs mlx_lm block (bfloat16 arithmetic):       relative %.3e  "
           "max abs %.3e\n", rel, maxabs);
    const int ok = rel32 < 1e-5;
    printf("%s\n", ok
           ? "MATCH: indexing, packing and dequantization are correct"
           : "MISMATCH: the kernel is not computing the same thing");

    for (int i = 0; i < nfiles; i++) close(fds[i]);
    return ok ? 0 : 1;
}
}
