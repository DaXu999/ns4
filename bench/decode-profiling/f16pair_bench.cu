/* What is holding matmul_f16_pair_ordered_chunks_kernel at 14-29% of roofline,
 * and what can a kernel reading this little actually reach?
 *
 * The shipped kernel is the compressor projection: one block per output row,
 * 32 threads, each thread walking a contiguous 1/32 slice of the row.  kprof
 * puts it at 230 GB/s (indexer, 4.19 MB), 381 (ratio-128, 8.39 MB) and 484
 * (ratio-4, 16.78 MB) against this box's 1642 GB/s streaming ceiling.
 *
 * Two things are conflated in that ratio and this separates them:
 *
 *   1. the access pattern.  Thread t reads w[t*chunk + i] for ascending i, so
 *      at any instant the 32 lanes of the warp sit 256 bytes apart -- every
 *      load instruction becomes 32 transactions.  With one warp per block and
 *      ~1.4 blocks per SM there is nothing in flight to hide the latency.
 *      Staging the row through shared memory with all 256 threads fixes the
 *      loads without touching one float operation: the chunk assignment, the
 *      order inside a chunk, and the final ordered 32-way sum all stay put, so
 *      the result is bit-identical (checked here with memcmp, not assumed).
 *
 *   2. the size.  1642 GB/s was measured on an 8 GB stream.  A 4.19 MB read is
 *      22 KB per SM; it cannot amortise DRAM ramp-up the same way, so the
 *      achievable ceiling *at this size* is lower and is the honest
 *      denominator.  The `ceiling` arm reads exactly the same bytes from
 *      exactly the same pool with a pure coalesced grid-stride walk and no
 *      arithmetic, which is the most any kernel of this shape could do.
 *
 * Every arm reads from a rotating pool larger than the 128 MB L2, so the
 * weights come from DRAM as they do in decode.  Times are reported three ways
 * (see fd_combine_bench.cu): graph-amortised is kernel work, eager is how
 * decode actually issues it, empty is the launch floor for that grid.
 *
 *   nvcc -O3 -arch=sm_120 -o f16pair_bench f16pair_bench.cu
 *   ./f16pair_bench [--pool-mb=512] [--reps=50]
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

#define ROOFLINE 1642.0   /* GB/s, bw_probe.cu, 8 GB stream */

/* ---- the shipped kernel, extracted verbatim ------------------------------
 * sed -n '4845,4886p' ns4_cuda.cu > f16pair_baseline.inc                  */
#include "f16pair_baseline.inc"

/* ---- variant: stage the row pair through shared, compute unchanged ------
 * NT threads load both rows coalesced (uint4, consecutive lanes -> adjacent
 * addresses).  Then lanes 0..31 run the *identical* chunk walk out of shared
 * and thread 0 runs the identical ordered 32-way sum.  Not one float add
 * changes position, so this is bit-identical by construction. */
template <uint32_t NT>
__global__ static void matmul_f16_pair_staged_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    extern __shared__ __half srow[];
    const uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __half *s0 = srow;
    __half *s1 = srow + in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;

    /* 16-byte coalesced staging; in_dim is a multiple of 8 at every live
     * shape (4096), so the vector loop covers the row exactly. */
    const uint64_t n8 = in_dim / 8u;
    const uint4 *v0 = (const uint4 *)wr0;
    const uint4 *v1 = (const uint4 *)wr1;
    uint4 *d0 = (uint4 *)s0;
    uint4 *d1 = (uint4 *)s1;
    for (uint64_t i = threadIdx.x; i < n8; i += NT) {
        d0[i] = v0[i];
        d1[i] = v1[i];
    }
    __syncthreads();

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    if (tid < 32u) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        const uint64_t chunk = (in_dim + 31u) / 32u;
        const uint64_t k0 = (uint64_t)tid * chunk;
        uint64_t k1 = k0 + chunk;
        if (k1 > in_dim) k1 = in_dim;
        for (uint64_t i = k0; i < k1; i++) {
            const float xv = x[i];
            if (row < out0_dim) sum0 += __half2float(s0[i]) * xv;
            if (row < out1_dim) sum1 += __half2float(s1[i]) * xv;
        }
        partial0[tid] = sum0;
        partial1[tid] = sum1;
    }
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

/* ---- variant: same chunk walk, 16-byte loads ----------------------------
 * The shipped kernel loads one __half per thread per instruction -- 2 bytes.
 * With only out_dim warps of parallelism in the whole algorithm (one per
 * output row) there is nothing else to hide DRAM latency with, so the bytes
 * in flight per SM are tiny.  Reading each chunk as uint4 gives 8x the bytes
 * per request and lets the compiler keep several in flight, and it does not
 * move a single float add: thread t still owns [t*chunk, (t+1)*chunk) and
 * still walks it in ascending i.  PF is how many 16-byte groups to fetch
 * before consuming any, i.e. the explicit ILP depth. */
template <uint32_t PF>
__global__ static void matmul_f16_pair_vec4_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    const uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;

    const uint64_t n8 = (k1 - k0) / 8u;
    const uint4 *v0 = (const uint4 *)(wr0 + k0);
    const uint4 *v1 = (const uint4 *)(wr1 + k0);
    uint64_t b = 0;
    for (; b + PF <= n8; b += PF) {
        uint4 a[PF], c[PF];
        #pragma unroll
        for (uint32_t p = 0; p < PF; p++) { a[p] = v0[b + p]; c[p] = v1[b + p]; }
        #pragma unroll
        for (uint32_t p = 0; p < PF; p++) {
            const __half *ha = (const __half *)&a[p];
            const __half *hc = (const __half *)&c[p];
            const float *xp = x + k0 + (b + p) * 8u;
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const float xv = xp[j];
                if (row < out0_dim) sum0 += __half2float(ha[j]) * xv;
                if (row < out1_dim) sum1 += __half2float(hc[j]) * xv;
            }
        }
    }
    for (uint64_t i = k0 + b * 8u; i < k1; i++) {      /* tail, same order */
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

/* ---- variant: NT chunks instead of 32, tree reduction -------------------
 * NOT bit-identical, and that is the point of measuring it.  The ordered form
 * has exactly 32 independent sequential chains per row, so the whole algorithm
 * carries only out_dim warps of parallelism -- 256 warps at the indexer shape,
 * 1.4 per SM.  Giving a row NT threads is the only way past that, and it
 * changes both the chunk boundaries and the shape of the final sum.  This arm
 * prices what the exactness contract costs. */
template <uint32_t NT>
__global__ static void matmul_f16_pair_wide_kernel(
        float *out0, float *out1, const __half *w0, const __half *w1,
        const float *x, uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim) {
    const uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    const uint64_t n8 = in_dim / 8u;
    const uint4 *v0 = (const uint4 *)wr0;
    const uint4 *v1 = (const uint4 *)wr1;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint64_t b = threadIdx.x; b < n8; b += NT) {
        const uint4 a = v0[b], c = v1[b];
        const __half *ha = (const __half *)&a;
        const __half *hc = (const __half *)&c;
        const float *xp = x + b * 8u;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const float xv = xp[j];
            sum0 += __half2float(ha[j]) * xv;
            sum1 += __half2float(hc[j]) * xv;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, off);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, off);
    }
    __shared__ float ws0[NT / 32], ws1[NT / 32];
    const uint32_t warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0u) { ws0[warp] = sum0; ws1[warp] = sum1; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float t0 = 0.0f, t1 = 0.0f;
        for (uint32_t i = 0; i < NT / 32u; i++) { t0 += ws0[i]; t1 += ws1[i]; }
        if (row < out0_dim) out0[row] = t0;
        if (row < out1_dim) out1[row] = t1;
    }
}

/* ---- ceiling: the same bytes, pure coalesced read, no arithmetic -------- */
__global__ static void ceiling_read_kernel(const __half *__restrict__ w0,
                                           const __half *__restrict__ w1,
                                           uint64_t n8_total, float *sink) {
    const uint4 *a = (const uint4 *)w0;
    const uint4 *b = (const uint4 *)w1;
    uint4 acc = make_uint4(0, 0, 0, 0);
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n8_total; i += stride) {
        const uint4 va = a[i];
        const uint4 vb = b[i];
        acc.x += va.x + vb.x; acc.y += va.y + vb.y;
        acc.z += va.z + vb.z; acc.w += va.w + vb.w;
    }
    if (acc.x == 0xdeadbeefu) sink[0] = (float)(acc.x + acc.y + acc.z + acc.w);
}

/* Pseudorandom __half fill.  memset gives every element the same value, which
 * makes any reduction order agree and would hide a real numerical error. */
__global__ static void fill_rand_kernel(__half *w, uint64_t n, uint32_t seed) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)(i * 2654435761u) ^ (seed * 40503u);
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        w[i] = __float2half(((float)(h % 20001u) / 10000.0f - 1.0f) * 0.05f);
    }
}

__global__ static void empty_kernel(float *sink) { if (threadIdx.x == 1u << 30) sink[0] = 1.f; }

/* ---- harness ------------------------------------------------------------ */
struct Pool {
    __half **w0, **w1;
    uint32_t n;
};

typedef void (*launch_fn)(const Pool &, int rep, float *o0, float *o1,
                          const float *x, uint64_t in_dim, uint64_t out_dim,
                          cudaStream_t s);

static void launch_base(const Pool &p, int rep, float *o0, float *o1,
                        const float *x, uint64_t in_dim, uint64_t out_dim,
                        cudaStream_t s) {
    const uint32_t k = rep % p.n;
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32, 0, s>>>(
        o0, o1, p.w0[k], p.w1[k], x, in_dim, out_dim, out_dim);
}

template <uint32_t NT>
static void launch_staged(const Pool &p, int rep, float *o0, float *o1,
                          const float *x, uint64_t in_dim, uint64_t out_dim,
                          cudaStream_t s) {
    const uint32_t k = rep % p.n;
    const size_t shmem = (size_t)in_dim * 2u * sizeof(__half);
    matmul_f16_pair_staged_kernel<NT><<<(unsigned)out_dim, NT, shmem, s>>>(
        o0, o1, p.w0[k], p.w1[k], x, in_dim, out_dim, out_dim);
}

template <uint32_t PF>
static void launch_vec4(const Pool &p, int rep, float *o0, float *o1,
                        const float *x, uint64_t in_dim, uint64_t out_dim,
                        cudaStream_t s) {
    const uint32_t k = rep % p.n;
    matmul_f16_pair_vec4_kernel<PF><<<(unsigned)out_dim, 32, 0, s>>>(
        o0, o1, p.w0[k], p.w1[k], x, in_dim, out_dim, out_dim);
}

template <uint32_t NT>
static void launch_wide(const Pool &p, int rep, float *o0, float *o1,
                        const float *x, uint64_t in_dim, uint64_t out_dim,
                        cudaStream_t s) {
    const uint32_t k = rep % p.n;
    matmul_f16_pair_wide_kernel<NT><<<(unsigned)out_dim, NT, 0, s>>>(
        o0, o1, p.w0[k], p.w1[k], x, in_dim, out_dim, out_dim);
}

static void launch_ceiling(const Pool &p, int rep, float *o0, float *o1,
                           const float *x, uint64_t in_dim, uint64_t out_dim,
                           cudaStream_t s) {
    (void)o1; (void)x;
    const uint32_t k = rep % p.n;
    const uint64_t n8 = out_dim * in_dim / 8u;
    ceiling_read_kernel<<<1504, 256, 0, s>>>(p.w0[k], p.w1[k], n8, o0);
}

static double time_graph(launch_fn fn, const Pool &p, float *o0, float *o1,
                         const float *x, uint64_t in_dim, uint64_t out_dim,
                         int reps) {
    cudaStream_t s; CHECK(cudaStreamCreate(&s));
    for (int i = 0; i < 4; i++) fn(p, i, o0, o1, x, in_dim, out_dim, s);
    CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < reps; i++) fn(p, i, o0, o1, x, in_dim, out_dim, s);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
    CHECK(cudaGraphLaunch(ge, s)); CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, s));
    CHECK(cudaGraphLaunch(ge, s));
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaStreamSynchronize(s));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    CHECK(cudaGraphExecDestroy(ge)); CHECK(cudaGraphDestroy(g));
    CHECK(cudaStreamDestroy(s));
    return (double)ms * 1000.0 / reps;      /* us/call */
}

static double time_eager(launch_fn fn, const Pool &p, float *o0, float *o1,
                         const float *x, uint64_t in_dim, uint64_t out_dim,
                         int reps) {
    for (int i = 0; i < 4; i++) fn(p, i, o0, o1, x, in_dim, out_dim, 0);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) fn(p, i, o0, o1, x, in_dim, out_dim, 0);
    CHECK(cudaEventRecord(b));
    CHECK(cudaDeviceSynchronize());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return (double)ms * 1000.0 / reps;
}

static double time_empty(uint32_t blocks, uint32_t threads, float *sink, int reps) {
    empty_kernel<<<blocks, threads>>>(sink);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) empty_kernel<<<blocks, threads>>>(sink);
    CHECK(cudaEventRecord(b));
    CHECK(cudaDeviceSynchronize());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return (double)ms * 1000.0 / reps;
}

int main(int argc, char **argv) {
    double pool_mb = 512.0; int reps = 50;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--pool-mb=", 10)) pool_mb = atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    printf("device: %d SMs, roofline %.0f GB/s (bw_probe, 8 GB stream)\n", sms, ROOFLINE);
    printf("pool %.0f MB per shape (L2 is ~128 MB), reps %d\n", pool_mb, reps);

    const uint64_t in_dim = 4096;
    struct { uint64_t out_dim; const char *what; uint32_t layers; } shapes[] = {
        {1024, "attn compressor, ratio-4 layers",   21},
        { 512, "attn compressor, ratio-128 layers", 20},
        { 256, "indexer compressor",                21},
    };

    for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
        const uint64_t out_dim = shapes[si].out_dim;
        const uint64_t row_bytes = in_dim * sizeof(__half);
        const uint64_t set_bytes = 2ull * out_dim * row_bytes;   /* both matrices */
        uint32_t np = (uint32_t)((pool_mb * 1e6) / (double)set_bytes);
        if (np < 4u) np = 4u;

        Pool p; p.n = np;
        p.w0 = (__half **)malloc(np * sizeof(__half *));
        p.w1 = (__half **)malloc(np * sizeof(__half *));
        for (uint32_t k = 0; k < np; k++) {
            CHECK(cudaMalloc(&p.w0[k], out_dim * row_bytes));
            CHECK(cudaMalloc(&p.w1[k], out_dim * row_bytes));
            fill_rand_kernel<<<256, 256>>>(p.w0[k], out_dim * in_dim, k * 2u + 1u);
            fill_rand_kernel<<<256, 256>>>(p.w1[k], out_dim * in_dim, k * 2u + 2u);
        }
        float *x, *o_ref0, *o_ref1, *o_new0, *o_new1;
        CHECK(cudaMalloc(&x, in_dim * sizeof(float)));
        CHECK(cudaMalloc(&o_ref0, out_dim * sizeof(float)));
        CHECK(cudaMalloc(&o_ref1, out_dim * sizeof(float)));
        CHECK(cudaMalloc(&o_new0, out_dim * sizeof(float)));
        CHECK(cudaMalloc(&o_new1, out_dim * sizeof(float)));
        float *hx = (float *)malloc(in_dim * sizeof(float));
        for (uint64_t i = 0; i < in_dim; i++) hx[i] = (float)((i * 2654435761u) % 1000) / 500.0f - 1.0f;
        CHECK(cudaMemcpy(x, hx, in_dim * sizeof(float), cudaMemcpyHostToDevice));

        printf("\n=== out_dim %4llu  (%s, x%u layers)  reads %.2f MB/call ===\n",
               (unsigned long long)out_dim, shapes[si].what, shapes[si].layers,
               set_bytes / 1e6);

        /* correctness: same pool entry, both kernels, bit-for-bit */
        launch_base(p, 0, o_ref0, o_ref1, x, in_dim, out_dim, 0);
        launch_staged<256>(p, 0, o_new0, o_new1, x, in_dim, out_dim, 0);
        CHECK(cudaDeviceSynchronize());
        float *h0 = (float *)malloc(out_dim * 4), *h1 = (float *)malloc(out_dim * 4);
        float *g0 = (float *)malloc(out_dim * 4), *g1 = (float *)malloc(out_dim * 4);
        CHECK(cudaMemcpy(h0, o_ref0, out_dim * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h1, o_ref1, out_dim * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(g0, o_new0, out_dim * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(g1, o_new1, out_dim * 4, cudaMemcpyDeviceToHost));
        const int same = !memcmp(h0, g0, out_dim * 4) && !memcmp(h1, g1, out_dim * 4);
        printf("staged<256> vs shipped: %s\n", same ? "BIT-IDENTICAL" : "*** MISMATCH ***");
        launch_vec4<4>(p, 0, o_new0, o_new1, x, in_dim, out_dim, 0);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(g0, o_new0, out_dim * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(g1, o_new1, out_dim * 4, cudaMemcpyDeviceToHost));
        const int same4 = !memcmp(h0, g0, out_dim * 4) && !memcmp(h1, g1, out_dim * 4);
        printf("vec4        vs shipped: %s\n", same4 ? "BIT-IDENTICAL" : "*** MISMATCH ***");
        {   /* numerical distance of every reordering arm from the shipped one */
            struct { const char *n; launch_fn f; } chk[] = {
                {"staged<256>", launch_staged<256>}, {"vec4 PF=8", launch_vec4<8>},
                {"wide<128>", launch_wide<128>}, {"wide<256>", launch_wide<256>},
                {"wide<512>", launch_wide<512>},
            };
            for (size_t ci = 0; ci < sizeof(chk)/sizeof(chk[0]); ci++) {
                chk[ci].f(p, 0, o_new0, o_new1, x, in_dim, out_dim, 0);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(g0, o_new0, out_dim * 4, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(g1, o_new1, out_dim * 4, cudaMemcpyDeviceToHost));
                double mx = 0.0, l2n = 0.0, l2d = 0.0; int nbit = 0;
                for (uint64_t i = 0; i < out_dim; i++) {
                    const float rr[2] = {h0[i], h1[i]}, gg[2] = {g0[i], g1[i]};
                    for (int q = 0; q < 2; q++) {
                        if (memcmp(&rr[q], &gg[q], 4)) nbit++;
                        const double d = fabs((double)rr[q] - (double)gg[q]);
                        const double rel = d / (fabs((double)rr[q]) + 1e-30);
                        if (rel > mx) mx = rel;
                        l2n += d * d; l2d += (double)rr[q] * (double)rr[q];
                    }
                }
                printf("  %-12s max_rel %.3e  rel_l2 %.3e  differing %d/%llu\n",
                       chk[ci].n, mx, sqrt(l2n / (l2d + 1e-30)), nbit,
                       (unsigned long long)(out_dim * 2));
            }
        }

        const double ceil_us = time_graph(launch_ceiling, p, o_new0, o_new1, x, in_dim, out_dim, reps);
        const double ceil_gbs = set_bytes / (ceil_us * 1e-6) / 1e9;
        printf("%-22s %9s %9s %9s %9s\n", "arm", "graph us", "eager us", "GB/s", "% ceil");

        struct { const char *name; launch_fn fn; uint32_t threads; } arms[] = {
            {"shipped (32 thr)",  launch_base,          32},
            {"staged<256>",       launch_staged<256>,  256},
            {"vec4 PF=1",         launch_vec4<1>,       32},
            {"vec4 PF=2",         launch_vec4<2>,       32},
            {"vec4 PF=4",         launch_vec4<4>,       32},
            {"vec4 PF=8",         launch_vec4<8>,       32},
            {"vec4 PF=16",        launch_vec4<16>,      32},
            {"wide<128>  (reorders)", launch_wide<128>, 128},
            {"wide<256>  (reorders)", launch_wide<256>, 256},
            {"wide<512>  (reorders)", launch_wide<512>, 512},
        };
        for (size_t ai = 0; ai < sizeof(arms)/sizeof(arms[0]); ai++) {
            const double gus = time_graph(arms[ai].fn, p, o_new0, o_new1, x, in_dim, out_dim, reps);
            const double eus = time_eager(arms[ai].fn, p, o_new0, o_new1, x, in_dim, out_dim, reps);
            const double gbs = set_bytes / (gus * 1e-6) / 1e9;
            printf("%-22s %9.2f %9.2f %9.0f %8.0f%%\n",
                   arms[ai].name, gus, eus, gbs, 100.0 * gbs / ceil_gbs);
        }
        const double empty_us = time_empty((uint32_t)out_dim, 256, o_new0, reps);
        printf("%-22s %9.2f %9s %9.0f %8s   <- pure read, no arithmetic\n",
               "ceiling (1504x256)", ceil_us, "-", ceil_gbs, "100%");
        printf("%-22s %9s %9.2f\n", "empty grid (eager)", "-", empty_us);
        printf("ceiling is %.0f%% of the 8 GB-stream roofline at this size\n",
               100.0 * ceil_gbs / ROOFLINE);

        for (uint32_t k = 0; k < np; k++) { cudaFree(p.w0[k]); cudaFree(p.w1[k]); }
        free(p.w0); free(p.w1); free(hx); free(h0); free(h1); free(g0); free(g1);
        cudaFree(x); cudaFree(o_ref0); cudaFree(o_ref1); cudaFree(o_new0); cudaFree(o_new1);
    }
    return 0;
}
