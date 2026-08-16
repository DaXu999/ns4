/* outa_bench.cu -- price the decode out_a grouped q8_0 GEMV at the live
 * DeepSeek-V4-Flash shape (group_dim 4096, rank 1024, 8 groups -> low_dim
 * 8192 rows, 128 q8_0 blocks per row, n_tokens 1) against a pure-read
 * ceiling, with an L2-defeating weight pool rotation.
 *
 * Variants:
 *   warp8<RPB>      production kernel re-templated on rows-per-block.  The
 *                   per-row schedule (lane stride 32, warp_sum tree) is
 *                   unchanged, so every RPB is bit-identical to production
 *                   RPB=8 -- only the block packing moves.
 *   warp8u<RPB>     same, with the 128-block loop count baked in so nvcc can
 *                   unroll/pipeline the 4 per-lane iterations.  Same
 *                   accumulation sequence per lane -> still bit-identical.
 *   wide<NT>        NT lanes per row (grid = low_dim), per-lane stride NT,
 *                   warp trees + cross-warp chain like
 *                   matmul_q8_0_pair_preq_wide_kernel.  Re-associated -> NOT
 *                   bit-identical; prints max |diff| instead.
 *   ceiling         grid-stride uint4 read of the same bytes.
 *
 * Build: nvcc -O3 --use_fast_math -arch=sm_120a -o outa_bench outa_bench.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

/* ---- verbatim device helpers from ds4_cuda.cu ---- */
__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}
__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}
__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}
__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}
__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

/* ---- shape ---- */
static const uint64_t GROUP_DIM = 4096;
static const uint64_t RANK = 1024;
static const uint32_t N_GROUPS = 8;
static const uint64_t LOW_DIM = (uint64_t)N_GROUPS * RANK; /* 8192 */
static const uint64_t BLOCKS = (GROUP_DIM + 31) / 32;      /* 128 */
static const uint64_t W_BYTES = LOW_DIM * BLOCKS * 34u;    /* 35.65 MB */

/* production kernel, RPB-templated (RPB=8 == shipped grouped_q8_0_a_preq_warp8) */
template <uint32_t RPB>
__global__ static void warp8_kernel(
        float *low, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, uint64_t blocks, int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * RPB + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim) return;
    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const int8_t *xqr = xq + group * blocks * 32;
    const float *xsr = xscale + group * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[row] = acc;
}

/* fixed-trip variant: 128 blocks known at compile time, unrolled */
template <uint32_t RPB>
__global__ static void warp8u_kernel(
        float *low, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t rank, uint32_t n_groups, int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * RPB + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim) return;
    const uint64_t group = row / rank;
    const unsigned char *wr = w + row * 128u * 34u;
    const int8_t *xqr = xq + group * 128u * 32u;
    const float *xsr = xscale + group * 128u;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < 4u; k++) {
        const uint64_t b = lane + 32u * k;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, 32u, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[row] = acc;
}

/* NT lanes per row, pair-wide style epilogue (re-associated) */
template <uint32_t NT>
__global__ static void wide_kernel(
        float *low, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t rank, uint32_t n_groups,
        uint64_t blocks, int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t group = row / rank;
    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr = xq + group * blocks * 32u;
    const float *xsr = xscale + group * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += NT) {
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, 32u, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    __shared__ float ws[NT / 32];
    const uint32_t warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0u) ws[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float total = 0.0f;
        for (uint32_t i = 0; i < NT / 32u; i++) total += ws[i];
        low[row] = total;
    }
}

__global__ static void ceiling_kernel(const uint4 *w, uint64_t n4, uint32_t *sink) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s = 0;
    for (; i < n4; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint4 v = w[i];
        s ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0xdeadbeefu) *sink = s;
}

typedef void (*launch_fn)(float *, const unsigned char *, const int8_t *, const float *);

int main(void) {
    const int POOL = 8; /* 8 x 35.65 MB = 285 MB > 2x the 128 MB L2 */
    unsigned char *w[POOL];
    srand(7);
    unsigned char *host_w = (unsigned char *)malloc(W_BYTES);
    for (uint64_t i = 0; i < W_BYTES; i++) host_w[i] = (unsigned char)(rand() & 0xff);
    /* plant sane f16 scales at each block head */
    for (uint64_t r = 0; r < LOW_DIM; r++) {
        for (uint64_t b = 0; b < BLOCKS; b++) {
            __half h = __float2half(0.01f + 0.001f * (float)((r + b) & 15));
            memcpy(host_w + (r * BLOCKS + b) * 34u, &h, 2);
        }
    }
    for (int p = 0; p < POOL; p++) {
        CK(cudaMalloc(&w[p], W_BYTES));
        CK(cudaMemcpy(w[p], host_w, W_BYTES, cudaMemcpyHostToDevice));
    }
    int8_t *xq; float *xs; float *low_ref, *low_tst;
    CK(cudaMalloc(&xq, N_GROUPS * BLOCKS * 32));
    CK(cudaMalloc(&xs, N_GROUPS * BLOCKS * sizeof(float)));
    CK(cudaMalloc(&low_ref, LOW_DIM * sizeof(float)));
    CK(cudaMalloc(&low_tst, LOW_DIM * sizeof(float)));
    int8_t *hxq = (int8_t *)malloc(N_GROUPS * BLOCKS * 32);
    float *hxs = (float *)malloc(N_GROUPS * BLOCKS * sizeof(float));
    for (uint64_t i = 0; i < N_GROUPS * BLOCKS * 32; i++) hxq[i] = (int8_t)((rand() % 255) - 127);
    for (uint64_t i = 0; i < N_GROUPS * BLOCKS; i++) hxs[i] = 0.005f + 0.0001f * (float)(i & 31);
    CK(cudaMemcpy(xq, hxq, N_GROUPS * BLOCKS * 32, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xs, hxs, N_GROUPS * BLOCKS * sizeof(float), cudaMemcpyHostToDevice));

    float *href = (float *)malloc(LOW_DIM * sizeof(float));
    float *htst = (float *)malloc(LOW_DIM * sizeof(float));

    /* reference = production RPB 8 */
    warp8_kernel<8><<<(unsigned)(LOW_DIM / 8), 256>>>(low_ref, w[0], xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(href, low_ref, LOW_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    struct Var { const char *name; int id; unsigned grid, block; };
    const Var vars[] = {
        {"warp8<8>  (production)", 0, (unsigned)(LOW_DIM / 8), 256},
        {"warp8<4>",              1, (unsigned)(LOW_DIM / 4), 128},
        {"warp8<2>",              2, (unsigned)(LOW_DIM / 2), 64},
        {"warp8<1>",              3, (unsigned)LOW_DIM, 32},
        {"warp8u<8> fixed128",    4, (unsigned)(LOW_DIM / 8), 256},
        {"warp8u<4> fixed128",    5, (unsigned)(LOW_DIM / 4), 128},
        {"wide<64>",              6, (unsigned)LOW_DIM, 64},
        {"wide<128>",             7, (unsigned)LOW_DIM, 128},
    };
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int WARM = 30, ITER = 300;
    for (const Var &v : vars) {
        /* correctness on instance 0 */
        CK(cudaMemset(low_tst, 0, LOW_DIM * sizeof(float)));
        switch (v.id) {
        case 0: warp8_kernel<8><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
        case 1: warp8_kernel<4><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
        case 2: warp8_kernel<2><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
        case 3: warp8_kernel<1><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
        case 4: warp8u_kernel<8><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, RANK, N_GROUPS, 1); break;
        case 5: warp8u_kernel<4><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, RANK, N_GROUPS, 1); break;
        case 6: wide_kernel<64><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, RANK, N_GROUPS, BLOCKS, 1); break;
        case 7: wide_kernel<128><<<v.grid, v.block>>>(low_tst, w[0], xq, xs, RANK, N_GROUPS, BLOCKS, 1); break;
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(htst, low_tst, LOW_DIM * sizeof(float), cudaMemcpyDeviceToHost));
        int exact = memcmp(href, htst, LOW_DIM * sizeof(float)) == 0;
        float mad = 0.0f;
        for (uint64_t i = 0; i < LOW_DIM; i++) {
            float d = fabsf(href[i] - htst[i]);
            if (d > mad) mad = d;
        }
        /* timing over the rotating pool */
        for (int i = 0; i < WARM; i++) {
            const unsigned char *wp = w[i % POOL];
            switch (v.id) {
            case 0: warp8_kernel<8><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 1: warp8_kernel<4><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 2: warp8_kernel<2><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 3: warp8_kernel<1><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 4: warp8u_kernel<8><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, 1); break;
            case 5: warp8u_kernel<4><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, 1); break;
            case 6: wide_kernel<64><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, BLOCKS, 1); break;
            case 7: wide_kernel<128><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, BLOCKS, 1); break;
            }
        }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < ITER; i++) {
            const unsigned char *wp = w[i % POOL];
            switch (v.id) {
            case 0: warp8_kernel<8><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 1: warp8_kernel<4><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 2: warp8_kernel<2><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 3: warp8_kernel<1><<<v.grid, v.block>>>(low_tst, wp, xq, xs, GROUP_DIM, RANK, N_GROUPS, BLOCKS, 1); break;
            case 4: warp8u_kernel<8><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, 1); break;
            case 5: warp8u_kernel<4><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, 1); break;
            case 6: wide_kernel<64><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, BLOCKS, 1); break;
            case 7: wide_kernel<128><<<v.grid, v.block>>>(low_tst, wp, xq, xs, RANK, N_GROUPS, BLOCKS, 1); break;
            }
        }
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = (double)ms * 1000.0 / ITER;
        const double gbs = (double)W_BYTES / (us * 1e-6) / 1e9;
        printf("%-24s grid %5u x %3u  %7.2f us  %7.1f GB/s  %s mad=%.3e\n",
               v.name, v.grid, v.block, us, gbs,
               exact ? "BITEXACT" : "differs ", mad);
    }

    /* pure-read ceiling at a saturating grid */
    uint32_t *sink; CK(cudaMalloc(&sink, 4));
    for (int i = 0; i < WARM; i++) ceiling_kernel<<<2048, 256>>>((const uint4 *)w[i % POOL], W_BYTES / 16, sink);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < ITER; i++) ceiling_kernel<<<2048, 256>>>((const uint4 *)w[i % POOL], W_BYTES / 16, sink);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = (double)ms * 1000.0 / ITER;
    printf("%-24s grid %5u x %3u  %7.2f us  %7.1f GB/s\n",
           "uint4 read ceiling", 2048u, 256u, us, (double)W_BYTES / (us * 1e-6) / 1e9);
    return 0;
}
