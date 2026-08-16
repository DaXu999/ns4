/* indexer_scan_bench.cu -- price the decode DSA indexer score scan.
 *
 * The production kernel (indexer_score_one_direct_kernel, verbatim in
 * scan_direct.inc) gives each compressed position a 128-thread block and runs
 * 64 warp reductions per position -- ~44M warp reductions/token at ctx 131072,
 * measured ~340 GB/s of a ~1450 GB/s size ceiling. Two candidate rebuilds:
 *
 *   V1 "qreg":     4 warps x 16 heads, q register-resident, the 16 reduction
 *                  chains per warp pipeline instead of serializing. Reads the
 *                  same f32 rows. Bit-identical by construction (same float4
 *                  dot order, same warp_sum_f32 tree, same group-of-4 tail).
 *   V2 "qreg+fp4": same compute, but reads a 4-bit+scale mirror of the row
 *                  cache. Rows are e2m1xPo2 grid points (indexer_hadamard_fp4
 *                  writes dequantized values), so codes+scale reproduce the
 *                  exact f32 bits: lut[code]*scale is the same fmul the qat
 *                  kernel performed. 80 B/row instead of 512 B.
 *
 * Every variant is memcmp'd against the production kernel's scores buffer.
 * Timings: eager and graph-amortised, L2 defeated by rotating k pools.
 *
 * Build: nvcc -O3 --use_fast_math -gencode arch=compute_120a,code=sm_120a \
 *          -o indexer_scan_bench indexer_scan_bench.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#include "scan_warp_sum.inc"
#include "scan_e2m1.inc"
#include "scan_hadamard.inc"
#include "scan_direct.inc"

/* ---------------- mirror encode (fused form of the qat kernel) ----------- */
/* Same math as indexer_hadamard_fp4_kernel, but also emits the 4-bit code
 * stream and the per-32-group f32 scale. The f32 output must stay
 * bit-identical to the original kernel's (verified by memcmp below): the
 * mirror is derived from the same (code, scale) the quantizer picked. */
__device__ static int dsv4_e2m1fn_code_dev(float x) {
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return best;
}

/* mirror row layout: 64 bytes of packed codes (low nibble = even element),
 * then 4 f32 scales. 80 bytes, uint4-aligned. */
#define MIRROR_ROW_BYTES 80u

__global__ static void indexer_hadamard_fp4_mirror_kernel(
        float *x, unsigned char *mirror, uint32_t first_row,
        uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    __shared__ unsigned char codes[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    float clamped = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int code = dsv4_e2m1fn_code_dev(clamped);
    /* sign bit carried separately so -0.0f survives the round-trip */
    if (clamped < 0.0f) code |= 8;
    xr[tid] = dsv4_e2m1fn_dequant_dev(clamped) * scale;

    unsigned char *mrow = mirror + (uint64_t)(first_row + row) * MIRROR_ROW_BYTES;
    codes[tid] = (unsigned char)code;
    __syncthreads();
    if (tid < 64u) {
        mrow[tid] = (unsigned char)((codes[2u * tid] & 0xFu) |
                                    ((codes[2u * tid + 1u] & 0xFu) << 4u));
    }
    if (lane == 0u) ((float *)(mrow + 64u))[fp4_block] = scale;
}

/* ---------------- V1: q register-resident, f32 rows ---------------------- */
/* Warp w owns heads [16w, 16w+16). Lane l holds q[h][4l..4l+3]. Each head's
 * dot is the same float4 expression and the same warp_sum_f32 tree as the
 * production kernel; groups of four heads are summed by lane 0 of the owning
 * warp in the production tail's association order, and thread 0 walks the 16
 * group sums in order. */
template <uint32_t STRIP, bool FP4>
__global__ static void indexer_scan_qreg_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        const unsigned char *mirror,
        uint32_t n_comp,
        float scale) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;

    float4 qreg[16];
    float wreg[16];
    #pragma unroll
    for (uint32_t i = 0; i < 16u; i++) {
        const uint32_t h = warp * 16u + i;
        qreg[i] = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        wreg[i] = weights[h];
    }

    __shared__ float krow[128];
    __shared__ float group_sum[16];
    __shared__ float lut[16];
    if (tid < 16u) {
        float mag = dsv4_e2m1fn_value_dev((int)(tid & 7u));
        lut[tid] = (tid & 8u) ? -mag : mag;
    }

    const uint32_t c_end = min(n_comp, (blockIdx.x + 1u) * STRIP);
    for (uint32_t c = blockIdx.x * STRIP; c < c_end; c++) {
        __syncthreads();
        if (FP4) {
            const unsigned char *mrow = mirror + (uint64_t)c * MIRROR_ROW_BYTES;
            const unsigned char byte = mrow[tid >> 1u];
            const unsigned int code = (tid & 1u) ? (byte >> 4u) : (byte & 0xFu);
            const float sc = ((const float *)(mrow + 64u))[tid >> 5u];
            krow[tid] = lut[code] * sc;
        } else {
            krow[tid] = index_comp[(uint64_t)c * 128u + tid];
        }
        __syncthreads();

        const float4 kv = ((const float4 *)krow)[lane];
        float dot[16];
        #pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            dot[i] = qreg[i].x * kv.x + qreg[i].y * kv.y +
                     qreg[i].z * kv.z + qreg[i].w * kv.w;
        }
        #pragma unroll
        for (uint32_t i = 0; i < 16u; i++) dot[i] = warp_sum_f32(dot[i]);
        if (lane == 0u) {
            /* __fmul_rn/__fadd_rn: in the production kernel the term muls are
             * separated from the tail adds by a shared-memory round trip, so
             * nvcc cannot contract them into FMAs. Here they are adjacent and
             * it does, which is a 1-ULP difference. Intrinsics forbid it. */
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const float t0 = __fmul_rn(__fmul_rn(fmaxf(dot[4u * j + 0u], 0.0f), wreg[4u * j + 0u]), scale);
                const float t1 = __fmul_rn(__fmul_rn(fmaxf(dot[4u * j + 1u], 0.0f), wreg[4u * j + 1u]), scale);
                const float t2 = __fmul_rn(__fmul_rn(fmaxf(dot[4u * j + 2u], 0.0f), wreg[4u * j + 2u]), scale);
                const float t3 = __fmul_rn(__fmul_rn(fmaxf(dot[4u * j + 3u], 0.0f), wreg[4u * j + 3u]), scale);
                group_sum[warp * 4u + j] = __fadd_rn(__fadd_rn(__fadd_rn(t0, t1), t2), t3);
            }
        }
        __syncthreads();
        if (tid == 0u) {
            float total = 0.0f;
            #pragma unroll
            for (uint32_t g = 0; g < 16u; g++) total += group_sum[g];
            scores[c] = total;
        }
    }
}

/* ---------------- V3: transpose, one thread per position ----------------- */
/* No shuffles, no per-head barriers: thread owns position c and computes the
 * 32 lane-partials of each head in registers, then replicates warp_sum_f32's
 * shfl_down pairing (p[l] += p[l+off], off = 16..1) serially. k rows are
 * decoded once into shared as [group][column] float4 so a warp's reads are
 * lane-consecutive; q is read via uniform L1 loads (same address across the
 * warp). Threads pair per position: half 0 takes heads 0..31, half 1 heads
 * 32..63 and parks its group sums for half 0 to fold in tail order. */
template <bool FP4>
__global__ static void indexer_scan_tpose_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        const unsigned char *mirror,
        uint32_t n_comp,
        float scale) {
    const uint32_t tid = threadIdx.x;
    const uint32_t col = tid & 63u;
    const uint32_t half = tid >> 6u;
    const uint32_t c0 = blockIdx.x * 64u;

    __shared__ float4 kdec[32][64];
    __shared__ float tail[64][8];
    __shared__ float lut[16];
    __shared__ unsigned char stage[64 * MIRROR_ROW_BYTES];
    if (tid < 16u) {
        float mag = dsv4_e2m1fn_value_dev((int)(tid & 7u));
        lut[tid] = (tid & 8u) ? -mag : mag;
    }

    const uint32_t rows = min(64u, n_comp - min(n_comp, c0));
    if (FP4) {
        const uint4 *src = (const uint4 *)(mirror + (uint64_t)c0 * MIRROR_ROW_BYTES);
        uint4 *dst = (uint4 *)stage;
        for (uint32_t i = tid; i < rows * (MIRROR_ROW_BYTES / 16u); i += 128u) dst[i] = src[i];
        __syncthreads();
        for (uint32_t i = tid; i < rows * 128u; i += 128u) {
            const uint32_t r = i >> 7u, e = i & 127u;
            const unsigned char *mrow = stage + r * MIRROR_ROW_BYTES;
            const unsigned char byte = mrow[e >> 1u];
            const unsigned int code = (e & 1u) ? (byte >> 4u) : (byte & 0xFu);
            const float sc = ((const float *)(mrow + 64u))[e >> 5u];
            ((float *)kdec)[(e >> 2u) * 256u + r * 4u + (e & 3u)] = lut[code] * sc;
        }
    } else {
        for (uint32_t i = tid; i < rows * 128u; i += 128u) {
            const uint32_t r = i >> 7u, e = i & 127u;
            ((float *)kdec)[(e >> 2u) * 256u + r * 4u + (e & 3u)] =
                index_comp[((uint64_t)c0 + r) * 128u + e];
        }
    }
    __syncthreads();

    const uint32_t c = c0 + col;
    if (c >= n_comp) return;

    float partial[32];
    float total = 0.0f;
    float t4[4];
    #pragma unroll 1
    for (uint32_t i = 0; i < 32u; i++) {
        const uint32_t h = half * 32u + i;
        const float4 *qh = (const float4 *)(q + (uint64_t)h * 128u);
        #pragma unroll
        for (uint32_t g = 0; g < 32u; g++) {
            const float4 qv = __ldg(qh + g);
            const float4 kv = kdec[g][col];
            partial[g] = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            for (uint32_t l = 0; l < off; l++) partial[l] += partial[l + off];
        t4[i & 3u] = __fmul_rn(__fmul_rn(fmaxf(partial[0], 0.0f), __ldg(weights + h)), scale);
        if ((i & 3u) == 3u) {
            const float g4 = __fadd_rn(__fadd_rn(__fadd_rn(t4[0], t4[1]), t4[2]), t4[3]);
            if (half == 0u) total = __fadd_rn(total, g4);
            else tail[col][(i >> 2u)] = g4;
        }
    }
    __syncthreads();
    if (half == 0u) {
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) total = __fadd_rn(total, tail[col][j]);
        scores[c] = total;
    }
}

/* ---------------- V4: transpose, 32-c tile, 4-way head split ------------- */
/* Same exactness construction as V3 but sized for occupancy: 32 positions per
 * block, 128 threads = 4 quarters of 16 heads, ~18 KB shared -> ~5 blocks/SM.
 * Mirror rows are loaded cooperatively (4 threads share one row's 64 code
 * bytes) and decoded once into the [group][column] float4 layout. */
template <bool FP4>
__global__ static void indexer_scan_tpose32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        const unsigned char *mirror,
        uint32_t n_comp,
        float scale) {
    const uint32_t tid = threadIdx.x;
    const uint32_t col = tid & 31u;
    const uint32_t quarter = tid >> 5u;
    const uint32_t c0 = blockIdx.x * 32u;

    __shared__ float4 kdec[32][32];
    __shared__ float tail[32][12];
    __shared__ float lut[16];
    if (tid < 16u) {
        float mag = dsv4_e2m1fn_value_dev((int)(tid & 7u));
        lut[tid] = (tid & 8u) ? -mag : mag;
    }
    __syncthreads();

    const uint32_t rows = n_comp > c0 ? min(32u, n_comp - c0) : 0u;
    if (FP4) {
        /* thread (col, quarter): row col's code quad `quarter` (16 bytes) and
         * its quarter scale -> decode 32 elements into kdec */
        if (col < rows) {
            /* __ldcs: the row stream is touched once; evict-first keeps q
             * resident in L1 across blocks, which is most of this kernel. */
            const unsigned char *mrow = mirror + (uint64_t)(c0 + col) * MIRROR_ROW_BYTES;
            const uint4 codes = __ldcs((const uint4 *)mrow + quarter);
            const float sc = __ldcs((const float *)(mrow + 64u) + quarter);
            const unsigned char *cb = (const unsigned char *)&codes;
            #pragma unroll
            for (uint32_t j = 0; j < 32u; j++) {
                const uint32_t e = quarter * 32u + j;
                const unsigned char byte = cb[j >> 1u];
                const unsigned int code = (j & 1u) ? (byte >> 4u) : (byte & 0xFu);
                ((float *)kdec)[(e >> 2u) * 128u + col * 4u + (e & 3u)] = lut[code] * sc;
            }
        }
    } else {
        for (uint32_t i = tid; i < rows * 128u; i += 128u) {
            const uint32_t r = i >> 7u, e = i & 127u;
            ((float *)kdec)[(e >> 2u) * 128u + r * 4u + (e & 3u)] =
                __ldcs(index_comp + ((uint64_t)c0 + r) * 128u + e);
        }
    }
    __syncthreads();

    const uint32_t c = c0 + col;
    if (c >= n_comp) return;

    float partial[32];
    float total = 0.0f;
    float t4[4];
    #pragma unroll 1
    for (uint32_t i = 0; i < 16u; i++) {
        const uint32_t h = quarter * 16u + i;
        const float4 *qh = (const float4 *)(q + (uint64_t)h * 128u);
        #pragma unroll
        for (uint32_t g = 0; g < 32u; g++) {
            const float4 qv = __ldg(qh + g);
            const float4 kv = kdec[g][col];
            partial[g] = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            for (uint32_t l = 0; l < off; l++) partial[l] += partial[l + off];
        t4[i & 3u] = __fmul_rn(__fmul_rn(fmaxf(partial[0], 0.0f), __ldg(weights + h)), scale);
        if ((i & 3u) == 3u) {
            const float g4 = __fadd_rn(__fadd_rn(__fadd_rn(t4[0], t4[1]), t4[2]), t4[3]);
            if (quarter == 0u) total = __fadd_rn(total, g4);
            else tail[col][(quarter - 1u) * 4u + (i >> 2u)] = g4;
        }
    }
    __syncthreads();
    if (quarter == 0u) {
        #pragma unroll
        for (uint32_t j = 0; j < 12u; j++) total = __fadd_rn(total, tail[col][j]);
        scores[c] = total;
    }
}

/* ---------------- V6: exact-integer dp4a scan over both mirrors ---------- */
/* Both q and k rows are e2m1xPo2 grid points, so 2x their code magnitudes are
 * small ints and a group dot is IntDot * (0.25 * sk * sq) with every factor
 * exact.  The result is the mathematically exact dot of the quantized
 * vectors -- tighter than the f32 tree, but NOT bit-identical to it, so this
 * arm reports accuracy against a float64 reference instead of a bit compare.
 * One thread per position; k codes unpacked once into registers, q unpacked
 * once per block into shared int8. */
__device__ static inline int e2m1_nibble_to_i8(unsigned int nib) {
    const unsigned int m = nib & 7u;
    int v;
    if (m == 0u) v = 0;
    else if (m == 1u) v = 1;
    else v = (int)((2u + (m & 1u)) << ((m >> 1u) - 1u));
    return (nib & 8u) ? -v : v;
}

__global__ static void indexer_scan_int_kernel(
        float *scores,
        const unsigned char *q_mirror,
        const float *weights,
        const unsigned char *k_mirror,
        uint32_t n_comp,
        float scale) {
    const uint32_t tid = threadIdx.x;
    const uint32_t col = tid & 31u;
    const uint32_t quarter = tid >> 5u;
    const uint32_t c0 = blockIdx.x * 32u;

    __shared__ uint32_t qdec[64][32];   /* int8x4 words, 8 KB */
    __shared__ float qsc[64][4];        /* q scale * 0.5 (fold one sqrt of 0.25) */
    __shared__ float tail[32][12];

    for (uint32_t i = tid; i < 64u * 32u; i += 128u) {
        const uint32_t r = i >> 5u, w = i & 31u;
        const unsigned char *mrow = q_mirror + (uint64_t)r * MIRROR_ROW_BYTES;
        const unsigned char b0 = mrow[2u * w];
        const unsigned char b1 = mrow[2u * w + 1u];
        const int v0 = e2m1_nibble_to_i8(b0 & 0xFu);
        const int v1 = e2m1_nibble_to_i8(b0 >> 4u);
        const int v2 = e2m1_nibble_to_i8(b1 & 0xFu);
        const int v3 = e2m1_nibble_to_i8(b1 >> 4u);
        qdec[r][w] = (uint32_t)(v0 & 0xFF) | ((uint32_t)(v1 & 0xFF) << 8) |
                     ((uint32_t)(v2 & 0xFF) << 16) | ((uint32_t)(v3 & 0xFF) << 24);
        if (w < 4u) qsc[r][w] = ((const float *)(mrow + 64u))[w] * 0.5f;
    }
    __syncthreads();

    const uint32_t c = c0 + col;
    if (c >= n_comp) return;

    uint32_t kdec[32];
    float ksc[4];
    {
        const unsigned char *mrow = k_mirror + (uint64_t)c * MIRROR_ROW_BYTES;
        #pragma unroll
        for (uint32_t w = 0; w < 32u; w++) {
            const unsigned char b0 = mrow[2u * w];
            const unsigned char b1 = mrow[2u * w + 1u];
            const int v0 = e2m1_nibble_to_i8(b0 & 0xFu);
            const int v1 = e2m1_nibble_to_i8(b0 >> 4u);
            const int v2 = e2m1_nibble_to_i8(b1 & 0xFu);
            const int v3 = e2m1_nibble_to_i8(b1 >> 4u);
            kdec[w] = (uint32_t)(v0 & 0xFF) | ((uint32_t)(v1 & 0xFF) << 8) |
                      ((uint32_t)(v2 & 0xFF) << 16) | ((uint32_t)(v3 & 0xFF) << 24);
        }
        #pragma unroll
        for (uint32_t g = 0; g < 4u; g++)
            ksc[g] = ((const float *)(mrow + 64u))[g] * 0.5f;
    }

    float total = 0.0f;
    float t4[4];
    #pragma unroll 1
    for (uint32_t i = 0; i < 16u; i++) {
        const uint32_t h = quarter * 16u + i;
        int acc[4] = {0, 0, 0, 0};
        #pragma unroll
        for (uint32_t g = 0; g < 4u; g++) {
            #pragma unroll
            for (uint32_t w = 0; w < 8u; w++) {
                acc[g] = __dp4a((int)kdec[g * 8u + w], (int)qdec[h][g * 8u + w], acc[g]);
            }
        }
        /* term_g = IntDot * (0.5 sk)(0.5 sq): both scales po2, IntDot < 2^24,
         * so each term is exact; the 3-add combine is the only rounding. */
        float dot = __fadd_rn(__fadd_rn(__fadd_rn(
                (float)acc[0] * __fmul_rn(ksc[0], qsc[h][0]),
                (float)acc[1] * __fmul_rn(ksc[1], qsc[h][1])),
                (float)acc[2] * __fmul_rn(ksc[2], qsc[h][2])),
                (float)acc[3] * __fmul_rn(ksc[3], qsc[h][3]));
        t4[i & 3u] = __fmul_rn(__fmul_rn(fmaxf(dot, 0.0f), __ldg(weights + h)), scale);
        if ((i & 3u) == 3u) {
            const float g4 = __fadd_rn(__fadd_rn(__fadd_rn(t4[0], t4[1]), t4[2]), t4[3]);
            if (quarter == 0u) total = __fadd_rn(total, g4);
            else tail[col][(quarter - 1u) * 4u + (i >> 2u)] = g4;
        }
    }
    __syncthreads();
    if (quarter == 0u) {
        #pragma unroll
        for (uint32_t j = 0; j < 12u; j++) total = __fadd_rn(total, tail[col][j]);
        scores[c] = total;
    }
}

/* ---------------- V5: warp per position, register k, prefetch ------------ */
/* One warp owns a strip of positions. Lane l holds k[4l..4l+3] as a float4 in
 * registers (the same float4 the production kernel's lane reads from its
 * shared krow), prefetched one position ahead so the DRAM/L2 latency of the
 * next row hides under the current row's 64 head reductions. No shared
 * memory, no barriers; the head loop and every reduction keep the production
 * association order. */
__constant__ float g_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

template <bool FP4>
__device__ static inline float4 wpc_load_row(
        const float *index_comp, const unsigned char *mirror,
        uint32_t c, uint32_t lane) {
    if (FP4) {
        const unsigned char *mrow = mirror + (uint64_t)c * MIRROR_ROW_BYTES;
        const unsigned short two = ((const unsigned short *)mrow)[lane];
        const float sc = ((const float *)(mrow + 64u))[lane >> 3u];
        float4 kv;
        kv.x = g_e2m1_lut[two & 15u] * sc;
        kv.y = g_e2m1_lut[(two >> 4u) & 15u] * sc;
        kv.z = g_e2m1_lut[(two >> 8u) & 15u] * sc;
        kv.w = g_e2m1_lut[(two >> 12u) & 15u] * sc;
        return kv;
    }
    return ((const float4 *)(index_comp + (uint64_t)c * 128u))[lane];
}

template <bool FP4, uint32_t STRIP>
__global__ static void indexer_scan_wpc_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        const unsigned char *mirror,
        uint32_t n_comp,
        float scale) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t c_base = (blockIdx.x * 4u + warp) * STRIP;
    if (c_base >= n_comp) return;
    const uint32_t c_end = min(n_comp, c_base + STRIP);

    float4 kv_next = wpc_load_row<FP4>(index_comp, mirror, c_base, lane);
    for (uint32_t c = c_base; c < c_end; c++) {
        const float4 kv = kv_next;
        if (c + 1u < c_end) kv_next = wpc_load_row<FP4>(index_comp, mirror, c + 1u, lane);
        float total = 0.0f;
        float t4[4];
        #pragma unroll 8
        for (uint32_t h = 0; h < 64u; h++) {
            const float4 qv = __ldg((const float4 *)(q + (uint64_t)h * 128u) + lane);
            float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            dot = warp_sum_f32(dot);
            t4[h & 3u] = __fmul_rn(__fmul_rn(fmaxf(dot, 0.0f), __ldg(weights + h)), scale);
            if ((h & 3u) == 3u) {
                total = __fadd_rn(total,
                        __fadd_rn(__fadd_rn(__fadd_rn(t4[0], t4[1]), t4[2]), t4[3]));
            }
        }
        if (lane == 0u) scores[c] = total;
    }
}

/* Rebuild a mirror from already-quantized rows: scale = 2^(E_max-2) always
 * lands stored values back on e2m1 codes exactly (production restore path;
 * validated here by decoding the mirror back and memcmp'ing). */
__global__ static void mirror_reencode_kernel(
        const float *x, unsigned char *mirror, uint32_t n_rows) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || tid >= 128u) return;
    __shared__ float absbuf[128];
    __shared__ unsigned char codes[128];
    const float v = x[(uint64_t)row * 128u + tid];
    uint32_t fp4_block = tid >> 5u, lane = tid & 31u, block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        __syncthreads();
    }
    const float amax = absbuf[block_base];
    float scale = 1.0f;
    if (amax > 0.0f) {
        int e_max;
        (void)frexpf(amax, &e_max);
        scale = exp2f((float)(e_max - 3 < -126 ? -126 : e_max - 3));
    }
    const float mag = fabsf(v) / scale;
    int code = 0;
    for (int i = 1; i < 8; i++)
        if (mag == dsv4_e2m1fn_value_dev(i)) code = i;
    if (signbit(v)) code |= 8;
    codes[tid] = (unsigned char)code;
    __syncthreads();
    unsigned char *mrow = mirror + (uint64_t)row * MIRROR_ROW_BYTES;
    if (tid < 64u)
        mrow[tid] = (unsigned char)((codes[2u * tid] & 0xFu) |
                                    ((codes[2u * tid + 1u] & 0xFu) << 4u));
    if (lane == 0u) ((float *)(mrow + 64u))[fp4_block] = scale;
}

__global__ static void mirror_decode_kernel(
        float *out, const unsigned char *mirror, uint32_t n_rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_rows * 128u) return;
    const uint32_t r = (uint32_t)(i >> 7u), e = (uint32_t)(i & 127u);
    const unsigned char *mrow = mirror + (uint64_t)r * MIRROR_ROW_BYTES;
    const unsigned char byte = mrow[e >> 1u];
    const unsigned int code = (e & 1u) ? (byte >> 4u) : (byte & 0xFu);
    const float sc = ((const float *)(mrow + 64u))[e >> 5u];
    float mag = dsv4_e2m1fn_value_dev((int)(code & 7u));
    out[i] = ((code & 8u) ? -mag : mag) * sc;
}

/* ---------------- harness ------------------------------------------------ */

static float frand(unsigned int *st) {
    *st = *st * 1664525u + 1013904223u;
    return ((float)(*st >> 9) / 8388608.0f - 1.0f) * 2.0f;
}

typedef void (*launch_fn)(float *, const float *, const float *, const float *,
                          const unsigned char *, uint32_t, float, cudaStream_t);

static void launch_direct(float *s, const float *q, const float *w, const float *k,
                          const unsigned char *m, uint32_t n, float sc, cudaStream_t st) {
    (void)m;
    indexer_score_one_direct_kernel<true><<<n, 128, 0, st>>>(s, q, w, k, n, 0, 1, sc, 0);
}
template <uint32_t STRIP, bool FP4>
static void launch_qreg(float *s, const float *q, const float *w, const float *k,
                        const unsigned char *m, uint32_t n, float sc, cudaStream_t st) {
    indexer_scan_qreg_kernel<STRIP, FP4><<<(n + STRIP - 1u) / STRIP, 128, 0, st>>>(
            s, q, w, k, m, n, sc);
}
template <bool FP4>
static void launch_tpose(float *s, const float *q, const float *w, const float *k,
                         const unsigned char *m, uint32_t n, float sc, cudaStream_t st) {
    indexer_scan_tpose_kernel<FP4><<<(n + 63u) / 64u, 128, 0, st>>>(s, q, w, k, m, n, sc);
}
template <bool FP4>
static void launch_tpose32(float *s, const float *q, const float *w, const float *k,
                           const unsigned char *m, uint32_t n, float sc, cudaStream_t st) {
    indexer_scan_tpose32_kernel<FP4><<<(n + 31u) / 32u, 128, 0, st>>>(s, q, w, k, m, n, sc);
}
template <bool FP4, uint32_t STRIP>
static void launch_wpc(float *s, const float *q, const float *w, const float *k,
                       const unsigned char *m, uint32_t n, float sc, cudaStream_t st) {
    indexer_scan_wpc_kernel<FP4, STRIP><<<(n + 4u * STRIP - 1u) / (4u * STRIP), 128, 0, st>>>(
            s, q, w, k, m, n, sc);
}

int main(int argc, char **argv) {
    uint32_t n_comp = argc > 1 ? (uint32_t)atoi(argv[1]) : 32768u;
    const int POOL = 20;         /* 20 x 16.8 MB = 336 MB >> 128 MB L2 */
    const int ITER = 40;         /* timed launches per measurement */
    const float scale = 1.0f / sqrtf(128.0f * 64.0f);

    float *q, *w, **kpool, *scores_ref, *scores;
    unsigned char **mpool;
    kpool = (float **)malloc(POOL * sizeof(float *));
    mpool = (unsigned char **)malloc(POOL * sizeof(unsigned char *));
    CK(cudaMalloc(&q, 64 * 128 * sizeof(float)));
    CK(cudaMalloc(&w, 64 * sizeof(float)));
    CK(cudaMalloc(&scores_ref, (uint64_t)n_comp * sizeof(float)));
    CK(cudaMalloc(&scores, (uint64_t)n_comp * sizeof(float)));

    unsigned int st = 12345u;
    float *host = (float *)malloc((uint64_t)n_comp * 128 * sizeof(float));
    for (int i = 0; i < 64 * 128; i++) host[i] = frand(&st);
    CK(cudaMemcpy(q, host, 64 * 128 * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < 64; i++) host[i] = frand(&st);
    CK(cudaMemcpy(w, host, 64 * sizeof(float), cudaMemcpyHostToDevice));

    /* q is qat'd in production before scoring (64 rows of 128) */
    indexer_hadamard_fp4_kernel<<<64, 128>>>(q, 64, 128);

    for (int p = 0; p < POOL; p++) {
        CK(cudaMalloc(&kpool[p], (uint64_t)n_comp * 128 * sizeof(float)));
        CK(cudaMalloc(&mpool[p], (uint64_t)n_comp * MIRROR_ROW_BYTES));
        for (uint64_t i = 0; i < (uint64_t)n_comp * 128; i++) host[i] = frand(&st);
        CK(cudaMemcpy(kpool[p], host, (uint64_t)n_comp * 128 * sizeof(float), cudaMemcpyHostToDevice));
    }
    free(host);

    /* Quantize pool 0 with the ORIGINAL kernel, pools 1.. with the fused
     * mirror kernel; then re-run the fused kernel on a copy of pool 0's
     * pre-quant data... simpler: quantize pool 0 twice, once each way, and
     * memcmp. Pool 0 gets a pristine copy first. */
    float *k0_copy;
    CK(cudaMalloc(&k0_copy, (uint64_t)n_comp * 128 * sizeof(float)));
    CK(cudaMemcpy(k0_copy, kpool[0], (uint64_t)n_comp * 128 * sizeof(float), cudaMemcpyDeviceToDevice));
    indexer_hadamard_fp4_kernel<<<n_comp, 128>>>(kpool[0], n_comp, 128);
    indexer_hadamard_fp4_mirror_kernel<<<n_comp, 128>>>(k0_copy, mpool[0], 0, n_comp, 128);
    CK(cudaDeviceSynchronize());
    {
        float *a = (float *)malloc((uint64_t)n_comp * 128 * sizeof(float));
        float *b = (float *)malloc((uint64_t)n_comp * 128 * sizeof(float));
        CK(cudaMemcpy(a, kpool[0], (uint64_t)n_comp * 128 * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b, k0_copy, (uint64_t)n_comp * 128 * sizeof(float), cudaMemcpyDeviceToHost));
        if (memcmp(a, b, (uint64_t)n_comp * 128 * sizeof(float)) != 0) {
            printf("FUSED QAT MISMATCH vs original hadamard_fp4 kernel\n");
            return 1;
        }
        printf("fused qat kernel: f32 output bit-identical to original\n");
        free(a); free(b);
    }
    CK(cudaFree(k0_copy));
    for (int p = 1; p < POOL; p++) {
        indexer_hadamard_fp4_mirror_kernel<<<n_comp, 128>>>(kpool[p], mpool[p], 0, n_comp, 128);
    }
    CK(cudaDeviceSynchronize());

    /* q mirror via the reencode path (validates the restore kernel), plus a
     * decode-back memcmp of both mirror producers. */
    unsigned char *q_mirror;
    float *decbuf;
    CK(cudaMalloc(&q_mirror, 64 * MIRROR_ROW_BYTES));
    CK(cudaMalloc(&decbuf, (uint64_t)n_comp * 128 * sizeof(float)));
    mirror_reencode_kernel<<<64, 128>>>(q, q_mirror, 64);
    mirror_decode_kernel<<<(64 * 128 + 255) / 256, 256>>>(decbuf, q_mirror, 64);
    CK(cudaDeviceSynchronize());
    {
        float *a = (float *)malloc(64 * 128 * sizeof(float));
        float *b = (float *)malloc(64 * 128 * sizeof(float));
        CK(cudaMemcpy(a, q, 64 * 128 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b, decbuf, 64 * 128 * 4, cudaMemcpyDeviceToHost));
        printf("reencode(q) decode-back: %s\n",
               memcmp(a, b, 64 * 128 * 4) == 0 ? "bit-identical" : "** DIFFERS **");
        free(a); free(b);
    }
    mirror_decode_kernel<<<((uint32_t)((uint64_t)n_comp * 128 + 255) / 256), 256>>>(
            decbuf, mpool[0], n_comp);
    CK(cudaDeviceSynchronize());
    {
        uint64_t bytes = (uint64_t)n_comp * 128 * 4;
        float *a = (float *)malloc(bytes);
        float *b = (float *)malloc(bytes);
        CK(cudaMemcpy(a, kpool[0], bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b, decbuf, bytes, cudaMemcpyDeviceToHost));
        printf("fused-mirror(k) decode-back: %s\n",
               memcmp(a, b, bytes) == 0 ? "bit-identical" : "** DIFFERS **");
        free(a); free(b);
    }
    CK(cudaFree(decbuf));

    /* reference scores from the production kernel on pool 0 */
    launch_direct(scores_ref, q, w, kpool[0], NULL, n_comp, scale, 0);
    CK(cudaDeviceSynchronize());

    struct { const char *name; launch_fn fn; int fp4; } arms[] = {
        { "direct(prod)      ", launch_direct,          0 },
        { "qreg f32 STRIP=8  ", launch_qreg<8u,  false>, 0 },
        { "qreg fp4 STRIP=8  ", launch_qreg<8u,  true>,  1 },
        { "tpose f32         ", launch_tpose<false>,     0 },
        { "tpose fp4         ", launch_tpose<true>,      1 },
        { "tpose32 f32       ", launch_tpose32<false>,   0 },
        { "tpose32 fp4       ", launch_tpose32<true>,    1 },
        { "wpc f32 STRIP=8   ", launch_wpc<false, 8u>,   0 },
        { "wpc f32 STRIP=32  ", launch_wpc<false, 32u>,  0 },
        { "wpc fp4 STRIP=8   ", launch_wpc<true, 8u>,    1 },
        { "wpc fp4 STRIP=32  ", launch_wpc<true, 32u>,   1 },
    };
    const int n_arms = (int)(sizeof(arms) / sizeof(arms[0]));

    float *sref = (float *)malloc((uint64_t)n_comp * sizeof(float));
    float *sgot = (float *)malloc((uint64_t)n_comp * sizeof(float));
    CK(cudaMemcpy(sref, scores_ref, (uint64_t)n_comp * sizeof(float), cudaMemcpyDeviceToHost));

    printf("n_comp=%u  bytes f32=%.2f MB  mirror=%.2f MB\n\n", n_comp,
           n_comp * 512.0 / 1e6, n_comp * (double)MIRROR_ROW_BYTES / 1e6);
    printf("%-20s %10s %10s %12s %12s  %s\n", "arm", "eager us", "graph us",
           "f32-GB/s", "read-GB/s", "bits");

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    const char *only = getenv("SCAN_ONLY");
    for (int a = 0; a < n_arms; a++) {
        if (only && !strstr(arms[a].name, only)) continue;
        /* correctness on pool 0 */
        CK(cudaMemset(scores, 0, (uint64_t)n_comp * sizeof(float)));
        arms[a].fn(scores, q, w, kpool[0], mpool[0], n_comp, scale, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        CK(cudaMemcpy(sgot, scores, (uint64_t)n_comp * sizeof(float), cudaMemcpyDeviceToHost));
        int same = memcmp(sref, sgot, (uint64_t)n_comp * sizeof(float)) == 0;
        if (!same && getenv("SCAN_DEBUG")) {
            uint32_t n_bad = 0, first = 0;
            for (uint32_t i = 0; i < n_comp; i++) {
                if (memcmp(&sref[i], &sgot[i], 4) != 0) {
                    if (n_bad == 0) first = i;
                    n_bad++;
                }
            }
            printf("  [%s] %u/%u mismatch, first at c=%u: ref=%.9g(%08x) got=%.9g(%08x) relerr=%.3g\n",
                   arms[a].name, n_bad, n_comp, first, sref[first],
                   *(unsigned *)&sref[first], sgot[first], *(unsigned *)&sgot[first],
                   fabs((double)sgot[first] - sref[first]) / fmax(1e-30, fabs((double)sref[first])));
        }

        /* eager, rotating pools */
        for (int i = 0; i < 8; i++)
            arms[a].fn(scores, q, w, kpool[i % POOL], mpool[i % POOL], n_comp, scale, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < ITER; i++)
            arms[a].fn(scores, q, w, kpool[i % POOL], mpool[i % POOL], n_comp, scale, 0);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms_eager;
        CK(cudaEventElapsedTime(&ms_eager, e0, e1));

        /* graph-amortised, rotating pools inside the capture */
        cudaStream_t cs;
        CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < ITER; i++)
            arms[a].fn(scores, q, w, kpool[i % POOL], mpool[i % POOL], n_comp, scale, cs);
        CK(cudaStreamEndCapture(cs, &gr));
        CK(cudaGraphInstantiate(&ge, gr, NULL, NULL, 0));
        CK(cudaGraphLaunch(ge, cs));
        CK(cudaStreamSynchronize(cs));
        CK(cudaEventRecord(e0, cs));
        CK(cudaGraphLaunch(ge, cs));
        CK(cudaEventRecord(e1, cs));
        CK(cudaEventSynchronize(e1));
        float ms_graph;
        CK(cudaEventElapsedTime(&ms_graph, e0, e1));
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(gr));
        CK(cudaStreamDestroy(cs));

        double us_e = ms_eager * 1000.0 / ITER;
        double us_g = ms_graph * 1000.0 / ITER;
        double f32_bytes = (double)n_comp * 512.0;
        double read_bytes = arms[a].fp4 ? (double)n_comp * MIRROR_ROW_BYTES : f32_bytes;
        printf("%-20s %10.2f %10.2f %12.0f %12.0f  %s\n", arms[a].name, us_e, us_g,
               f32_bytes / (us_g * 1000.0), read_bytes / (us_g * 1000.0),
               same ? "IDENTICAL" : "** DIFFERS **");
    }
    /* ---- exact-integer arm: timed like the others, accuracy vs float64 ---- */
    if (!only || strstr("int fp4", only)) {
        for (int i = 0; i < 8; i++)
            indexer_scan_int_kernel<<<(n_comp + 31u) / 32u, 128>>>(
                    scores, q_mirror, w, mpool[i % POOL], n_comp, scale);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < ITER; i++)
            indexer_scan_int_kernel<<<(n_comp + 31u) / 32u, 128>>>(
                    scores, q_mirror, w, mpool[i % POOL], n_comp, scale);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms_eager;
        CK(cudaEventElapsedTime(&ms_eager, e0, e1));

        indexer_scan_int_kernel<<<(n_comp + 31u) / 32u, 128>>>(
                scores, q_mirror, w, mpool[0], n_comp, scale);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(sgot, scores, (uint64_t)n_comp * sizeof(float), cudaMemcpyDeviceToHost));

        const uint32_t nref = n_comp < 8192u ? n_comp : 8192u;
        float *hq = (float *)malloc(64 * 128 * 4);
        float *hw = (float *)malloc(64 * 4);
        float *hk = (float *)malloc((uint64_t)nref * 128 * 4);
        CK(cudaMemcpy(hq, q, 64 * 128 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hw, w, 64 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hk, kpool[0], (uint64_t)nref * 128 * 4, cudaMemcpyDeviceToHost));
        double dmax_int = 0, dmax_ref = 0, smax = 0;
        for (uint32_t c = 0; c < nref; c++) {
            double total = 0;
            for (int h = 0; h < 64; h++) {
                double dot = 0;
                for (int d = 0; d < 128; d++)
                    dot += (double)hq[h * 128 + d] * hk[(uint64_t)c * 128 + d];
                if (dot > 0) total += dot * hw[h] * (double)scale;
            }
            double di = fabs((double)sgot[c] - total);
            double dr = fabs((double)sref[c] - total);
            if (di > dmax_int) dmax_int = di;
            if (dr > dmax_ref) dmax_ref = dr;
            if (fabs(total) > smax) smax = fabs(total);
        }
        printf("%-20s %10.2f          -    (accuracy vs f64 over %u pos, |score|max %.3g:\n"
               "%-20s   int-kernel max err %.3g, f32-tree max err %.3g)\n",
               "int fp4 dp4a      ", ms_eager * 1000.0 / ITER, nref, smax, "", dmax_int, dmax_ref);
        free(hq); free(hw); free(hk);
    }

    printf("\nf32-GB/s = effective bandwidth in f32-cache terms (comparable across arms);\n"
           "read-GB/s = bytes actually read. graph column is the production-relevant one.\n");
    return 0;
}
