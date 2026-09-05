/* FlashAttention-2 style fused decode attention for DeepSeek MLA.
 *
 * The decode shape here is one query token, 64 query heads, and a single
 * shared 512-wide latent KV row per position (MLA absorbs W_UK/W_UV, so V is
 * the same tensor as K).  That makes S = Q*K^T a real GEMM with the *heads*
 * on the M axis, which is what lets tensor cores do the work that the
 * CUDA-core kernel does with per-row shuffle reductions.
 *
 * FA2 properties implemented here:
 *   - one pass over KV: scores are never written to memory, softmax is online
 *   - warp-level work partitioning with no cross-warp reduction in the PV
 *     phase (each warp owns a disjoint slice of the 512 output dims)
 *   - the output accumulator is rescaled, never re-divided; the single
 *     normalisation happens once in the combine kernel
 *   - KV rows are split across blocks (flash-decoding) so the grid fills the
 *     SMs even though a decode step has only one query row
 *
 * Fragment/swizzle conventions are the ones already used by the token-tile
 * prefill kernel in ns4_cuda.cu.
 *
 * Included by ns4_cuda.cu after `fd_row_geom` / `fd_compute_geom` (which it
 * expects in scope) and launched from attention_flash_decode_launch for every
 * decode caller; on by default, NS4_CUDA_FLASH2_ATTN=0 falls back to the
 * CUDA-core flash kernel.  bench/decode-profiling/fd2_bench.cu exercises it
 * standalone.  Measured: 12.5us kernel + 7.1us combine vs 25us for the
 * score-split path it replaced, rel_l2 2.7e-4 against the fp32 reference
 * (bench/DECODE_LAUNCH_BUDGET.md section 4).
 */

#include <cuda_fp16.h>
#include <mma.h>

static constexpr uint32_t kFD2Dim         = 512u;
static constexpr uint32_t kFD2ChunkHalves = 8u;    /* 16B ldmatrix granule */
static constexpr uint32_t kFD2Chunks      = kFD2Dim / kFD2ChunkHalves;  /* 64 */
static constexpr uint32_t kFD2Heads       = 16u;   /* one MMA M tile */
static constexpr uint32_t kFD2Tile        = 16u;   /* KV rows staged per pass */
static constexpr uint32_t kFD2Warps       = 4u;
static constexpr uint32_t kFD2Threads     = kFD2Warps * 32u;
static constexpr uint32_t kFD2PStride     = kFD2Tile + 8u;
static constexpr uint32_t kFD2PvNTiles    = kFD2Dim / 8u / kFD2Warps;  /* 16 */

__device__ __forceinline__ unsigned fd2_smem_addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

/* XOR swizzle on the 16B chunk index, identical to the prefill kernel's. */
__device__ __forceinline__ uint32_t fd2_off_bytes(uint32_t row, uint32_t c) {
    return (row * kFD2Chunks + (c ^ (row & 7u))) * 16u;
}

__device__ __forceinline__ void fd2_ldmatrix_x4(uint32_t (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void fd2_ldmatrix_x2(uint32_t (&r)[2], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void fd2_ldmatrix_x2_trans(uint32_t (&r)[2], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void fd2_mma(float *d, const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
    uint32_t d0 = __float_as_uint(d[0]), d1 = __float_as_uint(d[1]);
    uint32_t d2 = __float_as_uint(d[2]), d3 = __float_as_uint(d[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0); d[1] = __uint_as_float(d1);
    d[2] = __uint_as_float(d2); d[3] = __uint_as_float(d3);
}

/* C-fragment element (i = M index, j = N index) held by `lane` at slot l. */
__device__ __forceinline__ uint32_t fd2_c_i(uint32_t lane, uint32_t l) {
    return ((l >> 1) << 3) + (lane >> 2);
}
__device__ __forceinline__ uint32_t fd2_c_j(uint32_t lane, uint32_t l) {
    return ((lane & 3u) << 1) + (l & 1u);
}

/* Convert 8 floats (two float4) to halves and store one swizzled 16B chunk. */
__device__ __forceinline__ void fd2_store_chunk_f4(half *base, uint32_t row,
                                                   uint32_t c, float4 v0, float4 v1) {
    const float sv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    half tmp[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) tmp[i] = __float2half(sv[i]);
    *reinterpret_cast<int4 *>(reinterpret_cast<char *>(base) + fd2_off_bytes(row, c)) =
        *reinterpret_cast<const int4 *>(tmp);
}

/* Convert 8 consecutive floats to halves and store one swizzled 16B chunk. */
__device__ __forceinline__ void fd2_store_chunk(half *base, uint32_t row,
                                                uint32_t c, const float *src) {
    /* Two float4 loads (rows are whole 512-float lines and chunks 32-byte
     * aligned) instead of eight scalars; same values, same conversions. */
    const float4 *s4 = reinterpret_cast<const float4 *>(src);
    fd2_store_chunk_f4(base, row, c, s4[0], s4[1]);
}

__device__ __forceinline__ void fd2_store_chunk_zero(half *base, uint32_t row,
                                                     uint32_t c) {
    *reinterpret_cast<int4 *>(reinterpret_cast<char *>(base) + fd2_off_bytes(row, c)) =
        make_int4(0, 0, 0, 0);
}

/* Combine the per-split partials.  The shipped combine runs one block per
 * head (64 blocks) and walks all 512 dims inside it, which leaves most of the
 * GPU idle for what is a pure streaming reduction; this one adds a dim axis to
 * the grid so the reduction is spread over n_head * (512/kFD2CombDims) blocks. */
static constexpr uint32_t kFD2CombDims = 128u;

__global__ static void attention_flash2_combine_kernel(
        float *heads,
        const float *sinks,
        const float *part_o,
        const float2 *part_ml,
        uint32_t n_head,
        uint32_t n_split) {
    const uint32_t head = blockIdx.x;
    const uint32_t d0   = blockIdx.y * kFD2CombDims;
    if (head >= n_head) return;

    extern __shared__ float fd2_w[];
    float m_max = sinks[head];
    for (uint32_t s = 0; s < n_split; s++)
        m_max = fmaxf(m_max, part_ml[(uint64_t)s * n_head + head].x);

    float denom = __expf(sinks[head] - m_max);
    for (uint32_t s = threadIdx.x; s < n_split; s += blockDim.x) {
        const float2 ml = part_ml[(uint64_t)s * n_head + head];
        fd2_w[s] = ml.x == -INFINITY ? 0.0f : __expf(ml.x - m_max);
    }
    for (uint32_t s = 0; s < n_split; s++) {
        const float2 ml = part_ml[(uint64_t)s * n_head + head];
        denom += ml.y * (ml.x == -INFINITY ? 0.0f : __expf(ml.x - m_max));
    }
    __syncthreads();

    const float inv = 1.0f / denom;
    const uint32_t d = d0 + threadIdx.x;
    if (d < kFD2Dim) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_split; s++)
            acc += part_o[((uint64_t)s * n_head + head) * kFD2Dim + d] * fd2_w[s];
        heads[(uint64_t)head * kFD2Dim + d] = acc * inv;
    }
}

__global__ static void __launch_bounds__(kFD2Threads, 2)
attention_flash2_decode_kernel(
        float *part_o,          /* [n_split][n_head][512] unnormalised */
        float2 *part_ml,        /* [n_split][n_head]      (max, denom)  */
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,    /* NULL for the dense (non-indexed) path */
        uint32_t n_head,
        uint32_t n_split,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio) {
    __shared__ half  qs[kFD2Heads * kFD2Dim];
    __shared__ half  ks[kFD2Tile * kFD2Dim];
    __shared__ float sp[2][kFD2Heads][kFD2Tile];   /* QK partials, per k-half */
    __shared__ half  ps[kFD2Heads * kFD2PStride];
    __shared__ float m_s[kFD2Heads], l_s[kFD2Heads], a_s[kFD2Heads];
    __shared__ uint32_t valid_s[kFD2Tile];

    const uint32_t tid   = threadIdx.x;
    const uint32_t lane  = tid & 31u;
    const uint32_t warp  = tid >> 5u;
    const uint32_t head0 = blockIdx.x * kFD2Heads;
    const uint32_t split = blockIdx.y;

    const fd_row_geom geom = fd_compute_geom(
        pos0, n_raw, n_comp, top_k, window, ratio, topk != NULL);

    const uint32_t tiles_total = (geom.n_score + kFD2Tile - 1u) / kFD2Tile;
    const uint32_t tiles_per   = (tiles_total + n_split - 1u) / n_split;
    const uint32_t row_begin   = split * tiles_per * kFD2Tile;
    uint32_t row_end = row_begin + tiles_per * kFD2Tile;
    if (row_end > geom.n_score) row_end = geom.n_score;

    /* ---- stage Q for this block's 16 heads, once ---- */
    for (uint32_t idx = tid; idx < kFD2Heads * kFD2Chunks; idx += kFD2Threads) {
        const uint32_t h = idx / kFD2Chunks;
        const uint32_t c = idx % kFD2Chunks;
        const uint32_t gh = head0 + h;
        if (gh < n_head) {
            fd2_store_chunk(qs, h, c, q + (uint64_t)gh * kFD2Dim + c * kFD2ChunkHalves);
        } else {
            fd2_store_chunk_zero(qs, h, c);
        }
    }

    float o_acc[kFD2PvNTiles][4];
#pragma unroll
    for (uint32_t t = 0; t < kFD2PvNTiles; t++)
#pragma unroll
        for (uint32_t l = 0; l < 4u; l++) o_acc[t][l] = 0.0f;

    if (tid < kFD2Heads) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }

    const float scale = rsqrtf((float)kFD2Dim);
    const uint32_t qk_ntile = warp >> 1u;          /* 0..1 : 8 rows each   */
    const uint32_t qk_khalf = warp & 1u;           /* 0..1 : 256 dims each */

    for (uint32_t row0 = row_begin; row0 < row_end; row0 += kFD2Tile) {
        const uint32_t nr = row_end - row0 < kFD2Tile ? row_end - row0 : kFD2Tile;
        __syncthreads();

        /* ---- stage KV rows (raw ring, then top-k / dense compressed) ---- */
        for (uint32_t idx = tid; idx < kFD2Tile * kFD2Chunks; idx += kFD2Threads) {
            const uint32_t r = idx / kFD2Chunks;
            const uint32_t c = idx % kFD2Chunks;
            uint32_t ok = 0u;
            const float *base = NULL;
            uint32_t src = 0u;
            if (r < nr) {
                const uint32_t gr = row0 + r;
                if (gr < geom.raw_count) {
                    src = (raw_start + geom.raw_first_idx + gr) % raw_cap;
                    base = raw_kv;
                    ok = 1u;
                } else {
                    const uint32_t slot = gr - geom.raw_count;
                    const int32_t cc = topk ? topk[slot] : (int32_t)slot;
                    ok = (cc >= 0 && (uint32_t)cc < geom.visible_comp) ? 1u : 0u;
                    src = ok ? (uint32_t)cc : 0u;
                    base = comp_kv;
                }
            }
            if (c == 0u) valid_s[r] = ok;
            if (ok) {
                fd2_store_chunk(ks, r, c, base + (uint64_t)src * kFD2Dim + c * kFD2ChunkHalves);
            } else {
                fd2_store_chunk_zero(ks, r, c);
            }
        }
        /* zero both score partial planes before the MMA writes land */
        for (uint32_t idx = tid; idx < 2u * kFD2Heads * kFD2Tile; idx += kFD2Threads) {
            sp[idx / (kFD2Heads * kFD2Tile)][(idx / kFD2Tile) % kFD2Heads][idx % kFD2Tile] = 0.0f;
        }
        __syncthreads();

        /* ---- S = Q * K^T on tensor cores ---- */
        {
            float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const unsigned q_base = fd2_smem_addr(qs);
            const unsigned k_base = fd2_smem_addr(ks);
            const uint32_t a_row = lane & 15u;              /* head    */
            const uint32_t a_half = lane >> 4u;             /* k half  */
            const uint32_t b_row = qk_ntile * 8u + (lane & 7u);   /* kv row */
            const uint32_t b_half = (lane >> 3u) & 1u;
            const uint32_t ksteps = (kFD2Dim / 16u) / 2u;   /* 16 per k-half */
#pragma unroll 4
            for (uint32_t kt = 0; kt < ksteps; kt++) {
                const uint32_t kchunk = (qk_khalf * ksteps + kt) * 2u;
                uint32_t a[4], b[2];
                fd2_ldmatrix_x4(a, q_base + fd2_off_bytes(a_row, kchunk + a_half));
                fd2_ldmatrix_x2(b, k_base + fd2_off_bytes(b_row, kchunk + b_half));
                fd2_mma(s_frag, a, b);
            }
#pragma unroll
            for (uint32_t l = 0; l < 4u; l++) {
                const uint32_t i = fd2_c_i(lane, l);
                const uint32_t j = qk_ntile * 8u + fd2_c_j(lane, l);
                sp[qk_khalf][i][j] = s_frag[l];
            }
        }
        __syncthreads();

        /* ---- online softmax, one lane per head ---- */
        if (warp == 0u && lane < kFD2Heads) {
            const uint32_t h = lane;
            float tile_max = -INFINITY;
            float s_row[kFD2Tile];
#pragma unroll
            for (uint32_t r = 0; r < kFD2Tile; r++) {
                float v = (r < nr && valid_s[r])
                              ? (sp[0][h][r] + sp[1][h][r]) * scale : -INFINITY;
                s_row[r] = v;
                tile_max = fmaxf(tile_max, v);
            }
            const float m_old = m_s[h];
            const float m_new = fmaxf(m_old, tile_max);
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            float l_add = 0.0f;
            if (tile_max == -INFINITY) {
                a_s[h] = 1.0f;                     /* nothing to add */
#pragma unroll
                for (uint32_t r = 0; r < kFD2Tile; r++) ps[h * kFD2PStride + r] = __float2half(0.0f);
            } else {
#pragma unroll
                for (uint32_t r = 0; r < kFD2Tile; r++) {
                    const float p = (s_row[r] == -INFINITY) ? 0.0f : __expf(s_row[r] - m_new);
                    l_add += p;
                    ps[h * kFD2PStride + r] = __float2half(p);
                }
                a_s[h] = alpha;
                m_s[h] = m_new;
                l_s[h] = l_s[h] * alpha + l_add;
            }
        }
        __syncthreads();

        /* ---- rescale the running output, then O += P * V ---- */
        {
            const unsigned p_base = fd2_smem_addr(ps);
            const unsigned k_base = fd2_smem_addr(ks);
            uint32_t a[4];
            fd2_ldmatrix_x4(a, p_base +
                            (unsigned)(((lane & 15u) * kFD2PStride + (lane >> 4u) * 8u) * sizeof(half)));
            const uint32_t v_row = lane & 15u;
#pragma unroll
            for (uint32_t t = 0; t < kFD2PvNTiles; t++) {
                const uint32_t nchunk = warp * kFD2PvNTiles + t;
#pragma unroll
                for (uint32_t l = 0; l < 4u; l++) {
                    o_acc[t][l] *= a_s[fd2_c_i(lane, l)];
                }
                uint32_t b[2];
                fd2_ldmatrix_x2_trans(b, k_base + fd2_off_bytes(v_row, nchunk));
                fd2_mma(o_acc[t], a, b);
            }
        }
    }

    /* ---- write this split's unnormalised partial ---- */
    __syncthreads();
#pragma unroll
    for (uint32_t t = 0; t < kFD2PvNTiles; t++) {
        const uint32_t nchunk = warp * kFD2PvNTiles + t;
#pragma unroll
        for (uint32_t l = 0; l < 4u; l++) {
            const uint32_t h = fd2_c_i(lane, l);
            const uint32_t d = nchunk * 8u + fd2_c_j(lane, l);
            const uint32_t gh = head0 + h;
            if (gh < n_head) {
                part_o[((uint64_t)split * n_head + gh) * kFD2Dim + d] = o_acc[t][l];
            }
        }
    }
    if (tid < kFD2Heads && head0 + tid < n_head) {
        part_ml[(uint64_t)split * n_head + head0 + tid] =
            make_float2(m_s[tid], l_s[tid]);
    }
}
