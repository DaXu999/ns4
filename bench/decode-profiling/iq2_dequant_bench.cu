/* Does a combined (grid, sign) LUT beat the live IQ2_XXS sign reconstruction?
 *
 * ns4-iq2-moe-dequant-is-alu-bound established that
 * moe_gate_up_mid_decode_lut_qwarp32_kernel runs at 348 GB/s (21% of this
 * box's 1642 GB/s roofline) and that neither the weight access pattern nor
 * shared-memory bank conflicts explain it: the cost is the ALU sequence in
 * dev_iq2_i8x8_lut, run 32 times per 256-element block.
 *
 * The live sequence per 8 weights is
 *
 *     signs[s7]                       // LDS.U8, s7 data-dependent over 0..127
 *     popc / xor / imad               // parity fix + byte broadcast
 *     2x __vcmpne4                    // 4 sign bits -> 4 byte masks, twice
 *     grid[g8]                        // LDS.64, g8 data-dependent over 0..255
 *     2x xor + 2x __vsub4             // negate the selected bytes
 *     2x __dp4a
 *
 * The (g8, s7) pair has only 256*128 = 32768 values, so the eight signed
 * int8s can be fully precomputed: 32768 * 8 B = 256 KiB.  That does not fit
 * in shared memory (100 KB/block cap), so the combined table has to live in
 * global memory and be served by L1/L2 -- trading ~13 ALU ops for a
 * data-dependent 8-byte gather over a 256 KiB working set.  Whether that is
 * a win is an empirical question about L1 wavefront throughput, which is what
 * this measures.
 *
 * Also measured, because they are the natural fallbacks if the gather loses:
 *   - the same sequence minus its redundant popcount (ksigns entries already
 *     carry even parity, so the parity fix is dead work);
 *   - a split table, grid_idx x sign_nibble -> one signed int32, 16 KiB per
 *     half, 32 KiB total, which does fit in shared memory.
 *
 * The kernel body is the live one (rows, lanes, tiles, block stride and
 * accumulation order all identical); only dev_iq2_i8x8_lut changes, so every
 * variant reads exactly the same bytes in the same order and must produce
 * bit-identical output.  That is checked, not assumed.
 *
 * Shape is DeepSeek-V4-Flash decode: expert_mid_dim 2048, xq_blocks 16,
 * n_expert 6, one token -> qgrid (64, 6) x 256 threads, 25.95 MB of gate+up
 * weights per call.  Experts are drawn from a pool far larger than the 128 MB
 * L2 so the weights come from DRAM as they do in the model.
 *
 *   nvcc -O3 -arch=sm_120 -o iq2_dequant_bench iq2_dequant_bench.cu
 *   ./iq2_dequant_bench
 *   ./iq2_dequant_bench --pool=64      # smaller expert pool (L2-resident)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

#define CUDA_QK_K 256

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "../../src/ns4_iq2_tables_cuda.inc"

/* Host mirrors of the two device __constant__ tables, for building the
 * derived tables and for the CPU reference. */
static uint64_t h_grid[256];
static uint8_t h_ksigns[128];

/* ---------------------------------------------------------------- tables */

/* combined[g * 128 + s7] -> the eight signed int8 values, packed low byte
 * first so that lo32 is weights 0..3 and hi32 is weights 4..7, matching the
 * (w0, w1) split the live code feeds to __dp4a. */
static uint64_t *build_combined(void) {
    uint64_t *t = (uint64_t *)malloc(256ull * 128ull * sizeof(uint64_t));
    for (uint32_t g = 0; g < 256u; g++) {
        const uint64_t gv = h_grid[g];
        for (uint32_t s = 0; s < 128u; s++) {
            const uint32_t k = h_ksigns[s];
            uint64_t out = 0;
            for (uint32_t i = 0; i < 8u; i++) {
                const int32_t v = (int32_t)((gv >> (8u * i)) & 0xffu);
                const int32_t sv = (k >> i) & 1u ? -v : v;
                out |= (uint64_t)(uint8_t)(int8_t)sv << (8u * i);
            }
            t[g * 128u + s] = out;
        }
    }
    return t;
}

/* half[h][g * 16 + n] -> four signed int8 values: half 0 signs grid bytes
 * 0..3 with mask bits 0..3, half 1 signs grid bytes 4..7 with mask bits 4..7
 * (n carries those four bits already shifted down). */
static void build_split(uint32_t *lo, uint32_t *hi) {
    for (uint32_t g = 0; g < 256u; g++) {
        const uint64_t gv = h_grid[g];
        for (uint32_t n = 0; n < 16u; n++) {
            uint32_t o0 = 0, o1 = 0;
            for (uint32_t i = 0; i < 4u; i++) {
                const int32_t a = (int32_t)((gv >> (8u * i)) & 0xffu);
                const int32_t b = (int32_t)((gv >> (8u * (i + 4u))) & 0xffu);
                o0 |= (uint32_t)(uint8_t)(int8_t)((n >> i) & 1u ? -a : a) << (8u * i);
                o1 |= (uint32_t)(uint8_t)(int8_t)((n >> i) & 1u ? -b : b) << (8u * i);
            }
            lo[g * 16u + n] = o0;
            hi[g * 16u + n] = o1;
        }
    }
}

/* Every 4-byte half of every grid entry is one of only 74 distinct magnitude
 * quads (the grid bytes are drawn from {8, 25, 43}), so one 128 x 16 table of
 * signed quads serves both halves: 8 KiB instead of 32, which is the
 * difference between 6 blocks/SM and 2.  The price is a 256-entry uint16 map
 * from grid index to the pair of quad ids, i.e. one extra dependent load. */
static uint32_t build_quad(uint32_t *tab, uint16_t *map) {
    uint32_t quads[128];
    uint32_t nq = 0;
    for (uint32_t g = 0; g < 256u; g++) {
        const uint64_t gv = h_grid[g];
        uint32_t id[2];
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t q = (uint32_t)(gv >> (32u * h));
            uint32_t k = 0;
            for (; k < nq; k++) if (quads[k] == q) break;
            if (k == nq) { quads[nq++] = q; }
            id[h] = k;
        }
        map[g] = (uint16_t)(id[0] | (id[1] << 8));
    }
    for (uint32_t q = 0; q < 128u; q++) {
        const uint32_t qv = q < nq ? quads[q] : 0u;
        for (uint32_t n = 0; n < 16u; n++) {
            uint32_t o = 0;
            for (uint32_t i = 0; i < 4u; i++) {
                const int32_t a = (int32_t)((qv >> (8u * i)) & 0xffu);
                o |= (uint32_t)(uint8_t)(int8_t)((n >> i) & 1u ? -a : a) << (8u * i);
            }
            tab[q * 16u + n] = o;
        }
    }
    return nq;
}

__device__ static const uint64_t *g_combined;
__device__ static const uint64_t *g_combined_half;
__device__ static const uint32_t *g_split_lo;
__device__ static const uint32_t *g_split_hi;
__device__ static const uint32_t *g_quad_tab;
__device__ static const uint16_t *g_quad_map;

/* ---------------------------------------------------------------- variants */

enum {
    V_BASE = 0,       /* verbatim ns4_cuda.cu */
    V_NOPOPC,         /* same, minus the dead parity fix */
    V_COMBINED,       /* 256 KiB combined LUT, global memory */
    V_COMBINED_HALF,  /* 128 KiB, negation symmetry folded out */
    V_SPLIT_KSIGN,    /* 2 x 16 KiB shared, nibbles via the ksigns byte */
    V_SPLIT_POPC,     /* 2 x 16 KiB shared, high nibble via popcount */
    V_QUADMAP,        /* 8 KiB shared quad table + 512 B grid->quad map */
    V_NOSIGN,         /* floor: grid lookup only, sign application deleted */
    V_ALIGNED,        /* base math, aligned-SoA weights: uint2 loads not u16 */
    V_ALIGNED16,      /* same, two 32-weight groups per uint4 load */
    V_ALIGNED16_P4,   /* aligned + 4 extra ALU/lookup: is dequant binding now? */
    V_ALIGNED16_QUAD, /* aligned + the 8 KiB quad table instead of sign math */
    V_PLUS4,          /* base + 4 identity rotations per 8 weights */
    V_PLUS12,         /* base + 12 */
    V_PLUS24,         /* base + 24 */
    V_COUNT
};

static const char *v_name[V_COUNT] = {
    "base (live)",
    "base -popc",
    "combined 256K global",
    "combined 128K global",
    "split 32K shared (ksign)",
    "split 32K shared (popc)",
    "quad 8K shared + map",
    "floor: no sign apply",
    "aligned SoA weights",
    "aligned SoA, uint4 loads",
    "  + 4 ALU/lookup",
    "  + quad 8K shared LUT",
    "base +4 ALU/lookup",
    "base +12 ALU/lookup",
    "base +24 ALU/lookup",
};

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

/* Shared-memory tables a variant needs; unused members cost nothing because
 * each variant is a separate template instantiation. */
#define V_USES_GRID(V) ((V) == V_BASE || (V) == V_NOPOPC || (V) == V_ALIGNED || (V) == V_ALIGNED16 || (V) == V_ALIGNED16_P4 || \
                        (V) == V_PLUS4 || (V) == V_PLUS12 || (V) == V_PLUS24)
#define V_USES_KSIGNS(V) (V_USES_GRID(V) || (V) == V_SPLIT_KSIGN)

template <int V>
struct iq2_smem {
    uint64_t grid[V_USES_GRID(V) ? 256 : 1];
    uint8_t signs[V_USES_KSIGNS(V) ? 128 : 1];
    uint32_t lo[(V == V_SPLIT_KSIGN || V == V_SPLIT_POPC) ? 256 * 16 : 1];
    uint32_t hi[(V == V_SPLIT_KSIGN || V == V_SPLIT_POPC) ? 256 * 16 : 1];
    uint32_t qtab[(V == V_QUADMAP || V == V_ALIGNED16_QUAD) ? 128 * 16 : 1];
    uint16_t qmap[(V == V_QUADMAP || V == V_ALIGNED16_QUAD) ? 256 : 1];
    uint64_t gmag[V == V_NOSIGN ? 256 : 1];
};

template <int V>
__device__ __forceinline__ static void iq2_smem_load(iq2_smem<V> *s) {
    if (V_USES_GRID(V)) {
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s->grid[i] = cuda_iq2xxs_grid[i];
    }
    if (V_USES_KSIGNS(V)) {
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s->signs[i] = cuda_ksigns_iq2xs[i];
    }
    if (V == V_SPLIT_KSIGN || V == V_SPLIT_POPC) {
        for (uint32_t i = threadIdx.x; i < 256u * 16u; i += blockDim.x) {
            s->lo[i] = g_split_lo[i];
            s->hi[i] = g_split_hi[i];
        }
    }
    if (V == V_QUADMAP || V == V_ALIGNED16_QUAD) {
        for (uint32_t i = threadIdx.x; i < 128u * 16u; i += blockDim.x) s->qtab[i] = g_quad_tab[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s->qmap[i] = g_quad_map[i];
    }
    if (V == V_NOSIGN) {
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s->gmag[i] = cuda_iq2xxs_grid[i];
    }
}

template <int V>
__device__ __forceinline__ static void iq2_i8x8(
        const iq2_smem<V> *sm, uint32_t g8, uint32_t s7, int32_t *w0, int32_t *w1) {
    if (V == V_BASE || V == V_ALIGNED || V == V_ALIGNED16) {
        const uint32_t s = dev_unpack_iq2_signs(sm->signs[s7]);
        const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
        const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
        const uint64_t g = sm->grid[g8];
        *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
        *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
    } else if (V == V_NOPOPC) {
        const uint32_t s = (uint32_t)sm->signs[s7] * 0x01010101u;
        const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
        const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
        const uint64_t g = sm->grid[g8];
        *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
        *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
    } else if (V == V_PLUS4 || V == V_PLUS12 || V == V_PLUS24 || V == V_ALIGNED16_P4) {
        const uint32_t s = dev_unpack_iq2_signs(sm->signs[s7]);
        const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
        const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
        const uint64_t g = sm->grid[g8];
        int32_t a = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
        int32_t b = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
        const int extra = (V == V_PLUS4 || V == V_ALIGNED16_P4) ? 1 : V == V_PLUS12 ? 3 : 6;
        for (int e = 0; e < extra; e++) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                a = (int32_t)__funnelshift_l((uint32_t)a, (uint32_t)a, 8);
                b = (int32_t)__funnelshift_l((uint32_t)b, (uint32_t)b, 8);
            }
        }
        *w0 = a;
        *w1 = b;
    } else if (V == V_COMBINED) {
        const uint64_t v = __ldg(g_combined + (g8 << 7) + s7);
        *w0 = (int32_t)(uint32_t)v;
        *w1 = (int32_t)(uint32_t)(v >> 32);
    } else if (V == V_COMBINED_HALF) {
        /* Complementing all eight sign bits preserves even parity, so
         * entry(g, ~s) == -entry(g, s) and half the table is redundant. */
        const uint32_t neg = (s7 >> 6) & 1u;
        const uint32_t m = 0u - neg;
        const uint64_t v = __ldg(g_combined_half + (g8 << 6) + ((s7 ^ m) & 63u));
        *w0 = __vsub4((int32_t)((uint32_t)v ^ m), (int32_t)m);
        *w1 = __vsub4((int32_t)((uint32_t)(v >> 32) ^ m), (int32_t)m);
    } else if (V == V_SPLIT_KSIGN) {
        const uint32_t k = sm->signs[s7];
        *w0 = (int32_t)sm->lo[(g8 << 4) | (k & 15u)];
        *w1 = (int32_t)sm->hi[(g8 << 4) | (k >> 4)];
    } else if (V == V_SPLIT_POPC) {
        const uint32_t hn = ((s7 >> 4) & 7u) | ((__popc(s7) & 1u) << 3u);
        *w0 = (int32_t)sm->lo[(g8 << 4) | (s7 & 15u)];
        *w1 = (int32_t)sm->hi[(g8 << 4) | hn];
    } else if (V == V_QUADMAP || V == V_ALIGNED16_QUAD) {
        const uint32_t c = sm->qmap[g8];
        const uint32_t hn = ((s7 >> 4) & 7u) | ((__popc(s7) & 1u) << 3u);
        *w0 = (int32_t)sm->qtab[((c & 0xffu) << 4) | (s7 & 15u)];
        *w1 = (int32_t)sm->qtab[((c >> 8) << 4) | hn];
    } else {
        /* Floor: the grid lookup with the sign application deleted.  Results
         * are wrong (all magnitudes positive), but the loads, the register
         * pressure and the dp4a chain are the live ones, and no real scheme
         * can beat two shared loads from a 2 KiB table and zero ALU.  What
         * base costs over this is the whole prize on offer. */
        (void)s7;
        const uint64_t g = sm->gmag[g8];
        *w0 = (int32_t)(uint32_t)g;
        *w1 = (int32_t)(uint32_t)(g >> 32);
    }
}

template <int V>
__device__ static float dot_iq2_block(
        const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y, const iq2_smem<V> *sm) {
    const float xd = __half2float(*reinterpret_cast<const __half *>(&x->d));
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        iq2_i8x8<V>(sm, aux0 & 0xffu,         (aux1 >> 0)  & 127u, &w[0], &w[1]);
        iq2_i8x8<V>(sm, (aux0 >> 8) & 0xffu,  (aux1 >> 7)  & 127u, &w[2], &w[3]);
        iq2_i8x8<V>(sm, (aux0 >> 16) & 0xffu, (aux1 >> 14) & 127u, &w[4], &w[5]);
        iq2_i8x8<V>(sm, (aux0 >> 24) & 0xffu, (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

/* Same arithmetic as dot_iq2_block, sourced from the aligned-SoA artifact
 * layout that ns4_repack_build_iq2_aligned already produces for the MMQ path:
 * [__half dq[nblk]][pad 64B][uint2 qs[nblk*8]].  Byte-neutral -- 2 + 64 vs the
 * raw block's 66 -- but qs is 8-byte aligned, so each 32-weight group is one
 * ld.global.v2.b32 instead of four unmergeable ld.global.b16. */
template <int V>
__device__ static float dot_iq2_block_aligned(
        const uint2 *qs, float xd, const cuda_block_q8_K *y, const iq2_smem<V> *sm) {
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    /* Each block's qs starts on a 64-byte boundary, so two groups can come in
     * on one 16-byte load. */
    uint4 pair;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        uint32_t aux0, aux1;
        if (V != V_ALIGNED) {
            if ((ib32 & 1) == 0) pair = *(const uint4 *)(qs + ib32);
            aux0 = (ib32 & 1) ? pair.z : pair.x;
            aux1 = (ib32 & 1) ? pair.w : pair.y;
        } else {
            const uint2 c = qs[ib32];
            aux0 = c.x;
            aux1 = c.y;
        }
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        iq2_i8x8<V>(sm, aux0 & 0xffu,         (aux1 >> 0)  & 127u, &w[0], &w[1]);
        iq2_i8x8<V>(sm, (aux0 >> 8) & 0xffu,  (aux1 >> 7)  & 127u, &w[2], &w[3]);
        iq2_i8x8<V>(sm, (aux0 >> 16) & 0xffu, (aux1 >> 14) & 127u, &w[4], &w[5]);
        iq2_i8x8<V>(sm, (aux0 >> 24) & 0xffu, (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float qwarp_sum(float v) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset, 8);
    return v;
}

#define ROW_TILES 1u
#define ROWS_PER_BLOCK (32u * ROW_TILES)

/* 1536 threads/SM on sm_120 caps this at 6 blocks of 256; pinning it keeps
 * the no-dequant floor from being measured at a different occupancy than the
 * variants it is a floor for. */
template <int V>
__global__ __launch_bounds__(256, 6) static void moe_gate_up_bench_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        const uint2 *qs_gate,
        const __half *dq_gate,
        const uint2 *qs_up,
        const __half *dq_up) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ iq2_smem<V> sm;
    for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
    iq2_smem_load<V>(&sm);
    __syncthreads();
    xqb = sxq;

    for (uint32_t rr = 0; rr < ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * ROWS_PER_BLOCK + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        if (V == V_ALIGNED || V == V_ALIGNED16 || V == V_ALIGNED16_P4 || V == V_ALIGNED16_QUAD) {
            const uint64_t rbase = ((uint64_t)expert * expert_mid_dim + row) * xq_blocks;
            for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                gate += dot_iq2_block_aligned<V>(qs_gate + (rbase + b) * 8u,
                                                 __half2float(dq_gate[rbase + b]), xqb + b, &sm);
                up += dot_iq2_block_aligned<V>(qs_up + (rbase + b) * 8u,
                                               __half2float(dq_up[rbase + b]), xqb + b, &sm);
            }
        } else {
            for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                gate += dot_iq2_block<V>(gr + b, xqb + b, &sm);
                up += dot_iq2_block<V>(ur + b, xqb + b, &sm);
            }
        }
        gate = qwarp_sum(gate);
        up = qwarp_sum(up);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* ---------------------------------------------------------------- harness */

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

typedef void (*launch_fn)(dim3, dim3, float *, const char *, const char *,
                          const cuda_block_q8_K *, const int32_t *, const float *,
                          uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float,
                          const uint2 *, const __half *, const uint2 *, const __half *);

template <int V>
static void launch(dim3 g, dim3 b, float *mid, const char *gw, const char *uw,
                   const cuda_block_q8_K *xq, const int32_t *sel, const float *w,
                   uint64_t eb, uint64_t rb, uint32_t nb, uint32_t md, uint32_t ne, float clamp,
                   const uint2 *qg, const __half *dg, const uint2 *qu, const __half *du) {
    moe_gate_up_bench_kernel<V><<<g, b>>>(mid, gw, uw, xq, sel, w, eb, rb, nb, md, ne, clamp,
                                          qg, dg, qu, du);
}

typedef void (*report_fn)(const char *);

template <int V>
static void report(const char *name) {
    cudaFuncAttributes attr;
    int blocks = 0;
    CHECK(cudaFuncGetAttributes(&attr, moe_gate_up_bench_kernel<V>));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, moe_gate_up_bench_kernel<V>, 256, 0));
    printf("  %-26s regs=%d smem=%zu B  %d blocks/SM\n", name, attr.numRegs, attr.sharedSizeBytes, blocks);
}

int main(int argc, char **argv) {
    uint32_t pool = 160;         /* experts in the rotation pool */
    uint32_t reps = 60;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--pool=", 7)) pool = (uint32_t)atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = (uint32_t)atoi(argv[i] + 7);
    }

    const uint32_t xq_blocks = 16;
    const uint32_t expert_mid_dim = 2048;
    const uint32_t n_expert = 6;
    const uint64_t gate_row_bytes = (uint64_t)xq_blocks * sizeof(cuda_block_iq2_xxs);
    const uint64_t gate_expert_bytes = gate_row_bytes * expert_mid_dim;
    const uint64_t pool_bytes = gate_expert_bytes * pool;

    CHECK(cudaMemcpyFromSymbol(h_grid, cuda_iq2xxs_grid, sizeof(h_grid)));
    CHECK(cudaMemcpyFromSymbol(h_ksigns, cuda_ksigns_iq2xs, sizeof(h_ksigns)));

    printf("shape: mid=%u xq_blocks=%u n_expert=%u row=%llu B expert=%.2f MB\n",
           expert_mid_dim, xq_blocks, n_expert,
           (unsigned long long)gate_row_bytes, (double)gate_expert_bytes / 1048576.0);
    printf("per call: %.2f MB gate+up; pool %u experts = %.0f MB x2\n",
           (double)(gate_expert_bytes * n_expert * 2) / 1048576.0,
           pool, (double)pool_bytes / 1048576.0);

    /* Weights: random IQ2_XXS payloads.  The grid/sign fields are pure
     * indices, so uniform random bytes exercise the LUT exactly like real
     * weights do; only the f16 scale needs to be sane. */
    uint8_t *h_w = (uint8_t *)malloc(pool_bytes);
    for (uint64_t i = 0; i < pool_bytes; i += 4) *(uint32_t *)(h_w + i) = rnd();
    for (uint64_t e = 0; e < pool; e++) {
        for (uint64_t r = 0; r < expert_mid_dim; r++) {
            cuda_block_iq2_xxs *blk = (cuda_block_iq2_xxs *)(h_w + e * gate_expert_bytes + r * gate_row_bytes);
            for (uint32_t b = 0; b < xq_blocks; b++) blk[b].d = 0x2000;   /* ~0.0078 */
        }
    }

    char *d_gate = NULL, *d_up = NULL;
    CHECK(cudaMalloc(&d_gate, pool_bytes));
    CHECK(cudaMalloc(&d_up, pool_bytes));
    CHECK(cudaMemcpy(d_gate, h_w, pool_bytes, cudaMemcpyHostToDevice));
    for (uint64_t i = 0; i < pool_bytes; i += 4) *(uint32_t *)(h_w + i) = rnd();
    for (uint64_t e = 0; e < pool; e++) {
        for (uint64_t r = 0; r < expert_mid_dim; r++) {
            cuda_block_iq2_xxs *blk = (cuda_block_iq2_xxs *)(h_w + e * gate_expert_bytes + r * gate_row_bytes);
            for (uint32_t b = 0; b < xq_blocks; b++) blk[b].d = 0x2000;
        }
    }
    CHECK(cudaMemcpy(d_up, h_w, pool_bytes, cudaMemcpyHostToDevice));
    free(h_w);

    cuda_block_q8_K *h_xq = (cuda_block_q8_K *)calloc(xq_blocks, sizeof(cuda_block_q8_K));
    for (uint32_t b = 0; b < xq_blocks; b++) {
        h_xq[b].d = 0.0123f;
        for (uint32_t i = 0; i < CUDA_QK_K; i++) h_xq[b].qs[i] = (int8_t)(rnd() & 0xffu);
    }
    cuda_block_q8_K *d_xq = NULL;
    CHECK(cudaMalloc(&d_xq, xq_blocks * sizeof(cuda_block_q8_K)));
    CHECK(cudaMemcpy(d_xq, h_xq, xq_blocks * sizeof(cuda_block_q8_K), cudaMemcpyHostToDevice));

    /* One selected[] row per rep, so consecutive calls touch different
     * experts and the weights keep coming from DRAM. */
    int32_t *h_sel = (int32_t *)malloc((size_t)reps * n_expert * sizeof(int32_t));
    for (uint32_t r = 0; r < reps; r++)
        for (uint32_t s = 0; s < n_expert; s++) h_sel[r * n_expert + s] = (int32_t)(rnd() % pool);
    int32_t *d_sel = NULL;
    CHECK(cudaMalloc(&d_sel, (size_t)reps * n_expert * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_sel, h_sel, (size_t)reps * n_expert * sizeof(int32_t), cudaMemcpyHostToDevice));

    float h_wt[16];
    for (uint32_t i = 0; i < n_expert; i++) h_wt[i] = 0.1f + 0.01f * (float)i;
    float *d_wt = NULL;
    CHECK(cudaMalloc(&d_wt, sizeof(h_wt)));
    CHECK(cudaMemcpy(d_wt, h_wt, sizeof(h_wt), cudaMemcpyHostToDevice));

    /* Aligned-SoA mirror of the same pool: dq[nblk] then qs[nblk*8], the
     * layout ns4_repack_build_iq2_aligned already emits.  Byte-neutral, so
     * both arms move identical DRAM traffic; only the load width differs. */
    const uint64_t nblk = (uint64_t)pool * expert_mid_dim * xq_blocks;
    uint2 *d_qs_gate = NULL, *d_qs_up = NULL;
    __half *d_dq_gate = NULL, *d_dq_up = NULL;
    CHECK(cudaMalloc(&d_qs_gate, nblk * 8u * sizeof(uint2)));
    CHECK(cudaMalloc(&d_qs_up, nblk * 8u * sizeof(uint2)));
    CHECK(cudaMalloc(&d_dq_gate, nblk * sizeof(__half)));
    CHECK(cudaMalloc(&d_dq_up, nblk * sizeof(__half)));
    {
        /* Transcribe on device straight from the raw pool, block by block. */
        const uint64_t total = nblk;
        uint2 *hqs = (uint2 *)malloc(1u << 22);
        uint16_t *hdq = (uint16_t *)malloc(1u << 19);
        const uint64_t step = (1u << 22) / (8u * sizeof(uint2));
        uint8_t *raw = (uint8_t *)malloc(pool_bytes);
        for (int which = 0; which < 2; which++) {
            CHECK(cudaMemcpy(raw, which ? (void *)d_up : (void *)d_gate, pool_bytes, cudaMemcpyDeviceToHost));
            for (uint64_t base = 0; base < total; base += step) {
                const uint64_t n = total - base < step ? total - base : step;
                for (uint64_t i = 0; i < n; i++) {
                    const cuda_block_iq2_xxs *blk =
                        (const cuda_block_iq2_xxs *)(raw + (base + i) * 66ull);
                    hdq[i] = blk->d;
                    memcpy(&hqs[i * 8u], blk->qs, 64);
                }
                CHECK(cudaMemcpy((which ? d_qs_up : d_qs_gate) + base * 8u, hqs,
                                 n * 8u * sizeof(uint2), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy((which ? d_dq_up : d_dq_gate) + base, hdq,
                                 n * sizeof(uint16_t), cudaMemcpyHostToDevice));
            }
        }
        free(raw);
        free(hqs);
        free(hdq);
    }

    const size_t mid_elems = (size_t)n_expert * expert_mid_dim;
    float *d_mid = NULL, *d_ref = NULL;
    CHECK(cudaMalloc(&d_mid, mid_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_ref, mid_elems * sizeof(float)));

    /* Derived tables. */
    uint64_t *hc = build_combined();
    uint64_t *dc = NULL;
    CHECK(cudaMalloc(&dc, 256ull * 128ull * sizeof(uint64_t)));
    CHECK(cudaMemcpy(dc, hc, 256ull * 128ull * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(g_combined, &dc, sizeof(dc)));

    uint64_t *hh = (uint64_t *)malloc(256ull * 64ull * sizeof(uint64_t));
    for (uint32_t g = 0; g < 256u; g++)
        for (uint32_t s = 0; s < 64u; s++) hh[g * 64u + s] = hc[g * 128u + s];
    uint64_t *dh = NULL;
    CHECK(cudaMalloc(&dh, 256ull * 64ull * sizeof(uint64_t)));
    CHECK(cudaMemcpy(dh, hh, 256ull * 64ull * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(g_combined_half, &dh, sizeof(dh)));
    free(hh);
    free(hc);

    uint32_t *hqt = (uint32_t *)malloc(128 * 16 * sizeof(uint32_t));
    uint16_t *hqm = (uint16_t *)malloc(256 * sizeof(uint16_t));
    const uint32_t nq = build_quad(hqt, hqm);
    uint32_t *dqt = NULL;
    uint16_t *dqm = NULL;
    CHECK(cudaMalloc(&dqt, 128 * 16 * sizeof(uint32_t)));
    CHECK(cudaMalloc(&dqm, 256 * sizeof(uint16_t)));
    CHECK(cudaMemcpy(dqt, hqt, 128 * 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dqm, hqm, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(g_quad_tab, &dqt, sizeof(dqt)));
    CHECK(cudaMemcpyToSymbol(g_quad_map, &dqm, sizeof(dqm)));
    free(hqt);
    free(hqm);
    printf("distinct magnitude quads: %u of 128 table slots\n", nq);

    uint32_t *hlo = (uint32_t *)malloc(256 * 16 * sizeof(uint32_t));
    uint32_t *hhi = (uint32_t *)malloc(256 * 16 * sizeof(uint32_t));
    build_split(hlo, hhi);
    uint32_t *dlo = NULL, *dhi = NULL;
    CHECK(cudaMalloc(&dlo, 256 * 16 * sizeof(uint32_t)));
    CHECK(cudaMalloc(&dhi, 256 * 16 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dlo, hlo, 256 * 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dhi, hhi, 256 * 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(g_split_lo, &dlo, sizeof(dlo)));
    CHECK(cudaMemcpyToSymbol(g_split_hi, &dhi, sizeof(dhi)));
    free(hlo);
    free(hhi);

    dim3 grid((expert_mid_dim + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK, n_expert, 1);
    dim3 block(256, 1, 1);
    const float clamp = 10.0f;

    launch_fn fns[V_COUNT] = {
        launch<V_BASE>, launch<V_NOPOPC>, launch<V_COMBINED>, launch<V_COMBINED_HALF>,
        launch<V_SPLIT_KSIGN>, launch<V_SPLIT_POPC>, launch<V_QUADMAP>, launch<V_NOSIGN>, launch<V_ALIGNED>, launch<V_ALIGNED16>, launch<V_ALIGNED16_P4>, launch<V_ALIGNED16_QUAD>,
        launch<V_PLUS4>, launch<V_PLUS12>, launch<V_PLUS24>,
    };
    report_fn reps_fn[V_COUNT] = {
        report<V_BASE>, report<V_NOPOPC>, report<V_COMBINED>, report<V_COMBINED_HALF>,
        report<V_SPLIT_KSIGN>, report<V_SPLIT_POPC>, report<V_QUADMAP>, report<V_NOSIGN>, report<V_ALIGNED>, report<V_ALIGNED16>, report<V_ALIGNED16_P4>, report<V_ALIGNED16_QUAD>,
        report<V_PLUS4>, report<V_PLUS12>, report<V_PLUS24>,
    };

    /* Correctness: every variant must be bit-identical to the live one. */
    float *ha = (float *)malloc(mid_elems * sizeof(float));
    float *hb = (float *)malloc(mid_elems * sizeof(float));
    fns[V_BASE](grid, block, d_ref, d_gate, d_up, d_xq, d_sel, d_wt,
                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, clamp,
               d_qs_gate, d_dq_gate, d_qs_up, d_dq_up);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(ha, d_ref, mid_elems * sizeof(float), cudaMemcpyDeviceToHost));
    for (int v = 1; v < V_COUNT; v++) {
        if (v == V_NOSIGN) continue;   /* deliberately wrong weights */
        fns[v](grid, block, d_mid, d_gate, d_up, d_xq, d_sel, d_wt,
               gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, clamp,
               d_qs_gate, d_dq_gate, d_qs_up, d_dq_up);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hb, d_mid, mid_elems * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  %-26s %s\n", v_name[v],
               memcmp(ha, hb, mid_elems * sizeof(float)) == 0 ? "bit-identical to base" : "*** MISMATCH ***");
    }

    for (int v = 0; v < V_COUNT; v++) reps_fn[v](v_name[v]);

    /* Timing: interleave the variants so drift hits them equally. */
    const int rounds = 5;
    double best[V_COUNT], sum[V_COUNT];
    for (int v = 0; v < V_COUNT; v++) { best[v] = 1e30; sum[v] = 0.0; }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    for (int v = 0; v < V_COUNT; v++) {   /* warm up every variant first */
        for (int i = 0; i < 3; i++)
            fns[v](grid, block, d_mid, d_gate, d_up, d_xq, d_sel, d_wt,
                   gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, clamp,
               d_qs_gate, d_dq_gate, d_qs_up, d_dq_up);
    }
    CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < rounds; r++) {
        for (int v = 0; v < V_COUNT; v++) {
            CHECK(cudaEventRecord(e0));
            for (uint32_t i = 0; i < reps; i++)
                fns[v](grid, block, d_mid, d_gate, d_up, d_xq, d_sel + (uint64_t)i * n_expert, d_wt,
                       gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, clamp,
               d_qs_gate, d_dq_gate, d_qs_up, d_dq_up);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = (double)ms * 1000.0 / (double)reps;
            if (us < best[v]) best[v] = us;
            sum[v] += us;
        }
    }

    const double bytes = (double)(gate_expert_bytes * n_expert * 2);
    printf("\n  %-26s %10s %10s %10s %8s\n", "variant", "best us", "mean us", "GB/s", "vs base");
    for (int v = 0; v < V_COUNT; v++) {
        printf("  %-26s %10.2f %10.2f %10.0f %7.2fx\n", v_name[v],
               best[v], sum[v] / rounds, bytes / (best[v] * 1e-6) / 1e9, best[V_BASE] / best[v]);
    }
    printf("\n  roofline on this box: 1642 GB/s\n");
    return 0;
}
