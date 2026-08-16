/* The DSA indexer's decode top-k: shipped chunked-bitonic vs a radix select.
 *
 * At decode n_tokens is 1, so every shipped path puts one token on one block:
 * indexer_topk_chunk_pow2_kernel<4096> runs a full 4096-element bitonic sort
 * per 4096-score chunk (grid 1 x ceil(n_comp/4096) -- three blocks at ctx
 * 32768) and indexer_topk_merge_pow2_kernel<4096> runs another one on a single
 * block. Four blocks of 188 SMs, 21 times a token, outside the graph islands.
 *
 * The comparator topk_score_better(av,ai,bv,bi) = av > bv || (av==bv && ai<bi)
 * is a strict total order on (score, index) and indices are unique, so the
 * top-k list AND its order are unique. Any exact algorithm is therefore
 * bit-identical to the shipped one -- which this bench checks with memcmp
 * rather than assuming.
 *
 *   nvcc -O3 -arch=sm_120 -o topk_bench topk_bench.cu
 *   ./topk_bench                 # live decode shapes
 *   ./topk_bench --n=109002      # the ctx 436000 frontier
 *   ./topk_bench --dist=narrow   # scores inside one binade: worst case for
 *                                # a radix select's first pass
 *
 * The baseline kernels are #included verbatim from ds4_cuda.cu so the
 * comparison cannot drift:
 *
 *   { sed -n '14052,14054p' ../../ds4_cuda.cu; echo;
 *     sed -n '14495,14686p' ../../ds4_cuda.cu; } > topk_baseline.inc
 *
 * (line numbers as of the radix-select commit's parent; re-extract if they
 * move). Each row is timed graph-amortised -- reps captured as one graph, so
 * the ~6 us WDDM submission floor costs ~0.8 -- and eager, which is how decode
 * actually issues these. Read a decode win off the eager column.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

enum { DS4_CUDA_TOPK_MERGE_GROUP = 8u };

#include "topk_baseline.inc"

/* ---------------------------------------------------------------- candidate
 *
 * Exact top-k by radix select on a sortable key, then one bounded bitonic
 * sort. One block per token, as the shipped kernels also are, but the work is
 * O(passes * n_comp) counting instead of O(n_comp log^2 4096) compare-exchange.
 */

/* Monotone float -> uint32: strictly increasing in the float order, so
 * comparing keys compares scores. -INF -> 0x007fffff, +INF -> 0xff800000. */
__device__ __forceinline__ static uint32_t topk_sortable_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* (score, index) packed so that a plain 64-bit > is exactly
 * topk_score_better: the score key dominates, and ~index makes the SMALLER
 * index compare greater on a tie. */
__device__ __forceinline__ static uint64_t topk_composite(uint32_t key, uint32_t idx) {
    return ((uint64_t)key << 32) | (uint64_t)(~idx);
}

__device__ __forceinline__ static uint32_t topk_composite_index(uint64_t c) {
    return ~(uint32_t)(c & 0xffffffffu);
}

/* Descending bitonic sort of n (a power of two) 64-bit composites. Same stage
 * structure as the shipped kernels, so the surviving order matches. */
template <uint32_t NT>
__device__ static void topk_bitonic_sort_desc(uint64_t *buf, uint32_t n, uint32_t tid) {
    for (uint32_t k = 2u; k <= n; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < n; i += NT) {
                const uint32_t o = i ^ j;
                if (o > i) {
                    const uint64_t a = buf[i], b = buf[o];
                    const bool swap = ((i & k) == 0u) ? (b > a) : (a > b);
                    if (swap) { buf[i] = b; buf[o] = a; }
                }
            }
            __syncthreads();
        }
    }
}

template <uint32_t CAP, uint32_t NT>
__global__ static void indexer_topk_radix_select_kernel(
        uint32_t *__restrict__ selected,
        const float *__restrict__ scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const float *__restrict__ row = scores + (uint64_t)t * n_comp;
    uint32_t *__restrict__ out = selected + (uint64_t)t * top_k;

    __shared__ uint32_t hist[256];
    __shared__ uint64_t buf[CAP];
    __shared__ uint32_t s_digit, s_above, s_ndigit, s_count;

    /* Radix select: narrow a 32-bit key prefix one byte at a time until the
     * elements at or above it fit the sort buffer. `above` counts elements
     * strictly greater than the prefix, `need` how many of the equal-prefix
     * ones are still wanted; above + need == top_k throughout. */
    uint32_t prefix = 0u, hi_mask = 0u, above = 0u, need = top_k, n_digit = 0u;
    int pass = 0;
    for (; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (uint32_t i = tid; i < 256u; i += NT) hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = tid; i < n_comp; i += NT) {
            const uint32_t k = topk_sortable_key(row[i]);
            if ((k & hi_mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        /* Suffix-scan the 256 bins from the top and stop where the running
         * count first reaches `need`. One warp: lane l owns the eight bins
         * ending at 255-8l, and a shuffle scan supplies each lane's running
         * total, so the crossing lane is unique and finds its own bin. Doing
         * this serially on thread 0 cost ~2.5 us a pass, which at two passes
         * was a fifth of the kernel. */
        if (tid < 32u) {
            uint32_t local[8], sum = 0u;
            for (uint32_t q = 0u; q < 8u; ++q) {
                local[q] = hist[255u - (tid * 8u + q)];
                sum += local[q];
            }
            uint32_t v = sum;
            for (uint32_t off = 1u; off < 32u; off <<= 1u) {
                const uint32_t o = __shfl_up_sync(0xffffffffu, v, off);
                if (tid >= off) v += o;
            }
            const uint32_t excl = v - sum;
            if (excl < need && excl + sum >= need) {
                uint32_t acc = excl;
                for (uint32_t q = 0u; q < 8u; ++q) {
                    if (acc + local[q] >= need) {
                        s_digit = 255u - (tid * 8u + q);
                        s_above = acc;
                        s_ndigit = local[q];
                        break;
                    }
                    acc += local[q];
                }
            }
        }
        __syncthreads();
        above  += s_above;
        need   -= s_above;
        prefix |= s_digit << shift;
        hi_mask = (shift == 0) ? 0xffffffffu : (hi_mask | (0xffu << shift));
        n_digit = s_ndigit;
        if (above + n_digit <= CAP) break;
    }

    /* Common exit: every element with key >= prefix fits the buffer, so one
     * sort settles both the selection and the tie order. */
    if (above + n_digit <= CAP) {
        if (tid == 0) s_count = 0u;
        __syncthreads();
        for (uint32_t i = tid; i < n_comp; i += NT) {
            const uint32_t k = topk_sortable_key(row[i]);
            if (k >= prefix) {
                const uint32_t p = atomicAdd(&s_count, 1u);
                if (p < CAP) buf[p] = topk_composite(k, i);
            }
        }
        __syncthreads();
        uint32_t m = s_count < CAP ? s_count : CAP;
        uint32_t m2 = 1u;
        while (m2 < m) m2 <<= 1u;
        if (m2 < top_k) m2 = top_k;
        for (uint32_t i = m + tid; i < m2; i += NT) buf[i] = 0ull;
        __syncthreads();
        topk_bitonic_sort_desc<NT>(buf, m2, tid);
        for (uint32_t i = tid; i < top_k; i += NT) out[i] = topk_composite_index(buf[i]);
        return;
    }

    /* Degenerate exit: more than CAP elements share the whole 32-bit key, so
     * the cut runs through a block of identical scores and the tie-break is
     * purely by index. `above` (< top_k) of them are settled by sort; the rest
     * are the `need` smallest indices with key == prefix, which is also their
     * output order since equal scores rank by ascending index. */
    if (tid == 0) s_count = 0u;
    __syncthreads();
    for (uint32_t i = tid; i < n_comp; i += NT) {
        const uint32_t k = topk_sortable_key(row[i]);
        if (k > prefix) {
            const uint32_t p = atomicAdd(&s_count, 1u);
            if (p < CAP) buf[p] = topk_composite(k, i);
        }
    }
    __syncthreads();
    {
        uint32_t m2 = 1u;
        while (m2 < above) m2 <<= 1u;
        if (m2 < 2u) m2 = 2u;
        for (uint32_t i = above + tid; i < m2; i += NT) buf[i] = 0ull;
        __syncthreads();
        topk_bitonic_sort_desc<NT>(buf, m2, tid);
    }
    for (uint32_t i = tid; i < above; i += NT) out[i] = topk_composite_index(buf[i]);

    __shared__ uint32_t s_warp[NT / 32u > 0u ? NT / 32u : 1u];
    __shared__ uint32_t s_base, s_tile;
    if (tid == 0) { s_base = 0u; s_tile = 0u; }
    __syncthreads();
    for (uint32_t tile = 0u; tile < n_comp; tile += NT) {
        const uint32_t i = tile + tid;
        const uint32_t pred =
            (i < n_comp && topk_sortable_key(row[i]) == prefix) ? 1u : 0u;
        const uint32_t lane = tid & 31u, warp = tid >> 5u;
        const uint32_t mask = __ballot_sync(0xffffffffu, pred);
        if (lane == 0u) s_warp[warp] = (uint32_t)__popc(mask);
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0u;
            for (uint32_t w = 0u; w < NT / 32u; ++w) {
                const uint32_t c = s_warp[w]; s_warp[w] = acc; acc += c;
            }
            s_tile = acc;
        }
        __syncthreads();
        const uint32_t rank =
            s_base + s_warp[warp] + (uint32_t)__popc(mask & ((1u << lane) - 1u));
        if (pred && rank < need) out[above + rank] = i;
        __syncthreads();
        if (tid == 0) s_base += s_tile;
        __syncthreads();
        if (s_base >= need) break;
    }
}

/* ------------------------------------------------------------------ driver */

static void fill_scores(std::vector<float> &v, const char *dist, unsigned seed) {
    uint64_t s = 0x9e3779b97f4a7c15ull ^ seed;
    auto next = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (size_t i = 0; i < v.size(); i++) {
        const float u = (float)((next() >> 11) * (1.0 / 9007199254740992.0));
        if (!strcmp(dist, "narrow")) {
            v[i] = 1.0f + u;                       /* one binade: worst case */
        } else if (!strcmp(dist, "normal")) {
            const float u2 = (float)((next() >> 11) * (1.0 / 9007199254740992.0));
            v[i] = sqrtf(-2.0f * logf(u + 1e-9f)) * cosf(6.2831853f * u2) * 8.0f;
        } else if (!strcmp(dist, "tied")) {
            v[i] = (float)((int)(u * 4.0f));       /* 4 distinct values */
        } else {
            v[i] = (u - 0.5f) * 40.0f;             /* wide */
        }
    }
}

struct Timing { double graph_us, eager_us; };

template <typename F>
static Timing time_it(F launch, int reps) {
    cudaStream_t st; CUDA_OK(cudaStreamCreate(&st));
    for (int i = 0; i < 5; i++) launch(st);
    CUDA_OK(cudaStreamSynchronize(st));

    cudaGraph_t g; cudaGraphExec_t ge;
    CUDA_OK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < reps; i++) launch(st);
    CUDA_OK(cudaStreamEndCapture(st, &g));
    CUDA_OK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    cudaEvent_t a, b; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b));
    CUDA_OK(cudaGraphLaunch(ge, st)); CUDA_OK(cudaStreamSynchronize(st));
    CUDA_OK(cudaEventRecord(a, st));
    CUDA_OK(cudaGraphLaunch(ge, st));
    CUDA_OK(cudaEventRecord(b, st));
    CUDA_OK(cudaStreamSynchronize(st));
    float gms = 0; CUDA_OK(cudaEventElapsedTime(&gms, a, b));

    CUDA_OK(cudaEventRecord(a, st));
    for (int i = 0; i < reps; i++) launch(st);
    CUDA_OK(cudaEventRecord(b, st));
    CUDA_OK(cudaStreamSynchronize(st));
    float ems = 0; CUDA_OK(cudaEventElapsedTime(&ems, a, b));

    CUDA_OK(cudaGraphExecDestroy(ge)); CUDA_OK(cudaGraphDestroy(g));
    CUDA_OK(cudaEventDestroy(a)); CUDA_OK(cudaEventDestroy(b));
    CUDA_OK(cudaStreamDestroy(st));
    return { gms * 1000.0 / reps, ems * 1000.0 / reps };
}

int main(int argc, char **argv) {
    uint32_t n_comp = 8202u, top_k = 512u, n_tokens = 1u;
    const char *dist = "wide";
    int reps = 200;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--n=", 4)) n_comp = (uint32_t)atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--k=", 4)) top_k = (uint32_t)atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--tokens=", 9)) n_tokens = (uint32_t)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--dist=", 7)) dist = argv[i] + 7;
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }
    printf("n_comp=%u top_k=%u n_tokens=%u dist=%s\n", n_comp, top_k, n_tokens, dist);

    std::vector<float> h((size_t)n_comp * n_tokens);
    fill_scores(h, dist, 12345u);
    float *d_scores; CUDA_OK(cudaMalloc(&d_scores, h.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_scores, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

    const uint32_t chunk_n = 4096u;
    const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
    const uint32_t cand_stride = n_chunks * top_k;
    uint32_t n_sets = n_chunks;
    uint64_t scratch_per_token = cand_stride;
    while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
        n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
        scratch_per_token += (uint64_t)n_sets * top_k;
    }
    uint32_t *d_scratch;
    CUDA_OK(cudaMalloc(&d_scratch, (size_t)n_tokens * scratch_per_token * sizeof(uint32_t)));
    uint32_t *d_ref, *d_new;
    CUDA_OK(cudaMalloc(&d_ref, (size_t)n_tokens * top_k * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_new, (size_t)n_tokens * top_k * sizeof(uint32_t)));

    auto launch_ref = [&](cudaStream_t st) {
        uint32_t *cur = d_scratch;
        uint32_t sets = n_chunks, cur_stride = cand_stride;
        dim3 gc(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<gc, 1024, 0, st>>>(
                cur, d_scores, n_comp, n_tokens, top_k, cand_stride);
        while (sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets =
                (sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 gm(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<gm, 1024, 0, st>>>(
                    next, cur, d_scores, n_comp, n_tokens, top_k, sets,
                    DS4_CUDA_TOPK_MERGE_GROUP, cur_stride, next_stride);
            cur = next; sets = next_sets; cur_stride = next_stride;
        }
        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024, 0, st>>>(
                d_ref, cur, d_scores, n_comp, n_tokens, top_k, sets * top_k, cur_stride);
    };

    printf("%-34s %10s %10s\n", "variant", "graph_us", "eager_us");
    Timing tr = time_it(launch_ref, reps);
    printf("%-34s %10.2f %10.2f\n", "shipped chunk+merge<4096>", tr.graph_us, tr.eager_us);

    std::vector<uint32_t> ref((size_t)n_tokens * top_k), got(ref.size());
    launch_ref(0); CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(ref.data(), d_ref, ref.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

#define TRY(CAP, NT)                                                              \
    do {                                                                          \
        auto lf = [&](cudaStream_t st) {                                          \
            indexer_topk_radix_select_kernel<CAP, NT><<<n_tokens, NT, 0, st>>>(   \
                    d_new, d_scores, n_comp, n_tokens, top_k);                    \
        };                                                                        \
        CUDA_OK(cudaMemset(d_new, 0xff, got.size() * sizeof(uint32_t)));          \
        lf(0);                                                                    \
        cudaError_t e = cudaDeviceSynchronize();                                  \
        if (e != cudaSuccess) { printf("%-34s  LAUNCH FAIL %s\n",                 \
                "radix<" #CAP "," #NT ">", cudaGetErrorString(e)); break; }        \
        CUDA_OK(cudaMemcpy(got.data(), d_new, got.size() * sizeof(uint32_t),      \
                           cudaMemcpyDeviceToHost));                              \
        const bool same = memcmp(ref.data(), got.data(),                          \
                                 got.size() * sizeof(uint32_t)) == 0;             \
        Timing tt = time_it(lf, reps);                                            \
        printf("%-34s %10.2f %10.2f   %s  %.2fx eager\n", "radix<" #CAP "," #NT ">",\
               tt.graph_us, tt.eager_us, same ? "identical" : "*** DIFFERS ***",   \
               tr.eager_us / tt.eager_us);                                        \
    } while (0)

    TRY(2048, 256);
    TRY(2048, 512);
    TRY(2048, 1024);
    TRY(1024, 512);
    TRY(1024, 1024);
    TRY(4096, 1024);
#undef TRY

    /* The floor: the same launch shape returning immediately. */
    auto empty = [&](cudaStream_t st) {
        indexer_topk_radix_select_kernel<2048, 1024><<<n_tokens, 1024, 0, st>>>(
                d_new, d_scores, n_comp, 0u, top_k);
    };
    Timing te = time_it(empty, reps);
    printf("%-34s %10.2f %10.2f\n", "empty launch (floor)", te.graph_us, te.eager_us);
    return 0;
}
