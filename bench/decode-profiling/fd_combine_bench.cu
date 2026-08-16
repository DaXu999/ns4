/* Standalone microbenchmark for the flash-decode *combine* kernel.
 *
 * End-to-end ns4-bench cannot resolve this kernel: it is ~1 of 37 kernels per
 * layer, and against +/-5% run-to-run drift the rewrite showed +2.1% mean with
 * 1 of 3 replicates favouring the old shape -- i.e. nothing.  This times the
 * kernel alone, both as decode issues it (eager, on the ~6 us WDDM submission
 * floor) and with that floor amortised away by a CUDA graph, and asserts the
 * two shapes are bit-identical.
 *
 * Verdict, at the n_split = 27 that every position past ctx 2048 produces:
 * 1.8x L2-warm / 1.6x L2-cold on the kernel, 1.3-2.1 us per layer once the
 * launch floor is paid, ~0.3-0.45% of a token over 43 layers.  Worth keeping
 * because it is free and bit-identical, not because it moves the benchmark.
 *
 * The production kernel comes from fd_combine.inc, extracted verbatim from
 * ns4_cuda.cu so the comparison stays honest:
 *
 *     sed -n '9811,9816p'   ns4_cuda.cu  > fd_combine.inc  # kFD* constants
 *     sed -n '10031,10032p' ns4_cuda.cu >> fd_combine.inc  # kFDCombDims
 *     sed -n '10034,10119p' ns4_cuda.cu >> fd_combine.inc  # the kernel
 *
 * (line numbers as of acf4a5e + the uncommitted combine rewrite; re-extract if
 * the kernel moves).  The control is frozen in this file instead -- it is no
 * longer in ns4_cuda.cu to extract.
 *
 *     nvcc -O3 -arch=sm_120 -o fd_combine_bench fd_combine_bench.cu
 *     ./fd_combine_bench                 # L2-warm, as in real decode
 *     ./fd_combine_bench --bufs=64       # partials evicted from L2
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

#include "fd_combine.inc"

/* The pre-rewrite combine, frozen here as the A/B control.  One block per head
 * (64 on 188 SMs), the dim axis walked by a loop, and all 256 threads re-running
 * both serial n_split loops so that thread 0 alone can store fd_w[]. */
__global__ static void attention_flash_decode_combine_legacy_kernel(
        float *heads,
        const float *sinks,
        const float *part_o,
        const float2 *part_ml,
        uint32_t n_head,
        uint32_t n_split) {
    extern __shared__ float fd_w_legacy[];
    const uint32_t head = blockIdx.x;
    if (head >= n_head) return;
    float m_max = sinks[head];
    for (uint32_t s = 0; s < n_split; s++)
        m_max = fmaxf(m_max, part_ml[(uint64_t)s * n_head + head].x);
    float denom = expf(sinks[head] - m_max);
    for (uint32_t s = 0; s < n_split; s++) {
        const float2 ml = part_ml[(uint64_t)s * n_head + head];
        const float w = ml.x == -INFINITY ? 0.0f : expf(ml.x - m_max);
        if (threadIdx.x == 0u) fd_w_legacy[s] = w;
        denom += ml.y * w;
    }
    __syncthreads();
    const float inv = 1.0f / denom;
    for (uint32_t d = threadIdx.x; d < kFDDim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_split; s++)
            acc += part_o[((uint64_t)s * n_head + head) * kFDDim + d] * fd_w_legacy[s];
        heads[(uint64_t)head * kFDDim + d] = acc * inv;
    }
}

/* Launch floor reference: same grid, no work. */
__global__ static void combine_empty_kernel(float *heads, const float *sinks,
                                            const float *part_o, const float2 *part_ml,
                                            uint32_t n_head, uint32_t n_split) {
    (void)heads; (void)sinks; (void)part_o; (void)part_ml; (void)n_head; (void)n_split;
}

static const uint32_t kNHead = 64u;

/* n_split as attention_flash_decode_launch computes it, so the sweep points
 * correspond to real sequence positions rather than round numbers. */
static uint32_t split_for_n_score(uint32_t n_score) {
    const uint32_t tiles_total = (n_score + kFDTile - 1u) / kFDTile;
    const uint32_t tiles_per_split = (kFDRowsPerSplit + kFDTile - 1u) / kFDTile;
    uint32_t n = (tiles_total + tiles_per_split - 1u) / tiles_per_split;
    if (n < 1u) n = 1u;
    if (n > kFDMaxSplit) n = kFDMaxSplit;
    return n;
}

static void fill_random(std::vector<float> &v, unsigned seed, float scale) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = (((float)(s >> 8) / 8388608.0f) - 1.0f) * scale;
    }
}

enum Variant { kNew = 0, kLegacy = 1, kEmpty = 2 };

static void launch(Variant v, float *heads, const float *sinks, const float *part_o,
                   const float2 *part_ml, uint32_t n_split, cudaStream_t stream) {
    if (v == kLegacy) {
        attention_flash_decode_combine_legacy_kernel
            <<<kNHead, 256, n_split * sizeof(float), stream>>>(
                heads, sinks, part_o, part_ml, kNHead, n_split);
    } else if (v == kNew) {
        const dim3 grid(kNHead, (kFDDim + kFDCombDims - 1u) / kFDCombDims, 1);
        attention_flash_decode_combine_kernel
            <<<grid, kFDCombDims, 2u * n_split * sizeof(float), stream>>>(
                heads, sinks, part_o, part_ml, kNHead, n_split);
    } else {
        const dim3 grid(kNHead, (kFDDim + kFDCombDims - 1u) / kFDCombDims, 1);
        combine_empty_kernel<<<grid, kFDCombDims, 0, stream>>>(
            heads, sinks, part_o, part_ml, kNHead, n_split);
    }
}

/* Time `reps` launches captured as one graph, replayed `iters` times, so the
 * ~5.5 us WDDM submission floor is amortised ~8x and what is left is kernel
 * work.  Buffers rotate across reps to control L2 residency. */
static double time_graph(Variant v, float *heads, const float *sinks,
                         const std::vector<float *> &part_o, const float2 *part_ml,
                         uint32_t n_split, int reps, int iters) {
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    for (int r = 0; r < reps; r++)          /* warm up + JIT the launch config */
        launch(v, heads, sinks, part_o[r % part_o.size()], part_ml, n_split, stream);
    CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int r = 0; r < reps; r++)
        launch(v, heads, sinks, part_o[r % part_o.size()], part_ml, n_split, stream);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; i++) CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaEventRecord(t1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    return (double)ms * 1000.0 / ((double)reps * iters);   /* us per call */
}

/* Same launches with no graph, i.e. how decode actually issues this kernel.
 * If both shapes land on the ~5.5 us WDDM submission floor, the kernel-level
 * win is real but invisible: the stream is submission-limited, not GPU-limited,
 * and the saved GPU time just becomes idle. */
static double time_eager(Variant v, float *heads, const float *sinks,
                         const std::vector<float *> &part_o, const float2 *part_ml,
                         uint32_t n_split, int calls) {
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    for (int r = 0; r < 32; r++)
        launch(v, heads, sinks, part_o[r % part_o.size()], part_ml, n_split, stream);
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < calls; i++)
        launch(v, heads, sinks, part_o[i % part_o.size()], part_ml, n_split, stream);
    CHECK(cudaEventRecord(t1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    CHECK(cudaStreamDestroy(stream));
    return (double)ms * 1000.0 / calls;
}

int main(int argc, char **argv) {
    int reps = 64, iters = 200, nbuf = 1;
    bool interleave = true;
    std::vector<uint32_t> scores;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--bufs=", 7))       nbuf = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--reps=", 7))  reps = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--nscore=", 9)) scores.push_back((uint32_t)atoi(argv[i] + 9));
        else if (!strcmp(argv[i], "--no-interleave")) interleave = false;
    }
    if (scores.empty()) {
        /* raw window 128 + top-k slots; 640 is every position past ctx 2048,
         * which is where decode actually lives. */
        scores = {64u, 160u, 320u, 640u};
    }
    if (nbuf < 1) nbuf = 1;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s: %d SMs, L2 %.0f MB\n", prop.name, prop.multiProcessorCount,
           prop.l2CacheSize / 1048576.0);
    printf("n_head=%u head_dim=%u  bufs=%d reps=%d iters=%d\n\n",
           kNHead, kFDDim, nbuf, reps, iters);

    const size_t o_elems = (size_t)kFDMaxSplit * kNHead * kFDDim;
    const size_t ml_elems = (size_t)kFDMaxSplit * kNHead;

    std::vector<float> h_o(o_elems), h_sinks(kNHead);
    std::vector<float2> h_ml(ml_elems);
    fill_random(h_o, 7u, 1.0f);
    fill_random(h_sinks, 11u, 0.5f);
    {   /* (m, l) pairs as the attention kernel leaves them: a plausible max and
         * a positive partial denominator, with a few empty splits. */
        unsigned s = 99u;
        for (size_t i = 0; i < ml_elems; i++) {
            s = s * 1664525u + 1013904223u;
            const float u = (float)(s >> 8) / 8388608.0f;   /* [0,2) */
            h_ml[i].x = (i % 37u == 0u) ? -INFINITY : (u * 4.0f - 2.0f);
            h_ml[i].y = 0.5f + u * 8.0f;
        }
    }

    std::vector<float *> d_o(nbuf);
    for (int b = 0; b < nbuf; b++) {
        CHECK(cudaMalloc(&d_o[b], o_elems * sizeof(float)));
        CHECK(cudaMemcpy(d_o[b], h_o.data(), o_elems * sizeof(float), cudaMemcpyHostToDevice));
    }
    float *d_sinks, *d_heads_a, *d_heads_b;
    float2 *d_ml;
    CHECK(cudaMalloc(&d_sinks, kNHead * sizeof(float)));
    CHECK(cudaMalloc(&d_ml, ml_elems * sizeof(float2)));
    CHECK(cudaMalloc(&d_heads_a, (size_t)kNHead * kFDDim * sizeof(float)));
    CHECK(cudaMalloc(&d_heads_b, (size_t)kNHead * kFDDim * sizeof(float)));
    CHECK(cudaMemcpy(d_sinks, h_sinks.data(), kNHead * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ml, h_ml.data(), ml_elems * sizeof(float2), cudaMemcpyHostToDevice));

    printf("%-8s %-8s %10s %10s %10s %8s %9s %9s %11s %10s %10s\n",
           "n_score", "n_split", "new(us)", "legacy(us)", "empty(us)",
           "speedup", "new GB/s", "bits", "eager new", "eager leg", "eager mt");
    for (uint32_t n_score : scores) {
        const uint32_t n_split = split_for_n_score(n_score);

        /* Bit-identity: the rewrite claims identical arithmetic, not just close. */
        launch(kNew, d_heads_a, d_sinks, d_o[0], d_ml, n_split, 0);
        CHECK(cudaDeviceSynchronize());
        launch(kLegacy, d_heads_b, d_sinks, d_o[0], d_ml, n_split, 0);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        std::vector<float> a((size_t)kNHead * kFDDim), b(a.size());
        CHECK(cudaMemcpy(a.data(), d_heads_a, a.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), d_heads_b, b.size() * 4, cudaMemcpyDeviceToHost));
        const bool bitwise = memcmp(a.data(), b.data(), a.size() * 4) == 0;

        /* Alternate the arms so any thermal drift inside the run hits both. */
        double t_new = 0.0, t_leg = 0.0;
        const int rounds = interleave ? 3 : 1;
        for (int r = 0; r < rounds; r++) {
            t_new += time_graph(kNew,    d_heads_a, d_sinks, d_o, d_ml, n_split, reps, iters);
            t_leg += time_graph(kLegacy, d_heads_b, d_sinks, d_o, d_ml, n_split, reps, iters);
        }
        t_new /= rounds; t_leg /= rounds;
        const double t_empty = time_graph(kEmpty, d_heads_a, d_sinks, d_o, d_ml, n_split, reps, iters);

        double e_new = 0.0, e_leg = 0.0;
        for (int r = 0; r < rounds; r++) {
            e_new += time_eager(kNew,    d_heads_a, d_sinks, d_o, d_ml, n_split, reps * 8);
            e_leg += time_eager(kLegacy, d_heads_b, d_sinks, d_o, d_ml, n_split, reps * 8);
        }
        e_new /= rounds; e_leg /= rounds;
        const double e_empty = time_eager(kEmpty, d_heads_a, d_sinks, d_o, d_ml, n_split, reps * 8);

        const double bytes = (double)n_split * kNHead * kFDDim * 4.0
                           + (double)n_split * kNHead * 8.0
                           + (double)kNHead * kFDDim * 4.0;
        printf("%-8u %-8u %10.2f %10.2f %10.2f %7.2fx %9.0f %9s %11.2f %10.2f %10.2f\n",
               n_score, n_split, t_new, t_leg, t_empty, t_leg / t_new,
               bytes / (t_new * 1e-6) / 1e9, bitwise ? "identical" : "DIFFER",
               e_new, e_leg, e_empty);
        fflush(stdout);
    }

    for (int b = 0; b < nbuf; b++) CHECK(cudaFree(d_o[b]));
    CHECK(cudaFree(d_sinks)); CHECK(cudaFree(d_ml));
    CHECK(cudaFree(d_heads_a)); CHECK(cudaFree(d_heads_b));
    return 0;
}
