/* Standalone microbenchmark for the ds4 CUDA flash decode attention kernel.
 * Baseline kernel text is extracted verbatim from ds4_cuda.cu so the timing
 * and the numerics comparison are honest. */
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

#include "fd_baseline.inc"

/* ------------------------------------------------------------------ */
/* Model geometry as served by DeepSeek-V4-Flash on this box.          */
struct Shape {
    uint32_t n_head   = 64;
    uint32_t head_dim = 512;
    uint32_t raw_cap  = 2304;
    uint32_t window   = 128;
    uint32_t top_k    = 512;
    uint32_t ratio    = 4;
    uint32_t n_comp   = 117002;   /* ctx 468000 */
    uint32_t pos0     = 8191;
};

static void fill_random(std::vector<float> &v, unsigned seed, float scale) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = (((float)(s >> 8) / 8388608.0f) - 1.0f) * scale;
    }
}

int main(int argc, char **argv) {
    Shape sh;
    int iters = 300;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--pos=", 6))    sh.pos0   = (uint32_t)atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--ncomp=", 8)) sh.n_comp = (uint32_t)atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i] + 8);
    }

    const uint32_t n_raw = std::min(sh.pos0 + 1u, sh.raw_cap);
    const uint32_t raw_start = (sh.pos0 + 1u) % sh.raw_cap;
    const fd_row_geom geom = fd_compute_geom(sh.pos0, n_raw, sh.n_comp, sh.top_k,
                                             sh.window, sh.ratio, 1u);
    printf("shape: pos0=%u n_raw=%u raw_cap=%u n_comp=%u visible_comp=%u "
           "raw_count=%u comp_slots=%u n_score=%u\n",
           sh.pos0, n_raw, sh.raw_cap, sh.n_comp, geom.visible_comp,
           geom.raw_count, geom.comp_slots, geom.n_score);

    /* Host buffers */
    std::vector<float> h_q((size_t)sh.n_head * sh.head_dim);
    std::vector<float> h_raw((size_t)sh.raw_cap * sh.head_dim);
    std::vector<float> h_comp((size_t)sh.n_comp * sh.head_dim);
    std::vector<float> h_sinks(sh.n_head);
    fill_random(h_q, 1u, 1.0f);
    fill_random(h_raw, 2u, 1.0f);
    fill_random(h_comp, 3u, 1.0f);
    fill_random(h_sinks, 4u, 0.5f);

    /* top-k: distinct random rows inside the visible compressed range, in the
     * arbitrary order the indexer emits them. */
    std::vector<int32_t> h_topk(sh.top_k);
    {
        const uint32_t vis = geom.visible_comp ? geom.visible_comp : 1u;
        unsigned s = 12345u;
        std::vector<uint32_t> pool;
        for (uint32_t i = 0; i < vis; i++) pool.push_back(i);
        for (uint32_t i = 0; i < sh.top_k; i++) {
            s = s * 1664525u + 1013904223u;
            const uint32_t j = i + (s % (uint32_t)std::max<size_t>(1, pool.size() - i));
            if (j < pool.size()) std::swap(pool[i], pool[j]);
            h_topk[i] = (int32_t)pool[std::min<size_t>(i, pool.size() - 1)];
        }
    }

    float *d_q, *d_raw, *d_comp, *d_sinks, *d_heads, *d_part_o;
    float2 *d_part_ml;
    int32_t *d_topk;
    CHECK(cudaMalloc(&d_q, h_q.size() * 4));
    CHECK(cudaMalloc(&d_raw, h_raw.size() * 4));
    CHECK(cudaMalloc(&d_comp, h_comp.size() * 4));
    CHECK(cudaMalloc(&d_sinks, h_sinks.size() * 4));
    CHECK(cudaMalloc(&d_heads, (size_t)sh.n_head * sh.head_dim * 4));
    CHECK(cudaMalloc(&d_topk, sh.top_k * 4));
    CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_raw, h_raw.data(), h_raw.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_comp, h_comp.data(), h_comp.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_topk, h_topk.data(), sh.top_k * 4, cudaMemcpyHostToDevice));

    /* split geometry exactly as the production launcher computes it */
    const uint32_t tiles_total = (geom.n_score + kFDTile - 1u) / kFDTile;
    const uint32_t tiles_per_split = (kFDRowsPerSplit + kFDTile - 1u) / kFDTile;
    uint32_t n_split = (tiles_total + tiles_per_split - 1u) / tiles_per_split;
    if (n_split < 1u) n_split = 1u;
    if (n_split > kFDMaxSplit) n_split = kFDMaxSplit;
    printf("launch: grid=(%u,%u) block=%u  n_split=%u  (%u blocks total)\n",
           (sh.n_head + kFDWarps - 1u) / kFDWarps, n_split, kFDWarps * 32u, n_split,
           ((sh.n_head + kFDWarps - 1u) / kFDWarps) * n_split);

    CHECK(cudaMalloc(&d_part_o, (size_t)n_split * sh.n_head * kFDDim * 4));
    CHECK(cudaMalloc(&d_part_ml, (size_t)n_split * sh.n_head * sizeof(float2)));

    const dim3 grid((sh.n_head + kFDWarps - 1u) / kFDWarps, n_split, 1);

    auto launch_baseline = [&]() {
        attention_flash_decode_kernel<kFDWarps, kFDTile><<<grid, kFDWarps * 32u>>>(
            d_part_o, d_part_ml, d_q, d_raw, d_comp, d_topk, sh.n_head, n_split,
            sh.pos0, n_raw, sh.raw_cap, raw_start, sh.n_comp, sh.top_k,
            sh.window, sh.ratio);
        attention_flash_decode_combine_kernel<<<sh.n_head, 256, n_split * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, sh.n_head, n_split);
    };

    /* warmup + correctness snapshot */
    for (int i = 0; i < 20; i++) launch_baseline();
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)sh.n_head * sh.head_dim);
    CHECK(cudaMemcpy(ref.data(), d_heads, ref.size() * 4, cudaMemcpyDeviceToHost));
    FILE *f = fopen("/home/xl/.claude/jobs/bbc3ef95/tmp/fd_ref.bin", "wb");
    if (f) { fwrite(ref.data(), 4, ref.size(), f); fclose(f); }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    /* whole path (score + combine), which is what a layer pays */
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) launch_baseline();
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_all = 0; CHECK(cudaEventElapsedTime(&ms_all, e0, e1));

    /* score kernel alone */
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) {
        attention_flash_decode_kernel<kFDWarps, kFDTile><<<grid, kFDWarps * 32u>>>(
            d_part_o, d_part_ml, d_q, d_raw, d_comp, d_topk, sh.n_head, n_split,
            sh.pos0, n_raw, sh.raw_cap, raw_start, sh.n_comp, sh.top_k,
            sh.window, sh.ratio);
    }
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_score = 0; CHECK(cudaEventElapsedTime(&ms_score, e0, e1));

    /* combine alone */
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) {
        attention_flash_decode_combine_kernel<<<sh.n_head, 256, n_split * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, sh.n_head, n_split);
    }
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms_comb = 0; CHECK(cudaEventElapsedTime(&ms_comb, e0, e1));

    const double us_all = ms_all * 1000.0 / iters;
    const double us_score = ms_score * 1000.0 / iters;
    const double us_comb = ms_comb * 1000.0 / iters;
    const double kv_bytes = (double)geom.n_score * kFDDim * 4.0;
    const double kv_bytes_read = kv_bytes * ((sh.n_head + kFDWarps - 1u) / kFDWarps);
    const double flops = 2.0 * 2.0 * sh.n_head * geom.n_score * kFDDim;

    printf("\nBASELINE per layer:  total %7.2f us   score %7.2f us   combine %7.2f us\n",
           us_all, us_score, us_comb);
    printf("  per token (43 layers): %7.3f ms   (%.1f%% of a 19.0 ms token)\n",
           us_all * 43.0 / 1000.0, us_all * 43.0 / 19000.0 * 100.0);
    printf("  KV bytes/layer: %.2f MB unique, %.2f MB actually fetched (x%u redundancy)\n",
           kv_bytes / 1e6, kv_bytes_read / 1e6, (sh.n_head + kFDWarps - 1u) / kFDWarps);
    printf("  effective BW: %.1f GB/s (fetched) | %.1f GFLOP/s | occupancy-limited work\n",
           kv_bytes_read / (us_score * 1e-6) / 1e9, flops / (us_score * 1e-6) / 1e9);
    return 0;
}
