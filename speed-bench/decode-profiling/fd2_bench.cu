/* Validate + time the FA2 decode kernel against the shipped CUDA-core one. */
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
#include "fd2.cuh"

static uint32_t g_n_head = 64, g_raw_cap = 2304, g_window = 128, g_top_k = 512,
                g_ratio = 4, g_n_comp = 117002, g_pos0 = 8191;
static float *d_q, *d_raw, *d_comp, *d_sinks, *d_heads, *d_part_o;
static float2 *d_part_ml;
static int32_t *d_topk;
static uint32_t g_n_raw, g_raw_start;
static fd_row_geom g_geom;

static void fill_random(std::vector<float> &v, unsigned seed, float scale) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = (((float)(s >> 8) / 8388608.0f) - 1.0f) * scale;
    }
}

static void compare(const std::vector<float> &a, const std::vector<float> &b,
                    const char *label) {
    double worst = 0.0, sum2 = 0.0, ref2 = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        const double r = d / (fabs((double)b[i]) + 1e-3);
        if (r > worst) { worst = r; at = i; }
        sum2 += d * d; ref2 += (double)b[i] * (double)b[i];
    }
    printf("  %s: max_rel=%.3e (head %zu dim %zu)  rel_l2=%.3e\n",
           label, worst, at / 512, at % 512, sqrt(sum2 / (ref2 + 1e-30)));
}

int main(int argc, char **argv) {
    int iters = 300;
    uint32_t fd2_split = 40;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--pos=", 6)) g_pos0 = (uint32_t)atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--split=", 8)) fd2_split = (uint32_t)atoi(argv[i] + 8);
    }
    g_n_raw = std::min(g_pos0 + 1u, g_raw_cap);
    g_raw_start = (g_pos0 + 1u) % g_raw_cap;
    g_geom = fd_compute_geom(g_pos0, g_n_raw, g_n_comp, g_top_k, g_window, g_ratio, 1u);
    printf("pos0=%u n_score=%u (raw %u + topk %u) visible_comp=%u\n",
           g_pos0, g_geom.n_score, g_geom.raw_count, g_geom.comp_slots, g_geom.visible_comp);

    std::vector<float> h_q((size_t)g_n_head * kFDDim), h_raw((size_t)g_raw_cap * kFDDim),
                       h_comp((size_t)g_n_comp * kFDDim), h_sinks(g_n_head);
    fill_random(h_q, 1u, 1.0f); fill_random(h_raw, 2u, 1.0f);
    fill_random(h_comp, 3u, 1.0f); fill_random(h_sinks, 4u, 0.5f);
    std::vector<int32_t> h_topk(g_top_k);
    { unsigned s = 12345u; const uint32_t vis = g_geom.visible_comp ? g_geom.visible_comp : 1u;
      for (uint32_t i = 0; i < g_top_k; i++) { s = s * 1664525u + 1013904223u; h_topk[i] = (int32_t)(s % vis); } }

    CHECK(cudaMalloc(&d_q, h_q.size() * 4));
    CHECK(cudaMalloc(&d_raw, h_raw.size() * 4));
    CHECK(cudaMalloc(&d_comp, h_comp.size() * 4));
    CHECK(cudaMalloc(&d_sinks, h_sinks.size() * 4));
    CHECK(cudaMalloc(&d_heads, (size_t)g_n_head * kFDDim * 4));
    CHECK(cudaMalloc(&d_topk, g_top_k * 4));
    CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_raw, h_raw.data(), h_raw.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_comp, h_comp.data(), h_comp.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_topk, h_topk.data(), g_top_k * 4, cudaMemcpyHostToDevice));

    const uint32_t max_split = std::max(64u, fd2_split);
    CHECK(cudaMalloc(&d_part_o, (size_t)max_split * g_n_head * kFDDim * 4));
    CHECK(cudaMalloc(&d_part_ml, (size_t)max_split * g_n_head * sizeof(float2)));

    /* ---------------- baseline (production settings) ---------------- */
    const uint32_t tiles_total = (g_geom.n_score + kFDTile - 1u) / kFDTile;
    const uint32_t tiles_per_split = (kFDRowsPerSplit + kFDTile - 1u) / kFDTile;
    uint32_t base_split = (tiles_total + tiles_per_split - 1u) / tiles_per_split;
    if (base_split > kFDMaxSplit) base_split = kFDMaxSplit;
    const dim3 base_grid((g_n_head + kFDWarps - 1u) / kFDWarps, base_split, 1);
    auto run_base = [&]() {
        attention_flash_decode_kernel<kFDWarps, kFDTile><<<base_grid, kFDWarps * 32u>>>(
            d_part_o, d_part_ml, d_q, d_raw, d_comp, d_topk, g_n_head, base_split,
            g_pos0, g_n_raw, g_raw_cap, g_raw_start, g_n_comp, g_top_k, g_window, g_ratio);
        attention_flash_decode_combine_kernel<<<g_n_head, 256, base_split * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, g_n_head, base_split);
    };

    /* ---------------- FA2 ---------------- */
    const uint32_t fd2_tiles = (g_geom.n_score + kFD2Tile - 1u) / kFD2Tile;
    uint32_t nsplit2 = std::min(fd2_split, fd2_tiles);
    const dim3 fd2_grid((g_n_head + kFD2Heads - 1u) / kFD2Heads, nsplit2, 1);
    auto run_fd2 = [&]() {
        attention_flash2_decode_kernel<<<fd2_grid, kFD2Threads>>>(
            d_part_o, d_part_ml, d_q, d_raw, d_comp, d_topk, g_n_head, nsplit2,
            g_pos0, g_n_raw, g_raw_cap, g_raw_start, g_n_comp, g_top_k, g_window, g_ratio);
        attention_flash2_combine_kernel<<<dim3(g_n_head, kFD2Dim / kFD2CombDims, 1),
                                         kFD2CombDims, nsplit2 * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, g_n_head, nsplit2);
    };

    /* isolate the two halves so the plateau can be attributed */
    auto run_fd2_main = [&]() {
        attention_flash2_decode_kernel<<<fd2_grid, kFD2Threads>>>(
            d_part_o, d_part_ml, d_q, d_raw, d_comp, d_topk, g_n_head, nsplit2,
            g_pos0, g_n_raw, g_raw_cap, g_raw_start, g_n_comp, g_top_k, g_window, g_ratio);
    };
    auto run_fd2_comb = [&]() {
        attention_flash2_combine_kernel<<<dim3(g_n_head, kFD2Dim / kFD2CombDims, 1),
                                         kFD2CombDims, nsplit2 * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, g_n_head, nsplit2);
    };
    auto run_old_comb = [&]() {
        attention_flash_decode_combine_kernel<<<g_n_head, 256, nsplit2 * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, g_n_head, nsplit2);
    };

    std::vector<float> out_base((size_t)g_n_head * kFDDim), out_fd2(out_base.size());
    for (int i = 0; i < 10; i++) run_base();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_base.data(), d_heads, out_base.size() * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10; i++) run_fd2();
    CHECK(cudaDeviceSynchronize());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("FA2 launch failed: %s\n", cudaGetErrorString(e)); return 1; }
    CHECK(cudaMemcpy(out_fd2.data(), d_heads, out_fd2.size() * 4, cudaMemcpyDeviceToHost));

    printf("\nnumerics (fp16 MMA inputs vs fp32 CUDA-core reference):\n");
    compare(out_fd2, out_base, "FA2 vs baseline");

    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    float ms_base = 0, ms_fd2 = 0;
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) run_base();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms_base, e0, e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) run_fd2();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms_fd2, e0, e1));

    const double us_b = ms_base * 1000.0 / iters, us_f = ms_fd2 * 1000.0 / iters;
    printf("\nbaseline (WARPS=8 TILE=8 split=%u, %u blocks): %7.2f us/layer  %.3f ms/token\n",
           base_split, base_grid.x * base_split, us_b, us_b * 43 / 1000.0);
    printf("FA2      (HEADS=16 TILE=16 split=%u, %u blocks): %7.2f us/layer  %.3f ms/token\n",
           nsplit2, fd2_grid.x * nsplit2, us_f, us_f * 43 / 1000.0);
    printf("speedup: %.2fx   token saving at 19.0 ms: %.2f%%\n",
           us_b / us_f, (us_b - us_f) * 43 / 19000.0 * 100.0);

    float ms_m = 0, ms_c = 0, ms_oc = 0;
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) run_fd2_main();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms_m, e0, e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) run_fd2_comb();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms_c, e0, e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) run_old_comb();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms_oc, e0, e1));
    printf("  breakdown: FA2 main %6.2f us   new combine %6.2f us   old combine %6.2f us\n",
           ms_m * 1000.0 / iters, ms_c * 1000.0 / iters, ms_oc * 1000.0 / iters);
    return 0;
}
