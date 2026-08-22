/* Sweep the existing flash-decode kernel's template/launch parameters to
 * separate "structural" cost from tuning cost before writing a new kernel. */
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

static uint32_t g_n_head = 64, g_raw_cap = 2304, g_window = 128, g_top_k = 512,
                g_ratio = 4, g_n_comp = 117002, g_pos0 = 8191;
static float *d_q, *d_raw, *d_comp, *d_sinks, *d_heads, *d_part_o;
static float2 *d_part_ml;
static int32_t *d_topk;
static uint32_t g_n_raw, g_raw_start;
static fd_row_geom g_geom;
static std::vector<float> g_ref;

static void fill_random(std::vector<float> &v, unsigned seed, float scale) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = (((float)(s >> 8) / 8388608.0f) - 1.0f) * scale;
    }
}

/* max |a-b| / (|ref|+1e-3) over the head outputs */
static double max_rel_err(const std::vector<float> &a, const std::vector<float> &b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        const double r = d / (fabs((double)b[i]) + 1e-3);
        if (r > worst) worst = r;
    }
    return worst;
}

template <uint32_t WARPS, uint32_t TILE>
static double time_variant(uint32_t rows_per_split, int iters, double *err_out) {
    const uint32_t tiles_total = (g_geom.n_score + TILE - 1u) / TILE;
    const uint32_t tiles_per_split = (rows_per_split + TILE - 1u) / TILE;
    uint32_t n_split = (tiles_total + tiles_per_split - 1u) / tiles_per_split;
    if (n_split < 1u) n_split = 1u;
    if (n_split > kFDMaxSplit) n_split = kFDMaxSplit;

    CHECK(cudaMalloc(&d_part_o, (size_t)n_split * g_n_head * kFDDim * 4));
    CHECK(cudaMalloc(&d_part_ml, (size_t)n_split * g_n_head * sizeof(float2)));
    const dim3 grid((g_n_head + WARPS - 1u) / WARPS, n_split, 1);

    auto go = [&]() {
        attention_flash_decode_kernel<WARPS, TILE><<<grid, WARPS * 32u>>>(
            d_part_o, d_part_ml, d_q, d_raw, d_comp, d_topk, g_n_head, n_split,
            g_pos0, g_n_raw, g_raw_cap, g_raw_start, g_n_comp, g_top_k,
            g_window, g_ratio);
        attention_flash_decode_combine_kernel<<<g_n_head, 256, n_split * sizeof(float)>>>(
            d_heads, d_sinks, d_part_o, d_part_ml, g_n_head, n_split);
    };

    for (int i = 0; i < 20; i++) go();
    CHECK(cudaDeviceSynchronize());
    if (cudaGetLastError() != cudaSuccess) { cudaFree(d_part_o); cudaFree(d_part_ml); return -1.0; }

    std::vector<float> out((size_t)g_n_head * kFDDim);
    CHECK(cudaMemcpy(out.data(), d_heads, out.size() * 4, cudaMemcpyDeviceToHost));
    if (err_out) *err_out = g_ref.empty() ? 0.0 : max_rel_err(out, g_ref);
    if (g_ref.empty()) g_ref = out;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) go();
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    cudaFree(d_part_o); cudaFree(d_part_ml);
    printf("  WARPS=%-3u TILE=%-3u rows/split=%-4u -> n_split=%-3u blocks=%-4u "
           "%7.2f us/layer   %6.3f ms/token   relerr=%.2e\n",
           WARPS, TILE, rows_per_split, n_split, grid.x * n_split,
           ms * 1000.0 / iters, ms * 43.0 / iters, err_out ? *err_out : 0.0);
    return ms * 1000.0 / iters;
}

int main(int argc, char **argv) {
    int iters = 300;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--pos=", 6)) g_pos0 = (uint32_t)atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = atoi(argv[i] + 8);
    }
    g_n_raw = std::min(g_pos0 + 1u, g_raw_cap);
    g_raw_start = (g_pos0 + 1u) % g_raw_cap;
    g_geom = fd_compute_geom(g_pos0, g_n_raw, g_n_comp, g_top_k, g_window, g_ratio, 1u);
    printf("pos0=%u n_score=%u (raw %u + topk %u), visible_comp=%u\n",
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

    double err;
    printf("\n[production setting first: WARPS=8 TILE=8 rows/split=24]\n");
    time_variant<8, 8>(24, iters, &err);
    printf("\n[warps sweep: fewer blocks in x = less KV re-read]\n");
    time_variant<8, 8>(24, iters, &err);
    time_variant<16, 8>(24, iters, &err);
    time_variant<32, 8>(24, iters, &err);
    printf("\n[tile sweep at WARPS=16]\n");
    time_variant<16, 8>(24, iters, &err);
    time_variant<16, 16>(24, iters, &err);
    /* TILE=32 exceeds the 48KB static shared cap */
    printf("\n[rows-per-split sweep at WARPS=16 TILE=16]\n");
    for (uint32_t r : {16u, 32u, 48u, 64u, 96u, 160u, 320u})
        time_variant<16, 16>(r, iters, &err);
    printf("\n[rows-per-split sweep at WARPS=32 TILE=16]\n");
    for (uint32_t r : {16u, 32u, 48u, 64u, 96u, 160u, 320u})
        time_variant<32, 16>(r, iters, &err);
    return 0;
}
