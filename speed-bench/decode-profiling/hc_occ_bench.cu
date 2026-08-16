/* hc_split_weighted_sum_norm_fused_kernel, priced honestly.
 *
 * kprof ranks this the second most expensive decode kernel: 34.4 ms of a
 * 400 ms window, 27.8 us/call, at grid 1x1 on a 188-SM GPU.  It is also the
 * one with a structural reason to be slow beyond its grid -- thread 0 runs
 * hc4_split_one (a serial Sinkhorn) while the other 255 threads sit at a
 * __syncthreads.  This separates the three candidate costs:
 *
 *   - the launch itself            (empty-grid column)
 *   - the serial Sinkhorn prologue (prologue-only variant)
 *   - the 64 KB weighted sum       (full kernel minus the above)
 *
 * Extraction recipe for the two verbatim pieces:
 *   sed -n '/^__device__ static void hc4_split_one/,/^}/p' ds4_cuda.cu
 *   sed -n '/^__global__ static void hc_split_weighted_sum_norm_fused_kernel/,/^}/p' ds4_cuda.cu
 *
 *   nvcc -O3 -arch=sm_120 -o hc_occ_bench hc_occ_bench.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

#include "hc4_split_one.inc"

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out, float *norm_out, float *split, const float *mix,
        const float *residual_hc, const float *scale, const float *base,
        const float *norm_w, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
        uint32_t sinkhorn_iters, float epsv, float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

/* Same kernel with the warp-parallel split.  Bit-identical by construction:
 * every reduction still sums in the serial version's order. */
__global__ static void hc_split_weighted_sum_norm_fused_warp_kernel(
        float *out, float *norm_out, float *split, const float *mix,
        const float *residual_hc, const float *scale, const float *base,
        const float *norm_w, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
        uint32_t sinkhorn_iters, float epsv, float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d < 32u) hc4_split_one_warp(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

/* Prologue only: isolates what the serial Sinkhorn costs. */
__global__ static void hc_prologue_only_kernel(
        float *split, const float *mix, const float *scale, const float *base,
        uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    const uint32_t t = blockIdx.x;
    if (t >= n_rows) return;
    if (threadIdx.x == 0)
        hc4_split_one(split + (uint64_t)t * 24u, mix + (uint64_t)t * 24u,
                      scale, base, sinkhorn_iters, epsv);
    __syncthreads();
}

__global__ static void empty_kernel(void) {}

template <typename L> static double time_graph(L launch, int reps) {
    cudaStream_t s; CHECK(cudaStreamCreate(&s));
    launch(s); CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g; cudaGraphExec_t e;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < reps; i++) launch(s);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&e, g, NULL, NULL, 0));
    CHECK(cudaGraphLaunch(e, s)); CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, s)); CHECK(cudaGraphLaunch(e, s)); CHECK(cudaEventRecord(b, s));
    CHECK(cudaStreamSynchronize(s));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaGraphExecDestroy(e)); CHECK(cudaGraphDestroy(g));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); CHECK(cudaStreamDestroy(s));
    return (double)ms * 1000.0 / reps;
}

template <typename L> static double time_eager(L launch, int reps) {
    cudaStream_t s; CHECK(cudaStreamCreate(&s));
    launch(s); CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < reps; i++) launch(s);
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaStreamSynchronize(s));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); CHECK(cudaStreamDestroy(s));
    return (double)ms * 1000.0 / reps;
}

int main(int argc, char **argv) {
    uint32_t n_embd = 4096, rows = 1, iters = 8;
    int reps = 2000;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--n=", 4)) n_embd = (uint32_t)atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--rows=", 7)) rows = (uint32_t)atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--iters=", 8)) iters = (uint32_t)atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    printf("device: %d SMs   n_embd=%u rows=%u sinkhorn_iters=%u  grid %ux1\n",
           sms, n_embd, rows, iters, rows);

    const size_t hc_elems = (size_t)rows * 4 * n_embd;
    std::vector<float> h(hc_elems);
    for (size_t i = 0; i < hc_elems; i++)
        h[i] = (float)((i * 1103515245u + 12345u) % 1000) / 500.0f - 1.0f;
    std::vector<float> hmix((size_t)rows * 24, 0.3f), hsc(4, 1.0f), hbase(24, 0.1f),
                       hnw(n_embd, 1.0f);

    float *d_res, *d_out, *d_norm, *d_split, *d_mix, *d_scale, *d_base, *d_nw;
    CHECK(cudaMalloc(&d_res, hc_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)rows * n_embd * sizeof(float)));
    CHECK(cudaMalloc(&d_norm, (size_t)rows * n_embd * sizeof(float)));
    CHECK(cudaMalloc(&d_split, (size_t)rows * 24 * sizeof(float)));
    CHECK(cudaMalloc(&d_mix, (size_t)rows * 24 * sizeof(float)));
    CHECK(cudaMalloc(&d_scale, 4 * sizeof(float)));
    CHECK(cudaMalloc(&d_base, 24 * sizeof(float)));
    CHECK(cudaMalloc(&d_nw, n_embd * sizeof(float)));
    CHECK(cudaMemcpy(d_res, h.data(), hc_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mix, hmix.data(), hmix.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, hsc.data(), hsc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_base, hbase.data(), hbase.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nw, hnw.data(), hnw.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto full = [&](cudaStream_t s) {
        hc_split_weighted_sum_norm_fused_kernel<<<rows, 256, 0, s>>>(
            d_out, d_norm, d_split, d_mix, d_res, d_scale, d_base, d_nw,
            n_embd, 4u, rows, iters, 1e-6f, 1e-6f);
    };
    auto prologue = [&](cudaStream_t s) {
        hc_prologue_only_kernel<<<rows, 256, 0, s>>>(
            d_split, d_mix, d_scale, d_base, rows, iters, 1e-6f);
    };
    auto empty = [&](cudaStream_t s) { empty_kernel<<<rows, 256, 0, s>>>(); };

    float *d_out2, *d_norm2, *d_split2;
    CHECK(cudaMalloc(&d_out2, (size_t)rows * n_embd * sizeof(float)));
    CHECK(cudaMalloc(&d_norm2, (size_t)rows * n_embd * sizeof(float)));
    CHECK(cudaMalloc(&d_split2, (size_t)rows * 24 * sizeof(float)));
    auto full_warp = [&](cudaStream_t s) {
        hc_split_weighted_sum_norm_fused_warp_kernel<<<rows, 256, 0, s>>>(
            d_out2, d_norm2, d_split2, d_mix, d_res, d_scale, d_base, d_nw,
            n_embd, 4u, rows, iters, 1e-6f, 1e-6f);
    };

    /* bit-identity, not tolerance */
    full(0); full_warp(0);
    CHECK(cudaDeviceSynchronize());
    const size_t on = (size_t)rows * n_embd, sn = (size_t)rows * 24;
    std::vector<float> a(on), b(on), sa(sn), sb(sn);
    CHECK(cudaMemcpy(a.data(), d_norm, on * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(), d_norm2, on * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sa.data(), d_split, sn * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sb.data(), d_split2, sn * sizeof(float), cudaMemcpyDeviceToHost));
    const bool same_norm = memcmp(a.data(), b.data(), on * sizeof(float)) == 0;
    const bool same_split = memcmp(sa.data(), sb.data(), sn * sizeof(float)) == 0;
    printf("correctness: split %s, norm_out %s\n",
           same_split ? "BIT-IDENTICAL" : "DIFFERS",
           same_norm ? "BIT-IDENTICAL" : "DIFFERS");
    if (!same_split) {
        for (size_t i = 0; i < sn && i < 24; i++)
            if (sa[i] != sb[i]) printf("  split[%zu] %.9g vs %.9g\n", i, (double)sa[i], (double)sb[i]);
    }

    const double fg = time_graph(full, reps), fe = time_eager(full, reps);
    const double wg = time_graph(full_warp, reps), we = time_eager(full_warp, reps);
    const double pg = time_graph(prologue, reps), pe = time_eager(prologue, reps);
    const double eg = time_graph(empty, reps), ee = time_eager(empty, reps);

    printf("\n                       graph-amortised      eager\n");
    printf("  full kernel (serial)   %8.2f us  %8.2f us\n", fg, fe);
    printf("  full kernel (warp)     %8.2f us  %8.2f us\n", wg, we);
    printf("  Sinkhorn prologue only %8.2f us  %8.2f us\n", pg, pe);
    printf("  empty grid             %8.2f us  %8.2f us\n", eg, ee);
    printf("\ndecomposition of the graph-amortised serial kernel:\n");
    printf("  launch/dispatch      %6.2f us\n", eg);
    printf("  serial Sinkhorn      %6.2f us\n", pg - eg);
    printf("  weighted sum + norm  %6.2f us\n", fg - pg);
    printf("\nwarp split saves %.2f us/call graph-amortised (%.2fx), %.2f us eager\n",
           fg - wg, fg / wg, fe - we);
    printf("at 2 calls/layer x 43 layers that is %.0f us/token\n", (fe - we) * 86.0);
    return (same_norm && same_split) ? 0 : 2;
}
