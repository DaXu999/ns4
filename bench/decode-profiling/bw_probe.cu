/* What bandwidth can this box actually reach, and with how many SMs?
 *
 * Decode moves ~195 MB of weights per layer, ~8.4 GB per token, and achieves
 * ~420-440 GB/s of a nominal ~1.8 TB/s.  Before treating that as "idle SMs
 * wasting bandwidth" it is worth separating three different ceilings:
 *
 *   1. what a pure streaming read reaches at full grid   (the real roofline,
 *      which on WSL2 need not be the spec number)
 *   2. how many blocks it takes to get there             (if DRAM saturates
 *      well below 188 SMs, idle SMs cannot be the reason bandwidth is low)
 *   3. what it costs to read with a strided / small-tile access pattern,
 *      which is what a q8_0 or IQ2 weight walk actually does
 *
 * Reads a buffer far larger than L2 (128 MB on GB202) so nothing is cached.
 *
 *   nvcc -O3 -arch=sm_120 -o bw_probe bw_probe.cu
 *   ./bw_probe [--gb=8] [--reps=5]
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

/* Grid-stride float4 read.  One accumulator per thread so nothing is
 * optimised away, and the store is predicated off so only the read counts. */
__global__ static void read_f4_kernel(const float4 *__restrict__ src,
                                      uint64_t n4, float *sink) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4; i += stride) {
        const float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;
}

/* Same bytes, but each warp walks 32-byte granules with a gap -- the shape a
 * quantised weight row walk has when a block owns scattered output rows. */
__global__ static void read_strided_kernel(const float4 *__restrict__ src,
                                           uint64_t n4, uint32_t gap,
                                           float *sink) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x * gap;
    for (uint64_t i = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * gap;
         i < n4; i += stride) {
        const float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;
}

static double time_kernel(void (*fn)(const float4 *, uint64_t, float *),
                          const float4 *src, uint64_t n4, float *sink,
                          uint32_t blocks, uint32_t threads, int reps) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    fn<<<blocks, threads>>>(src, n4, sink);          /* warm */
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) fn<<<blocks, threads>>>(src, n4, sink);
    CHECK(cudaEventRecord(b));
    CHECK(cudaDeviceSynchronize());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return (double)ms / reps;
}

int main(int argc, char **argv) {
    double gb = 8.0; int reps = 5;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--gb=", 5)) gb = atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    const uint64_t bytes = (uint64_t)(gb * 1024.0 * 1024.0 * 1024.0) & ~255ull;
    const uint64_t n4 = bytes / sizeof(float4);
    float4 *src = NULL; float *sink = NULL;
    CHECK(cudaMalloc(&src, bytes));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(src, 0, bytes));
    printf("device: %d SMs, streaming %.2f GB (L2 is ~128 MB, so this is DRAM)\n\n",
           sms, bytes / (1024.0 * 1024.0 * 1024.0));

    printf("contiguous float4 read, 256 threads/block\n");
    printf("  blocks   blocks/SM        ms      GB/s   %% of best\n");
    double best = 0.0;
    const uint32_t grids[] = {32, 64, 94, 128, 188, 376, 564, 752, 1504, 3008};
    double gbs[sizeof(grids)/sizeof(grids[0])];
    for (size_t i = 0; i < sizeof(grids)/sizeof(grids[0]); i++) {
        const double ms = time_kernel(read_f4_kernel, src, n4, sink, grids[i], 256, reps);
        gbs[i] = bytes / (ms * 1e-3) / 1e9;
        if (gbs[i] > best) best = gbs[i];
    }
    for (size_t i = 0; i < sizeof(grids)/sizeof(grids[0]); i++) {
        printf("  %6u   %8.2f   %7.3f  %8.1f   %6.1f%%\n",
               grids[i], (double)grids[i] / sms,
               bytes / (gbs[i] * 1e9) * 1e3, gbs[i], 100.0 * gbs[i] / best);
    }

    printf("\nbest contiguous read: %.1f GB/s\n", best);
    printf("ns4 decode achieves ~436 GB/s (8.4 GB/token at 19.2 ms) = %.1f%% of that\n",
           100.0 * 436.0 / best);

    printf("\nstrided reads at full grid (1504 blocks), same total bytes touched:\n");
    printf("  gap (float4)   effective GB/s   %% of contiguous\n");
    const uint32_t gaps[] = {1, 2, 4, 8};
    for (size_t i = 0; i < sizeof(gaps)/sizeof(gaps[0]); i++) {
        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        read_strided_kernel<<<1504, 256>>>(src, n4, gaps[i], sink);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int r = 0; r < reps; r++) read_strided_kernel<<<1504, 256>>>(src, n4, gaps[i], sink);
        CHECK(cudaEventRecord(b));
        CHECK(cudaDeviceSynchronize());
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
        const double touched = (double)bytes / gaps[i];
        const double eff = touched / ((ms / reps) * 1e-3) / 1e9;
        printf("  %12u   %14.1f   %13.1f%%\n", gaps[i], eff, 100.0 * eff / best);
    }
    return 0;
}
