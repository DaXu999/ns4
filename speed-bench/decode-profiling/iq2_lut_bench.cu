/* Is the IQ2_XXS dequant bound by shared-memory bank conflicts on its LUT?
 *
 * iq2_access_bench.cu already showed the weight *reads* are not the problem:
 * the same 66-byte block-per-lane pattern streams at 1596 GB/s, 97% of this
 * box's roofline, while the live kernel manages 348 GB/s.  So the 4.6x is
 * inside dev_dot_iq2_xxs_q8_K_block_lut, whose hot operation is
 *
 *     grid[idx8]        // uint64, idx8 data-dependent over 0..255
 *     signs[sign7]      // uint8,  data-dependent over 0..127
 *
 * both from shared memory, 32 times per 256-element block.  Shared memory has
 * 32 banks of 4 bytes, so a uint64 at index i covers banks (2i, 2i+1) mod 32 --
 * random indices therefore land in only 16 distinct bank-pairs and a full warp
 * cannot avoid conflicts.  Splitting the table into two uint32 arrays gives all
 * 32 banks and should cut the replay count.
 *
 * This measures the LUT traffic alone, at the rate the real kernel issues it,
 * with no weight reads and no dp4a.
 *
 *   nvcc -O3 -arch=sm_120 -o iq2_lut_bench iq2_lut_bench.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

/* Live layout: one uint64 table, indices random over 0..255. */
__global__ static void lut_u64_kernel(const uint8_t *__restrict__ idx,
                                      uint32_t iters, uint64_t *sink) {
    __shared__ uint64_t grid[256];
    __shared__ uint8_t signs[128];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x)
        grid[i] = 0x0101010101010101ull * (i + 1u);
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x)
        signs[i] = (uint8_t)i;
    __syncthreads();

    const uint8_t *my = idx + (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 64u;
    uint64_t acc = 0;
    for (uint32_t it = 0; it < iters; it++) {
#pragma unroll 8
        for (uint32_t k = 0; k < 64u; k++) {
            const uint8_t i8 = my[k] ^ (uint8_t)it;
            acc += grid[i8] + signs[i8 & 127u];
        }
    }
    if (acc == 0xffffffffffffffffull) sink[0] = acc;
}

/* Same table split into two uint32 halves: every index reaches all 32 banks. */
__global__ static void lut_u32x2_kernel(const uint8_t *__restrict__ idx,
                                        uint32_t iters, uint64_t *sink) {
    __shared__ uint32_t grid_lo[256];
    __shared__ uint32_t grid_hi[256];
    __shared__ uint8_t signs[128];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        grid_lo[i] = 0x01010101u * (i + 1u);
        grid_hi[i] = 0x01010101u * (i + 1u);
    }
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x)
        signs[i] = (uint8_t)i;
    __syncthreads();

    const uint8_t *my = idx + (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 64u;
    uint64_t acc = 0;
    for (uint32_t it = 0; it < iters; it++) {
#pragma unroll 8
        for (uint32_t k = 0; k < 64u; k++) {
            const uint8_t i8 = my[k] ^ (uint8_t)it;
            const uint64_t v = (uint64_t)grid_lo[i8] | ((uint64_t)grid_hi[i8] << 32);
            acc += v + signs[i8 & 127u];
        }
    }
    if (acc == 0xffffffffffffffffull) sink[0] = acc;
}

/* Control: conflict-free by construction (every lane reads its own bank). */
__global__ static void lut_noconflict_kernel(const uint8_t *__restrict__ idx,
                                             uint32_t iters, uint64_t *sink) {
    __shared__ uint64_t grid[256];
    __shared__ uint8_t signs[128];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x)
        grid[i] = 0x0101010101010101ull * (i + 1u);
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x)
        signs[i] = (uint8_t)i;
    __syncthreads();

    const uint8_t *my = idx + (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 64u;
    const uint32_t lane = threadIdx.x & 31u;
    uint64_t acc = 0;
    for (uint32_t it = 0; it < iters; it++) {
#pragma unroll 8
        for (uint32_t k = 0; k < 64u; k++) {
            const uint32_t i8 = (lane + k * 32u + it) & 255u;   /* lane-strided */
            acc += grid[i8] + signs[i8 & 127u] + my[k];
        }
    }
    if (acc == 0xffffffffffffffffull) sink[0] = acc;
}

template <typename L> static double time_it(L launch, int reps) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    launch(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaDeviceSynchronize());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return (double)ms / reps;
}

int main(int argc, char **argv) {
    int reps = 20; uint32_t iters = 64; uint32_t blocks = 384;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--iters=", 8)) iters = (uint32_t)atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--blocks=", 9)) blocks = (uint32_t)atoi(argv[i] + 9);
    }
    const uint32_t threads = 256;
    const uint64_t n_idx = (uint64_t)blocks * threads * 64u;
    uint8_t *idx = NULL; uint64_t *sink = NULL;
    CHECK(cudaMalloc(&idx, n_idx));
    CHECK(cudaMalloc(&sink, sizeof(uint64_t)));
    uint8_t *h = (uint8_t *)malloc(n_idx);
    for (uint64_t i = 0; i < n_idx; i++) h[i] = (uint8_t)((i * 1103515245u + 12345u) >> 16);
    CHECK(cudaMemcpy(idx, h, n_idx, cudaMemcpyHostToDevice));
    free(h);

    const double lookups = (double)blocks * threads * 64.0 * iters;
    printf("grid %u x %u, %.2f G LUT lookups per launch\n\n", blocks, threads, lookups / 1e9);

    const double t_u64 = time_it([&]{ lut_u64_kernel<<<blocks, threads>>>(idx, iters, sink); }, reps);
    const double t_u32 = time_it([&]{ lut_u32x2_kernel<<<blocks, threads>>>(idx, iters, sink); }, reps);
    const double t_nc  = time_it([&]{ lut_noconflict_kernel<<<blocks, threads>>>(idx, iters, sink); }, reps);

    printf("layout                       ms     G lookups/s   vs conflict-free\n");
    printf("  uint64 grid (live)     %7.3f   %11.1f   %13.2fx\n",
           t_u64, lookups / (t_u64 * 1e-3) / 1e9, t_nc / t_u64);
    printf("  uint32 x2 split        %7.3f   %11.1f   %13.2fx\n",
           t_u32, lookups / (t_u32 * 1e-3) / 1e9, t_nc / t_u32);
    printf("  lane-strided (control) %7.3f   %11.1f   %13.2fx\n",
           t_nc, lookups / (t_nc * 1e-3) / 1e9, 1.0);
    printf("\nsplit vs live: %.2fx\n", t_u64 / t_u32);
    return 0;
}
