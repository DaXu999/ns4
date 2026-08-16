/* Why moe_gate_up_mid_decode_lut_qwarp32_kernel only reaches 348 GB/s.
 *
 * That kernel is 11.5% of decode GPU time and runs at 21% of this box's
 * measured 1642 GB/s roofline (bw_probe.cu), while its sibling
 * moe_down_sum6_qwarp32 reaches 811 GB/s on the same kind of data.  Three
 * candidate causes, and they want different fixes:
 *
 *   A. the 8-lane block loop  (for b = lane; b < xq_blocks; b += 8) --
 *      only 8 lanes per row, 2 iterations at xq_blocks = 16
 *   B. sector waste from the 66-byte block_iq2_xxs stride -- a warp's 16-byte
 *      loads land at addresses 66 apart, so each 128 bytes of useful data
 *      drags in far more sectors than it uses
 *   C. bank conflicts on the shared uint64 s_iq2_grid[256] LUT
 *
 * This bench isolates B, the one that is purely an addressing property, by
 * stripping out the dot product and the LUT entirely.  Same bytes, same thread
 * geometry, only the access shape changes:
 *
 *   block_per_lane   what the kernel does now: lane L owns block L (66 B)
 *   coalesced        the 8 lanes sweep the same row as consecutive 16 B chunks
 *   padded68         block_per_lane, but blocks padded 66 -> 68 (4 B aligned)
 *   padded72         block_per_lane, but blocks padded 66 -> 72 (8 B aligned)
 *
 * If block_per_lane is far below coalesced, B is the cause and the fix is to
 * stage rows through shared memory (or pad the block stride).  If they are
 * close, the 4x is A or C and this bench says so by exclusion.
 *
 *   nvcc -O3 -arch=sm_120 -o iq2_access_bench iq2_access_bench.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

/* the live shape: 4096-wide input -> 16 superblocks of 256 elements */
static constexpr uint32_t kBlocks = 16;
static constexpr uint32_t kIQ2Bytes = 66;    /* fp16 d + 32 x uint16 qs */

/* A: lane L owns whole 66-byte blocks, striding by 8 -- today's kernel. */
template <uint32_t BLKB>
__global__ static void block_per_lane_kernel(const char *__restrict__ base,
                                             uint32_t rows, float *sink) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 3u) + row_lane;
    if (row >= rows) return;
    const char *r = base + (uint64_t)row * kBlocks * BLKB;
    uint32_t acc = 0;
    for (uint32_t b = lane; b < kBlocks; b += 8u) {
        const uint16_t *q = (const uint16_t *)(r + (uint64_t)b * BLKB);
#pragma unroll
        for (uint32_t i = 0; i < 33u; i++) acc += q[i];   /* 66 bytes */
    }
    if (acc == 0xffffffffu) sink[0] = (float)acc;
}

/* B: the same row bytes, swept as consecutive 16-byte chunks by the 8 lanes. */
__global__ static void coalesced_kernel(const char *__restrict__ base,
                                        uint32_t rows, float *sink) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 3u) + row_lane;
    if (row >= rows) return;
    const char *r = base + (uint64_t)row * kBlocks * kIQ2Bytes;
    const uint32_t row_bytes = kBlocks * kIQ2Bytes;         /* 1056 */
    uint32_t acc = 0;
    /* 1056 B = 66 chunks of 16 B; 8 lanes sweep them consecutively */
    for (uint32_t c = lane; c * 16u + 16u <= row_bytes; c += 8u) {
        uint4 v = *(const uint4 *)(r + (uint64_t)c * 16u);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xffffffffu) sink[0] = (float)acc;
}

__global__ static void empty_kernel(void) {}

template <typename L>
static double time_it(L launch, int reps) {
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
    int reps = 20;
    double gb = 4.0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--gb=", 5)) gb = atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    /* enough rows that the working set is far past L2 */
    const uint64_t want = (uint64_t)(gb * 1024.0 * 1024.0 * 1024.0);
    const uint32_t rows = (uint32_t)(want / (kBlocks * 72));   /* size for the widest stride */
    const uint32_t threads = 256;
    const uint32_t blocks = (rows + (threads >> 3) - 1) / (threads >> 3);

    char *buf = NULL; float *sink = NULL;
    CHECK(cudaMalloc(&buf, (uint64_t)rows * kBlocks * 72));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(buf, 1, (uint64_t)rows * kBlocks * 72));

    const double useful = (double)rows * kBlocks * kIQ2Bytes;   /* bytes the kernel needs */
    printf("device: %d SMs   rows=%u  useful bytes=%.2f GB  grid=%u x %u threads\n\n",
           sms, rows, useful / 1e9, blocks, threads);

    const double t_a  = time_it([&]{ block_per_lane_kernel<66><<<blocks, threads>>>(buf, rows, sink); }, reps);
    const double t_c  = time_it([&]{ coalesced_kernel<<<blocks, threads>>>(buf, rows, sink); }, reps);
    const double t_68 = time_it([&]{ block_per_lane_kernel<68><<<blocks, threads>>>(buf, rows, sink); }, reps);
    const double t_72 = time_it([&]{ block_per_lane_kernel<72><<<blocks, threads>>>(buf, rows, sink); }, reps);
    const double t_e  = time_it([&]{ empty_kernel<<<blocks, threads>>>(); }, reps);

    printf("pattern            ms      useful GB/s   vs coalesced\n");
    const double bw_c = useful / (t_c * 1e-3) / 1e9;
    struct { const char *n; double t; } rowsv[] = {
        {"block_per_lane (66)", t_a}, {"coalesced       ", t_c},
        {"padded68        ", t_68},   {"padded72        ", t_72},
    };
    for (auto &r : rowsv) {
        const double bw = useful / (r.t * 1e-3) / 1e9;
        printf("  %s %7.3f   %10.1f   %9.2fx\n", r.n, r.t, bw, bw / bw_c);
    }
    printf("\nempty grid: %.3f ms (launch is not a factor here)\n", t_e);
    printf("roofline on this box is 1642 GB/s (bw_probe.cu)\n");
    printf("the live kernel achieves 348 GB/s including LUT + dot product\n");
    return 0;
}
