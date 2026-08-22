/* Occupancy of the single-block decode reductions, priced honestly.
 *
 * Two of the three most expensive kernels in a decode token launch ONE block
 * on a 188-SM GPU, because their grid is `rows` and decode has one row:
 *
 *     rms_norm_plain_batch8_kernel            <<<1, 256>>>   ns4_cuda.cu:16718
 *     hc_split_weighted_sum_norm_fused_kernel <<<n_rows,256>>> ns4_cuda.cu:26264
 *
 * kprof ranks them at 17.6 and 27.8 us/call, but kprof brackets each launch
 * with events and a 1-block grid absorbs WDDM submission backpressure into
 * that bracket, so those numbers are an upper bound, not kernel time.  This
 * bench is the control: same kernels, timed three ways.
 *
 *   graph-amortised  reps launches replayed as one captured graph, so the
 *                    ~6 us submission floor is paid once.  This is kernel work.
 *   eager            the same launches unbatched, i.e. how decode issues them
 *                    outside an island.  Only part of a kernel win survives here.
 *   empty            the same grid doing nothing: the floor itself.
 *
 * The variant under test keeps one launch and adds a *column* axis to the
 * grid, sized from the device's SM count rather than from any context
 * threshold -- rows*chunks blocks, so prefill (many rows) is untouched and
 * decode (one row) spreads over the machine.  Every block redundantly runs
 * the identical full reduction, in the baseline's exact association order,
 * and writes only its own slice of the output.  The redundant reads are
 * served by L2 after the first block touches them; the point of paying them
 * is that the alternative -- splitting the reduction -- needs either a second
 * launch or a grid-wide barrier, and at this size a launch costs more than
 * the arithmetic does.  Because every block sums in the same order, the
 * result is bit-identical to the baseline, which this bench asserts by
 * memcmp rather than by tolerance.
 *
 *   nvcc -O3 -arch=sm_120 -o occ_bench occ_bench.cu
 *   ./occ_bench                 # n_embd 4096, the live decode shape
 *   ./occ_bench --chunks=N      # pin the column split instead of auto
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

/* ---------------- baselines, copied verbatim from ns4_cuda.cu ------------- */

__global__ static void rms_norm_plain_batch8_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        sum += v0 * v0;
        sum += v1 * v1;
        sum += v2 * v2;
        sum += v3 * v3;
        sum += v4 * v4;
        sum += v5 * v5;
        sum += v6 * v6;
        sum += v7 * v7;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        orow[i] = v0 * scale;
        orow[i + 256u] = v1 * scale;
        orow[i + 512u] = v2 * scale;
        orow[i + 768u] = v3 * scale;
        orow[i + 1024u] = v4 * scale;
        orow[i + 1280u] = v5 * scale;
        orow[i + 1536u] = v6 * scale;
        orow[i + 1792u] = v7 * scale;
    }
}

/* ---------------- variant: same reduction, grid gains a column axis ------- */

__global__ static void rms_norm_plain_batch8_occ_kernel(
        float *out, const float *x, uint32_t n, uint32_t rows, float eps,
        uint32_t chunks) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;

    /* Identical to the baseline, in every block: same strides, same eight
     * accumulate steps, same tree reduction, therefore the same float. */
    float sum = 0.0f;
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        sum += v0 * v0;
        sum += v1 * v1;
        sum += v2 * v2;
        sum += v3 * v3;
        sum += v4 * v4;
        sum += v5 * v5;
        sum += v6 * v6;
        sum += v7 * v7;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);

    /* Only this block's slice is written.  x[i]*scale does not depend on the
     * order the slices are visited, so the output is bit-identical. */
    const uint32_t per = (n + chunks - 1u) / chunks;
    const uint32_t c0 = blockIdx.y * per;
    const uint32_t c1 = c0 + per < n ? c0 + per : n;
    for (uint32_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void empty_kernel(void) {}

/* ---------------------------- timing harness ----------------------------- */

struct Timing { double graph_us, eager_us, empty_us; };

template <typename Launch>
static double time_graph(Launch launch, int reps) {
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    launch(s);                                   /* warm */
    CHECK(cudaStreamSynchronize(s));
    cudaGraph_t g; cudaGraphExec_t e;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < reps; i++) launch(s);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&e, g, NULL, NULL, 0));
    CHECK(cudaGraphLaunch(e, s));                /* warm the exec */
    CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, s));
    CHECK(cudaGraphLaunch(e, s));
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaStreamSynchronize(s));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaGraphExecDestroy(e)); CHECK(cudaGraphDestroy(g));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    CHECK(cudaStreamDestroy(s));
    return (double)ms * 1000.0 / reps;
}

template <typename Launch>
static double time_eager(Launch launch, int reps) {
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    launch(s);
    CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < reps; i++) launch(s);
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaStreamSynchronize(s));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    CHECK(cudaStreamDestroy(s));
    return (double)ms * 1000.0 / reps;
}

int main(int argc, char **argv) {
    uint32_t n = 4096, rows = 1, chunks = 0;
    int reps = 2000;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--n=", 4)) n = (uint32_t)atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--rows=", 7)) rows = (uint32_t)atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--chunks=", 9)) chunks = (uint32_t)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }

    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (chunks == 0) {
        /* Target the machine, not a context length: enough column chunks that
         * rows*chunks covers the SMs, capped so a chunk stays >= one warp of
         * useful width. */
        const uint32_t want = ((uint32_t)sms + rows - 1u) / rows;
        const uint32_t cap = n / 256u ? n / 256u : 1u;
        chunks = want < 1u ? 1u : (want > cap ? cap : want);
    }
    printf("device: %d SMs   n=%u rows=%u chunks=%u  (grid %ux%u vs %ux1)\n",
           sms, n, rows, chunks, rows, chunks, rows);

    const size_t elems = (size_t)n * rows;
    std::vector<float> hx(elems);
    for (size_t i = 0; i < elems; i++) hx[i] = (float)((i * 1103515245u + 12345u) % 1000) / 500.0f - 1.0f;

    float *dx, *d_base, *d_occ;
    CHECK(cudaMalloc(&dx, elems * sizeof(float)));
    CHECK(cudaMalloc(&d_base, elems * sizeof(float)));
    CHECK(cudaMalloc(&d_occ, elems * sizeof(float)));
    CHECK(cudaMemcpy(dx, hx.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_base, 0, elems * sizeof(float)));
    CHECK(cudaMemset(d_occ, 0, elems * sizeof(float)));

    const float eps = 1e-6f;
    auto base_launch = [&](cudaStream_t s) {
        rms_norm_plain_batch8_kernel<<<rows, 256, 0, s>>>(d_base, dx, n, rows, eps);
    };
    auto occ_launch = [&](cudaStream_t s) {
        dim3 g(rows, chunks, 1);
        rms_norm_plain_batch8_occ_kernel<<<g, 256, 0, s>>>(d_occ, dx, n, rows, eps, chunks);
    };
    auto empty_base = [&](cudaStream_t s) { empty_kernel<<<rows, 256, 0, s>>>(); };
    auto empty_occ = [&](cudaStream_t s) { dim3 g(rows, chunks, 1); empty_kernel<<<g, 256, 0, s>>>(); };

    /* correctness first: bit-identical, not merely close */
    base_launch(0); occ_launch(0);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> hb(elems), ho(elems);
    CHECK(cudaMemcpy(hb.data(), d_base, elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ho.data(), d_occ, elems * sizeof(float), cudaMemcpyDeviceToHost));
    const bool bitwise = memcmp(hb.data(), ho.data(), elems * sizeof(float)) == 0;
    double maxrel = 0.0;
    for (size_t i = 0; i < elems; i++) {
        const double d = fabs((double)hb[i] - (double)ho[i]);
        const double r = fabs((double)hb[i]) > 1e-30 ? d / fabs((double)hb[i]) : d;
        if (r > maxrel) maxrel = r;
    }
    printf("correctness: %s (max rel %.3g)\n",
           bitwise ? "BIT-IDENTICAL" : "DIFFERS", maxrel);

    Timing bt, ot;
    bt.graph_us = time_graph(base_launch, reps);
    bt.eager_us = time_eager(base_launch, reps);
    bt.empty_us = time_eager(empty_base, reps);
    ot.graph_us = time_graph(occ_launch, reps);
    ot.eager_us = time_eager(occ_launch, reps);
    ot.empty_us = time_eager(empty_occ, reps);

    printf("\nrms_norm_plain_batch8   graph-amortised      eager   empty-grid\n");
    printf("  baseline  <<<%2u,256>>>   %8.2f us  %8.2f us  %8.2f us\n",
           rows, bt.graph_us, bt.eager_us, bt.empty_us);
    printf("  occupancy <<<%2ux%2u>>>    %8.2f us  %8.2f us  %8.2f us\n",
           rows, chunks, ot.graph_us, ot.eager_us, ot.empty_us);
    printf("  speedup                  %8.2fx %9.2fx\n",
           bt.graph_us / ot.graph_us, bt.eager_us / ot.eager_us);
    printf("\nkernel work above the launch floor (eager - empty):\n");
    printf("  baseline %.2f us   occupancy %.2f us\n",
           bt.eager_us - bt.empty_us, ot.eager_us - ot.empty_us);
    return bitwise ? 0 : 2;
}
