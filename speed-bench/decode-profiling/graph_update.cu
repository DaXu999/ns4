/* What it costs to keep a captured graph usable when a scalar argument
 * changes every replay.
 *
 * The decode islands ds4 captures today are the position-INdependent parts of
 * a layer.  The ~600 launches per token that no island covers are the middle
 * of the layer -- rope, the KV store, the DSA compressor, attention -- and
 * they are uncovered for exactly one reason: they take `pos` / `n_kv` as
 * by-value kernel arguments, which a captured graph freezes.
 *
 * There are two ways to keep such a graph correct, and this benchmark prices
 * both against the thing they replace (an ordinary launch):
 *
 *   A. cudaGraphExecKernelNodeSetParams per position-dependent node, then one
 *      replay.  Costs CPU time per node but no GPU submission.
 *   B. Leave the graph alone and read the changing scalar from device memory,
 *      so nothing needs updating at all.  Costs one tiny H2D per token.
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -o graph_update graph_update.cu
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ static void touch_kernel(float *out, int pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += (float)pos;
}

__global__ static void touch_kernel_devpos(float *out, const int *pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += (float)*pos;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const int n_nodes = argc > 1 ? atoi(argv[1]) : 600;   /* uncovered launches/token */
    const int iters   = argc > 2 ? atoi(argv[2]) : 200;   /* tokens */
    const int n = 1024;

    float *buf = NULL;
    int *dpos = NULL;
    CHECK(cudaMalloc(&buf, n * sizeof(float)));
    CHECK(cudaMalloc(&dpos, sizeof(int)));
    CHECK(cudaMemset(buf, 0, n * sizeof(float)));

    const dim3 grid((n + 255) / 256), block(256);

    /* ---- reference: plain launches, the thing we want to replace ---- */
    for (int i = 0; i < n_nodes; i++) touch_kernel<<<grid, block>>>(buf, i, n);
    CHECK(cudaDeviceSynchronize());
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n_nodes; i++) touch_kernel<<<grid, block>>>(buf, it, n);
        CHECK(cudaDeviceSynchronize());
    }
    const double eager_ms = (now_ms() - t0) / iters;

    /* ---- A: capture once, patch every node's params per replay ---- */
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
    for (int i = 0; i < n_nodes; i++) touch_kernel<<<grid, block, 0, s>>>(buf, 0, n);
    CHECK(cudaStreamEndCapture(s, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

    size_t n_graph_nodes = 0;
    CHECK(cudaGraphGetNodes(graph, NULL, &n_graph_nodes));
    std::vector<cudaGraphNode_t> nodes(n_graph_nodes);
    CHECK(cudaGraphGetNodes(graph, nodes.data(), &n_graph_nodes));

    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());

    t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        for (size_t i = 0; i < n_graph_nodes; i++) {
            int pos = it;
            void *args[3] = { &buf, &pos, (void *)&n };
            cudaKernelNodeParams p = {};
            p.func = (void *)touch_kernel;
            p.gridDim = grid; p.blockDim = block; p.sharedMemBytes = 0;
            p.kernelParams = args; p.extra = NULL;
            CHECK(cudaGraphExecKernelNodeSetParams(exec, nodes[i], &p));
        }
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
    }
    const double patch_ms = (now_ms() - t0) / iters;

    /* ---- B: scalar lives in device memory; graph never changes ---- */
    cudaGraph_t graph2;
    cudaGraphExec_t exec2;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
    for (int i = 0; i < n_nodes; i++) touch_kernel_devpos<<<grid, block, 0, s>>>(buf, dpos, n);
    CHECK(cudaStreamEndCapture(s, &graph2));
    CHECK(cudaGraphInstantiate(&exec2, graph2, NULL, NULL, 0));
    CHECK(cudaGraphLaunch(exec2, 0));
    CHECK(cudaDeviceSynchronize());

    t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        CHECK(cudaMemcpyAsync(dpos, &it, sizeof(int), cudaMemcpyHostToDevice, 0));
        CHECK(cudaGraphLaunch(exec2, 0));
        CHECK(cudaDeviceSynchronize());
    }
    const double dev_ms = (now_ms() - t0) / iters;

    /* ---- C: re-capture every token, then cudaGraphExecUpdate ----
     * The interesting one for ds4: the encode path is untouched, it just runs
     * in capture mode exactly as islands 0/1 already do.  Whatever `pos` the
     * kernels bake in is whatever this token's capture recorded, so the
     * position dependence that keeps the middle of the layer out of a graph
     * today stops mattering.  Also varies the grid between tokens, the way an
     * attention split over a growing KV would. */
    double capture_ms = 0.0;
    bool update_ok = true;
    {
        cudaGraph_t g3;
        cudaGraphExec_t exec3;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        for (int i = 0; i < n_nodes; i++) touch_kernel<<<grid, block, 0, s>>>(buf, 0, n);
        CHECK(cudaStreamEndCapture(s, &g3));
        CHECK(cudaGraphInstantiate(&exec3, g3, NULL, NULL, 0));
        CHECK(cudaGraphLaunch(exec3, 0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGraphDestroy(g3));

        t0 = now_ms();
        for (int it = 0; it < iters && update_ok; it++) {
            const dim3 g_it((n + 255) / 256 + (it & 3));   /* grid changes too */
            cudaGraph_t gt;
            CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
            for (int i = 0; i < n_nodes; i++)
                touch_kernel<<<g_it, block, 0, s>>>(buf, it, n);
            CHECK(cudaStreamEndCapture(s, &gt));
            cudaGraphExecUpdateResult ur;
            cudaGraphNode_t bad = NULL;
#if CUDART_VERSION >= 12000
            cudaGraphExecUpdateResultInfo info = {};
            cudaError_t ue = cudaGraphExecUpdate(exec3, gt, &info);
            ur = info.result; bad = info.errorNode;
#else
            cudaError_t ue = cudaGraphExecUpdate(exec3, gt, &bad, &ur);
#endif
            if (ue != cudaSuccess || ur != cudaGraphExecUpdateSuccess) {
                (void)cudaGetLastError();
                update_ok = false;
                printf("  [C] cudaGraphExecUpdate failed: %s (result=%d)\n",
                       cudaGetErrorString(ue), (int)ur);
            }
            CHECK(cudaGraphDestroy(gt));
            CHECK(cudaGraphLaunch(exec3, 0));
            CHECK(cudaDeviceSynchronize());
        }
        capture_ms = (now_ms() - t0) / iters;
    }

    printf("nodes/token = %d, iters = %d\n\n", n_nodes, iters);
    printf("  %-46s %9.3f ms   %7.3f us/node\n",
           "eager launches (what decode does today)", eager_ms, eager_ms * 1000.0 / n_nodes);
    printf("  %-46s %9.3f ms   %7.3f us/node\n",
           "A: graph + SetParams on every node", patch_ms, patch_ms * 1000.0 / n_nodes);
    printf("  %-46s %9.3f ms   %7.3f us/node\n",
           "B: graph + device-resident scalar", dev_ms, dev_ms * 1000.0 / n_nodes);
    printf("  %-46s %9.3f ms   %7.3f us/node   %s\n",
           "C: re-capture + cudaGraphExecUpdate", capture_ms,
           capture_ms * 1000.0 / n_nodes, update_ok ? "" : "(FAILED)");
    printf("\n  A saves %.3f ms/token (%.2fx), B saves %.3f ms/token (%.2fx),"
           " C saves %.3f ms/token (%.2fx)\n",
           eager_ms - patch_ms, eager_ms / patch_ms,
           eager_ms - dev_ms, eager_ms / dev_ms,
           eager_ms - capture_ms, eager_ms / capture_ms);
    return 0;
}
