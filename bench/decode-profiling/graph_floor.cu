/* Does a CUDA graph amortise this box's ~5.5us launch floor?
 * Compares N individual launches against one graph replay containing the same
 * N nodes, for chain lengths that bracket ns4's decode islands (a handful of
 * kernels) and a whole decode layer (~37) and a whole token (~1600). */
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void tiny_kernel(float *p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.000001f + 1e-7f;
}

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("%d: %s\n",__LINE__,cudaGetErrorString(e_));return 1;} } while(0)

int main() {
    float *d; CK(cudaMalloc(&d, 4096 * 4));
    CK(cudaMemset(d, 0, 4096 * 4));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    const int chains[] = {4, 8, 37, 128, 400, 1600};
    printf("%-8s %14s %14s %10s\n", "kernels", "eager_us", "graph_us", "speedup");
    for (int n : chains) {
        const int reps = n >= 400 ? 20 : 200;

        for (int i = 0; i < n; i++) tiny_kernel<<<4, 128, 0, s>>>(d, 512);
        CK(cudaStreamSynchronize(s));

        CK(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; r++)
            for (int i = 0; i < n; i++) tiny_kernel<<<4, 128, 0, s>>>(d, 512);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms_eager = 0; CK(cudaEventElapsedTime(&ms_eager, e0, e1));

        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < n; i++) tiny_kernel<<<4, 128, 0, s>>>(d, 512);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));

        CK(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; r++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms_graph = 0; CK(cudaEventElapsedTime(&ms_graph, e0, e1));

        const double us_e = ms_eager * 1000.0 / reps, us_g = ms_graph * 1000.0 / reps;
        printf("%-8d %14.2f %14.2f %9.2fx   (%.2f vs %.2f us per kernel)\n",
               n, us_e, us_g, us_e / us_g, us_e / n, us_g / n);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
    return 0;
}
