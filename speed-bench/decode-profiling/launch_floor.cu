/* How much of the microbenchmark plateau is CPU launch submission rather than
 * GPU work?  Times an empty kernel in exactly the same back-to-back loop. */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void empty_kernel() {}
__global__ void empty_kernel_grid(float *p) { if (threadIdx.x == 1u << 30) p[0] = 1.0f; }

int main() {
    const int iters = 2000;
    float *d; cudaMalloc(&d, 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 100; i++) empty_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    struct Case { const char *name; int grid; int block; };
    Case cases[] = {{"1 block x 32", 1, 32}, {"80 blocks x 128", 80, 128},
                    {"216 blocks x 256", 216, 256}, {"256 blocks x 128", 256, 128}};

    for (auto &c : cases) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++) empty_kernel_grid<<<c.grid, c.block>>>(d);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        printf("empty kernel %-18s : %6.2f us per launch (GPU timeline)\n",
               c.name, ms * 1000.0 / iters);
    }

    /* two launches per iteration, like the attention path */
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        empty_kernel_grid<<<80, 128>>>(d);
        empty_kernel_grid<<<256, 128>>>(d);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("two empty launches         : %6.2f us per iteration\n", ms * 1000.0 / iters);

    /* CPU-side submission cost alone */
    cudaEventRecord(e0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) empty_kernel_grid<<<80, 128>>>(d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double cpu_us = ((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3) / iters;
    cudaDeviceSynchronize();
    printf("CPU submit cost            : %6.2f us per launch\n", cpu_us);
    return 0;
}
