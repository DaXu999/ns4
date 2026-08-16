// Discriminator: tpose32-fp4 with q loads replaced by a constant.
// If this collapses the time, per-head q re-reads are the wall.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MIRROR_ROW_BYTES 80u
#define CK(x) do{cudaError_t e=(x); if(e){printf("err %s\n",cudaGetErrorString(e));exit(1);} }while(0)
template <int MODE>  // 0=normal q loads, 1=no q loads, 2=no kdec reads
__global__ static void probe_kernel(float *scores, const float *q, const float *w,
                                    const unsigned char *mirror, uint32_t n_comp, float scale) {
    const uint32_t tid = threadIdx.x, col = tid & 31u, quarter = tid >> 5u;
    const uint32_t c0 = blockIdx.x * 32u;
    __shared__ float4 kdec[32][32];
    __shared__ float tail[32][12];
    const uint32_t rows = n_comp > c0 ? min(32u, n_comp - c0) : 0u;
    if (col < rows) {
        const unsigned char *mrow = mirror + (uint64_t)(c0 + col) * MIRROR_ROW_BYTES;
        const uint4 codes = ((const uint4 *)mrow)[quarter];
        const float sc = ((const float *)(mrow + 64u))[quarter];
        const unsigned char *cb = (const unsigned char *)&codes;
        #pragma unroll
        for (uint32_t j = 0; j < 32u; j++) {
            const uint32_t e = quarter * 32u + j;
            const unsigned char byte = cb[j >> 1u];
            const unsigned int code = (j & 1u) ? (byte >> 4u) : (byte & 0xFu);
            const float mag = (code & 7u) == 0 ? 0.f : (code & 7u) == 1 ? .5f : (1.f + .5f * (code & 1u)) * (float)(1 << (((code & 7u) >> 1u) - 1u));
            ((float *)kdec)[(e >> 2u) * 128u + col * 4u + (e & 3u)] = ((code & 8u) ? -mag : mag) * sc;
        }
    }
    __syncthreads();
    const uint32_t c = c0 + col;
    if (c >= n_comp) return;
    float partial[32]; float total = 0.0f; float t4[4];
    #pragma unroll 1
    for (uint32_t i = 0; i < 16u; i++) {
        const uint32_t h = quarter * 16u + i;
        const float4 *qh = (const float4 *)(q + (uint64_t)h * 128u);
        #pragma unroll
        for (uint32_t g = 0; g < 32u; g++) {
            const float4 qv = MODE == 1 ? make_float4(1.f, 2.f, 3.f, 4.f) : __ldg(qh + g);
            const float4 kv = MODE == 2 ? make_float4(1.f, 2.f, 3.f, 4.f) : kdec[g][col];
            partial[g] = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            for (uint32_t l = 0; l < off; l++) partial[l] += partial[l + off];
        t4[i & 3u] = __fmul_rn(__fmul_rn(fmaxf(partial[0], 0.0f), __ldg(w + h)), scale);
        if ((i & 3u) == 3u) {
            const float g4 = __fadd_rn(__fadd_rn(__fadd_rn(t4[0], t4[1]), t4[2]), t4[3]);
            if (quarter == 0u) total = __fadd_rn(total, g4);
            else tail[col][(quarter - 1u) * 4u + (i >> 2u)] = g4;
        }
    }
    __syncthreads();
    if (quarter == 0u) {
        #pragma unroll
        for (uint32_t j = 0; j < 12u; j++) total = __fadd_rn(total, tail[col][j]);
        scores[c] = total;
    }
}
int main() {
    uint32_t n = 32768; const int POOL = 20, ITER = 40;
    float *q, *w, *s; unsigned char *mp[POOL];
    CK(cudaMalloc(&q, 64*128*4)); CK(cudaMalloc(&w, 64*4)); CK(cudaMalloc(&s, n*4));
    for (int p = 0; p < POOL; p++) CK(cudaMalloc(&mp[p], (uint64_t)n*MIRROR_ROW_BYTES));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const char *names[3] = {"normal", "no-q  ", "no-k  "};
    for (int m = 0; m < 3; m++) {
        for (int i = 0; i < 8; i++) {
            if (m==0) probe_kernel<0><<<(n+31)/32,128>>>(s,q,w,mp[i%POOL],n,0.01f);
            if (m==1) probe_kernel<1><<<(n+31)/32,128>>>(s,q,w,mp[i%POOL],n,0.01f);
            if (m==2) probe_kernel<2><<<(n+31)/32,128>>>(s,q,w,mp[i%POOL],n,0.01f);
        }
        CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
        for (int i = 0; i < ITER; i++) {
            if (m==0) probe_kernel<0><<<(n+31)/32,128>>>(s,q,w,mp[i%POOL],n,0.01f);
            if (m==1) probe_kernel<1><<<(n+31)/32,128>>>(s,q,w,mp[i%POOL],n,0.01f);
            if (m==2) probe_kernel<2><<<(n+31)/32,128>>>(s,q,w,mp[i%POOL],n,0.01f);
        }
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("%s  %8.2f us\n", names[m], ms*1000.0/ITER);
    }
    return 0;
}
