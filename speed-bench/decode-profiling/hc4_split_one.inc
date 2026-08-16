__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}
__device__ static void hc4_split_one_warp(
        float *out, const float *mix, const float *scale, const float *base,
        uint32_t sinkhorn_iters, float epsv) {
    const uint32_t lane = threadIdx.x & 31u;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (lane < 4u) {
        const float z = mix[lane] * pre_scale + base[lane];
        out[lane] = 1.0f / (1.0f + expf(-z)) + epsv;
    } else if (lane < 8u) {
        const float z = mix[lane] * post_scale + base[lane];
        out[lane] = 2.0f / (1.0f + expf(-z));
    }

    const uint32_t r = lane >> 2;
    const uint32_t col = lane & 3u;
    float v = (lane < 16u) ? mix[8u + lane] * comb_scale + base[8u + lane] : 0.0f;

    /* row softmax */
    float m = -INFINITY;
#pragma unroll
    for (uint32_t k = 0; k < 4u; k++)
        m = fmaxf(m, __shfl_sync(0xffffffffu, v, r * 4u + k));
    v = expf(v - m);
    float rs = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < 4u; k++)
        rs += __shfl_sync(0xffffffffu, v, r * 4u + k);
    v = v / rs + epsv;

    /* one column pass, then (sinkhorn_iters - 1) row/column pairs */
    float cs = epsv;
#pragma unroll
    for (uint32_t k = 0; k < 4u; k++)
        cs += __shfl_sync(0xffffffffu, v, col + 4u * k);
    v /= cs;

#pragma unroll 1
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        float s = epsv;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            s += __shfl_sync(0xffffffffu, v, r * 4u + k);
        v /= s;
        s = epsv;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            s += __shfl_sync(0xffffffffu, v, col + 4u * k);
        v /= s;
    }
    if (lane < 16u) out[8u + lane] = v;
}
