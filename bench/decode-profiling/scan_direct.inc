template <bool HOIST_ACCUM>
__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    /* One slot per head rather than one per warp. The four warps used to
     * rendezvous on a shared partial[4] every four heads -- two
     * __syncthreads() and a four-term add on thread 0, sixteen times, so 32
     * barriers per block and a serial tail interleaved through the whole
     * loop. Giving each head its own slot lets the warps run the 16 rounds
     * with no barrier at all and defers the accumulation to one pass at the
     * end. Every head's term is still the same expression over the same warp
     * reduction, and the final sum still walks h in the same order and
     * associates the same way, so this is bit-identical -- what changes is
     * when the adds happen, not which. */
    __shared__ float head_term[HOIST_ACCUM ? 64 : 4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        const float term = fmaxf(dot, 0.0f) * weights[h] * scale;
        if (HOIST_ACCUM) {
            if (lane == 0) head_term[h] = term;
        } else {
            if (lane == 0) head_term[warp] = term;
            __syncthreads();
            if (tid == 0) total += head_term[0] + head_term[1] + head_term[2] + head_term[3];
            __syncthreads();
        }
    }
    if (HOIST_ACCUM) {
        __syncthreads();
        if (tid == 0) {
            for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
                total += head_term[h0] + head_term[h0 + 1u] +
                         head_term[h0 + 2u] + head_term[h0 + 3u];
            }
        }
    }
    if (tid == 0) scores[c] = total;
}
