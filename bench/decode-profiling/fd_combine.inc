static constexpr uint32_t kFDDim        = 512u;   /* head_dim this path serves */
static constexpr uint32_t kFDLaneDims   = kFDDim / 32u;
static constexpr uint32_t kFDWarps      = 8u;     /* query heads per block */
static constexpr uint32_t kFDTile       = 8u;     /* KV rows staged per pass */
static constexpr uint32_t kFDMaxSplit   = 32u;
static constexpr uint32_t kFDRowsPerSplit = 24u;  /* split target, measured */

/* Dim slice one combine block owns.  512 / 128 = 4 blocks per head. */
static constexpr uint32_t kFDCombDims = 128u;

/* Fold the per-split partials into the final heads.
 *
 * Two structural problems with the shape this replaced:
 *
 *   - it launched exactly n_head = 64 blocks on a 188-SM GPU, so two thirds of
 *     the machine sat idle and the dim axis was walked by a loop instead of by
 *     the grid;
 *   - every one of the 256 threads in a block re-ran *both* serial n_split
 *     loops -- the running max and the expf rescale weights -- to compute
 *     values identical across the whole block, purely so that thread 0 could
 *     store fd_w[].  That is a 256x redundant transcendental prologue per
 *     head, and it sat on the critical path ahead of every output element.
 *
 * Now warp 0 computes the prologue once, cooperatively, and the dim axis moves
 * into the grid.  The arithmetic each output sees is untouched: fmaxf is exact
 * and order-independent, the denominator still accumulates from
 * expf(sink - m_max) over ascending s, and each output's weighted sum still
 * runs over ascending s.  Results are bit-identical to the old kernel, which
 * fd_combine_bench.cu asserts by memcmp.
 *
 * Size the win honestly before extending this to other decode kernels.  kprof
 * ranked the old shape at 359 us/call at ctx 32768 -- 35% of all eager decode
 * GPU time -- and that was an artifact of ranking a 64-block kernel by event
 * bracket, which absorbs WDDM submission backpressure instead of kernel time.
 * fd_combine_bench.cu prices the real thing, at the live n_split = 27:
 *
 *                     graph-amortised      eager, incl. ~6.0 us launch floor
 *     old  L2-warm       5.2 us                    9.2 us
 *     new  L2-warm       3.0 us  (1.8x)            7.8 us
 *     old  L2-cold       8.1 us                   11.1 us
 *     new  L2-cold       5.2 us  (1.6x)            9.0 us
 *
 * Decode issues this eagerly, so what lands is the eager column: 1.3-2.1 us
 * per layer, 43 layers, ~0.3-0.45% of a 19 ms token.  Real, free, and an order
 * of magnitude under the +/-5% this box drifts between runs -- which is why it
 * has to be measured standalone and cannot be read off a ds4-bench pair. */
__global__ __launch_bounds__(kFDCombDims)
static void attention_flash_decode_combine_kernel(
        float *heads,
        const float *sinks,
        const float *part_o,
        const float2 *part_ml,
        uint32_t n_head,
        uint32_t n_split) {
    /* [n_split] rescale weights, then [n_split] partial denominators. */
    extern __shared__ float fd_sh[];
    float *fd_w = fd_sh;
    float *fd_y = fd_sh + n_split;
    __shared__ float fd_inv;

    const uint32_t head = blockIdx.x;
    if (head >= n_head) return;
    const uint32_t d0 = blockIdx.y * kFDCombDims;

    if (threadIdx.x < 32u) {
        const uint32_t lane = threadIdx.x;
        float m = -INFINITY;
        for (uint32_t s = lane; s < n_split; s += 32u)
            m = fmaxf(m, part_ml[(uint64_t)s * n_head + head].x);
        for (uint32_t off = 16u; off; off >>= 1u)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        const float m_max = fmaxf(sinks[head], __shfl_sync(0xffffffffu, m, 0));
        for (uint32_t s = lane; s < n_split; s += 32u) {
            const float2 ml = part_ml[(uint64_t)s * n_head + head];
            fd_w[s] = ml.x == -INFINITY ? 0.0f : expf(ml.x - m_max);
            fd_y[s] = ml.y;
        }
        __syncwarp();
        if (lane == 0u) {
            float denom = expf(sinks[head] - m_max);
            for (uint32_t s = 0; s < n_split; s++) denom += fd_y[s] * fd_w[s];
            fd_inv = 1.0f / denom;
        }
    }
    __syncthreads();

    const float inv = fd_inv;
    const uint32_t dend = d0 + kFDCombDims < kFDDim ? d0 + kFDCombDims : kFDDim;
    for (uint32_t d = d0 + threadIdx.x; d < dend; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_split; s++) {
            acc += part_o[((uint64_t)s * n_head + head) * kFDDim + d] * fd_w[s];
        }
        heads[(uint64_t)head * kFDDim + d] = acc * inv;
    }
}
