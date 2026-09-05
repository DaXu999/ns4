#ifndef NS4_GPU_H
#define NS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the NS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
#ifndef NS4_GPU_TENSOR_DEFINED
#define NS4_GPU_TENSOR_DEFINED
typedef struct ns4_gpu_tensor ns4_gpu_tensor;
#endif

#ifndef NS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define NS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define NS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ns4_gpu_attention_decode_row;
#endif

int ns4_gpu_init(void);
void ns4_gpu_cleanup(void);

ns4_gpu_tensor *ns4_gpu_tensor_alloc(uint64_t bytes);
ns4_gpu_tensor *ns4_gpu_tensor_alloc_managed(uint64_t bytes);
ns4_gpu_tensor *ns4_gpu_tensor_view(const ns4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ns4_gpu_tensor_free(ns4_gpu_tensor *tensor);
uint64_t ns4_gpu_tensor_bytes(const ns4_gpu_tensor *tensor);
int ns4_gpu_tensor_fill_f32(ns4_gpu_tensor *tensor, float value, uint64_t count);
int ns4_gpu_tensor_write(ns4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ns4_gpu_tensor_read(const ns4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ns4_gpu_tensor_copy(ns4_gpu_tensor *dst, uint64_t dst_offset,
                          const ns4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
/* Up to four same-tier device copies in one launch on the decode stream,
 * asynchronous to the host (unlike ns4_gpu_tensor_copy's blocking
 * cudaMemcpy). Unused pairs pass NULL/NULL. */
/* Up to 16 int32 values written from a by-value argument block, one async
 * launch on the decode stream (no host-side blocking copy). */
int ns4_gpu_tensor_write_i32_small_tensor(ns4_gpu_tensor *dst,
                                          const int32_t *vals,
                                          uint32_t n);
int ns4_gpu_tensor_copy4_async_tensor(
        ns4_gpu_tensor *dst0, uint64_t doff0, const ns4_gpu_tensor *src0, uint64_t soff0, uint64_t b0,
        ns4_gpu_tensor *dst1, uint64_t doff1, const ns4_gpu_tensor *src1, uint64_t soff1, uint64_t b1,
        ns4_gpu_tensor *dst2, uint64_t doff2, const ns4_gpu_tensor *src2, uint64_t soff2, uint64_t b2,
        ns4_gpu_tensor *dst3, uint64_t doff3, const ns4_gpu_tensor *src3, uint64_t soff3, uint64_t b3);
/* Up to 256 same-tier device copies (16-byte-aligned source offsets --
 * src_off may be NULL for all-zero -- into offset 0 of each destination,
 * 16-byte-multiple sizes) in ONE launch on the decode stream, asynchronous
 * to the host.  Refuses (returns 0) on any shape it cannot take; the caller
 * keeps its per-copy loop as the fallback. */
int ns4_gpu_tensor_copy_batch(ns4_gpu_tensor *const *dst,
                              const ns4_gpu_tensor *const *src,
                              const uint64_t *src_off,
                              const uint64_t *bytes,
                              uint32_t n);
int ns4_gpu_tensor_copy_f32_to_f16(ns4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ns4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
int ns4_gpu_moe_handoff_pack_tensor(
        ns4_gpu_tensor       *packed,
        const ns4_gpu_tensor *ffn_norm,
        const ns4_gpu_tensor *selected,
        const ns4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert);
/* Gather `n_rows` rows starting at `row0` from every slot of a
 * [n_slots][slot_cap][width] buffer into contiguous [row][slot * width] rows.
 * `row0` lets a caller pack a window that does not start at the slot base,
 * which is how a speculative block's verified rows are read back. */
int ns4_gpu_pack_slot_rows_f32_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *slots,
        uint32_t                row0,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

int ns4_gpu_begin_commands(void);
int ns4_gpu_flush_commands(void);
int ns4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ns4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ns4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
int ns4_gpu_end_commands(void);
/* Same boundary without the host wait: for phases whose consumer is more
 * device work (launches or the stream-ordered blocking tensor reads/writes/
 * copies); execution errors surface at the next read. */
int ns4_gpu_end_commands_async(void);
int ns4_gpu_synchronize(void);
/* Side stream inside an eager decode island: begin forks (later launches go
 * to the side stream; 0 = refused, launches stay on the decode stream),
 * leave returns launches to the decode stream with the side work in flight,
 * join makes the decode stream wait for it.  Callers keep the two streams'
 * buffers disjoint between fork and join. */
int ns4_gpu_side_begin(void);
void ns4_gpu_side_leave(void);
int ns4_gpu_side_join(void);

int ns4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ns4_gpu_set_model_fd(int fd);
int ns4_gpu_set_model_fd_for_map(int fd, const void *model_map);
/* Size the weight arena's first chunk to exactly this many bytes (the
 * engine's sum of the startup tensor spans); 0 keeps chunked growth. */
void ns4_gpu_set_weight_arena_reserve(uint64_t bytes);
/* Build the aligned IQ2_XXS/Q2_K expert artifacts of one GGUF before its
 * weight upload; one build per model map.  model_id labels the log lines:
 * "base" for the main model, "support" for the runtime support model. */
int ns4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size,
                                    const char *model_path, const char *model_id);
int ns4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes);
int ns4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ns4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ns4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ns4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ns4_gpu_q8_cache_suppressed(void);
void ns4_gpu_set_q8_cache_suppressed(int suppressed);

/* Model-file ranges assigned to CUDA devices by the multi-GPU placement
 * planner. Declared unconditionally for the shared engine interface. */
#ifndef NS4_MAX_GPUS
#define NS4_MAX_GPUS 16
#endif
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int target_device;
} ns4_tensor_range;

int ns4_gpu_device_cache_tensors(int device_id,
                                 const ns4_tensor_range *ranges,
                                 int n_ranges);
int ns4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias);
int ns4_gpu_device_cache_support_tensors(int device_id,
                                         int entry_device_id,
                                         const ns4_tensor_range *ranges,
                                         int n_ranges,
                                         int from_main_map);
uint64_t ns4_gpu_tier_free_vram(int logical_tier);
int ns4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                         int *out_device_id, void **out_device_ptr);
int ns4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes);

int ns4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ns4_gpu_set_quality(bool quality);
void ns4_gpu_set_ssd_streaming(bool enabled);
typedef struct ns4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ns4_gpu_stream_expert_table;
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ns4_gpu_stream_expert_cache_reset_route_hotness(void);
int ns4_gpu_stream_expert_cache_begin_selected_load(
        const ns4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
#if !defined(NS4_NO_GPU)
int ns4_gpu_stream_expert_cache_prepare_selected_batch(
        const ns4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
#endif
void ns4_gpu_print_memory_report(const char *label);

/* Sliced projections used by the CUDA tensor-parallel path
 * (--cuda-tensor-parallel), where each device owns one contiguous slice.
 * k offsets/counts are multiples of 32 (Q8_0 block) and the slices' partial
 * results sum to the full projection.
 *
 * ns4_gpu_matmul_q8_0_kslice_rows_tensor: each input row contains only the
 * owned contiguous K slice, while each output row spans the full projection
 * width.
 *
 * ns4_gpu_attention_output_q8_tp_tensor is the group-sliced attention output
 * pair: low projection for groups [group0, group0+group_cnt) plus the
 * matching k-slice of the expand projection, producing this slice's partial
 * attention block output (n_tokens == 1 only). */
int ns4_gpu_matmul_q8_0_kslice_rows_tensor(
        ns4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              full_in_dim,
        uint64_t              out_dim,
        uint64_t              k_off,
        uint64_t              k_cnt,
        const ns4_gpu_tensor *x,
        uint64_t              n_rows);
int ns4_gpu_attention_output_q8_tp_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ns4_gpu_tensor *heads);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ns4_gpu_embed_token_hc_tensor(
        ns4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ns4_gpu_embed_tokens_hc_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* All indexer scoring reads the fp4 mirror rows (the only persistent store);
 * single-token decode scans them directly, batch scorers expand them into an
 * internal f32 scratch (exact dequant) or transcode them for the MXF4 path. */
int ns4_gpu_indexer_score_one_tensor(
        ns4_gpu_tensor       *scores,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *weights,
        const ns4_gpu_tensor *index_mirror,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ns4_gpu_indexer_scores_prefill_tensor(
        ns4_gpu_tensor       *scores,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *weights,
        const ns4_gpu_tensor *index_mirror,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ns4_gpu_indexer_scores_decode_batch_tensor(
        ns4_gpu_tensor       *scores,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *weights,
        const ns4_gpu_tensor *index_mirror,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ns4_gpu_dspark_markov_argmax_tensor(ns4_gpu_tensor *out_idx,
                                        const ns4_gpu_tensor *logits_row,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t w1_offset,
                                        uint64_t w2_offset,
                                        uint32_t prev_token,
                                        uint32_t vocab,
                                        uint32_t rank);
/* Whole-proposal chain of the above: keys of slots 1..n_slots-1 (u64 at
 * scratch+0, 0 = failed) and their confidence logits (f32 at scratch+128),
 * enqueued back to back on the device; the host reads once.  Slot 0's
 * token is `first_prev_token`. */
int ns4_gpu_dspark_markov_chain_tensor(ns4_gpu_tensor *scratch,
                                       const ns4_gpu_tensor *logits_rows,
                                       const ns4_gpu_tensor *hidden_rows,
                                       const void *model_map,
                                       uint64_t model_size,
                                       uint64_t w1_offset,
                                       uint64_t w2_offset,
                                       uint64_t conf_offset,
                                       uint32_t first_prev_token,
                                       uint32_t n_slots,
                                       uint32_t vocab,
                                       uint32_t rank,
                                       uint32_t n_embd);
int ns4_gpu_indexer_topk_tensor(
        ns4_gpu_tensor       *selected,
        const ns4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* Every verify row's indexer scan + top-k in three launches, bit-identical
 * to the per-row int-scan chain; returns 0 (caller keeps its per-row
 * calls) when that chain is not the live per-row path.  iq_batch holds
 * n_rows x 64 x 128 query floats, iw_batch n_rows x 64 weights; row r
 * scores land at r * scores_row_stride floats and its top-k at
 * r * selected_row_stride u32s. */
int ns4_gpu_indexer_scan_topk_rows_tensor(
        ns4_gpu_tensor       *scores_buf,
        uint64_t                scores_row_stride,
        const ns4_gpu_tensor *iq_batch,
        const ns4_gpu_tensor *iw_batch,
        const ns4_gpu_tensor *index_mirror,
        ns4_gpu_tensor       *selected_batch,
        uint32_t                selected_row_stride,
        const uint32_t       *n_comp_rows,
        uint32_t                n_rows,
        uint32_t                top_k,
        float                   scale);

int ns4_gpu_indexer_top1_value_tensor(
        ns4_gpu_tensor       *selected,
        ns4_gpu_tensor       *values,
        const ns4_gpu_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              index_offset);

int ns4_gpu_matmul_q8_0_top1_tensor(
        ns4_gpu_tensor       *selected,
        ns4_gpu_tensor       *values,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ns4_gpu_tensor *x,
        uint32_t              index_offset);

int ns4_gpu_set_decode_fast_attention(int enabled);
int ns4_gpu_set_decode_score_vec4(int enabled);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ns4_gpu_argmax_tensor(
        ns4_gpu_tensor       *out_idx,
        const ns4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ns4_gpu_dsv4_topk_mask_tensor(
        ns4_gpu_tensor       *mask,
        const ns4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and NS4's tail-only RoPE.
 */

int ns4_gpu_matmul_q8_0_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

int ns4_gpu_matmul_quant_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ns4_gpu_matmul_q8_0_pair_tensor(
        ns4_gpu_tensor       *out0,
        ns4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

/* Small-batch preference: while set, 2..8-row dense GEMMs, the attention
 * output projection and the routed MoE take the weight-shared decode rows
 * kernels instead of the MMQ/cuBLAS batch paths.  The DSpark proposer sets
 * it around its support-model work (its numerics only shape drafts). */
void ns4_gpu_small_batch_rows_pref(int on);

/* Multi-row decode projections that preserve the one-row reduction order. */
int ns4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ns4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ns4_gpu_tensor *x,
        uint32_t              n_rows);
int ns4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ns4_gpu_tensor       *out0,
        ns4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ns4_gpu_tensor *x,
        uint32_t              n_rows);

int ns4_gpu_matmul_q8_0_f16_out_tensor(
        ns4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

int ns4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ns4_gpu_tensor       *gate,
        ns4_gpu_tensor       *up,
        ns4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        float                   clamp);
int ns4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ns4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        float                   clamp,
        const ns4_gpu_tensor *selected,
        const ns4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank);

int ns4_gpu_matmul_f16_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

/* CUDA batch path: fold an input RMS normalization into the FP16 activation
 * conversion used by the following projection. Returns 0 without touching
 * out when the optimized path is unavailable. */
int ns4_gpu_matmul_f16_rms_fold_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   norm_eps);

/* Exact multi-row form of the DeepSeek 4096x256 F16 router projection. */
int ns4_gpu_matmul_f16_router_rows_exact_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        const ns4_gpu_tensor *x,
        uint32_t                n_rows);

int ns4_gpu_matmul_f16_pair_tensor(
        ns4_gpu_tensor       *out_a,
        ns4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

/* Rows-exact f16 GEMVs for the DSpark verify block: succeed only when the
 * one-row entry would take the wide ordered kernel under the current env,
 * reproducing its per-row arithmetic bit-exactly with one weight read per
 * block; return 0 otherwise so the caller keeps its per-row loop. */
int ns4_gpu_matmul_f16_pair_wide_rows_exact_tensor(
        ns4_gpu_tensor       *out_a,
        ns4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        uint64_t                out_stride,
        const ns4_gpu_tensor *x,
        uint32_t                n_rows);

int ns4_gpu_matmul_f16_one_wide_rows_exact_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint32_t                n_rows);

/* Optional decode fusion. Returns 1 when the paired projection and
 * recurrent compressor-state store were encoded, 0 when the optimized path
 * is unavailable, and -1 on an attempted-path error. */
int ns4_gpu_matmul_f16_pair_compressor_store_tensor(
        ns4_gpu_tensor       *out_kv,
        ns4_gpu_tensor       *out_score,
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ns4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos);

int ns4_gpu_matmul_f32_tensor(
        ns4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        uint64_t                n_tok);

int ns4_gpu_repeat_hc_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_repeat_hc_rows_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *rows,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_rms_norm_plain_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ns4_gpu_rms_norm_plain_rows_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ns4_gpu_rms_norm_weight_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ns4_gpu_rms_norm_weight_rows_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ns4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ns4_gpu_tensor       *q_out,
        const ns4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ns4_gpu_tensor       *kv_out,
        const ns4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ns4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ns4_gpu_tensor       *q_out,
        const ns4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ns4_gpu_tensor       *kv_out,
        const ns4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

int ns4_gpu_head_rms_norm_tensor(
        ns4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ns4_gpu_head_rms_norm_rope_tail_tensor(
        ns4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps);

int ns4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ns4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ns4_gpu_dsv4_fp8_kv_quantize_tensor(
        ns4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ns4_gpu_dsv4_indexer_qat_tensor(
        ns4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);

/* Pinned host memory for latency-critical D2H targets (the per-token logits
 * readback).  Returns NULL when unavailable; callers fall back to a plain
 * allocation. */
void *ns4_gpu_host_alloc(uint64_t bytes);
void ns4_gpu_host_free(void *p);

/* Quantize the raw f32 rows staged at stage[0 .. n_rows) like
 * ns4_gpu_dsv4_indexer_qat_tensor (Hadamard + e2m1 QAT, quantized values
 * written back to the stage), and emit the lossless fp4 mirror rows
 * [first_row, first_row + n_rows) -- the indexer cache's only persistent
 * form.  Mirror rows are 80 bytes: 128 packed 4-bit codes then 4 f32
 * per-32-element scales. */
int ns4_gpu_dsv4_indexer_qat_mirror_tensor(
        ns4_gpu_tensor *stage,
        ns4_gpu_tensor *mirror,
        uint32_t          first_row,
        uint32_t          n_rows,
        uint32_t          head_dim);

/* Rebuild mirror rows [first_row, first_row + n_rows) from already-quantized
 * f32 rows staged at stage[0 .. n_rows) (session restore reads the on-disk
 * f32 payload back through the stage). */
int ns4_gpu_indexer_mirror_reencode_tensor(
        const ns4_gpu_tensor *stage,
        ns4_gpu_tensor       *mirror,
        uint32_t              first_row,
        uint32_t              n_rows,
        uint32_t              head_dim);

/* Expand mirror rows [first_row, first_row + n_rows) into f32 rows at
 * dst[0 .. n_rows); bit-identical to what the retired persistent f32 cache
 * held (session save keeps the f32 payload format on disk). */
int ns4_gpu_indexer_mirror_dequant_tensor(
        const ns4_gpu_tensor *mirror,
        ns4_gpu_tensor       *dst,
        uint32_t              first_row,
        uint32_t              n_rows,
        uint32_t              head_dim);

int ns4_gpu_rope_tail_tensor(
        ns4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

int ns4_gpu_rope_tail_decode_rows_tensor(
        ns4_gpu_tensor                     *x,
        const ns4_gpu_attention_decode_row *rows,
        uint32_t                            n_rows,
        uint32_t                            n_head,
        uint32_t                            head_dim,
        uint32_t                            n_rot,
        uint32_t                            n_ctx_orig,
        bool                                inverse,
        float                               freq_base,
        float                               freq_scale,
        float                               ext_factor,
        float                               attn_factor,
        float                               beta_fast,
        float                               beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs NS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ns4_gpu_kv_fp8_store_raw_tensor(
        ns4_gpu_tensor *kv,
        ns4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Exact multi-session form of the decode KV finalizer. KV rows are
 * contiguous, while each output row is written to its session-private cache. */
int ns4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ns4_gpu_tensor        *kv,
        ns4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ns4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ns4_gpu_store_raw_kv_tensor(
        ns4_gpu_tensor       *raw_cache,
        const ns4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ns4_gpu_store_raw_kv_batch_tensor(
        ns4_gpu_tensor       *raw_cache,
        const ns4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ns4_gpu_compressor_update_tensor(
        const ns4_gpu_tensor *kv_cur,
        const ns4_gpu_tensor *sc_cur,
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        ns4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        int                     fp8_emit,
        int                    *fp8_done);

int ns4_gpu_compressor_store_batch_tensor(
        const ns4_gpu_tensor *kv,
        const ns4_gpu_tensor *sc,
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

/* One verify row's attention call shape, recorded by the rows verifier at
 * the point the per-row call would have been made.  Dense rows use the
 * decode caller's single_all convention (pos0/top_k/window/ratio zero);
 * indexed rows carry the row's position, top-k and window. */
typedef struct {
    uint32_t pos0;
    uint32_t n_raw;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ns4_gpu_attn_row_desc;

/* Every verify row's flash2 decode attention in one kernel pair, each row
 * reproducing its one-row launch bit for bit.  Returns 0 (caller keeps the
 * per-row calls) when any row could not take the one-row flash2 path.
 * q/heads rows live at row_stride_f32 floats; indexed rows read their
 * top-k list at row r * topk_row_stride int32s of topk_batch. */
int ns4_gpu_attention_flash2_decode_rows_tensor(
        ns4_gpu_tensor       *heads_batch,
        const ns4_gpu_tensor *q_batch,
        uint64_t                row_stride_f32,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *raw_kv,
        uint32_t                raw_cap,
        const ns4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ns4_gpu_tensor *topk_batch,
        uint32_t                topk_row_stride,
        const ns4_gpu_attn_row_desc *rows,
        uint32_t                n_rows,
        uint32_t                n_head,
        uint32_t                head_dim);

/* One verify block's frontier stores, ratio-4 emits (fp8 for the attn
 * cache, fp4 mirror encode for the indexer) and per-row prefix snapshots
 * in a single block-sequential launch.  Bit-identical to the per-row
 * chain; returns 0 (caller keeps its per-row loop) when any per-row
 * dispatch under the current env would differ.  kv/sc rows are read at
 * row_stride; ratio!=4 blocks must contain no emit.
 *
 * ns4_gpu_compressor_block_desc carries the same arguments as a struct;
 * the pair entry runs two independent programs (the attention and the
 * indexer frontier) in one launch and returns a bitmask: 1 = the first
 * ran, 2 = the second ran (only ever together with the first). */
typedef struct ns4_gpu_compressor_block_desc {
    const ns4_gpu_tensor *kv_rows;
    const ns4_gpu_tensor *sc_rows;
    uint32_t row_stride;
    ns4_gpu_tensor *state_kv;
    ns4_gpu_tensor *state_score;
    ns4_gpu_tensor *emit_out;
    ns4_gpu_tensor *mirror;
    ns4_gpu_tensor *cap_kv;
    ns4_gpu_tensor *cap_score;
    uint32_t cap_slots;
    const void *model_map;
    uint64_t model_size;
    uint64_t ape_offset;
    uint32_t ape_type;
    uint64_t norm_offset;
    uint32_t norm_type;
    uint32_t head_dim;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_rows;
    uint32_t emit_row0;
    uint32_t n_rot;
    uint32_t n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    float rms_eps;
    int fp8_emit;
} ns4_gpu_compressor_block_desc;
int ns4_gpu_compressor_block_rows_pair_tensor(
        const ns4_gpu_compressor_block_desc *a,
        const ns4_gpu_compressor_block_desc *b);
int ns4_gpu_compressor_block_rows_tensor(
        const ns4_gpu_tensor *kv_rows,
        const ns4_gpu_tensor *sc_rows,
        uint32_t              row_stride,
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        ns4_gpu_tensor       *emit_out,
        ns4_gpu_tensor       *mirror,
        ns4_gpu_tensor       *cap_kv,
        ns4_gpu_tensor       *cap_score,
        uint32_t              cap_slots,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_rows,
        uint32_t                emit_row0,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        int                     fp8_emit);

int ns4_gpu_compressor_prefill_tensor(
        ns4_gpu_tensor       *comp_cache,
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        const ns4_gpu_tensor *kv,
        const ns4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ns4_gpu_compressor_prefill_ratio4_replay_tensor(
        ns4_gpu_tensor       *comp_cache,
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        const ns4_gpu_tensor *kv,
        const ns4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ns4_gpu_compressor_prefill_state_ratio4_tensor(
        ns4_gpu_tensor       *state_kv,
        ns4_gpu_tensor       *state_score,
        const ns4_gpu_tensor *kv_tail,
        const ns4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ns4_gpu_attention_decode_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ns4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ns4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Multi-session decode over contiguous Q/head rows and private KV caches.
 * The row table is copied into CUDA launch parameters, so no device-side
 * descriptor upload or synchronization is required. */
int ns4_gpu_attention_decode_rows_rope_tensor(
        ns4_gpu_tensor                       *heads,
        const void                           *model_map,
        uint64_t                              model_size,
        uint64_t                              sinks_offset,
        const ns4_gpu_tensor                 *q,
        const ns4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow);

int ns4_gpu_attention_prefill_raw_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ns4_gpu_attention_decode_raw_batch_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ns4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim);

int ns4_gpu_attention_decode_mixed_batch_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        const ns4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ns4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ns4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        const ns4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ns4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ns4_gpu_attention_prefill_static_mixed_heads_tensor(
        ns4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ns4_gpu_tensor *q,
        const ns4_gpu_tensor *raw_kv,
        const ns4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ns4_gpu_attention_output_q8_batch_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *low,
        ns4_gpu_tensor       *group_tmp,
        ns4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ns4_gpu_tensor *heads,
        uint32_t                n_tokens);
int ns4_gpu_attention_output_q4_K_batch_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *low,
        ns4_gpu_tensor       *group_tmp,
        ns4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint32_t                out_b_type,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ns4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ns4_gpu_attention_output_q8_batch_f16_tensor(
        ns4_gpu_tensor       *out_h,
        ns4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ns4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ns4_gpu_attention_output_low_q8_tensor(
        ns4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ns4_gpu_tensor *heads);
int ns4_gpu_attention_output_low_q4_K_slice_tensor(
        ns4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ns4_gpu_tensor *heads);

int ns4_gpu_attention_output_low_q8_rows_exact_tensor(
        ns4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ns4_gpu_tensor *heads,
        uint32_t                n_rows);

int ns4_gpu_attention_output_q8_tp_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ns4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ns4_gpu_swiglu_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *gate,
        const ns4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ns4_gpu_add_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *a,
        const ns4_gpu_tensor *b,
        uint32_t                n);

int ns4_gpu_directional_steering_project_tensor(
        ns4_gpu_tensor       *x,
        const ns4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int ns4_gpu_router_select_tensor(
        ns4_gpu_tensor       *selected,
        ns4_gpu_tensor       *weights,
        ns4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ns4_gpu_tensor *logits);

int ns4_gpu_router_select_batch_tensor(
        ns4_gpu_tensor       *selected,
        ns4_gpu_tensor       *weights,
        ns4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ns4_gpu_tensor *logits,
        const ns4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ns4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected);
int ns4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ns4_gpu_tensor       *out_hc,
        ns4_gpu_tensor       *block_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        uint64_t              in_start,
        uint64_t              in_count,
        const ns4_gpu_tensor *x,
        const ns4_gpu_tensor *block_add,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ns4_gpu_routed_moe_one_owned_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *gate,
        ns4_gpu_tensor       *up,
        ns4_gpu_tensor       *mid,
        ns4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ns4_gpu_tensor *selected,
        const ns4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ns4_gpu_tensor *x,
        ns4_gpu_tensor       *down_output,
        bool                  pack_fixed3,
        ns4_gpu_tensor       *shared_prequant);

int ns4_gpu_routed_moe_batch_owned_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *gate,
        ns4_gpu_tensor       *up,
        ns4_gpu_tensor       *mid,
        ns4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        ns4_gpu_tensor       *selected,
        ns4_gpu_tensor       *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ns4_gpu_tensor *x,
        uint32_t              layer_index,
        uint32_t              n_tokens,
        bool                 *mid_is_f16);

int ns4_gpu_routed_moe_owned_slots_combine_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *home_slots,
        const ns4_gpu_tensor *peer_slots,
        const ns4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ns4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *home_slots,
        const ns4_gpu_tensor *peer_slots,
        const ns4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split,
        uint32_t              rows);

int ns4_gpu_routed_moe_owned_packed_combine_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *home_slots,
        const ns4_gpu_tensor *peer_packed,
        const ns4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ns4_gpu_routed_moe_one_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *gate,
        ns4_gpu_tensor       *up,
        ns4_gpu_tensor       *mid,
        ns4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ns4_gpu_tensor *selected,
        const ns4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ns4_gpu_tensor *x,
        const ns4_gpu_tensor *add_in,
        uint32_t                layer_index,
        bool                    force_resident);

/* DSpark verify rows batched through the decode-exact aligned path: per-row
 * results are bit-identical to n_tokens calls of the one-token decode form. */
int ns4_gpu_routed_moe_rows_exact_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *gate,
        ns4_gpu_tensor       *up,
        ns4_gpu_tensor       *mid,
        ns4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ns4_gpu_tensor *selected,
        const ns4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ns4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens);

int ns4_gpu_routed_moe_batch_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *gate,
        ns4_gpu_tensor       *up,
        ns4_gpu_tensor       *mid,
        ns4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ns4_gpu_tensor *selected,
        const ns4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ns4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        bool                    force_resident);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ns4_gpu_hc_split_sinkhorn_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ns4_gpu_hc_weighted_sum_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_hc_weighted_sum_split_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ns4_gpu_hc_split_weighted_sum_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *split,
        const ns4_gpu_tensor *mix,
        const ns4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ns4_gpu_hc_split_weighted_sum_norm_tensor(
        ns4_gpu_tensor       *out,
        ns4_gpu_tensor       *norm_out,
        ns4_gpu_tensor       *split,
        const ns4_gpu_tensor *mix,
        const ns4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

int ns4_gpu_output_hc_weights_tensor(
        ns4_gpu_tensor       *out,
        const ns4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ns4_gpu_hc_expand_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *post,
        const ns4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);
int ns4_gpu_hc_expand_add_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out,
        const ns4_gpu_tensor *block_add,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *post,
        const ns4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);


int ns4_gpu_hc_expand_add_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out,
        const ns4_gpu_tensor *block_add,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *post,
        const ns4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_hc_expand_split_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_hc_expand_split_half_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out_h,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_hc_expand_add_split_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out,
        const ns4_gpu_tensor *block_add,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_hc_expand_add_split_half_add_tensor(
        ns4_gpu_tensor       *out_hc,
        const ns4_gpu_tensor *block_out,
        const ns4_gpu_tensor *block_add_h,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_shared_down_hc_expand_q8_0_tensor(
        ns4_gpu_tensor       *out_hc,
        ns4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *shared_mid,
        const ns4_gpu_tensor *routed_out,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_shared_down_hc_expand_rows_exact_tensor(
        ns4_gpu_tensor       *out_hc,
        ns4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *shared_mid,
        const ns4_gpu_tensor *routed_out,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                split_stride,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                n_rows);

int ns4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ns4_gpu_tensor       *out_hc,
        ns4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *shared_mid,
        const ns4_gpu_tensor *routed_out,
        const ns4_gpu_tensor *routed_add,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
        ns4_gpu_tensor       *out_hc,
        ns4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *shared_mid,
        const ns4_gpu_tensor *home_slots,
        const ns4_gpu_tensor *peer_packed,
        const ns4_gpu_tensor *selected,
        uint32_t                expert_split,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ns4_gpu_matmul_q8_0_hc_expand_tensor(
        ns4_gpu_tensor       *out_hc,
        ns4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ns4_gpu_tensor *x,
        const ns4_gpu_tensor *residual_hc,
        const ns4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Decode-island CUDA graph capture (CUDA backend; the CPU build stubs it
 * out and stays eager).  Design ported from the Entrpi/ds4 batched-serving
 * fork's per-layer decode graph capture.  The key identifies a captured
 * island: layer, island index, and the activation buffers whose addresses
 * the captured kernels bake in.  ns4_cuda.cu mirrors this struct
 * byte-for-byte (it does not include this header); keep both in sync. */
typedef struct ns4_decode_graph_key {
    uint32_t il;
    uint32_t island;    /* 0: layer top to pre-rope; 1: attn-out to layer end */
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ns4_decode_graph_key;

int  ns4_gpu_decode_graphs_supported(void);
/* 1: replayed (island already executed; skip encoding it)
 * 0: capturing (encode the island, then call _end)
 * -1: run eagerly */
int  ns4_gpu_decode_graph_begin(const ns4_decode_graph_key *key);
/* 0: capture committed and launched; -1: capture failed (entry retired;
 * the caller must re-encode the island eagerly -- no work was executed). */
int  ns4_gpu_decode_graph_end(const ns4_decode_graph_key *key);
void ns4_gpu_decode_graph_abort(const ns4_decode_graph_key *key);
void ns4_gpu_decode_graphs_invalidate(void);

#ifdef __cplusplus
}
#endif

#endif
