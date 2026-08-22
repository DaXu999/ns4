#ifndef NS4_H
#define NS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ns4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ns4_engine as the loaded model and
 * ns4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ns4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    NS4_BACKEND_CUDA,
    NS4_BACKEND_CPU,
} ns4_backend;

typedef enum {
    NS4_THINK_NONE,
    NS4_THINK_HIGH,
    NS4_THINK_MAX,
} ns4_think_mode;

typedef enum {
    NS4_LOG_DEFAULT,
    NS4_LOG_PREFILL,
    NS4_LOG_GENERATION,
    NS4_LOG_KVCACHE,
    NS4_LOG_TOOL,
    NS4_LOG_WARNING,
    NS4_LOG_TIMING,
    NS4_LOG_OK,
    NS4_LOG_ERROR,
} ns4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ns4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ns4_token_score;

#define NS4_DEFAULT_TEMPERATURE 1.0f
#define NS4_DEFAULT_TOP_P 1.0f
#define NS4_DEFAULT_MIN_P 0.05f

typedef struct ns4_engine ns4_engine;
typedef struct ns4_session ns4_session;

typedef void (*ns4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ns4_session_cancel_fn)(void *ud);

#define NS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    NS4_DISTRIBUTED_NONE = 0,
    NS4_DISTRIBUTED_COORDINATOR,
    NS4_DISTRIBUTED_WORKER,
} ns4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ns4_distributed_layers;

typedef struct {
    ns4_distributed_role role;
    ns4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ns4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ns4_backend backend;
    int n_threads;
    int context_size;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    float dspark_confidence_threshold;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool dspark;
    bool dspark_strict;
    bool dspark_confidence_threshold_set;
    bool cuda_tensor_parallel;
    bool ssd_streaming;
    bool inspect_only;
    /* Multi-GPU placement uses this to price per-layer KV storage. */
    int placement_ctx_hint;
    /* Server batch mode serializes execution and can share prefill scratch. */
    bool share_session_prefill_workspace;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ns4_distributed_options distributed;
} ns4_engine_options;

typedef void (*ns4_token_emit_fn)(void *ud, int token);
typedef void (*ns4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ns4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ns4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ns4_session_payload_file;

int ns4_engine_open(ns4_engine **out, const ns4_engine_options *opt);

/* Multi-GPU entry point.
 *
 * Accepts an optional ns4_gpu_config (defined in ns4_gpu_mgpu.h) that
 * lets callers describe a multi-GPU placement target. Passing NULL is
 * back-compatible with ns4_engine_open and produces identical engine
 * state — bit-equivalent execution at runtime.
 *
 * With a non-NULL config the computed layout is printed first. Multi-GPU
 * placements open and run; a placement that spills layers to the CPU is
 * refused with a non-zero return and a stderr notice, because CPU-tier
 * execution is not wired. */
/* ns4_gpu_config is declared in ns4_gpu_mgpu.h, which callers should
 * include separately. We forward-declare it here so this header can be
 * used as-is (callers passing NULL don't need the struct definition). */
struct ns4_gpu_config;
int ns4_engine_create_with_gpu_config(ns4_engine **out,
                                       const ns4_engine_options *opt,
                                       const struct ns4_gpu_config *gpu_cfg);
void ns4_engine_close(ns4_engine *e);
void ns4_engine_summary(ns4_engine *e);
int ns4_engine_vocab_size(ns4_engine *e);
uint32_t ns4_engine_prefill_chunk(ns4_engine *e);
int ns4_engine_power(ns4_engine *e);
int ns4_engine_set_power(ns4_engine *e, int power_percent);
const char *ns4_engine_model_name(ns4_engine *e);
int ns4_engine_layer_count(ns4_engine *e);
uint32_t ns4_engine_layer_compress_ratio(ns4_engine *e, uint32_t layer);
uint64_t ns4_engine_hidden_f32_values(ns4_engine *e);
int ns4_engine_embd_dim(ns4_engine *e);
uint64_t ns4_engine_model_bytes(ns4_engine *e);
/* Stable id for cache compatibility.  0 is the Flash shape, so old KV files
 * with the previously-zero reserved byte remain Flash-compatible; any future
 * non-Flash shape must use a nonzero id. */
int ns4_engine_model_id(ns4_engine *e);
const char *ns4_backend_name(ns4_backend backend);
bool ns4_think_mode_enabled(ns4_think_mode mode);
const char *ns4_think_mode_name(ns4_think_mode mode);
const char *ns4_think_max_prefix(void);
uint32_t ns4_think_max_min_context(void);
ns4_think_mode ns4_think_mode_for_context(ns4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ns4_engine_open(); call after opening
 * the GGUF so the model dimensions are known. */
ns4_context_memory ns4_context_memory_estimate(ns4_backend backend, int ctx_size);
ns4_context_memory ns4_context_memory_estimate_with_prefill(
        ns4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
ns4_context_memory ns4_context_memory_estimate_with_prefill_mode(
        ns4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        bool ssd_streaming);
bool ns4_log_is_tty(FILE *fp);
void ns4_log(FILE *fp, ns4_log_type type, const char *fmt, ...);
int ns4_engine_generate_argmax(ns4_engine *e, const ns4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ns4_token_emit_fn emit,
                               ns4_generation_done_fn done,
                               void *emit_ud,
                               ns4_session_progress_fn progress,
                               void *progress_ud);
int ns4_engine_collect_imatrix(ns4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ns4_engine_dump_tokens(ns4_engine *e, const ns4_tokens *tokens);
int ns4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ns4_engine_head_test(ns4_engine *e, const ns4_tokens *prompt);
int ns4_engine_first_token_test(ns4_engine *e, const ns4_tokens *prompt);
int ns4_engine_graph_test(ns4_engine *e, const ns4_tokens *prompt);
int ns4_engine_graph_full_test(ns4_engine *e, const ns4_tokens *prompt);
int ns4_engine_graph_prompt_test(ns4_engine *e, const ns4_tokens *prompt, int ctx_size);

void ns4_tokens_push(ns4_tokens *tv, int token);
void ns4_tokens_free(ns4_tokens *tv);
void ns4_tokens_copy(ns4_tokens *dst, const ns4_tokens *src);
bool ns4_tokens_starts_with(const ns4_tokens *tokens, const ns4_tokens *prefix);

void ns4_tokenize_text(ns4_engine *e, const char *text, ns4_tokens *out);
void ns4_tokenize_rendered_chat(ns4_engine *e, const char *text, ns4_tokens *out);
void ns4_chat_begin(ns4_engine *e, ns4_tokens *tokens);
void ns4_encode_chat_prompt(
        ns4_engine *e,
        const char *system,
        const char *prompt,
        ns4_think_mode think_mode,
        ns4_tokens *out);
void ns4_chat_append_max_effort_prefix(ns4_engine *e, ns4_tokens *tokens);
void ns4_chat_append_message(ns4_engine *e, ns4_tokens *tokens, const char *role, const char *content);
void ns4_chat_append_assistant_prefix(ns4_engine *e, ns4_tokens *tokens, ns4_think_mode think_mode);

char *ns4_token_text(ns4_engine *e, int token, size_t *len);
int ns4_token_eos(ns4_engine *e);
bool ns4_token_is_stop(ns4_engine *e, int token);
bool ns4_token_is_thinking_control(ns4_engine *e, int token);
bool ns4_token_is_stop_for_think_mode(ns4_engine *e,
                                      int token,
                                      ns4_think_mode mode);
int ns4_token_user(ns4_engine *e);
int ns4_token_assistant(ns4_engine *e);

int ns4_session_create(ns4_session **out, ns4_engine *e, int ctx_size);
void ns4_session_free(ns4_session *s);
int ns4_session_power(ns4_session *s);
int ns4_session_set_power(ns4_session *s, int power_percent);
bool ns4_session_is_distributed(ns4_session *s);
void ns4_session_set_progress(ns4_session *s, ns4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ns4_session_set_display_progress(ns4_session *s, ns4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ns4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns NS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ns4_session_set_cancel(ns4_session *s, ns4_session_cancel_fn fn, void *ud);
void ns4_session_report_progress(ns4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ns4_session_distributed_route_ready(ns4_session *s, char *err, size_t errlen);

typedef enum {
    NS4_SESSION_REWRITE_ERROR = -1,
    NS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    NS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ns4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
#define NS4_SESSION_SYNC_INTERRUPTED 2
int ns4_session_sync(ns4_session *s, const ns4_tokens *prompt, char *err, size_t errlen);
bool ns4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ns4_session_rewrite_result ns4_session_rewrite_from_common(
        ns4_session *s, const ns4_tokens *prompt, int common,
        char *err, size_t errlen);
int ns4_session_common_prefix(ns4_session *s, const ns4_tokens *prompt);
int ns4_session_argmax(ns4_session *s);
int ns4_session_argmax_excluding(ns4_session *s, int excluded_id);
int ns4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ns4_session_sample(ns4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
#ifdef NS4_TEST_HOOKS
int ns4_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch);
uint64_t ns4_test_mixed_native_count(void);
#endif
int ns4_session_top_logprobs(ns4_session *s, ns4_token_score *out, int k);
int ns4_session_token_logprob(ns4_session *s, int token, ns4_token_score *out);
int ns4_session_copy_logits(ns4_session *s, float *out, int cap);
int ns4_session_set_logits(ns4_session *s, const float *logits, int n);
/* Pay the one-time first-submission GPU cost outside any measured window;
 * used by the TP worker right after session create (no-op on CPU). */
void ns4_session_gpu_warmup(ns4_session *s);
int ns4_session_eval(ns4_session *s, int token, char *err, size_t errlen);

typedef struct {
    ns4_session *session;
    int token;
} ns4_decode_item;

/* Advance independent sessions by one token each. Batch size one is exactly
 * ns4_session_eval(). Backends without native batching use a correctness-first
 * sequential fallback. */
int ns4_sessions_eval_batch(ns4_decode_item *items, int count,
                            char *err, size_t errlen);
/* Advance one resumed prefill suffix and an independent decode batch as one
 * scheduling step. Unsupported combinations use the ordinary serialized
 * session operations. */
int ns4_sessions_eval_batch_with_prefill(
        ns4_decode_item *items, int count,
        ns4_session *prefill_session, const ns4_tokens *prefill_prompt,
        char *err, size_t errlen);
int ns4_session_eval_speculative_argmax(ns4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ns4_session_invalidate(ns4_session *s);
void ns4_session_rewind(ns4_session *s, int pos);
int ns4_session_pos(ns4_session *s);
int ns4_session_ctx(ns4_session *s);
int ns4_session_prefill_cap(ns4_session *s);
int ns4_engine_routed_quant_bits(ns4_engine *e);
bool ns4_engine_has_output_head(ns4_engine *e);
bool ns4_engine_has_mtp(ns4_engine *e);
int ns4_engine_mtp_draft_tokens(ns4_engine *e);
const ns4_tokens *ns4_session_tokens(ns4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ns4_distributed.c. */
int ns4_session_layer_slice_reset(ns4_session *s, char *err, size_t errlen);
int ns4_session_eval_layer_slice(ns4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ns4_session_eval_output_head_from_hc(ns4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the NS4-specific serialized graph state. */
#define NS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define NS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define NS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define NS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define NS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define NS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ns4_session_payload_bytes(ns4_session *s);
int ns4_session_stage_payload(ns4_session *s, ns4_session_payload_file *out,
                              char *err, size_t errlen);
int ns4_session_write_staged_payload(const ns4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ns4_session_payload_file_free(ns4_session_payload_file *payload);
int ns4_session_save_payload(ns4_session *s, FILE *fp, char *err, size_t errlen);
int ns4_session_load_payload(ns4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ns4_session_save_snapshot(ns4_session *s, ns4_session_snapshot *snap, char *err, size_t errlen);
int ns4_session_load_snapshot(ns4_session *s, const ns4_session_snapshot *snap, char *err, size_t errlen);
void ns4_session_snapshot_free(ns4_session_snapshot *snap);

uint64_t ns4_session_layer_payload_bytes(ns4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ns4_session_save_layer_payload(ns4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ns4_session_load_layer_payload(ns4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
