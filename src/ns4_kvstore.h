#ifndef NS4_KVSTORE_H
#define NS4_KVSTORE_H

#include "ns4.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define NS4_KVSTORE_FIXED_HEADER 48u
#define NS4_KVSTORE_DEFAULT_MB 4096
#define NS4_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define NS4_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define NS4_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define NS4_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define NS4_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    NS4_KVSTORE_REASON_UNKNOWN   = 0,
    NS4_KVSTORE_REASON_COLD      = 1,
    NS4_KVSTORE_REASON_CONTINUED = 2,
    NS4_KVSTORE_REASON_EVICT     = 3,
    NS4_KVSTORE_REASON_SHUTDOWN  = 4,
    NS4_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    NS4_KVSTORE_REASON_AGENT_SESSION = 6,
} ns4_kvstore_reason;

typedef enum {
    NS4_KVSTORE_LOG_DEFAULT,
    NS4_KVSTORE_LOG_KVCACHE,
    NS4_KVSTORE_LOG_WARNING,
} ns4_kvstore_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    /* Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} ns4_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} ns4_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    ns4_kvstore_options opt;
    int continued_last_store_tokens;
    ns4_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, ns4_kvstore_log_type type, const char *msg);
} ns4_kvstore;

typedef struct {
    const char *text;
    size_t text_len;
    uint8_t model_id;
    uint8_t quant_bits;
    uint32_t ctx_size;
    bool reject_different_quant;
} ns4_kvstore_eviction_context;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} ns4_kvstore_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    char *path;
} ns4_kvstore_load_result;

ns4_kvstore_options ns4_kvstore_default_options(void);
uint8_t ns4_kvstore_reason_code(const char *reason);
const char *ns4_kvstore_key_kind(uint8_t ext_flags);

bool ns4_kvstore_open(ns4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ns4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ns4_kvstore_log_type type, const char *msg),
                      void *log_ud);
void ns4_kvstore_close(ns4_kvstore *kc);
void ns4_kvstore_clear(ns4_kvstore *kc);
void ns4_kvstore_entry_free(ns4_kvstore_entry *e);

char *ns4_kvstore_render_tokens_text(ns4_engine *engine,
                                     const ns4_tokens *tokens,
                                     size_t *out_len);
bool ns4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void ns4_kvstore_tokens_copy_prefix(ns4_tokens *dst, const ns4_tokens *src, int n);
void ns4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ns4_engine *engine,
        const ns4_tokens *exact_prefix,
        const char *suffix_text,
        ns4_tokens *out);

int ns4_kvstore_store_len(const ns4_kvstore *kc, int tokens);
int ns4_kvstore_chat_anchor_pos(const ns4_kvstore *kc,
                                const ns4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int ns4_kvstore_continued_store_target(const ns4_kvstore *kc, int live_tokens);
void ns4_kvstore_note_store(ns4_kvstore *kc, int tokens);
int ns4_kvstore_suppress_continued_store(ns4_kvstore *kc, int tokens);
void ns4_kvstore_restore_suppressed_continued(ns4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool ns4_kvstore_file_size_fits(const ns4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double ns4_kvstore_entry_eviction_score(const ns4_kvstore_entry *e,
                                        const ns4_tokens *live,
                                        uint64_t now,
                                        const ns4_kvstore_eviction_context *incoming);
void ns4_kvstore_evict(ns4_kvstore *kc, const ns4_tokens *live,
                       uint64_t extra_bytes,
                       const ns4_kvstore_eviction_context *incoming);
int ns4_kvstore_find_text_prefix(ns4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);

bool ns4_kvstore_store_live_prefix_text(ns4_kvstore *kc,
                                        ns4_engine *engine,
                                        ns4_session *session,
                                        const ns4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ns4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool ns4_kvstore_store_live_prefix(ns4_kvstore *kc,
                                   ns4_engine *engine,
                                   ns4_session *session,
                                   const ns4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ns4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool ns4_kvstore_maybe_store_continued(ns4_kvstore *kc,
                                       ns4_engine *engine,
                                       ns4_session *session,
                                       const ns4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int ns4_kvstore_try_load_text(ns4_kvstore *kc,
                              ns4_engine *engine,
                              ns4_session *session,
                              const char *prompt_text,
                              ns4_tokens *effective_prompt,
                              ns4_kvstore_load_result *result,
                              const ns4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void ns4_kvstore_load_result_free(ns4_kvstore_load_result *result);

bool ns4_kvstore_read_header(FILE *fp, ns4_kvstore_entry *e,
                             uint32_t *text_bytes);
bool ns4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ns4_kvstore_entry *out);
void ns4_kvstore_fill_header(uint8_t h[NS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool ns4_kvstore_touch_file(const char *path, uint32_t hits);
bool ns4_kvstore_sha_hex_name(const char *name, char sha[41]);
void ns4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *ns4_kvstore_path_join(const char *dir, const char *name);
char *ns4_kvstore_path_for_sha(ns4_kvstore *kc, const char sha[41]);
void ns4_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t ns4_kvstore_le_get32(const uint8_t *p);

#endif
