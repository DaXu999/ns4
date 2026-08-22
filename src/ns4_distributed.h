#ifndef NS4_DISTRIBUTED_H
#define NS4_DISTRIBUTED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ns4.h"

/* Distributed inference is an engine backend, not a separate frontend API.
 * Programs parse distributed options here, then keep using the normal
 * ns4_session_* calls. Only worker/coordinator serving modes call
 * ns4_dist_run() directly.
 */
typedef ns4_distributed_options ns4_dist_options;
typedef struct ns4_dist_session ns4_dist_session;

/* Options used by standalone `./ns4 --role coordinator -p ...` generation.
 * Interactive tools and the server go through the normal ns4_session API.
 */
typedef struct {
    const char *prompt;
    const char *system;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    ns4_think_mode think_mode;
} ns4_dist_generation_options;

typedef enum {
    NS4_DIST_CLI_ERROR = -1,
    NS4_DIST_CLI_NOT_MATCHED = 0,
    NS4_DIST_CLI_MATCHED = 1,
} ns4_dist_cli_parse_result;

/* Shared option parsing. */
bool ns4_dist_enabled(const ns4_dist_options *opt);
ns4_dist_options *ns4_dist_options_create(void);
void ns4_dist_options_free(ns4_dist_options *opt);
void ns4_dist_usage(FILE *fp);
ns4_dist_cli_parse_result ns4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ns4_dist_options *opt,
        char *err,
        size_t errlen);

/* Applies distributed layer-loading choices to the engine options before the
 * model is loaded.
 */
int ns4_dist_prepare_engine_options(
        const ns4_dist_options *opt,
        ns4_engine_options *engine,
        char *err,
        size_t errlen);

/* Coordinator session backend used by ns4.c. These mirror the normal session
 * operations; callers outside the engine should not need to call them directly.
 */
int ns4_dist_session_create(
        ns4_dist_session **out,
        ns4_engine *engine,
        const ns4_dist_options *opt,
        ns4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen);
void ns4_dist_session_free(ns4_dist_session *d);

/* Returns 1 when the coordinator has full layer coverage, 0 when workers are
 * still missing, and -1 for configuration or internal errors.
 */
int ns4_dist_session_route_ready(ns4_dist_session *d, char *err, size_t errlen);

/* Synchronize the distributed KV state to the requested prompt timeline. */
int ns4_dist_session_sync(
        ns4_dist_session *d,
        ns4_session *owner,
        const ns4_tokens *checkpoint,
        const ns4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen);

/* Evaluate one additional token across the current distributed route. */
int ns4_dist_session_eval(
        ns4_dist_session *d,
        ns4_session *owner,
        const ns4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen);

/* Save/load use the normal DSV4 payload format. The coordinator gathers or
 * pushes remote layer shards internally so saved files are topology-neutral.
 */
int ns4_dist_session_save_payload(
        ns4_dist_session *d,
        ns4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen);
int ns4_dist_session_load_payload(
        ns4_dist_session *d,
        ns4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen);

/* Standalone distributed mode. Workers stay in this loop; coordinator one-shot
 * mode uses it for `./ns4 --role coordinator -p ...`.
 */
int ns4_dist_run(ns4_engine *engine, const ns4_dist_options *opt, const ns4_dist_generation_options *gen);

#endif
