/* ns4_gpu_args.h — shared CLI argument parser for --gpu-vram and
 * --gpu-devices flags. Used by ns4_cli, ns4_server, ns4_agent,
 * ns4_bench, and ns4_eval. See mgpu-cli-wiring task (wave 2) for design.
 *
 * The parser does not link CUDA headers directly. On CUDA builds the
 * "auto" path is gated and dispatches to ns4_gpu_args_probe_auto_cuda,
 * which is defined in ns4_cuda.cu. On CPU builds, "auto" returns
 * an error.
 */
#ifndef NS4_GPU_ARGS_H
#define NS4_GPU_ARGS_H

#include <stdbool.h>
#include <stddef.h>

#include "ns4_gpu_mgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single canonical parser. Populates *out and *out_skip_cuda on
 * success; returns 0 on success, nonzero on parse error (errbuf
 * populated). Caller zeroes out and out_skip_cuda before calling.
 *
 *   vram_arg may be:
 *     NULL       -- flag unset
 *     "0"        -- skip CUDA entirely; sets *out_skip_cuda = true
 *     "auto"     -- query free VRAM via CUDA (CUDA build only)
 *     "N[,N...]" -- explicit GB budgets per device
 *
 *   devices_arg may be:
 *     NULL       -- 0..n-1 of visible devices (or 0..vram_count-1)
 *     "N[,N...]" -- explicit CUDA device indices
 *
 * Constraints:
 *   - If both args are given and counts differ, error.
 *   - Negative or non-integer values error.
 *   - Counts > NS4_MAX_GPUS error.
 */
int parse_gpu_vram_arg(
    const char     *vram_arg,
    const char     *devices_arg,
    ns4_gpu_config *out,
    bool           *out_skip_cuda,
    char           *errbuf,
    size_t          errbuflen);

/* Format the layout echo line written to stdout right before
 * ns4_engine_create_with_gpu_config is called.
 *
 * Example output:
 *   ns4: GPU config: 2 devices [0,1] requested, budgets 40,12 GB; auto=true
 *
 * Returns the number of bytes written (excluding the trailing NUL), or
 * -1 if out is NULL/outlen is 0.
 */
int format_gpu_layout_line(const ns4_gpu_config *cfg,
                           bool                  was_auto,
                           char                 *out,
                           size_t                outlen);

/* CUDA-only auto probe. Defined in ns4_cuda.cu under the same gate.
 * Returns 0 on success, nonzero on failure (errbuf populated).
 * device_filter may be NULL (use all visible devices) or a list of
 * length filter_len (use only those device indices).
 *
 * Declared here for the parser's use; not called by application code
 * directly. The Mac/CPU builds skip this branch via the
 * #if !defined(NS4_NO_GPU) gate in the parser. */
#if !defined(NS4_NO_GPU)
int ns4_gpu_args_probe_auto_cuda(const int      *device_filter,
                                  int             filter_len,
                                  ns4_gpu_config *out,
                                  size_t          safety_margin_bytes,
                                  char           *errbuf,
                                  size_t          errbuflen);
#endif

#ifdef __cplusplus
}
#endif

#endif /* NS4_GPU_ARGS_H */
