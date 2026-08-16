#!/usr/bin/env bash
#
# Serve DeepSeek-V4-Flash IQ2XXS from one RTX PRO 6000 Blackwell (95.59 GiB
# usable, 80.76 GiB of that the resident model) at the largest context that
# still prefills at full speed.
#
# WHY --ctx 436000 AND NOT MORE
#
# There are two separate limits above this and the useful one is not the one
# that fails loudly.
#
#   1. context buffers >= 8192 MiB -- the KV cache silently becomes
#      cudaMallocManaged and paging costs 3-40x. Hard failure boundary.
#      --ctx 468000 sits at 8164 MiB, just under it.
#   2. free VRAM for prefill scratch -- prefill's transient buffers and cuBLAS
#      workspace come out of whatever the planned KV leaves behind, and the
#      VRAM reserve does not guard them (ds4_cuda.cu:1417). This one binds
#      well before (1), degrades gradually rather than failing, and is the
#      reason for 436000.
#
# Prefill tk/s at ctx 8192, prefill-chunk 2048, warm-up run discarded:
#
#     ctx-alloc   context buffers   prefill   vs full
#     412,000        7211 MiB        3865
#     424,000        7415 MiB        3842
#     436,000        7619 MiB        3847      full     <-- here
#     448,000        7824 MiB        3598      -6%
#     456,000        7960 MiB        3326      -13%
#     464,000        8096 MiB        2909      -24%
#     468,000        8164 MiB       ~2150      -44%
#
# 436000 costs nothing: it tracks the smallest allocation that holds a given
# sweep to within +-2% at every frontier out to 262k. Verified on this server,
# same two prompts at 436k vs 468k:
#
#                        436k                468k
#     20,591 tok    3442 tk/s prefill   2158 tk/s   (-37%)   13.2s vs 17.3s
#     68,069 tok    3404 tk/s prefill   2233 tk/s   (-34%)   25.8s vs 37.4s
#
# Decode does not care about the allocation (70.3 vs 68.6 t/s, 61.4 vs 59.2),
# so if a workload only ever decodes, the larger context is free. Anything that
# prefills should stay at 436000.
#
# Prefill decays with sequence length regardless of allocation -- ~3600 tk/s to
# 65k, ~3400 at 131k, ~2500 at 262k. That is per-token attention and the DSA
# indexer scan growing, not memory pressure, and no allocation avoids it.
#
# The three env exports are not tuning niceties. Without them the q8->f16
# weight cache grows until free VRAM hits the reserve, which on this model is
# zero, and throughput collapses ~20x (prefill 2016 -> 99 tk/s, decode
# 68.8 -> 3.2 t/s). The tells in the log are `free=0.00 GiB` on the
# `q8 fp16 cache budget exhausted` line.
#
# Every value below can be overridden from the environment; extra ds4-server
# flags are passed through.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"

MODEL="${DS4_MODEL:-$ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
CTX="${DS4_CTX:-436000}"
PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-2048}"
GPU_VRAM="${DS4_GPU_VRAM:-94}"
SERVER_HOST="${DS4_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${DS4_SERVER_PORT:-8000}"

die() { printf 'run-cuda-436k-server: %s\n' "$*" >&2; exit 1; }

[[ -f "$MODEL" ]] || die "model not found: $MODEL (set DS4_MODEL)"
[[ -x ./ds4-server ]] || die "./ds4-server not built -- make ds4-server CUDA_ARCH=sm_120"

# ds4-server links ds4_cuda.o but has its own link step, so a rebuild of the
# CUDA object does not update it. This has silently served a stale binary
# before -- once after the FA2 work landed, and again after the 2026-08-12
# decode kernels. Refuse rather than serve last week's kernels.
if [[ -f ./ds4_cuda.o && ./ds4_cuda.o -nt ./ds4-server ]]; then
    die "ds4-server is older than ds4_cuda.o -- relink first:
       make ds4-server CUDA_ARCH=sm_120   (takes <1s if the objects are current)"
fi

export DS4_CUDA_NO_Q8_F16_CACHE="${DS4_CUDA_NO_Q8_F16_CACHE:-1}"
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-5184}"
export DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB="${DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB:-512}"

# Decode levers validated 2026-08-15 (summation-order NLL protocol; see the
# memory notes).  All are numerically different from the bit-exact default
# paths -- deterministic, but greedy continuations can differ from a run with
# these unset.  Unset any of them to fall back:
#   SCAN_INT     exact-integer dp4a indexer scan over the fp4 mirrors
#   F16_ONE_WIDE replaces the cuBLAS one-token f16 GEMVs (router, indexer
#                projection, hc fns) and their f32->f16 feed conversions
#   F16_PAIR_WIDE / Q8_PAIR_WIDE  wide compressor / q8 pair GEMVs
export DS4_CUDA_INDEXER_SCAN_INT="${DS4_CUDA_INDEXER_SCAN_INT:-1}"
export DS4_CUDA_F16_ONE_WIDE="${DS4_CUDA_F16_ONE_WIDE:-256}"
export DS4_CUDA_F16_PAIR_WIDE="${DS4_CUDA_F16_PAIR_WIDE:-256}"
export DS4_CUDA_Q8_PAIR_WIDE="${DS4_CUDA_Q8_PAIR_WIDE:-256}"
# +7.5% decode at ctx 4096-8192, inert at 16k and above (accepted range ends
# at 4096; the dense path cannot reach long context -- shared-memory bound).
export DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD="${DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD:-4096}"

printf 'run-cuda-436k-server: ctx=%s prefill-chunk=%s gpu-vram=%s -> %s:%s\n' \
    "$CTX" "$PREFILL_CHUNK" "$GPU_VRAM" "$SERVER_HOST" "$SERVER_PORT"
printf 'run-cuda-436k-server: expect "context buffers 7619.27 MiB"; over ~7800 and prefill starts to drop\n'

exec ./ds4-server --cuda -m "$MODEL" \
    --gpu-vram "$GPU_VRAM" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --ctx "$CTX" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    "$@"
