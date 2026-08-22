#!/usr/bin/env bash
#
# Serve DeepSeek-V4-Flash IQ2XXS from one RTX PRO 6000 Blackwell (95.59 GiB
# usable, 80.76 GiB of that the resident model) at a large context with full
# prefill speed and ~3 GiB of VRAM headroom.
#
# WHY --ctx 436000
#
# Headroom, not speed. Re-measured on the fp4-mirror-only indexer-cache
# build (it retired the f32 indexer rows, freeing ~1 GiB of KV-side VRAM at
# this scale): prefill holds full speed at EVERY allocation from 436,000
# through 536,000 -- every 4-16k step tested reads within +-0.15% of the
# 436k control at the 8k/16k probe, and 500k/532k track the control within
# +-0.08% at every frontier out to 262k. The
# progressive roll-off this header used to document (-6% at 448k, -44% at
# 468k, measured on the f32-indexer-cache build) is gone on the mirror-only
# engine: it was prefill
# scratch starved of free VRAM, and that starvation band no longer exists
# before the hard limits below (the retired f32 indexer rows account for
# about half of the gained range; the rest is that the old sensitivity
# itself no longer reproduces).
#
# The two limits that remain, in the order they bind:
#
#   1. ctx >= ~537,900 (at prefill-chunk 2048): the KV cache switches to
#      cudaMallocManaged. Deterministic size rule -- 536,000 stays on device
#      KV, 540,000 flips, and session create prints "using managed KV cache"
#      to stderr when it happens. Measured cost today is only ~1% on prefill
#      and decode, flat to 262k depth -- not the old 3-40x paging cliff --
#      but the margin to real paging is unmeasured. Stay below it.
#   2. free-VRAM slack: a hard floor, not a slope. Session creation fails
#      loudly ("CUDA tensor alloc failed: out of memory") once transient +
#      persistent allocations no longer fit. Measured slack: ~3 GiB at
#      436,000, ~1.0-1.4 GiB at 536,000, shrinking ~18.3 KiB per token of
#      allocation.
#
# 436000 stays the default for the ~3 GiB of headroom -- room for driver and
# runtime variance across boots, and for anything else that wants residency.
# NS4_CTX=532000 is measured-good when the workload needs the context: full
# prefill speed to 262k depth, live-server verified (24.7k-token prompt at
# ~3.3k tk/s), ~1.1-1.5 GiB slack, 5.9k tokens under the managed flip.
#
# Decode is not affected by the allocation (74.8-74.9 t/s at 131k, 71.0 at
# 262k, 67.4 - 64.9 from 436k through 532k).
#
# Prefill decays with sequence length regardless of allocation -- on the
# boot behind the README Speed table ~3,460 tk/s at 8k, ~3,290 at 65k,
# ~3,150 at 131k, ~2,530 at 262k (bench/rtx_pro_6000.csv), and past
# the table ~2,310 at 410k, ~2,140 at 524k (a 409,600-token prompt prefills at
# ~2,626 tk/s aggregate, ~2.6 min, identical at alloc 436k and 532k).
# Absolute prefill is boot-scoped on this instance (up to ~12% across
# stop/start boots); the flatness across allocations is not. The decay is
# per-token attention and the DSA indexer scan growing, not memory pressure,
# and no allocation avoids it.
#
# WHERE THE VRAM GOES AT 436,000 (single card, exact-size weight arena build)
#
# Measured with a cudaMalloc interposer, the engine's own allocations peak at
# 89.4 GiB: 72.56 GiB of aligned expert artifacts, 8.20 GiB of other weight
# spans in the arena, 4.86 GiB of KV state (compressed rows, fp4 indexer
# mirror, SWA ring), and 3.65 GiB of session work buffers at prefill chunk
# 2048 (indexer scores + comp mask 1.66 GiB, batch_* prefill buffers 2.0 GiB,
# half of each for chunk 1024). The CUDA runtime keeps ~1.5 GiB more outside
# those numbers, and session creation fails loudly once the engine's
# allocations reach ~94.2 GiB on this card. The weight arena used to be
# grown in NS4_CUDA_WEIGHT_ARENA_CHUNK_MB chunks and stranded 1.9 GiB at this
# model's 8.20 GiB of spans (the experts left the arena with the aligned
# artifacts); the exact-size arena removes that, so the slack figures above
# gain ~1.9 GiB. NS4_CUDA_WEIGHT_ARENA_EXACT=0 restores the chunked layout.
#
# OPTIONAL DSPARK (NS4_DSPARK=1)
#
# The Flash 0731 DSpark support GGUF adds 5.58 GiB of resident weights (its
# three draft blocks, routed experts included; the arena is sized for main
# plus support) and ~0.2 GiB of runtime state -- nothing that scales with
# the context. At prefill chunk 2048 that lands at ~95 GiB and session
# creation fails; at chunk 1024 the engine peaks at 93.1 GiB and serves
# (measured: 9.4k-token prompt, DSpark active), with ~1 GiB of slack. So
# NS4_DSPARK=1 defaults the prefill chunk to 1024 unless NS4_PREFILL_CHUNK
# is set. The price is prefill throughput: ~3080 tk/s at 1024 against ~3415
# at 2048 on a 9.4k prompt (chunk 512 buys another 1.0 GiB of slack at ~2500
# tk/s); decode is unaffected by the chunk. DSpark only drafts for
# temperature-0 requests and is silently inactive under --batched-session.
# Set NS4_DSPARK_SUPPORT to another support GGUF path. Whether drafting pays
# is a separate question: the first A/B runs on this card decoded slower with
# DSpark (80.9 -> 54.8 t/s on a code prompt at ctx 8k, 76.9 -> 64.5 at a 32k
# prose prompt), with byte-identical greedy output.
#
# The three env exports are not tuning niceties. Without them the q8->f16
# weight cache grows until free VRAM hits the reserve, which on this model is
# zero, and throughput collapses ~20x (prefill 2016 -> 99 tk/s, decode
# 68.8 -> 3.2 t/s). The tells in the log are `free=0.00 GiB` on the
# `q8 fp16 cache budget exhausted` line.
#
# Every value below can be overridden from the environment; extra ns4-server
# flags are passed through.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

MODEL="${NS4_MODEL:-$ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
CTX="${NS4_CTX:-436000}"
DSPARK="${NS4_DSPARK:-0}"
DSPARK_SUPPORT="${NS4_DSPARK_SUPPORT:-$ROOT/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}"
if [[ "$DSPARK" == "1" ]]; then
    PREFILL_CHUNK="${NS4_PREFILL_CHUNK:-1024}"
else
    PREFILL_CHUNK="${NS4_PREFILL_CHUNK:-2048}"
fi
GPU_VRAM="${NS4_GPU_VRAM:-94}"
SERVER_HOST="${NS4_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${NS4_SERVER_PORT:-8000}"

die() { printf 'run-cuda-436k-server: %s\n' "$*" >&2; exit 1; }

[[ -f "$MODEL" ]] || die "model not found: $MODEL (set NS4_MODEL)"
[[ -x ./ns4-server ]] || die "./ns4-server not built -- make ns4-server CUDA_ARCH=sm_120"
DSPARK_ARGS=()
if [[ "$DSPARK" == "1" ]]; then
    [[ -f "$DSPARK_SUPPORT" ]] || die "DSpark support GGUF not found: $DSPARK_SUPPORT
       (./scripts/download_model.sh ds4f-dspark, or set NS4_DSPARK_SUPPORT)"
    DSPARK_ARGS=(--mtp "$DSPARK_SUPPORT" --dspark)
    if (( PREFILL_CHUNK > 1024 && CTX >= 436000 )); then
        printf 'run-cuda-436k-server: warning: DSpark at prefill-chunk %s and ctx %s was measured not to fit the 96 GB card (session creation fails); use NS4_PREFILL_CHUNK=1024\n' \
            "$PREFILL_CHUNK" "$CTX" >&2
    fi
fi

# ns4-server links ns4_cuda.o but has its own link step, so a rebuild of the
# CUDA object does not update it. This has silently served a stale binary
# before -- once after the FA2 work landed, and again after the 32-lane MoE
# and wide-pair decode kernels. Refuse rather than serve last week's kernels.
if [[ -f ./src/ns4_cuda.o && ./src/ns4_cuda.o -nt ./ns4-server ]]; then
    die "ns4-server is older than src/ns4_cuda.o -- relink first:
       make ns4-server CUDA_ARCH=sm_120   (takes <1s if the objects are current)"
fi

export NS4_CUDA_NO_Q8_F16_CACHE="${NS4_CUDA_NO_Q8_F16_CACHE:-1}"
# The arena is sized exactly for the startup spans; the chunk size below only
# governs the chunked fallback (NS4_CUDA_WEIGHT_ARENA_EXACT=0) and any spill.
export NS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${NS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-5184}"
export NS4_CUDA_WEIGHT_PRELOAD_SPAN_MB="${NS4_CUDA_WEIGHT_PRELOAD_SPAN_MB:-512}"

# Decode levers from the fp4-mirror build, validated with the summation-order
# NLL protocol (an NLL delta inside the spread between equally valid summation
# orders is noise, not a quality change).  Quality-neutral and deterministic, but
# every one except SCAN_INT is numerically different from the bit-exact
# default paths, so greedy continuations eventually diverge from a run with
# them off.  Export any of them as 0 to fall back (the threshold only takes
# 64..4096: export it empty to keep the engine default):
#   SCAN_INT      bit-exact.  Exact-integer dp4a indexer scan over the fp4
#                 mirrors; matches the default f32 tree, short and long
#   F16_ONE_WIDE  NOT bit-exact.  Replaces the cuBLAS one-token f16 GEMVs
#                 (router, indexer projection, hc fns) and their f32->f16
#                 feed conversions
#   F16_PAIR_WIDE NOT bit-exact.  Wide compressor pair GEMV
#   Q8_PAIR_WIDE  NOT bit-exact.  Wide q8 pair GEMV
# Dropping the non-bit-exact levers (including the sparse threshold below)
# costs nothing on prefill and 4.5-4.8% of decode from 16k up, 10-12% at 4-8k.
export NS4_CUDA_INDEXER_SCAN_INT="${NS4_CUDA_INDEXER_SCAN_INT:-1}"
export NS4_CUDA_F16_ONE_WIDE="${NS4_CUDA_F16_ONE_WIDE:-256}"
export NS4_CUDA_F16_PAIR_WIDE="${NS4_CUDA_F16_PAIR_WIDE:-256}"
export NS4_CUDA_Q8_PAIR_WIDE="${NS4_CUDA_Q8_PAIR_WIDE:-256}"
# +7.5% decode at ctx 4096-8192, inert at 16k and above (accepted range ends
# at 4096; the dense path cannot reach long context -- shared-memory bound).
# NOT bit-exact: identical to the default path on short prompts, but it
# changes results once the sequence reaches its 4k-8k band.
export NS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD="${NS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD-4096}"

printf 'run-cuda-436k-server: ctx=%s prefill-chunk=%s gpu-vram=%s dspark=%s -> %s:%s\n' \
    "$CTX" "$PREFILL_CHUNK" "$GPU_VRAM" "$DSPARK" "$SERVER_HOST" "$SERVER_PORT"
case "$PREFILL_CHUNK" in
    2048) EXPECT_CTX_BUFFERS="7619.27" ;;
    1024) EXPECT_CTX_BUFFERS="6681.19" ;;
    512)  EXPECT_CTX_BUFFERS="6212.15" ;;
    *)    EXPECT_CTX_BUFFERS="" ;;
esac
[[ -n "$EXPECT_CTX_BUFFERS" ]] && printf 'run-cuda-436k-server: expect "context buffers %s MiB" at the default ctx; full prefill speed holds through ctx 536000, managed KV past ~537900\n' "$EXPECT_CTX_BUFFERS"

exec ./ns4-server --cuda -m "$MODEL" \
    --gpu-vram "$GPU_VRAM" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --ctx "$CTX" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    "${DSPARK_ARGS[@]}" \
    "$@"
