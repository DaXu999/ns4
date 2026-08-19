#!/usr/bin/env bash
#
# Context-frontier sweep in the tuned serving configuration -- the run behind
# the Speed table in the top-level README.
#
# ns4-bench measures instantaneous prefill and decode at context frontiers,
# but its defaults are not the served configuration. This script pins what
# the table needs:
#
#   * the tuned NS4_CUDA_* environment stack (keep in sync with
#     scripts/run-cuda-436k-server.sh, the source of truth for it),
#   * --ctx-alloc 436000 -- prefill depends on the allocated context, not
#     just the sequence position, so measure at the served allocation,
#   * --prefill-chunk 2048, matching the launcher.
#
# Protocol: NS4_BENCH_RUNS full frontier ladders (default 4). Run 1 is
# warm-up and excluded from the summary -- the first run of a session reads
# low on both metrics. The summary prints per-frontier medians of
# prefill_tps and gen_steady_tps over the remaining runs. The README table
# quotes the 8192..262144 rows; the 4096 frontier exists to absorb per-run
# warm-up so every quoted prefill interval is a clean non-first one.
#
# Every value below can be overridden from the environment; extra ns4-bench
# flags are passed through.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

MODEL="${NS4_MODEL:-$ROOT/ns4flash.gguf}"
CTX_ALLOC="${NS4_CTX:-436000}"
PREFILL_CHUNK="${NS4_PREFILL_CHUNK:-2048}"
GPU_VRAM="${NS4_GPU_VRAM:-94}"
RUNS="${NS4_BENCH_RUNS:-4}"
GEN_TOKENS="${NS4_BENCH_GEN_TOKENS:-512}"
CTX_START="${NS4_BENCH_CTX_START:-4096}"
CTX_MAX="${NS4_BENCH_CTX_MAX:-262144}"
PROMPT_FILE="${NS4_BENCH_PROMPT_FILE:-$ROOT/bench/promessi_sposi.txt}"
OUT_DIR="${NS4_BENCH_OUT_DIR:-/tmp/ns4-ctx-sweep}"

die() { printf 'ctx-sweep: %s\n' "$*" >&2; exit 1; }

[[ -f "$MODEL" ]] || die "model not found: $MODEL (set NS4_MODEL)"
[[ -f "$PROMPT_FILE" ]] || die "prompt file not found: $PROMPT_FILE"
[[ -x ./ns4-bench ]] || die "./ns4-bench not built -- make ns4-bench CUDA_ARCH=sm_120"

# Same stale-link trap as ns4-server: ns4-bench has its own link step, so a
# rebuilt CUDA object does not update it. Refuse to measure last week's
# kernels.
if [[ -f ./src/ns4_cuda.o && ./src/ns4_cuda.o -nt ./ns4-bench ]]; then
    die "ns4-bench is older than src/ns4_cuda.o -- relink first:
       make ns4-bench CUDA_ARCH=sm_120   (takes <1s if the objects are current)"
fi

export NS4_CUDA_NO_Q8_F16_CACHE="${NS4_CUDA_NO_Q8_F16_CACHE:-1}"
export NS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${NS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-5184}"
export NS4_CUDA_WEIGHT_PRELOAD_SPAN_MB="${NS4_CUDA_WEIGHT_PRELOAD_SPAN_MB:-512}"
export NS4_CUDA_INDEXER_SCAN_INT="${NS4_CUDA_INDEXER_SCAN_INT:-1}"
export NS4_CUDA_F16_ONE_WIDE="${NS4_CUDA_F16_ONE_WIDE:-256}"
export NS4_CUDA_F16_PAIR_WIDE="${NS4_CUDA_F16_PAIR_WIDE:-256}"
export NS4_CUDA_Q8_PAIR_WIDE="${NS4_CUDA_Q8_PAIR_WIDE:-256}"
export NS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD="${NS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD:-4096}"

mkdir -p "$OUT_DIR"
(( RUNS >= 3 )) || printf 'ctx-sweep: warning: %s run(s); the protocol wants 4 (1 warm-up + 3 measured)\n' "$RUNS" >&2

for (( i = 1; i <= RUNS; i++ )); do
    printf 'ctx-sweep: run %d/%d start %s\n' "$i" "$RUNS" "$(date +%H:%M:%S)"
    ./ns4-bench --cuda -m "$MODEL" \
        --gpu-vram "$GPU_VRAM" \
        --prefill-chunk "$PREFILL_CHUNK" \
        --ctx-alloc "$CTX_ALLOC" \
        --prompt-file "$PROMPT_FILE" \
        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --step-mul 2 \
        --gen-tokens "$GEN_TOKENS" \
        --csv "$OUT_DIR/run$i.csv" \
        "$@"
done

FIRST_MEASURED=2
(( RUNS == 1 )) && FIRST_MEASURED=1
MEASURED=()
for (( i = FIRST_MEASURED; i <= RUNS; i++ )); do
    MEASURED+=("$OUT_DIR/run$i.csv")
done

printf 'ctx-sweep: medians over runs %d-%d (run 1 = warm-up, discarded); per-run CSVs in %s\n' \
    "$FIRST_MEASURED" "$RUNS" "$OUT_DIR"
awk -F, '
    function median(s,    a, m, i, j, t) {
        m = split(s, a, " ")
        for (i = 2; i <= m; i++) {
            t = a[i] + 0
            for (j = i - 1; j >= 1 && a[j] + 0 > t; j--) a[j + 1] = a[j]
            a[j + 1] = t
        }
        if (m % 2) return a[(m + 1) / 2]
        return (a[m / 2] + a[m / 2 + 1]) / 2
    }
    FNR == 1 { next }
    {
        if (!($1 in seen)) { seen[$1] = 1; order[++n] = $1 }
        p[$1] = p[$1] " " $3
        g[$1] = g[$1] " " $8
    }
    END {
        printf "%10s  %11s  %14s\n", "ctx_tokens", "prefill_tps", "gen_steady_tps"
        for (i = 1; i <= n; i++) {
            f = order[i]
            printf "%10s  %11.1f  %14.2f\n", f, median(p[f]), median(g[f])
        }
    }
' "${MEASURED[@]}"
