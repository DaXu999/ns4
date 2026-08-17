#!/bin/sh
set -e

REPO="antirez/deepseek-v4-gguf"
DS4F_Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
DS4F_Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf"
DS4F_MXFP4_FILE="DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf"
DS4F_Q2_Q4_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
DS4F_DSPARK_FILE="DeepSeek-V4-Flash-DSpark-support-0731.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${NS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DwarfStar GGUF downloader

Usage:
  ./download_model.sh ds4f-q2 [--token TOKEN]
  ./download_model.sh ds4f-q2-q4 [--token TOKEN]
  ./download_model.sh ds4f-q4 [--token TOKEN]
  ./download_model.sh ds4f-mxfp4 [--token TOKEN]
  ./download_model.sh ds4f-dspark [--token TOKEN]

Targets:

  ds4f-q2
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  ds4f-q2-q4
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB MacBooks. Works on DGX Spark but loading
       may struggle compared to ds4f-q2.

  ds4f-q4
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  ds4f-mxfp4
       Native DeepSeek V4 Flash MXFP4 routed experts, about 156 GB on disk.
       Supported by CUDA; Blackwell uses FP4 tensor cores for batched
       expert work, while CUDA decode keeps Q8 activations.

  ds4f-dspark
       Optional DSpark speculative decoding support GGUF for Flash 0731, about
       6 GB. Enable it with --dspark and --mtp when running ns4 or ns4-server.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  NS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After main-model downloads the script updates:
  ./ns4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ns4 -p "Hello"
  ./ns4-server --ctx 100000

After downloading DSpark support, enable it explicitly in greedy mode:
  ./ns4 --dspark --mtp <download directory>/$DS4F_DSPARK_FILE --temp 0

The ds4f-mxfp4 file is downloaded with the official Hugging Face downloader
instead of the curl path used by the other Flash GGUF files.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
LINK_MODEL=1
FORCE_HF_DOWNLOAD=0
FLATTEN_DOWNLOADS=0

case "$MODEL" in
    ds4f-q2) MODEL_FILE=$DS4F_Q2_FILE ;;
    ds4f-q2-q4) MODEL_FILE=$DS4F_Q2_Q4_FILE ;;
    ds4f-q4) MODEL_FILE=$DS4F_Q4_FILE ;;
    ds4f-mxfp4) MODEL_FILE=$DS4F_MXFP4_FILE; FORCE_HF_DOWNLOAD=1 ;;
    ds4f-dspark) MODEL_FILE=$DS4F_DSPARK_FILE; LINK_MODEL=0 ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

needs_hf_download() {
    if [ "${FORCE_HF_DOWNLOAD:-0}" -eq 1 ]; then
        return 0
    fi
    return 1
}

find_hf_command() {
    if command -v hf >/dev/null 2>&1; then
        printf '%s\n' hf
        return 0
    fi
    for dir in "$HOME"/Library/Python/*/bin "$HOME"/.local/bin; do
        if [ -x "$dir/hf" ]; then
            printf '%s\n' "$dir/hf"
            return 0
        fi
    done
    return 1
}

local_download_name() {
    if [ "${FLATTEN_DOWNLOADS:-0}" -eq 1 ]; then
        basename "$1"
    else
        printf '%s\n' "$1"
    fi
}

download_one_hf() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    hf_out="$OUT_DIR/$file"
    part="$out.part"

    mkdir -p "$(dirname "$out")"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    if [ -e "$part" ]; then
        echo "Found curl partial download: $part" >&2
        echo "The Hugging Face downloader cannot resume curl .part files." >&2
        echo "Move or remove that partial download before retrying this target." >&2
        exit 1
    fi

    HF_CMD=$(find_hf_command || true)
    if [ -z "$HF_CMD" ]; then
        echo "Large GGUF downloads require the official Hugging Face CLI." >&2
        echo "Install it with:" >&2
        echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
        exit 1
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "using $HF_CMD download"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    else
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR"
    fi

    if [ "$hf_out" != "$out" ] && [ -s "$hf_out" ]; then
        mv "$hf_out" "$out"
        rmdir "$(dirname "$hf_out")" 2>/dev/null || true
    fi

    if [ ! -s "$out" ]; then
        echo "Hugging Face download finished but expected file is missing: $out" >&2
        exit 1
    fi
}

download_one() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    if needs_hf_download "$file"; then
        download_one_hf "$file"
        return
    fi

    mkdir -p "$(dirname "$out")"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_one "$MODEL_FILE"

if [ "$MODEL" = "ds4f-dspark" ]; then
    echo
    echo "DSpark support downloaded. Enable it explicitly in greedy mode:"
    echo "  ./ns4 --dspark -m ./ns4flash.gguf --mtp $OUT_DIR/$DS4F_DSPARK_FILE --temp 0"
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ns4flash.gguf
    echo "Linked ./ns4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
