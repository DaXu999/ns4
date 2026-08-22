# NS4 GGUF Tools

This directory contains the offline tools used to build and evaluate DeepSeek
V4 Flash GGUF files for `ns4`.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer.  It implements the NS4 output formats we actually ship:
  `q8_0`, `q8_K`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with `ns4`.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C tools/gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Generate An Imatrix

First inspect the tracked calibration dataset in
`tools/gguf-tools/imatrix/dataset/` (regenerating it needs a ds4 checkout; see
`tools/gguf-tools/imatrix/README.md`).

Then collect activation statistics with the NS4 runtime:

```sh
./ns4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset tools/gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ns4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with this NS4 quantizer.  Generic GGUF
tools need NS4-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `ns4 --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the NS4
prefill graph over the full calibration corpus and reads routed-MoE activation
statistics back from the GPU.  The latest published imatrix-generated GGUF files
are available in the antirez Hugging Face repository:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/tree/main
```

## Generate Q2 And Q4 GGUFs

The template GGUF supplies metadata, tokenizer, tensor order, and logical
shapes.  Tensor bytes are regenerated from the Hugging Face safetensors.  Full
generation is intentionally offline and heavy: expect roughly 80-90 GB outputs
for the 2-bit template family and roughly 150-170 GB for the 4-bit routed-expert
family, plus enough free disk for the temporary output.  Use `--dry-run` and
`--compare-tensor` before starting a full write, and use `--overwrite` only when
you really mean to replace an existing GGUF.

Q2 routed experts with imatrix:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ns4.dat
```

Q4 routed experts with imatrix:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ns4.dat
```

True Q8_K routed experts:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --out gguf/DeepSeek-V4-Flash-Q8KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --experts q8_K \
  --threads 8
```

You can override tensor families:

```sh
--experts iq2_xxs
--routed-w2 q2_k
--attention-proj q8_0
--shared q8_0
--output q8_0
```

Useful checks before writing a full model:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template MODEL.gguf \
  --compare-tensor blk.0.attn_q_a.weight
```

`--compare-tensor` regenerates a single tensor and byte-compares it against the
template or `--compare-gguf`.  `--threads N` controls routed-expert workers.

## Convert A DSpark Support Checkpoint

The DSpark Flash checkpoint is published as Hugging Face safetensors and stores
the draft module under `mtp.0`, `mtp.1`, and `mtp.2`.  Before writing a support
GGUF, inspect the official index and verify that every DSpark tensor name is
understood by the converter:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash-0731 \
  --dspark-manifest > /tmp/dspark-manifest.tsv
```

The manifest reads only `model.safetensors.index.json`; it does not require the
large shard files to be present.  The final summary should report three DSpark
stages and zero unknown DSpark tensors before attempting a full conversion.

To build the support GGUF used by `ns4 --mtp`, run the DSpark support mode.  This
mode writes standalone DSpark metadata plus the packed `mtp.*` tensor payloads;
it does not require a base-model GGUF template:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash-0731 \
  --dspark-support \
  --out DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

`--dspark-support --dry-run` reads safetensors shard headers to derive exact
GGUF shapes and types, but it does not read tensor payloads.  The DSpark metadata
defaults match the published Flash DSpark config: block size 5, target layers
40,41,42, Markov rank 256, and noise token 128799.  Override them with
`--dspark-block-size`, `--dspark-target-layers`, `--dspark-markov-rank`, and
`--dspark-noise-token-id` if converting a different checkpoint.

Before a full write, regenerate one support tensor and record its checksum:

```sh
tools/gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash-0731 \
  --dspark-support \
  --compare-tensor mtp.0.main_proj.weight
```

This reads only the payloads needed for that tensor.  Add `--compare-gguf
DeepSeek-V4-Flash-DSpark-support-0731.gguf` to byte-compare against an existing
support GGUF.

## When No Imatrix Is Given

`iq2_xxs` requires an importance vector.  If `--imatrix` is not provided and
the target type requires one, `deepseek4-quantize` computes a synthetic fallback
from the dequantized weight itself:

```text
importance[column] = sum(row[column]^2) over all rows
```

This is a weight-energy heuristic.  It is not as good as measuring real NS4
activations, but it gives the quantizer a stable column weighting and was good
enough for the first working 2-bit GGUFs.

## Quality Testing

See `quality-testing/README.md`.  The short version is:

```sh
python3 tools/gguf-tools/quality-testing/collect_official.py
make -C tools/gguf-tools quality-score
tools/gguf-tools/quality-testing/score_official MODEL.gguf tools/gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/model.tsv 4096
python3 tools/gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```
