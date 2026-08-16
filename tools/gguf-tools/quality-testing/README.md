# Official-Continuation Quality Testing

This directory contains the prompts, tracked official fixtures, and scripts used
to compare local GGUF variants against hosted-model continuations.

The metric is target-token negative log likelihood: collect a deterministic
official continuation, then ask each local GGUF how much probability it assigns
to that exact continuation token by token.  This avoids judging quality from one
sampled answer.

## 1. Tracked Fixture Sets

Curated fixtures are kept in the repository so release QA can run without
calling hosted APIs:

- `data/flash`: 100 DeepSeek V4 Flash 0731 continuations collected from the
  official DeepSeek API with `top_logprobs=20`.
- `data/pro`: 100 DeepSeek V4 PRO continuations collected from the official
  DeepSeek API with `top_logprobs=20`.

DeepSeek V4 Flash also has tracked official smoke vectors in
`tests/test-vectors/`.  Those vectors drive `./ns4_test --logprob-vectors` and
include short prompts plus long-prompt attention cases.

The hosted APIs expose output-token logprobs and top-logprob alternatives, not
full vocabulary logits.

## 2. Collect Official Continuations

For the tracked DeepSeek V4 Flash 0731 fixture:

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-flash \
  --endpoint https://api.deepseek.com/chat/completions \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/flash \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --thinking disabled \
  --reasoning-effort omit
```

The scorer links against the NS4 runtime and uses CUDA by default.

Build the optional llama.cpp control scorer with:

```sh
make -C gguf-tools quality-llama-score
```

## 4. Score GGUF Variants

```sh
gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/OLD.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/old.tsv \
  4096

gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/NEW.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/new.tsv \
  4096
```

Use `data/flash/manifest.tsv` for Flash GGUFs and
`data/pro/manifest.tsv` for PRO GGUFs.  The scorer and comparator do not care
which model produced the manifest; the manifest path selects the continuation
set.

Add `--quality` to disable NS4's speed-oriented numerical shortcuts. For an
independent llama.cpp comparison of a DeepSeek V4 GGUF, use the same manifest
and the token-identical NS4 prompt renderer:

```sh
gguf-tools/quality-testing/score_llama \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/llama.tsv \
  4096 \
  deepseek-ns4
```

For a full-residency vs SSD-streaming comparison, score the same model twice and
add the streaming flags to one run:

```sh
gguf-tools/quality-testing/score_official \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/streaming.tsv \
  4096 \
  --ssd-streaming
```

## 5. Compare

```sh
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Output fields:

- `avg_nll`: average negative log likelihood; lower is better.
- `delta_new_minus_old`: negative means the new GGUF fits the official
  continuation better.
- `case_wins_new_old_ties`: per-prompt NLL wins.
- `first_token_matches`: how often the local greedy first token matches the
  official first token.
- `avg_greedy_lcp`: average greedy longest common prefix against the official
  continuation.
- `api_target_mae`: when the manifest includes `response_file`, absolute
  local-vs-API logprob delta for aligned official output tokens.
- `api_top_coverage`: fraction of API top-logprob alternatives that map exactly
  to one local tokenizer token.
- `api_top1_rate`: how often the API top alternative equals the local greedy
  token.
- `api_topn_recall`: fraction of mapped API top-N alternatives found in the
  local top-N for the same position.
- `api_top_mae`: local-vs-API logprob MAE over mapped API top alternatives.
- `api_pair_rate`: pairwise ordering agreement among mapped API alternatives.
