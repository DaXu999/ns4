# QA Before Releases

This is the release gate for ns4.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: CUDA graph inference,
SSD streaming, distributed execution, disk KV cache, and server APIs.

Do not run multiple huge model processes at the same time.  Record the commit,
hardware, GGUF file, context size, and any non-default flags for every manual
run.

Release test hardware:

- Primary: a single RTX PRO 6000 Blackwell class GPU (local).
- Experimental: DGX Spark / GB10.
- Optional: a multi-GPU CUDA host for the tensor-parallel gates, when one is
  available.

Ask before running QA on any remote or shared host, and ask again for each
new QA pass; earlier permission does not carry over.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the normal local target:
  `make clean && make cuda CUDA_ARCH=sm_NN` (or `make cuda-generic`).
- Build CPU-only binaries as a compile check only:
  `make clean && make cpu`.
- Treat compiler warnings as build failures. Save each release and test build's
  complete output and require no `warning:` or NVCC `warning #` lines. Fix the
  source when possible; use a narrow target-specific suppression only when a
  test deliberately compiles a partial translation unit.
- Repeat the warning-free build gate on every validated release target:
  `make clean && make cuda CUDA_ARCH=sm_120` on the primary host and
  `make clean && make cuda-spark` on DGX Spark.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./ns4 --help` and `./ns4-server --help` render cleanly, with
  readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test`.
- Run `tests/test_gpu_args_cli.sh` explicitly after changing executable option
  parsing or multi-GPU placement. Invalid values and device/budget count
  mismatches must reach the shared GPU parser in all four binaries; an
  `unknown option` response from a binary that advertises the flag is a
  release blocker. On CUDA, also start `ns4-server` once with
  `--gpu-vram auto` and the intended `--gpu-devices` list and preserve the
  resolved layout line.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `NS4_TEST_MODEL=/path/to/0731.gguf
  NS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ns4_test --logprob-vectors`
  and
  `NS4_TEST_MODEL=/path/to/0731.gguf
  NS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/flash-0731/local-golden.vec
  ./ns4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ns4_test --server`.
- Run `./ns4-eval --self-test-extractors`.

### Critical Input And Server Regression Pass

Run these checks after changing parsers, server generation, model loading,
distributed snapshots, caches, DSpark, or CUDA build rules. Keep the item
numbers in the QA report so omissions are visible.

1. Send malformed OpenAI, Responses, and Anthropic requests with repeated
   owned string or array fields under ASan. Each request must fail cleanly and
   a following valid request must still work. Run `./ns4_test --server` too.
2. Replay at least 4,096 assistant/tool-result pairs through the Responses and
   Anthropic validators. Validation must remain linear-time and preserve the
   same accepted and rejected histories as a short replay.
3. Feed a distributed worker a snapshot header whose declared lengths exceed
   the configured and protocol limits. It must reject the header before a
   large allocation or payload read, without growing RSS materially.
4. Run malformed safetensors and GGUF fixtures through the loader and
   `tools/gguf-tools/deepseek4-quantize` under ASan and UBSan. Truncated files,
   impossible dimensions, and overflowing tensor sizes must be rejected.
5. With a checkpoint-matched DSpark drafter, compare temperature-zero output
   against a no-drafter run for 400 and 800 accepted target tokens. Output must
   be byte-identical. Record acceptance and decode speed because correctness
   replay can change the useful speedup.
6. Exercise unterminated and twice-closed reasoning in streaming and
   non-streaming OpenAI, Responses, and Anthropic requests, with and without
   tools. Reasoning must never leak into answer content.
7. On real Blackwell hardware, build the CUDA targets for `sm_120` or `sm_120a`
   and for DGX Spark `sm_121`. Confirm the emitted architecture flags retain
   the architecture-specific feature suffix and run `make cuda-regression`.
8. Build with CUDA 12.8 or newer and require the CUDA translation units to
   compile warning-free, including the `FLT_MAX` users.
9. Force a conversation past the in-memory KV threshold, restore the same disk
   checkpoint twice, and confirm the checkpoint file remains present after
   both successful loads. Corrupt checkpoints must still be rejected.
10. Run `make dspark-verify-depth` with matching 0731 target and drafter files.
    Strict capture must skip layers without a compressor and compare every
    captured compressor layer.
11. Send the same long prompt twice to one server session. The second
    request must report `cache_source: memory-rewind`, reuse through one token
    before the prompt boundary, and produce the same greedy output as a fresh
    session.
12. Run `./ns4_test --think-tool-recovery`, then repeat through all three HTTP
    APIs. A complete tool block inside unclosed reasoning must be recovered
    once, preceding prose must remain reasoning, and no synthetic continuation
    may be generated.
13. Run the server parser tests under UBSan with `NaN`, positive infinity, and
    negative infinity where integer JSON fields are expected. Conversion must
    be defined and clamped, with no sanitizer report.

## 3. Official Continuation Quality Gates

These tests are release-blocking after tokenizer, template, KV-cache, attention,
MoE routing, quantization, logit, or model-graph changes.  They are
teacher-forced continuation checks against hosted-model output and API
top-logprob slices, so do not replace them with one sampled chat answer.

- Build the scorer:
  `make -C tools/gguf-tools quality-score`.
- Match every Flash GGUF to the fixture captured from the same checkpoint.
  The current release checkpoint is 0731 and uses
  `tests/test-vectors/flash-0731/`; the older undated GGUF uses the preserved
  `tests/test-vectors/flash-pre-0731/` fixture. Never report a cross-checkpoint
  failure as a quality regression. New checkpoints require a new
  `flash-CHECKPOINT/` directory before release QA; do not replace an older
  fixture. Checkpoint-labelled GGUFs such as `-0731` must use the fixture with
  the same label.
- Run the tracked DeepSeek V4 Flash 0731 smoke vectors:
  `NS4_TEST_MODEL=/path/to/0731.gguf
  NS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ns4_test --logprob-vectors`.
  This covers short prompts and long-prompt attention cases. The runner defaults
  to this fixture, but release logs should keep the path explicit.
- Run the 100-case DeepSeek V4 Flash fixture for every released Flash GGUF:
  `tools/gguf-tools/quality-testing/score_official /path/to/deepseek-v4-flash.gguf tools/gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/flash.tsv 4096`.
  This manifest is also for the 0731 checkpoint. A later checkpoint needs a
  separately named 100-case fixture and must not be scored against this one.
- Treat the native MXFP4 Flash GGUF as a separate release artifact. Run the
  same 100-case fixture on resident CUDA and CUDA SSD streaming; compare the
  streaming result with the resident baseline.
  For resident multi-GPU CUDA, pass the normal placement flags to the scorer,
  for example `--gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7
  --cuda-tensor-parallel`.
- For SSD streaming, run the same official-continuation scorer once with full
  residency and once with `--ssd-streaming` for the release model.  The summary
  and API agreement should stay in the same quality band.
- Compare any candidate against the previous release or last-known-good output:
  `python3 tools/gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv`.
  Treat a large first-token-match drop, a clear NLL regression, or a material
  API top-1/pair-order regression as a blocker unless the release notes call out
  an intentional quality tradeoff.
- Keep the raw `summary` and `api_summary` lines in the release notes or QA log.

## 4. Flash Path

Use the normal Flash GGUF that 128 GB users run.

- One-shot CLI:
  `./ns4 -m ns4flash.gguf --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./ns4 --nothink --temp 0 --dump-logprobs /tmp/ns4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `ns4-bench` with `bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.
- For native MXFP4 changes, run `make mxfp4-dot-test test-mxfp4-cuda`, then a
  short greedy prompt and the section 3 continuation fixture with the MXFP4
  GGUF. The synthetic fused-MoE test and full-model quality gate must both pass.

### DSpark / DeepSpec Runtime

DSpark is opt-in, but it mutates the verifier, target-hidden capture, support
model loading, and scheduler paths.  Run these whenever DSpark support,
speculative verification, confidence/scheduler policy, target hidden capture,
tiny routed-MoE verifier kernels, or shared `--mtp` support-model code changes:

Use the 0731 DSpark support GGUF only with a Flash 0731 target. A support model
from another checkpoint can have plausible acceptance statistics while
producing a different greedy continuation.

- Default greedy acceptance fixture:
  `NS4_DSPARK_MODEL=$NS4_GGUF_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf NS4_DSPARK_SUPPORT=$NS4_GGUF_DIR/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- 64-token guardrail:
  `NS4_DSPARK_FIXTURE_TOKENS=64 NS4_DSPARK_MODEL=$NS4_GGUF_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf NS4_DSPARK_SUPPORT=$NS4_GGUF_DIR/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- Fixed-block partial-accept fallback:
  `NS4_DSPARK_FIXTURE_CONFIDENCE=0 NS4_DSPARK_FIXTURE_TOKENS=8 NS4_DSPARK_FIXTURE_REQUIRE_PARTIAL=1 NS4_DSPARK_MODEL=$NS4_GGUF_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf NS4_DSPARK_SUPPORT=$NS4_GGUF_DIR/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- DSpark verifier invariant smoke:
  `NS4_TEST_MODEL=$NS4_GGUF_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf NS4_DSPARK_SUPPORT=$NS4_GGUF_DIR/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-verify-depth`.
- If shared support-model or verifier structures changed, also run legacy MTP:
  `make mtp-verify-depth` with `NS4_TEST_MTP` set to a one-stage MTP support
  GGUF, or confirm the target skips only because the optional file is missing.
- Record `c_add` `accepted_draft`, `errors=0`, `verify_layer`, and `net_saved`
  for both 32-token and 64-token runs.  A faster run with lower proposal
  quality is a regression unless it was an intentional scheduler change.
- If verifier MoE kernels changed, run one diagnostic `c_add` profile with
  `NS4_DSPARK_VERIFY_SELECTED_PROFILE` set (any value enables it) or the MoE
  stage profiler and record the selected-expert footprint or stage timing in
  the DSpark log.

### Session Microbatching

Run these gates whenever session scheduling, batched decode, mixed
prefill/decode, QKV projection, shared or routed experts, tensor parallelism,
or backend fallback selection changes.

- Run the CUDA session-batch and mixed-batch oracles:
  `NS4_TEST_MODEL=/path/to/ns4flash.gguf make test-cuda-session-batch` and
  `NS4_TEST_MODEL=/path/to/ns4flash.gguf make test-cuda-mixed-batch`.
- The oracle must cover reversed row ordering, at least six decode steps, and a
  mixed prefill/decode call. Any nonzero differing-logit count is a blocker;
  argmax-only agreement is insufficient.
- Benchmark 1, 2, 4, 8, and 16 simultaneous resident sessions on the same host
  and model. Record model-step latency and aggregate decode tokens/second, not
  only request completion speed.
- Verify unsupported combinations explicitly: DSpark/MTP support models,
  SSD streaming, quality/reference modes, steering, resident Q4 expert
  overlap, and CPU-router modes must use the established exact fallback or
  reject the combination before evaluation. They must not partially activate
  native batching.

## 5. SSD Streaming

SSD streaming is a capacity path, so test both correctness and user experience.

- Flash q2/q2-q4 streaming:
  `./ns4 -m ns4flash.gguf --ssd-streaming -p "..."`
- Regression test mixed-quant Flash SSD streaming. Use the mixed q2/q4 GGUF
  with boosted Q4 routed-expert layers and a prompt long enough to exercise the
  selected-address prefill path; it must not fail with "model range is not
  covered by mapped model views":
  `./ns4 -m gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf --ssd-streaming --ctx 4096 --tokens 1 --nothink --prompt-file /tmp/ns4_600tok_prompt.txt`.
- Confirm generation does not stall on repeated expert misses for a small
  interactive prompt.
- If streaming cache internals changed, test the same prompt twice and compare
  first-token/logprob sanity between runs.

## 6. CUDA / DGX Spark

Do not claim a CUDA target is release-ready without this pass on that target.

- Fetch or push the exact release commit to the target machine.
- Build:
  `make clean && make cuda-spark`.
- Require every validated CUDA build to complete without compiler warnings.
- Run:
  `make cuda-regression`.
- For native MXFP4 changes, run
  `make test-mxfp4-cuda CUDA_ARCH=native` on the primary host and
  `make test-mxfp4-cuda CUDA_ARCH=sm_121` on DGX Spark. Dense MMQ, routed MMQ,
  routed MMVQ, fused gate/up, and fused down must pass. The Spark run must also
  pass the Blackwell K-tile guard. This synthetic parity test does not replace
  full-model continuation scoring.
- Run the native MXFP4 GGUF resident where it fits, and run it with
  `--ssd-streaming` on DGX Spark. Use the same greedy prompt and continuation
  fixture on both. Record prefill and generation speed, require finite logits,
  and compare quality with the resident MXFP4 result. Blackwell MMQ quantizes
  activations to native FP4 for batched work; decode MMVQ keeps Q8 activations,
  so quality must be checked rather than inferred from kernel-only parity.
- Run a short CLI prompt with the Flash GGUF and record generation t/s.
- Run a longer prompt that exercises routed experts past a few thousand tokens.
- When a multi-GPU CUDA host is available, run the full-vocabulary decode
  oracle there:
  `NS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-session-batch`.
  Preserve the per-batch timing for 2, 4, and 8 rows and require
  `nonexact_logits=0`. Run the released Q4 file and the reduced-precision Q2
  file: Q4 exercises grouped routed/shared stages, while unsupported Q2 native
  MoE shapes must retain the ordered exact fallback.
- With CUDA TP attention enabled, compatible Q4 runs must use grouped
  attention-core, QKV, KV-store, and attention-post by default and remain
  full-vocabulary exact against isolated decode. The 16-row decode step must
  not regress against the host's recorded baseline. Repeat once with
  `NS4_CUDA_TP_ATTN=0` only as rollback coverage; it is not the production
  configuration.
- Run native mixed prefill/decode at the default frontier and at compressed
  context:
  `NS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch` and
  `NS4_TEST_CONTEXT=4096 NS4_TEST_MIXED_INITIAL=2048 NS4_TEST_MIXED_ROUNDS=8
  NS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch`.
  Every round must report exact logits and `mode=native`; a serialized fallback
  is a failure for the multi-GPU TP/EP topology. Under CUDA TP attention, the
  native mixed step must use the same exact grouped decode stages when their
  capability checks pass; record correctness and speedup separately. Also
  run with `NS4_TEST_MIXED_QUANTUM=800 NS4_TEST_ALLOW_FALLBACK=1`: the 800-row
  prefill quantum must report the serialized safety fallback, which
  `NS4_TEST_ALLOW_FALLBACK` tolerates instead of failing the round.
- On the multi-GPU host, start `ns4-server` with 8
  and 16 batched sessions and issue at least that many simultaneous requests
  with mixed prompt lengths. Verify no session mix-up, deadlock, or starvation
  and record aggregate generation throughput.
  `tests/test_server_batching.py --url http://HOST:PORT --pairs 8` drives this:
  it submits seeded duplicate pairs in one concurrent wave and requires
  identical outputs per pair.
- On DGX Spark, verify the same public batch API and server concurrency use the
  single-GPU fallback without creating peer-only TP/EP state. The multi-GPU
  native oracle is not a valid Spark test because its topology is intentionally
  unavailable there.
- If CUDA Q4, distributed, streaming hooks, tensor span loading, or model cache
  code changed, test the specific GGUF and split mode that uses that path.

## 7. Distributed Inference

Distributed code has regressed around route setup, KV snapshots, request IDs,
and split model loading.  Test it whenever distributed, KV, session, or model
loading code changes.

- Start workers first, then the coordinator.
- Test a small prompt and a longer prompt.
- Verify the coordinator waits for a complete route and exits cleanly.
- Verify `Ctrl+C` returns control after the current distributed token or chunk
  drains.
- Save and restore a distributed KV snapshot if that code changed.
- If CUDA distributed is relevant, test across the CUDA hosts and record
  generation speed, not just "it works".

## 8. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./ns4-server --ctx 100000 --kv-disk-dir /tmp/ns4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.

## 9. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` should serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- When a multi-GPU CUDA TP host is available, start `ns4-server` with the
  release TP options and verify all 16 100k-context sessions allocate.
  Startup must report a 2048-token prefill cap; a silent fallback to 4096 is
  an OOM regression.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.

## 10. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test a Flash target enough to verify URL, resume, Hugging
  Face CLI/curl behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 11. Performance And Power

- Run `ns4-bench` on the release machine and compare with tracked CSV baselines.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, eval, and bench
  where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 12. Speed Regression

Performance is a release gate. A correct result that is unexpectedly much
slower still needs an explanation before release.

Use the same commit, GGUF checksum, prompt, context frontier, generated-token
count, power setting, and backend flags as the reference run. Let the machine
become idle, discard the first warm-up run, then record the median of three
runs. Do not compare different model checkpoints or quantizations. For batched
tests, record aggregate and per-session decode speed.

- A slowdown over 5% requires a clean rerun and investigation.
- A repeatable slowdown over 10% in prefill, decode, or aggregate batched
  decode is a release blocker unless the change and tradeoff are documented.
- Keep the complete `ns4-bench` CSV. A single short-prompt average is not enough
  to detect a context-dependent regression.
- Compare startup time and peak memory as well as tokens per second when model
  loading, caches, streaming, or temporary arenas changed.
- Run the backend-specific batch tests in sections 4 and 6. Fast single-session
  decode does not substitute for aggregate multi-session throughput.

These are the last known good observations available when this gate was added.
They are reference points for matching hardware and workloads, not performance
claims across different models or contexts.

| System and backend | Model and workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| RTX PRO 6000 Blackwell, CUDA | Flash q2 imatrix 0731, ctx 131,072, tuned serving config | ~3,150 t/s | 74.9 t/s |

The RTX PRO 6000 row uses the tuned launcher configuration in
`scripts/run-cuda-436k-server.sh`; compare against it with the same
environment exports.

## 13. Release Sign-off

Do not sign off until:

- The Flash path passed.
- Official-quality, MTP, batching-fallback, and CUDA gates passed.
- Official continuation quality gates passed for every released model family.
- CUDA was tested on the CUDA machine or the release notes explicitly say CUDA
  was not validated.
- CUDA, CPU-only, and test builds completed without compiler warnings on every
  release target that was validated.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- The speed-regression gate passed on every validated backend, with any skipped
  baseline or intentional slowdown documented.
- CUDA native decode and mixed batching passed, including the forced fallback
  gates.
- Any skipped item is written down with the reason.
