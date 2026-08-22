# Agent Notes

ns4 (NEUTRON STAR) is a lightning speed CUDA optimized inference engine and
server for DeepSeek V4 Flash, a rework of the original ds4 project. It is not
a generic GGUF runner. The goal is a small, readable, high-performance C
codebase with CUDA kernels in `src/ns4_cuda.cu`.

## Goals

- Keep the production path as whole-model CUDA graph inference.
- Always make sure that the SSD streaming, CUDA, and distributed inference paths are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long-running sessions practical through live KV reuse and disk KV checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++ outside the CUDA translation units (`.cu`/`.cuh`, which nvcc compiles as C++).

## Safety

- Avoid very large CPU inference runs; the CPU path has previously exposed host kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

- `src/ns4.c`: model loading, tokenizer, CPU reference code, graph scheduling,
  sessions, disk-cache payload serialization.
- `src/ns4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `src/ns4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue,
  streaming, tool-call mapping, disk KV cache policy.
- `src/ns4_cuda.cu`: CUDA runtime and compute kernels; `src/cuda/mmq/` vendors
  the llama.cpp quantized-matmul prefill tier (see its VENDOR.md).
- `vendor/`: linenoise (CLI REPL) and rax (disk-KV index).
- `tools/`: gguf-tools (quantizer, imatrix, quality scoring) and dir-steering.
- `bench/`: speed benchmarking recipes, corpora, and decode profiling.
- `tests/`: unit and live integration tests; `docs/`: operational docs;
  `scripts/`: server launchers and the model download script.

This list is not complete, check the files for more info.

## Testing

Use `make cuda-generic` for build validation. Use `make test` for
unit/regression tests when a model and a CUDA device are available. Use live server tests only when intentionally
testing the API surface.

At every major change where one of the following could be affected, make sure to:

1. Test the normal CUDA path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected.
4. Ask the user before testing on any remote or shared host.
