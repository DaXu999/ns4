# Provenance

ns4 is a standalone DeepSeek-V4-on-CUDA inference engine. It began as a
verbatim import of the `neutron_star` branch of ds4
(`github.com/DaXu999/ds4`, commit `964117d`); the import commit in this
repository records the matching tree hash. Everything since is ns4's own
history.

## What ns4 deliberately does not carry

- **Other backends.** The Metal (macOS), ROCm, and GLM-model support that
  existed in ds4 was removed upstream before the import. ns4 is CUDA-only
  (plus the `make cpu` reference build) and its sources are written as such.
- **Two-machine tensor parallelism** (`--tensor-parallel`, RDMA/TCP): was
  Metal-gated end to end. Single-host `--cuda-tensor-parallel` is a
  different mechanism and is fully supported.
- **Resident expert-cache budgeting.** ds4's Metal backend kept a
  fixed-size resident routed-expert cache for SSD streaming, with CLI
  budgeting (`--ssd-streaming-cache-experts`, `--ssd-streaming-preload-experts`),
  an automatic budget, a 13k-entry built-in preload hotlist, and a
  cache-pressure regression test. On CUDA the underlying cache was never
  implemented (every budget entry point was a no-op), so ns4 removed the
  whole surface rather than ship inert flags. CUDA SSD streaming uses the
  per-token selected-expert cache, which needs no budget. A future
  resident-cache project would reintroduce this deliberately.
- **PRO Q4 expert-table preload**: likewise double-stubbed on CUDA;
  removed.
- **The agent front end** (`ns4-agent` and its web-tool layer): deferred,
  not dropped — serving is the first-priority path. It can be
  reintroduced from the ds4 sources when agent work resumes.

## Where recovery lives

The ds4 repository retains the full pre-import history, including the
last-with-Metal commit (`2e18805`) from which the removed backend, the
two-machine TP transport, and the resident-cache implementation can be
recovered, and the base commit (`1965844`) against which the branch was
verified bit-identical.

## Vendored code

`src/cuda/mmq/` vendors CUDA matmul kernels from llama.cpp (MIT); see
`src/cuda/mmq/VENDOR.md` for the upstream commit pin and per-file
verbatim/patched status. The LICENSE file preserves the upstream
attribution lines.

## Frozen data

`tools/gguf-tools/imatrix/dataset/`, `tools/gguf-tools/quality-testing/data/`,
`tests/test-vectors/`, and the long-context prompt fixtures are content
imported byte-for-byte from ds4 (several deliberately contain the old
`ds4` spellings inside prompt text and fixture headers). Do not rename
inside them: the calibration dataset feeds imatrix collection, the quality
data and test vectors are scoring fixtures, and changing their bytes
changes what they measure.
