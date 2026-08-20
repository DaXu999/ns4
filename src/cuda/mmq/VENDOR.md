# Vendored llama.cpp mmq kernels

This directory contains source files copied verbatim from
[llama.cpp's `ggml-cuda` backend](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda),
plus the ns4-side adapter and kernels (`ns4_ggml_stubs.{h,cu}`,
`ns4_mmq.{h,cu}`, `ns4_mmq_d2r.{cu,cuh}`, `ns4_repack.{cu,h}`) that let the
templated CUDA kernels compile and link without the full ggml runtime.

## Why these files are vendored, not submoduled

ns4 is a flat, self-contained C/CUDA codebase with no third-party build
dependencies. A submodule would force consumers to clone, configure, and
build all of llama.cpp just to use a few quantized-matmul kernels. Vendoring
keeps ns4 self-contained at the cost of a periodic re-sync.

## Upstream pin

| Field         | Value                                                                                                              |
|---------------|--------------------------------------------------------------------------------------------------------------------|
| Source        | https://github.com/ggml-org/llama.cpp                                                                              |
| Commit        | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`                                                                         |
| Commit date   | 2026-05-14                                                                                                         |
| Commit title  | `ggml-hexagon: cpy: add contiguous fast-path in reshape copy (#23076)`                                             |
| License       | MIT (`LICENSE` at the repository root, copyright "2023-2026 The ggml authors"). Compatible with ns4's MIT license. |

## File inventory

| File                    | Origin in llama.cpp                          | Status                                                                   | Lines |
|-------------------------|----------------------------------------------|--------------------------------------------------------------------------|------:|
| `mmq.cuh`               | `ggml/src/ggml-cuda/mmq.cuh`                 | patched: `NS4_CUDA_MMQ_X_MAX` tile hook, aligned-SoA `load_tiles_*` twins, partial last-tile clamps | 4415 |
| `mma.cuh`               | `ggml/src/ggml-cuda/mma.cuh`                 | verbatim                                                                 |  1456 |
| `vecdotq.cuh`           | `ggml/src/ggml-cuda/vecdotq.cuh`             | verbatim                                                                 |  1317 |
| `quantize.cuh`          | `ggml/src/ggml-cuda/quantize.cuh`            | verbatim                                                                 |    41 |
| `quantize.cu`           | `ggml/src/ggml-cuda/quantize.cu`             | verbatim                                                                 |   443 |
| `mmid.cuh`              | `ggml/src/ggml-cuda/mmid.cuh`                | patched: `ns4_mmid_large_enabled` prototype                              |     9 |
| `mmid.cu`               | `ggml/src/ggml-cuda/mmid.cu`                 | patched: large-n global path and case-1 fast path (`NS4_MMID_LARGE`, `NS4_MMID_CASE1`) | 339 |
| `mmvq.cuh`              | `ggml/src/ggml-cuda/mmvq.cuh`                | patched: `mul_mat_vec_q_switch_type` proto exposed; ggml-tensor entries gated on `NS4_MMVQ_INCLUDE_GGML_ENTRIES` | 34 |
| `mmvq.cu`               | `ggml/src/ggml-cuda/mmvq.cu`                 | patched: `mul_mat_vec_q_switch_type` promoted from `static`; -1 router id guards; `ggml_cuda_mul_mat_vec_q` + `ggml_cuda_op_mul_mat_vec_q` gated on `NS4_MMVQ_INCLUDE_GGML_ENTRIES` | 1183 |
| `unary.cuh`             | `ggml/src/ggml-cuda/unary.cuh`               | verbatim (needed by `mmvq.cu` for inline GLU epilogues)                  |   114 |
| `common.cuh`            | `ggml/src/ggml-cuda/common.cuh`              | verbatim                                                                 |  1489 |
| `ggml-common.h`         | `ggml/src/ggml-common.h`                     | verbatim                                                                 |  1900 |
| `vendors/cuda.h`        | `ggml/src/ggml-cuda/vendors/cuda.h`          | verbatim                                                                 |    28 |
| `ggml.h`                | (new)                                        | redirect to `ns4_ggml_stubs.h`                                           |     5 |
| `ggml-impl.h`           | (new)                                        | redirect to `ns4_ggml_stubs.h`                                           |     4 |
| `ggml-cuda.h`           | (new)                                        | redirect to `ns4_ggml_stubs.h`                                           |     4 |
| `ns4_ggml_stubs.h`      | (new)                                        | shim: ggml_type enum, macros, info struct, type_size lookups             |   303 |
| `ns4_ggml_stubs.cu`     | (new)                                        | shim impls: `ggml_cuda_info`, naive pool, `ggml_backend_cuda_context::*` |   160 |
| `ns4_mmq.h`             | (new)                                        | public C ABI for ns4 to call                                             |   857 |
| `ns4_mmq.cu`            | (new)                                        | host entries and ns4 kernels: dense Q8_0 (plain / aligned / D2R dispatch), routed-MoE entries, aligned-SoA decode matvecs | 4588 |
| `ns4_mmq_d2r.cu`/`.cuh` | (new)                                        | D2R (dequant-to-registers) MoE-down and dense Q8_0 GEMMs                 | 3370 / 91 |
| `ns4_repack.cu`/`.h`    | (new)                                        | aligned-artifact builders (IQ2_XXS gate/up, Q2_K down, Q8_0 dense) run at load | 1122 / 126 |

**Upstream (verbatim + patched):** ~12,800 lines of CUDA. **ns4-side:** ~10,600 lines.

## What llama.cpp's `mmq.cu` does that we don't vendor

The upstream `ggml/src/ggml-cuda/mmq.cu` (372 lines) is the ggml-backend
dispatch entry. It talks to `ggml_tensor` / `ggml_backend_buffer_get_usage`,
does its own activation Q8_1 quantization, and wires into the ggml op graph.
We **don't vendor it.** Instead `ns4_mmq.cu` provides ns4-style entries
that bypass the ggml graph and call the per-type `mul_mat_q_case<T>` directly.

## Symbol-resolution table

Symbols the vendored files reference, and how they resolve in this directory:

| Symbol category                                | Resolution                                                                                       |
|------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `GGML_ASSERT`, `GGML_ABORT`, `GGML_UNUSED`     | Macros in `ns4_ggml_stubs.h`                                                                     |
| `GGML_PAD`, `GGML_UNUSED_VARS`                 | Macros in `ns4_ggml_stubs.h`                                                                     |
| `GGML_CUDA_CC_*` constants                     | Defined in vendored `common.cuh`                                                                 |
| `GGML_TYPE_*` enum values                      | Defined in `ns4_ggml_stubs.h::enum ggml_type`                                                    |
| `GGML_CUDA_MAX_DEVICES`, `GGML_CUDA_NAME`      | Macros in `ns4_ggml_stubs.h`                                                                     |
| `GGML_MAX_DIMS`, `GGML_MAX_SRC`                | Macros in `ns4_ggml_stubs.h`                                                                     |
| `block_q8_0`, `block_q2_K`, `block_iq2_xxs`    | Defined in vendored `ggml-common.h` (gated by `GGML_COMMON_IMPL_CUDA`)                           |
| `ggml_half`, `ggml_half2`                      | `uint16_t` typedefs in `ns4_ggml_stubs.h`                                                        |
| `ggml_type_size`, `ggml_blck_size`             | Inline implementations in `ns4_ggml_stubs.h`                                                     |
| `ggml_cuda_info()`                             | Singleton in `ns4_ggml_stubs.cu`, populated via `cudaGetDeviceProperties`                        |
| `ggml_cuda_get_device`, `ggml_cuda_set_device` | Thin wrappers in `ns4_ggml_stubs.cu`                                                             |
| `ggml_time_us`                                 | `std::chrono::steady_clock` in `ns4_ggml_stubs.cu`                                               |
| `ggml_cuda_pool`, `ggml_cuda_pool_alloc`       | Defined in vendored `common.cuh`. Concrete `ns4_naive_pool` subclass in `ns4_ggml_stubs.cu`      |
| `ggml_backend_cuda_context`                    | Defined in vendored `common.cuh`. `new_pool_for_device` and `~dtor` provided in `ns4_ggml_stubs.cu` |
| `ggml_tensor`                                  | Forward declaration in `ns4_ggml_stubs.h`. Never dereferenced - only held as pointer.            |
| `ggml_glu_op`                                  | Enum stub in `ns4_ggml_stubs.h` (fusion args present in common.cuh but unused by ns4)            |
| `CUDA_CHECK`, `CUBLAS_CHECK`                   | Macros in `ns4_ggml_stubs.h`                                                                     |
| `ggml_cuda_launch_mm_ids_helper`               | Defined in vendored `mmid.cu`                                                                    |
| `ggml_cuda_should_use_mmq`                     | Defined in vendored `mmq.cu` - **we re-implement in `ns4_mmq.cu`** since we don't vendor mmq.cu  |
| `ggml_cuda_mul_mat_q*`                         | Upstream's host entry. **Replaced** by `ns4_mmq_q8_0_dense` and family.                          |

## Things we deliberately do NOT support

- **CUDA graphs.** ns4 manages its own streams; `USE_CUDA_GRAPH` is undefined.
- **HIP / MUSA / AMD backends.** vendored code's HIP/MUSA `#ifdef` branches are dead in our build but kept for upstream diff-cleanliness.
- **The full `ggml_tensor` type.** No tensor introspection - shapes and strides come in via raw arguments to `ns4_mmq_*`.
- **`ggml_op` graph evaluation.** We call kernels directly.

## Re-syncing with upstream

When upstream lands a bugfix or perf improvement we want, the procedure is:

```sh
cd /tmp/llama-research && git fetch && git checkout <NEW_COMMIT>
cd <ns4-checkout>/src/cuda/mmq
cp /tmp/llama-research/ggml/src/ggml-cuda/{mmq.cuh,mma.cuh,vecdotq.cuh,quantize.cuh,quantize.cu,mmid.cuh,mmid.cu,common.cuh} .
cp /tmp/llama-research/ggml/src/ggml-common.h .
cp /tmp/llama-research/ggml/src/ggml-cuda/vendors/cuda.h vendors/
# Verify the shim still covers all referenced symbols:
grep -hoE 'GGML_[A-Z_]+|ggml_[a-z_]+' *.cuh *.cu *.h | sort -u > /tmp/symbols.new
diff /tmp/symbols.last /tmp/symbols.new  # check for newly-introduced names
# Update this VENDOR.md's commit pin and run `make cuda CUDA_ARCH=sm_120`.
```

If the new symbols are minor (e.g., new `GGML_CUDA_CC_*` constants), they
likely come from `common.cuh` which we vendor and don't need any shim
changes. If they're new `ggml_*` host functions (rare, but possible if
upstream adds a new helper), extend `ns4_ggml_stubs.h`.

## Testing matrix

| Test                                          | Status                       | Phase     |
|-----------------------------------------------|------------------------------|-----------|
| `nvcc -c src/cuda/mmq/ns4_mmq.cu` builds cleanly | **passes** (sm_120, nvcc 13) | Phase 0   |
| Q8_0 dense parity vs CPU reference            | **passes** (4 shapes)        | Phase 2   |
| Q2_K dense parity                             | **passes** (4 shapes)        | Phase 3   |
| IQ2_XXS dense parity                          | **passes** (4 shapes)        | Phase 3   |
| MoE `_id` Q8_0 parity                         | **passes** (3 shapes)        | Phase 4   |
| MoE `_id` Q2_K parity                         | **passes** (3 shapes)        | Phase 4   |
| MoE `_id` IQ2_XXS parity                      | **passes** (4 shapes)        | Phase 4   |
| `make ns4-bench` with mmq integration         | **builds and runs**          | Phase 5/6 |
| Frontier sweep, ctx 2k-16k, V4 Flash IQ2XXS   | **see results below**        | Phase 7   |

## Validated performance

Fork-era baseline on a PRO 6000 Blackwell (sm_120), CUDA 13.0, V4 Flash
IQ2_XXS GGUF (86.7 GB), mmq on the prefill tier only (decode on the legacy
paths); current numbers are in the top-level README:

| ctx    | baseline pf t/s | mmq pf t/s | speedup    | baseline gen t/s | mmq gen t/s | gen ratio |
|--------|-----------------|------------|------------|------------------|-------------|-----------|
|  2048  | 373.21          | 1033.42    | **2.77x**  | 40.48            | 39.33       | 0.972x    |
|  4096  | 366.25          | 1041.39    | **2.84x**  | 39.64            | 38.64       | 0.975x    |
|  6144  | 364.81          | 1025.24    | **2.81x**  | 39.50            | 38.55       | 0.976x    |
|  8192  | 364.01          | 1026.71    | **2.82x**  | 38.81            | 37.88       | 0.976x    |
| 10240  | 361.75          | 1019.53    | **2.82x**  | 38.45            | 37.57       | 0.977x    |
| 12288  | 360.52          | 1013.17    | **2.81x**  | 38.31            | 37.22       | 0.972x    |
| 14336  | 359.15          | 1004.45    | **2.80x**  | 38.09            | 37.18       | 0.976x    |
| 16384  | 357.99          | 1001.29    | **2.80x**  | 38.71            | 37.86       | 0.978x    |

Sustained ~2.80x prefill speedup across the swept context range; gen
within 2.5% of baseline (run-to-run variance).

## ns4 local rename

The ns4 standalone renamed its local patch surface inside the patched files
(`ds4_*` identifiers, `DS4_*` env switches and the `ds4:` log prefix became
`ns4_*`/`NS4_*`/`ns4:`). Upstream identifiers that merely contain the letters
"ds4" -- `MMQ_Q8_1_DS_LAYOUT_DS4` and the `block_q8_1_mmq` member `ds4[4]`
("4 delta-sum pairs") -- are upstream vocabulary and were left untouched, as
were the verbatim files.
