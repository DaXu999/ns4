# Metal-only features and their CUDA porting status

Removing the Metal backend dropped several features that were gated to
`DS4_BACKEND_METAL` or to `__APPLE__` and therefore never ran on CUDA. This
file records what they did, why they were Metal-only, and what porting each one
would actually cost. Ordered by value-per-effort.

All Metal sources are recoverable from commit `2e18805` (the last commit before
the backend was removed): `ds4_metal.m`, `metal/*.metal`, `ds4_tp.c`,
`ds4_tp.h`, and the `#if defined(__APPLE__)` regions of `ds4.c` and
`tests/ds4_test.c`.

---

## 1. SSD-streaming automatic cache budget — **best port candidate**

**What it did.** With `--ssd-streaming` and no explicit cache size, the engine
sized the routed-expert cache itself: take 80% of the backend's recommended
working set, subtract the non-routed weights, reserve headroom for the two full
routed layers used by overlapped streaming prefill, then convert the remainder
into a dynamic expert-slot count for the current GGUF.

**Why it was Metal-only.** One boolean:

```c
static bool ds4_backend_supports_streaming_auto_cache(ds4_backend backend) {
    if (backend == DS4_BACKEND_METAL) return true;
    return false;
}
```

`ds4_engine_configure_streaming_auto_cache()` returned early for every other
backend, so the ~90 lines of budget math never executed on CUDA.

**Why the port is cheap.** The one backend primitive it needs,
`ds4_gpu_recommended_working_set_size()`, **is already implemented for CUDA**
(`ds4_cuda.cu:28443`) and returns aggregate VRAM across devices:

```c
if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return 0;
if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
return (uint64_t)total_b * (uint64_t)n;
```

So the port is: restore `ds4_engine_configure_streaming_auto_cache` from
`2e18805`, gate it on `ds4_backend_uses_graph()` instead of the deleted
predicate, and re-add the call in `ds4_engine_open_internal`.

**Effort:** small. **Value:** high — today `--ssd-streaming` on CUDA silently
requires the user to pass `--ssd-streaming-cache-experts N|NGB`, and getting it
wrong either OOMs or wastes VRAM.

**Caveat before trusting it:** `cudaMemGetInfo` misreports under WSL2, so
validate the resulting budget against real allocation on the target host rather
than believing the first number it prints.

---

## 2. Metal-only test coverage

`tests/ds4_test.c` carried a 2560-line `#if defined(__APPLE__)` region — 38% of
the file — that no Linux build ever compiled. It backed four subtests still
advertised in `--help`:

| subtest | what it covered |
|---|---|
| `--metal-kernels` | isolated kernel numeric regressions (Q8_0 output NR4 exactness, etc.) |
| `--metal-tensor-equivalence` | fast vs quality prompt-logit and greedy-token equivalence |
| `--metal-short-prefill` | ratio-4 short prefill regression |
| `--metal-ssd-streaming-cache-pressure` | layer-batched decode under cache pressure (issue #384) |

It also held a float-pattern table worth keeping for any conversion test:
signed zeros, infinities, quiet/signalling NaNs, F32 subnormals, the minimum
F16 subnormal boundary, the subnormal/normal boundary, round-to-nearest ties,
and the overflow boundary.

CUDA's existing equivalents are `test_mxfp4_cuda`, `test_cuda_session_batch` and
`test_cuda_mixed_batch` — none of which cover fast/quality tensor equivalence or
per-kernel numeric exactness.

**Effort:** medium. **Value:** high — this is the largest coverage gap the
removal opened.

### The kernel group was already CUDA-compiled but never ran there

Separately from the Apple-gated block, `test_gpu_kernel_group` (formerly
`--metal-kernels`) sat *outside* `#if defined(__APPLE__)`, so it was linked into
every CUDA `ds4_test` — yet it had never passed on CUDA. Two causes, both fixed
or recorded here:

1. It allocates device tensors directly instead of going through
   `test_get_engine()`, and nothing called `ds4_gpu_init()`. Metal allocated
   lazily; CUDA returns NULL from `ds4_gpu_tensor_alloc()` until the device is
   up, so all 20 assertions failed immediately. **Fixed** — the group now calls
   `ds4_gpu_init()` and skips cleanly if no GPU is present. Five of seven cases
   pass on CUDA. All five drive real CUDA kernels on the live DeepSeek path
   (`matmul_f16`, `matmul_q8_0`, `matmul_q8_0_pair`, `store_raw_kv_batch`,
   `pack_slot_rows_f32`) against an inline CPU reference. They add no detection
   power over `--logprob-vectors`, which already exercises the same kernels
   end to end; their value is that they run in 0.55 s with no GGUF, so a kernel
   change can be localized without loading the 80 GiB model.
2. `test_gpu_q8_0_prefill_matmul` passes a plain `malloc` buffer to
   `ds4_gpu_set_model_map()`. Metal addressed it directly through unified
   memory; CUDA registers nothing (`registered 0.00 GiB model mapping`) and the
   quantize launch fails with `invalid argument`. **Not ported** — it now skips
   with an explicit reason. Porting it means staging the synthetic weights into
   a device-visible mapping first.

Why it went unnoticed: `--all` runs slowest-first, so `--long-context` aborts on
a missing `./ds4flash.gguf` before the kernel group is reached, and every
`ds4_test` invocation in QA_BEFORE_RELEASES.md is a targeted subtest with an
explicit `DS4_TEST_MODEL=` path. The old name (`--metal-kernels`, "isolated
Metal kernel numeric regressions") also made any failure on a CUDA host look
like expected noise.

---

## 3. PRO Q4 expert-table preload

**What it did.** `ds4_engine_preload_pro_q4_expert_tables()` preloaded routed
expert dequant tables at startup for the PRO Q4 model when weights were
resident. Gated on `backend == DS4_BACKEND_METAL && DS4_MODEL_VARIANT ==
DS4_VARIANT_PRO && !ssd_streaming`, with `DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_PRELOAD`
as the escape hatch.

**Why the port is not just a gate flip.** Unlike (1), the CUDA side is a
placeholder — `ds4_gpu_preload_q4_expert_tables` exists at `ds4_cuda.cu:28770`
but ignores every argument and returns success:

```c
extern "C" int ds4_gpu_preload_q4_expert_tables(...) {
    (void)model_map; (void)model_size; ...
    return 1;
}
```

A port needs a real CUDA implementation first.

**Effort:** medium. **Value:** PRO-Q4-only, and only for the first tokens after
load.

---

## 4. Two-machine tensor parallelism (`--tensor-parallel`)

**What it did.** Ran one logical session across two machines working on the
*same* token simultaneously — latency reduction, not capacity. Each rank kept a
contiguous half of the routed experts resident; dense, attention, shared-expert,
embedding and output weights were replicated.

The splits it implemented, all in backend-agnostic graph code:

- attention head split (`tp_heads`, `tp_head0`)
- attention-output group slice with a k-window of the expand projection
- shared-expert intermediate-lane slice plus a down-projection k-slice
- routed-expert ownership halves, with the shared partial folded into the
  routed sum so no separate combine dispatch was needed
- vocab-split logits head — each rank computed its half of the head rows
- prefill row split, with optional sub-chunk gate pipelining at `n_tokens % 256 == 0`

Transport was RDMA verbs (over Thunderbolt) with a TCP fallback, one bulk
partial-sum exchange per layer per stage.

**Why it was Metal-only.** Two hard gates plus a missing backend:
`ds4_tp_validate_engine_options()` rejected `opt->backend != DS4_BACKEND_METAL`;
`ds4_engine_tp_bind()` was compiled to an unconditional failure under
`#if defined(DS4_NO_GPU) || !defined(__APPLE__)`; and the whole
`ds4_gpu_tp_*` kernel API (`tp_init`, `tp_gate_encode`, `tp_batch_gate_encode`,
`tp_set_batch_exchange`, `tp_set_big_exchange`, `tp_set_session_batch_mode`,
`tp_suspend_expert_sharding`, `tp_failed`, `tp_shutdown`) existed **only** in
`ds4_metal.m`. `ds4_cuda.cu` has zero references to `tp_world`, `tp_rank`,
`tp_out`, `tp_batch_out` or `tp_zero`.

**What CUDA has instead.** `--cuda-tensor-parallel` — single-host, multi-GPU,
driven by the separate `cuda_tp_decode` / `cuda_tp_moe` / `cuda_tp_ep` fields.
It is unrelated code and is unaffected by this removal.

**Effort:** large. The split math is the valuable part and is backend-agnostic,
so a port is mostly transport work: replace the verbs gate exchange with NCCL
(or CUDA-aware MPI/verbs) and re-implement the per-layer gate ordering
guarantees. **Value:** only if multi-node CUDA is a goal; for one host,
`--cuda-tensor-parallel` already covers it.

---

## 5. Re-pointed at CUDA in this change — **unvalidated, needs testing**

Two features were gated to Metal by policy rather than by implementation. Their
bodies use the shared graph API, so rather than delete them I re-gated them on
`ds4_backend_uses_graph()`. Neither has been exercised on CUDA:

- **imatrix collection** (`--imatrix-out` / `--imatrix-dataset`). The old error
  was literally `"imatrix collection currently requires --metal"`. Its body uses
  `metal_graph_prefill_cap_for_prompt`, `metal_graph_raw_cap_for_context` and a
  plain `ds4_gpu_graph`, all shared. Note `--imatrix-out` also used to force
  `backend = DS4_BACKEND_METAL` in the CLI; that override is gone, so it now
  runs on the default backend.
- **expert profile / hotlist** (`DS4_EXPERT_PROFILE`, `DS4_EXPERT_HOTLIST`,
  `--expert-profile`). The old message said "Metal-only *for now*", and
  `ds4_expert_profile_init` is plain C in `ds4.c`, not Metal code.

Validate both before relying on them.

---

## 6. Superseded — do not port

`ds4_sessions_eval_batch_metal` and `ds4_sessions_eval_batch_with_prefill_metal`
(plus their `_supported` predicates). CUDA has its own
`ds4_sessions_eval_batch_cuda` / `ds4_sessions_eval_batch_with_prefill_cuda`,
and the dispatcher already returned through the CUDA path before reaching the
Metal check.

Likewise these Metal-internal kernels had no shared-code callers outside
`__APPLE__` blocks, so there is no feature behind them to port:
`ds4_gpu_matmul_q8_0_decode_mpp_tensor`, `ds4_gpu_matmul_q8_0_rows_scalar_tensor`,
`ds4_gpu_sort_i32_rows_asc_tensor`, `ds4_gpu_flash_kv_stage_f16_tensor`,
`ds4_gpu_hc_rms_scale_project_f16_tensor`, `ds4_gpu_hc_weighted_sum_norm_tensor`,
`ds4_gpu_stream_expert_cache_seed_experts_gpu_copy`,
`ds4_gpu_stream_expert_cache_note_service_thread`,
`ds4_gpu_release_zero_prefix_prefill_mask_cache`.
