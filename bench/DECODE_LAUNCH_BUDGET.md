# Where a decode token actually goes (RTX PRO 6000 Blackwell, WSL2)

DeepSeek-V4-Flash IQ2XXS, 43 layers, single-token decode. Prefill is untouched
by everything here.

Commit hashes cited below (`8890139`, `a71fc64`, `acf4a5e`, `4a5a6e7`,
`12072a0`) refer to the ancestor ds4 history and are not reachable from this
repository; the measurements and conclusions carry over unchanged.

Originally measured 2026-08-06 at `8890139`; sections 1, 3 and 6 were
re-measured 2026-08-12 at `a71fc64` and several of the original conclusions
turned out to be wrong. Where that happened the old claim is kept and marked,
because the reason it was wrong is usually the useful part.

## 1. The byte budget, and the roofline that is not 1.8 TB/s

Per token, from the GGUF tensor table:

| group                                                     | bytes/token |
|-----------------------------------------------------------|------------:|
| attention projections `q_a`,`kv`,`q_b`,`output_a`,`output_b` (Q8_0, 43x) | 4887 MB |
| shared expert `gate`/`up`/`down` (Q8_0, 43x)              |     1149 MB |
| 6 routed experts of 256, `gate`+`up` (IQ2_XXS, 43x)       |     1116 MB |
| 6 routed experts, `down` (Q2_K, 43x)                      |      710 MB |
| output head `output.weight` (Q8_0, **once per token**)    |      563 MB |
| attention compressors (F16; 21 ratio-4 + 20 ratio-128)    |      520 MB |
| router `ffn_gate_inp` (F16, 43x)                          |       90 MB |
| indexer compressors (F16, 21 layers)                      |       88 MB |
| hyper-connection `hc_*_fn` (F16, 43x)                     |       68 MB |
| **total**                                                 | **9191 MB** |

**Corrected 2026-08-12.** This table used to read 195 MB/layer x 43 = 8.4 GB,
which was wrong twice. It took layer 3 as representative, but layer 3 is a
ratio-128 layer and the compressors alternate -- the 21 ratio-4 layers carry
`attn_compressor_kv/gate` at (4096,1024) rather than (4096,512), and 21 layers
carry an indexer compressor as well. And it omitted the output head entirely,
which is 563 MB read on every token, 6.1% of the budget on its own.

The denominator was wrong too. `bw_probe.cu` measures **1642 GB/s** on an 8 GB
stream, not the ~1.8 TB/s spec figure, and DRAM saturates at 1-2 blocks per SM
(188 blocks = 94.2% of peak, 376 = 99.6%, 752-3008 add nothing; below that it
falls off a cliff, 94 blocks = 63%, 32 = 22%).

So the batch-1 arithmetic ceiling is **9.191 GB / 1642 GB/s = 5.60 ms =
~179 t/s**, not the ~196 the old figure implied. Decode measures 62.8 t/s at
ctx 32768 (15.9 ms), i.e. **577 GB/s = 35% of achievable**.

The shape is worth keeping in mind: *attention projections outweigh the routed
experts 2.7:1*. Decode here is not "a MoE problem".

## 2. Launch overhead

An **empty** kernel, measured on the GPU timeline in a back-to-back loop:

| launch shape        | us/launch |
|---------------------|----------:|
| 1 block x 32        |      6.01 |
| 80 blocks x 128     |      5.94 |
| 216 blocks x 256    |      5.34 |
| CPU-side submit only|      4.13 |

Decode issues **1662 launches per token** (counted with an `LD_PRELOAD`
interposer over `__cudaLaunchKernel`; ~38.6 kernels x 43 layers). At ~5.5 us
eager that is most of a token, which is why so many of the per-layer kernels
are tiny elementwise work on one token's 4096-wide activation.

Where the kernel time goes, by category (share of profiled kernel time, graphs
off, ctx 32768):

| category                        | ms/token | % kernel time | launches/token |
|---------------------------------|---------:|--------------:|---------------:|
| dense / attention / shared matmuls |   8.04 |         30.1% |          320.9 |
| format conversion / quantize    |     5.31 |         19.8% |          565.0 |
| DSA lightning indexer           |     3.46 |         13.0% |           89.0 |
| routed MoE experts              |     2.54 |          9.5% |           86.0 |
| hyper-connections (Sinkhorn)    |     2.19 |          8.2% |           88.9 |
| RMS norms                       |     2.02 |          7.5% |          184.0 |
| attention (FA2 + combine)       |     1.15 |          4.3% |           85.9 |
| compressor store/update         |     0.62 |          2.3% |           82.2 |
| rope                            |     0.62 |          2.3% |           74.1 |
| router select                   |     0.58 |          2.2% |           43.0 |
| swiglu                          |     0.22 |          0.8% |           43.0 |

**565 launches per token do nothing but change number format.** That is the
largest non-matmul category and the standing fusion lever: the matmuls already
consume pre-quantised activations, so the fix is to have the *producer* of an
f32 activation emit `xq`/`xscale` directly. Untried.

## 3. CUDA graphs remove ~8x of the launch tax

Same kernels, individually launched vs. replayed as one captured graph:

| kernels in chain | eager     | graph    | per-kernel        |
|------------------|----------:|---------:|-------------------|
| 4                |  22.8 us  |   6.3 us | 5.70 -> 1.57 us   |
| 8                |  44.0 us  |   9.1 us | 5.49 -> 1.13 us   |
| 37 (one layer)   | 251.3 us  |  35.2 us | 6.79 -> 0.95 us   |
| 128              | 871.8 us  | 100.7 us | 6.81 -> 0.79 us   |
| 1600 (one token) |10653.4 us | 1290.4 us| 6.66 -> 0.81 us   |

Islands used to measure *slower* than graphs-off. The cause was the **replay
stream**: islands replayed on `g_decode_graph_stream`, created by
`cudaStreamCreate()`, which returns a *blocking* stream, while everything an
island does not cover stays on the legacy NULL stream -- and the legacy stream
implicitly synchronises with blocking streams both ways, forcing ~86 pipeline
drains per token. `cudaGraphLaunch` may target the legacy stream even though
capture may not, so replaying there keeps decode on one stream. Fixed in
`acf4a5e`; `NS4_CUDA_DECODE_GRAPH_OWN_STREAM=1` restores the old behaviour.

| ctx  | graphs ON, legacy stream | graphs ON, own stream | graphs OFF |
|------|-------------------------:|----------------------:|-----------:|
| 2048 |               **61.70**  |                49.59  |     53.11  |
| 4096 |               **57.29**  |                46.00  |     49.12  |
| 8192 |               **54.93**  |                47.34  |     47.64  |

**Superseded (2026-08-07).** This section used to end by calling wider island
coverage "the next lever, worth more than any single kernel". That was
implemented and measured, and it is only true below sequence position 4096:
**+7.7% at ctx 2048 but -11 to -13% at 4096 and above**. Past the full-attention
cap the DSA indexer keeps the GPU busy enough to hide the launches, and capture
is CPU work that cannot overlap with them. See `graph_update.cu` for per-node
costs of all four capture strategies.

## 4. Attention

`attention_flash_decode_kernel` used to be gated behind a `topk != NULL`
caller, so for this model's decode the live path was
`attention_decode_score_split_*`, which materialises the whole
`[n_head x n_score]` score matrix to global memory and reads KV twice.

`src/cuda/flash2_decode.cuh` (FA2: heads on the MMA M axis, fp16 `m16n8k16` tensor
cores, online softmax, no score materialisation) now serves **both** decode
callers as of `12072a0`, and `attention_decode_score_split_*` no longer appears
in decode at all. 25.0 us -> 12.5 us of kernel + 7.1 us of combine per layer.
Attention is only ~4.3% of a token, so this was worth ~1% end to end.

Worth keeping from the original sweep: the shipped kernel was
**parallelism-starved, not bandwidth-starved** -- *more* heads per block was
much worse (WARPS=16 -> 37.5 us, WARPS=32 -> 98.8 us).

## 5. Achieved bandwidth per weight-reading kernel

kprof, graphs off, ctx 32768, 468k allocation. Bytes are exact from the tensor
table; `hc_expand` is 2 calls/layer over two different tensors that share a
grid, so its row is the per-layer aggregate.

| kernel | grid/blk | us/call | reads | GB/s | % of 1642 |
|---|---|---|---|---|---|
| `matmul_q8_0_preq_warp8` (output head) | 16160x1/256 | 352.5 | 562.63 MB | **1596** | **97%** |
| `matmul_q8_0_preq_warp8` (`attn_q_b`) | 4096x1/256 | 31.8 | 35.65 MB | 1121 | 68% |
| `matmul_q8_0_hc_expand_preq_warp8` | 512x1/256 | 20.8 | 22.28 MB | 1071 | 65% |
| `grouped_q8_0_a_preq_warp8` (`attn_output_a`) | 1024x1/256 | 37.2 | 35.65 MB | 958 | 58% |
| `matmul_q8_0_pair_preq_warp8` (shared gate+up) | 256x1/256 | 18.7 | 17.83 MB | 952 | 58% |
| `moe_gate_up_mid_decode_aligned_w32` | 256x6/256 | 32.8 | 25.95 MB | 791 | 48% |
| `moe_down_sum6_aligned_qwarp32` | 128x1/256 | 26.2 | 16.52 MB | 631 | 38% |
| `matmul_q8_0_pair_preq_warp8` (`q_a`+`kv`) | 128x1/256 | 13.6 | 6.68 MB | 492 | 30% |
| `matmul_f16_pair_ordered_chunks` (compressor, ratio-4) | 1024x1/**32** | 34.7 | 16.78 MB | 484 | 29% |
| `matmul_f16_pair_ordered_chunks` (compressor, ratio-128) | 512x1/**32** | 22.0 | 8.39 MB | 381 | 23% |
| `matmul_f16_pair_ordered_chunks` (indexer compressor) | 256x1/**32** | 18.2 | 4.19 MB | 230 | 14% |

**The output head at 97% is the calibration point.** A single Q8_0 GEMV reading
563 MB hits 1596 GB/s, which proves both the 1642 figure and that
`matmul_q8_0_preq_warp8` is essentially optimal. Every shortfall elsewhere is
about read size, grid width or block width -- not kernel quality.

**kprof is blind to cuBLAS.** It hooks `cuLaunchKernel`, but CUDA 13's cuBLAS
dispatches through `cuLaunchKernelEx`, so its kernels land in the idle-gap
column instead. The live instance is the router: `ns4_gpu_matmul_f16_tensor`
takes the `cublas_one_token` branch for `ffn_gate_inp`, so 90 MB/token never
appears above.

## 6. 1642 GB/s is the wrong denominator for a small read

`f16pair_bench.cu` includes a `ceiling` arm: the same bytes from the same
rotating pool, pure coalesced grid-stride read, no arithmetic at all. That is
the most any kernel of that shape could reach.

| read size | ceiling | % of the 8 GB-stream 1642 |
|---|---|---|
| 16.78 MB | 1433-1460 GB/s | 87-89% |
| 8.39 MB  | 1291-1314 GB/s | 79-80% |
| 4.19 MB  | 1065-1117 GB/s | 65-68% |

A 4.19 MB read is 22 KB per SM; it cannot amortise DRAM ramp-up the way an 8 GB
stream does. **So no version of these kernels can approach 90% of 1642**, and
the "% of 1642" column in section 5 understates the small kernels. Judge them
against the size-appropriate ceiling.

## 7. What moved on 2026-08-12

Three kernel changes, all bit-identical (teacher-forced NLL unchanged in every
printed digit on two corpora: 4045.607689345 and 2121.782793653).

| kernel | before | after | factor | commit |
|---|---|---|---|---|
| `moe_down_sum6_aligned` | 634 GB/s | **887** | 1.40x | `4a5a6e7` |
| `matmul_f16_pair` out_dim 1024 | 554 | **1168** | 2.11x | `a71fc64` |
| `matmul_f16_pair` out_dim 512 | 447 | **879** | 1.97x | `a71fc64` |
| `matmul_f16_pair` out_dim 256 | 293 | **534** | 1.82x | `a71fc64` |
| `matmul_q8_0_pair` max_out 2048 | 660 | **767** | 1.16x | `a71fc64` |
| `matmul_q8_0_pair` max_out 1024 | 453 | **502** | 1.11x | `a71fc64` |

Three distinct causes, and none of them was the arithmetic:

* **Lanes per row.** The MoE down leg gave a row 8 lanes of a 256-thread block
  where its gate/up sibling had 32. Widening it also took the grid from 128 to
  512 blocks, crossing the ~188-block saturation point, so one change lifted
  both caps.
* **Bytes per request.** `matmul_f16_pair_ordered_chunks` loaded one `__half`
  per thread per instruction -- 2 bytes -- and the algorithm carries only
  `out_dim` warps in total (256 at the indexer shape, 1.4 per SM), so there was
  nothing else in flight to hide DRAM latency with. Reading each chunk as
  `uint4` is 8x the bytes per request and moves no float operation.
* **Grid width.** `matmul_q8_0_pair_preq_warp8` already gave each row a warp,
  but 8 rows per block put `q_a`+`kv` on 128 blocks, so 60 of 188 SMs got
  nothing. Rows per block is now chosen from `max_out`.

End to end: **+7.0%** decode (59.3 -> 62.8 t/s at ctx 32768), measured with all
three arms against all three off in one binary, interleaved, replicates
agreeing to 0.1%. That matches the +7.1% predicted from summing the kernel time
saved, i.e. the GPU-work savings converted essentially in full -- which is not a
given here. The hc Sinkhorn rewrite was 1.87x standalone and converted to
+0.85%, because those kernels are submission-bound rather than execution-bound.

**What is left is the exactness contract, not the loads.**
`matmul_f16_pair_wide_kernel` (opt-in, `NS4_CUDA_F16_PAIR_WIDE=128|256|512`)
gives a row NT threads instead of 32 and reaches 95-100% of the achievable
ceiling at all three shapes, against 50-82% for the bit-identical form. It is
off by default on cost, not accuracy: worth 1.17% of a token, inside this box's
run-to-run drift, in exchange for giving up bit-exactness. Its NLL moves +-0.5%
but the sign flips across corpora (-0.348%/+0.032%/-0.302% on prose,
+0.362%/+0.531%/+0.287% on source, for NT=128/256/512), which is this repo's
signature for summation-order noise rather than a quality change -- the same
test that cleared FA2 at +0.129%/-0.087%. An 1800-token sample and NT=256 alone
read as a clean 0.4% regression and were not enough to see that; use ~8k tokens
and all three widths.

## Reproducing

Harnesses under `bench/decode-profiling/`, none built by the Makefile:

* `kprof.c` -- `LD_PRELOAD` launch interposer, per-kernel GPU time. Read its
  four traps in the README before trusting a report.
* `bw_probe.cu` -- achievable bandwidth vs block count, and the strided penalty.
* `launch_floor.cu`, `graph_floor.cu`, `graph_update.cu` -- the launch floor,
  what a graph buys, and what each capture strategy costs.
* `f16pair_bench.cu` -- the compressor projection: shipped vs vec4 vs wide vs a
  pure-read ceiling, `memcmp`-checked for bit-identity on pseudorandom weights.
* `iq2_dequant_bench.cu` -- what the IQ2_XXS gate/up kernel is bound by, and
  every LUT scheme that lost.
* `occ_bench.cu`, `hc_occ_bench.cu` -- why kprof over-reports 1-block kernels.
* `fd_bench.cu`, `fd_sweep.cu`, `fd2_bench.cu`, `fd_combine_bench.cu` --
  attention.
