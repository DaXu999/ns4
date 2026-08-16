# Decode profiling harnesses

Standalone tools used to produce `speed-bench/DECODE_LAUNCH_BUDGET.md`. None of these
are built by the Makefile; each is a single translation unit.

There is no `nsys`/`ncu` on the WSL2 box these were written for, which is why
`kprof.c` exists.

## kprof.c - per-kernel GPU time without a profiler

`LD_PRELOAD` interposer over `cudaLaunchKernel`, `__cudaLaunchKernel`
(CUDA 13's `<<<>>>` lowering; the older name alone catches nothing) and
`cuLaunchKernel`. Records a pooled event pair per launch and aggregates GPU
time per kernel symbol, resolving names with `cuKernelGetName`.

    gcc -O2 -fPIC -shared -o kprof.so kprof.c \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
        -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -ldl

    KPROF_SKIP=1500 KPROF_MAX=32000 KPROF_OUT=decode.txt \
      LD_PRELOAD=./kprof.so ./ds4-bench --cuda -m MODEL ... --gen-tokens 40

`KPROF_SKIP` steps over prefill, `KPROF_MAX` caps the pool, `KPROF_AFTER_MS`
delays arming, `KPROF_ROLL` recycles the pool so the surviving report covers
the *last* window (decode). The report is written when the pool fills, not at
exit: the library destructor runs after the CUDA context is torn down and
every `cudaEventElapsedTime` there fails with "driver shutting down".

Identify decode kernels by launch count - one per layer per token is
`43 * n_tokens`. Recording inflates decode by ~10-30%, so read the ranking
rather than the absolute microseconds.

## launch_floor.cu - what an empty kernel costs

Establishes the ~5.5 us/launch floor that dominates single-token decode on
this box. Run it before optimising any decode kernel: if the kernel is faster
than the floor, the kernel is not the problem.

## graph_floor.cu - what a CUDA graph buys

Same kernels launched individually vs. replayed as one captured graph, for
chain lengths from 4 to 1600 (a decode token). Shows the ~8x amortisation.

## fd_bench.cu / fd_sweep.cu - the shipped flash decode kernel

Both `#include "fd_baseline.inc"`, which is extracted verbatim from
`ds4_cuda.cu` so the comparison stays honest:

    sed -n '9635,9640p' ds4_cuda.cu  > fd_baseline.inc
    sed -n '9666,9713p' ds4_cuda.cu >> fd_baseline.inc
    sed -n '9715,9888p' ds4_cuda.cu >> fd_baseline.inc

(line numbers as of `8890139`; re-extract if the kernel moves). `fd_sweep`
walks the kernel's `<WARPS, TILE>` template parameters and the rows-per-split
launch parameter. `TILE=32` exceeds the 48 KB static shared-memory cap.

## fd2_bench.cu - the FA2 kernel

Validates `cuda/flash2_decode.cuh` against the shipped kernel (same inputs,
same combine semantics) and reports max relative error, relative L2, and a
main/combine time split. `--split=N` sweeps the KV-row split; `--pos=N` moves
the sequence position, which sets how many rows the geometry selects.

Shapes are hardcoded to DeepSeek-V4-Flash: 64 heads, 512-wide latent,
`raw_cap` 2304, window 128, top-k 512, ratio 4.

## fd_combine_bench.cu - the combine kernel, and how to size a decode win

Times `attention_flash_decode_combine_kernel` against the shape it replaced,
and `memcmp`s their outputs to hold the "bit-identical" claim honest. The
production kernel is `#include`d from `fd_combine.inc` (extracted verbatim from
`ds4_cuda.cu`, recipe in the file header); the pre-rewrite control is frozen in
the bench, since it is no longer in `ds4_cuda.cu` to extract.

    nvcc -O3 -arch=sm_120 -o fd_combine_bench fd_combine_bench.cu
    ./fd_combine_bench              # L2-warm, as in real decode
    ./fd_combine_bench --bufs=64    # 256 MB rotation, partials evicted

Each row is timed three ways, and the spread between them is the point:

* **graph-amortised** - `reps` launches captured as one graph, so the ~6 us
  WDDM submission floor costs 0.7 us instead. This is kernel work.
* **eager** - the same launches with no graph, i.e. how decode actually issues
  this. Only *part* of a GPU-time saving survives here.
* **empty** - the same grid doing nothing, which is the floor itself.

At the live `n_split = 27` the rewrite is 1.7x graph-amortised (5.2 -> 3.0 us)
but only 1.5 us/call eager (9.6 -> 8.1), because submission and execution
partly overlap. Over 43 layers that is ~0.3-0.45% of a token -- an order of
magnitude under this box's run-to-run drift. Read a kernel win off the eager
column before assuming ds4-bench could ever show it.

`--nscore=N` picks the sweep points; the default set ends at 640 (raw window
128 + 512 top-k slots), which is every position past ctx 2048.

## iq2_dequant_bench.cu - what the IQ2_XXS decode kernel is actually bound by

Runs `moe_gate_up_mid_decode_lut_qwarp32_kernel`'s body at its live decode
shape (mid 2048, `xq_blocks` 16, 6 experts, grid 64x6 x 256) with only the
dequant/load path swapped between variants, so every arm reads the same bytes
in the same order and must be bit-identical - which is checked, not assumed.
Experts come from a rotating pool (`--pool=N`, default 160) so the weights
come from DRAM rather than the 128 MB L2.

    nvcc -O3 -arch=sm_120 -o iq2_dequant_bench iq2_dequant_bench.cu
    ./iq2_dequant_bench --pool=400

It answers two questions that look alike and are not.

**Can the (grid, sign) pair be precomputed?** No. The pair has 256*128 = 32768
values, so all eight signed int8s fit a 256 KiB table - but that is 2.5x the
100 KB shared-memory cap, so it has to be read through L1, and a
data-dependent 8-byte gather serialises into up to 32 cache-line wavefronts per
warp. It measures **0.22x**; the 128 KiB half-table (negation symmetry folded
out) is 0.20x, 32 KiB of split tables 0.87x (occupancy 6 -> 2 blocks/SM), and
the 8 KiB quad table 0.95x - the grid's 256 entries hold only 74 distinct
4-byte magnitude quads, so one table serves both halves. Nothing beat the
shipped arithmetic. Shared memory is a 32-bank gather engine and L1 is not;
growing a LUT out of shared memory trades a cheap bank conflict for an
expensive wavefront replay.

**Then what is it bound by?** Load width. The `base +N ALU/lookup` arms add N
identity rotations per lookup - bit-identical, so nothing else can have moved -
and the first ~4 ops per lookup are *free* (35.77 -> 35.41 us), which is what
load-stall slack looks like. `cuda_block_iq2_xxs` is `{uint16 d; uint16
qs[32];}`, so `qs` sits at offset 2 of a 66-byte block and block b's quants
land at `66b + 2`, which is 2 mod 4 for even b: the compiler cannot merge the
four `q2[0..3]` reads and the PTX shows `ld.global.b16`. Sourcing the same
kernel from the aligned-SoA artifact `ds4_repack_build_iq2_aligned` already
emits - `[__half dq[nblk]][pad 64B][uint2 qs[nblk*8]]`, byte-neutral - is
**1.36x with uint2 loads and 1.40x with uint4**, and only then does ALU begin
to bind (+4 ops then costs 11%).

Read absolute numbers here with care: this bench runs the kernel back-to-back
at a sustained 2827 MHz / 350 W and gets 34 us, while the same kernel under
live decode measures 78 us, where the GPU idles ~48% of the time on a bimodal
clock averaging 1760 MHz.

## topk_bench.cu - the DSA indexer's decode top-k

Times the shipped chunked-bitonic selection against a radix select at the live
decode shape (`n_comp` 8202, `top_k` 512, one token) and at the frontiers a
long context reaches. Baselines are `#include`d from `topk_baseline.inc`,
extracted verbatim from `ds4_cuda.cu` (recipe in the file header).

    nvcc -O3 -arch=sm_120 -o topk_bench topk_bench.cu
    ./topk_bench --n=16394          # ctx 65536
    ./topk_bench --dist=tied        # four distinct scores: the degenerate exit
    ./topk_bench --tokens=2048      # prefill width

`topk_score_better` is a strict total order on (score, index) and indices are
unique, so the top-k list *and its order* are unique: any exact algorithm is
bit-identical, and the bench `memcmp`s every variant against the shipped one
rather than assuming it. Four distributions matter because they select
different exits -- `tied` is the only one that reaches the equal-key path, and
`narrow` (one binade) is the worst case for a radix select's first pass.

## gguf_info.py

Dumps GGUF metadata and per-tensor byte sizes for one layer, which is where
the 195 MB/layer active-weight figure comes from.
