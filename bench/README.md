# Benchmarking

Prefill and generation speed at context frontiers on CUDA hardware. The
primary target is the RTX PRO 6000 Blackwell class; DGX Spark / GB10 is
experimental and has not been hardware-tested in this build.

Run `ns4-bench` as:

```
./ns4-bench \
  -m ns4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

## Reproducing The Top-Level README Speed Table

The speed numbers in the top-level README are for the tuned serving
configuration on an RTX PRO 6000 (`scripts/run-cuda-436k-server.sh`). The
default sweep above does not reproduce them: the table needs the same
`NS4_CUDA_*` environment stack that launcher exports, the served
`--ctx-alloc 436000` (so context buffers and free VRAM match serving; prefill
speed itself is flat across allocations), and `--prefill-chunk 2048`.

`scripts/ctx-sweep.sh` runs the whole protocol: the tuned environment
stack, a 4096 -> 262144 frontier ladder (x2 steps), four runs with the
first discarded as warm-up, and a per-frontier median summary of
`prefill_tps` and `gen_steady_tps`. The table quotes its medians at the
8192..262144 frontiers (the 4096 frontier only absorbs per-run warm-up,
so every quoted prefill interval is a clean non-first one). Every value
is env-overridable and extra `ns4-bench` flags pass through; the script
header documents the details.

```sh
./scripts/ctx-sweep.sh
```

The current reference results are `bench/rtx_pro_6000.csv` (per-frontier
medians of that protocol's measured runs at `4ef8515`) with the chart in
`bench/rtx_pro_6000_ts.svg`. Prefill throughput is sensitive to host and
clock state, so treat cross-machine or cross-boot prefill deltas of a few
percent as environment, not code, unless measured back-to-back.

If your CUDA hardware is not already covered, provide a PR including your
numbers. Name the CSV after the hardware, for example `rtx_pro_6000.csv`, so
it is clear what produced it.

To generate an SVG graph from a CSV file:

```
python3 bench/plot_speed.py bench/rtx_pro_6000.csv --title "RTX PRO 6000 t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `bench/rtx_pro_6000_ts.svg`.
It plots `gen_tps`, the whole-run decode average (first token included), where
the README table quotes `gen_steady_tps`.

`decode-profiling/` contains the standalone kernel-level harnesses used to
produce `DECODE_LAUNCH_BUDGET.md`; see its README before trusting or
extending those measurements.
