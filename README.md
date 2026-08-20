<p align="center">
  <img src="logo.svg" alt="Neutron Star logo" width="220">
</p>

# NEUTRON STAR (ns4)

**Lightning speed CUDA optimized inference server for DeepSeek V4 Flash,
based on the original ds4 project.**

ns4 is a small, self-contained native inference engine with one task: run
DeepSeek V4 Flash as fast as the hardware allows. It is deliberately narrow and is
not a general GGUF runner. Model loading, prompt rendering, tool calls, KV
state, and the HTTP server are built and tested together, and the repository
also includes the offline tools and data for GGUF generation, imatrix
collection, quality scoring, and speed measurement.

The only compute backend is **NVIDIA CUDA**. A CPU build exists strictly as a
reference/debug path.

Supported hardware:

* **RTX PRO 6000 Blackwell class GPUs** - the primary development and release
  target. A single 96 GB card serves the 2-bit Flash quant at up to 436k
  tokens of context (see [Speed](#speed)).

Experimental:

* **DGX Spark / GB10** - experimental support, not hardware tested.
* **Multi-GPU CUDA servers** - an even number of cards can run
  tensor/expert-parallel serving with batched sessions
  (see [Tensor Parallelism](#tensor-parallelism-across-cuda-gpus-experimental)).

## Core Features

* OpenAI-, Responses-, and Anthropic-compatible HTTP server with SSE
  streaming, thinking-mode handling, and DSML tool-call mapping.
* Disk KV cache with exact tool-call replay, so agent clients reuse long
  prefixes across sessions and server restarts.
* Session microbatching: multiple resident KV sessions with exact batched
  decode and mixed prefill/decode scheduling.
* SSD streaming of routed MoE experts for models larger than memory.
* Distributed pipeline parallelism over plain TCP to split a model across
  machines and pipeline long prefills (experimental).
* Single-host CUDA tensor/expert parallelism across an even number of GPUs
  (experimental).
* Directional activation steering, GPU power capping, benchmark and
  capability-evaluation tools.

## Speed

Primary target, single **RTX PRO 6000 Blackwell (96 GB)**, Flash q2 imatrix
0731, tuned serving configuration
([scripts/run-cuda-436k-server.sh](scripts/run-cuda-436k-server.sh)):

| Sequence position | Prefill (chunk 2048) | Decode |
| ---: | ---: | ---: |
| 8k | ~3,458 t/s | 82.2 t/s |
| 16k | ~3,425 t/s | 77.8 t/s |
| 65k | ~3,289 t/s | 76.8 t/s |
| 131k | ~3,149 t/s | 74.9 t/s |
| 262k | ~2,533 t/s | 71.0 t/s |

The launcher serves `--ctx 436000` from that one card for ~3 GiB of VRAM
headroom, not for speed: prefill holds full speed at every allocation through
536k, and its header documents the two limits that remain (the managed-KV flip
near 538k and the free-VRAM floor). Prefill decay with sequence position is
per-token attention and DSA indexer cost, not memory pressure. The tuned decode
levers it enables are deterministic but not bit-exact against the default
paths; unset them to fall back.

See [bench/README.md](bench/README.md) for how numbers are collected and how
to contribute results for other CUDA hardware.

## More Documentation

- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md): correctness and speed
  regression testing guide. **Read this before sending a pull request**.
- [docs/QA_BEFORE_RELEASES.md](docs/QA_BEFORE_RELEASES.md): the complete
  release test matrix.
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md): synopsis of the official
  DeepSeek-V4-Flash model card with the facts that matter for local inference.
- [tools/gguf-tools/README.md](tools/gguf-tools/README.md): offline GGUF
  generation, imatrix collection, quantization tooling, and quality checks.
- [tools/gguf-tools/imatrix/README.md](tools/gguf-tools/imatrix/README.md):
  how the routed-MoE imatrix is collected and used.
- [tools/gguf-tools/quality-testing/README.md](tools/gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash continuations.
- [tools/dir-steering/README.md](tools/dir-steering/README.md): directional
  steering data, vector generation, and usage.
- [bench/README.md](bench/README.md): benchmark commands, charts, and CSV
  generation.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation only works with the DeepSeek V4 GGUFs listed below. It is
not a general GGUF loader: arbitrary GGUF files will not have the tensor
layout, quantization mix, metadata, or optional support-model state expected
by the engine. The 2-bit quantizations provided here are verified to be high
quality: they behave well, work under coding agents, and call tools reliably.

The 2-bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of the model's bytes; the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download one main model. **Prefer the imatrix versions.**

```sh
./scripts/download_model.sh ds4f-q2      # 96/128 GB machines
./scripts/download_model.sh ds4f-q2-q4   # q2 with the last 6 expert layers at q4
./scripts/download_model.sh ds4f-q4      # >= 256 GB machines
./scripts/download_model.sh ds4f-mxfp4   # native MXFP4 experts, about 156 GB
```

The MXFP4 GGUF preserves DeepSeek's released MXFP4 routed-expert weights
rather than requantizing them.

The script downloads from `https://huggingface.co/antirez/deepseek-v4-gguf`,
stores files under `./gguf/`, resumes partial downloads with `curl -C -`, and
updates `./ns4flash.gguf` to point at the selected main model. Authentication
is optional for public downloads, but `--token TOKEN`, `HF_TOKEN`, or the
local Hugging Face token cache are used when present.

To regenerate GGUF files or collect a new imatrix, see
[tools/gguf-tools/README.md](tools/gguf-tools/README.md). Those tools are
meant for offline model-building work and can take a long time on the full
DeepSeek V4 Flash weights.

## Building

On Linux, plain `make` prints the available build targets instead of
selecting one implicitly:

```sh
make cuda-generic            # normal local CUDA build
make cuda CUDA_ARCH=sm_120   # explicit nvcc -arch (RTX PRO 6000 Blackwell)
make cuda CUDA_ARCH=native   # let nvcc pick the local device
make cuda-spark              # DGX Spark / GB10 (experimental)
make cpu                     # CPU-only reference/debug build
```

`./ns4flash.gguf` is the default model path used by all binaries. Pass `-m`
to select another supported GGUF from `./gguf/`. Run `./ns4 --help` and
`./ns4-server --help` for the full flag list.

CUDA builds accept `--gpu-vram N[,N,...]` and `--gpu-devices N[,N,...]` in
the CLI, server, benchmark, and eval. VRAM values are per-device GiB budgets;
`--gpu-vram auto` uses the free memory reported by CUDA. Placement reserves
graph and KV memory for the requested context and refuses to start if model
layers would spill to the CPU. Do not treat the CPU path as a production
target: it shares the same KV session and snapshot format as CUDA, but exists
for reference and debugging.

## CLI

One-shot prompt:

```sh
./ns4 -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
./ns4
ns4>
```

The interactive CLI is a real multi-turn chat. It keeps the rendered chat
transcript and the live graph KV checkpoint, so each turn extends the
previous conversation. Useful commands are `/help`, `/think`, `/think-max`,
`/nothink`, `/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the
current generation and returns to `ns4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./ns4-server --ctx 100000 --kv-disk-dir /tmp/ns4-kv --kv-disk-space-mb 8192
```

Use `--chdir /path/to/ns4` when launching `ns4-server` from another
directory, so relative runtime files resolve from the project tree. For the
tuned single-card high-context configuration, use
`./scripts/run-cuda-436k-server.sh`.

By default the server keeps one mutable backend/KV checkpoint in memory, so
stateless clients that resend a longer version of the same prompt reuse the
shared prefix instead of pre-filling from token zero.

`--batched-session N` preallocates `N` independent resident KV sessions.
Ready decode steps are evaluated together, while long prefills alternate in
bounded chunks so one request does not block every decoder. Requests beyond
`N` wait for a resident slot. If disk KV caching is enabled, an idle slot is
persisted before reuse and can be restored when that conversation returns; an
active request is never evicted. Choose `N` and `--ctx` so all resident KV
allocations fit in GPU memory. Decode batching is exact: when a native
batched kernel is unavailable, ns4 runs the affected rows in a fixed order
and returns the same full logits as separate session evaluations. Native
batched kernels are used on supported multi-GPU tensor-parallel layouts;
single-GPU hosts (including DGX Spark) use the ordered exact fallback. While
generation is active, prefill yields every 128 tokens by default
(`--mixed-prefill-quantum N` changes the interval).

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

The Flash model endpoint is a compatibility alias: it reports the model
currently loaded from the GGUF passed with `-m`.

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`min_p`, `seed`, `stream`, `stream_options.include_usage`, `tools`, and
`tool_choice`. Tool schemas are rendered into DeepSeek's DSML tool format,
and generated DSML tool calls are mapped back to OpenAI tool calls.

`/v1/responses` accepts OpenAI Responses-style `input`, `instructions`,
`tools`, `tool_choice`, `max_output_tokens`, `temperature`, `top_p`,
`stream`, and `reasoning`. It is the preferred endpoint for Codex CLI.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code style
clients. It accepts `system`, `messages`, `tools`, `tool_choice`,
`max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`,
and thinking controls. Tool uses are returned as Anthropic `tool_use` blocks.

Default sampled API generation uses `temperature=1`, `top_p=1`, and
`min_p=0.05`. In thinking mode ns4 applies those fixed sampling defaults to
any knob the request omits, matching DeepSeek's fixed-thinking API behavior,
but sampling parameters set explicitly in the request always win: a
`temperature=0` request is greedy through the whole reasoning phase, so
benchmark harnesses get deterministic thinking-mode output.

All three chat-style endpoints support SSE streaming. In thinking mode,
reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat streaming also streams tool calls as soon as the DSML
invocation is recognized; the Anthropic endpoint streams thinking and text
live, then emits structured `tool_use` blocks when the generated tool block
is complete; the Responses endpoint streams the full Responses event
lifecycle expected by Codex.

For browser JavaScript clients served from another origin, start the server
with `--cors` to emit `Access-Control-Allow-*` headers. This only changes
HTTP headers; use `--host 0.0.0.0` explicitly when remote machines should be
able to connect.

### Thinking Modes

DeepSeek V4 Flash has distinct non-thinking, thinking, and Think Max modes.
The server defaults to thinking mode. `reasoning_effort=max` requests Think
Max, but it is only applied when the context size is large enough for the
model card recommendation (384k); smaller contexts fall back to normal
thinking. OpenAI `reasoning_effort=xhigh` maps to normal thinking, not Think
Max. For direct replies, use `thinking: {"type":"disabled"}`, `think:false`,
or a non-thinking model alias such as `deepseek-chat`.

### Tool Call Handling And Canonicalization

DeepSeek V4 emits tool calls as
[DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md).
Agent clients do not send that same text back on the next request: they send
normalized OpenAI/Anthropic JSON tool-call objects. **If the server
re-rendered those objects slightly differently, the rendered byte prefix
would no longer match the live KV checkpoint** and the next turn would have
to be rebuilt.

The first line of defense is exact replay. Every tool call gets an
unguessable API tool ID, and the server remembers
`tool id -> exact sampled DSML block` in a bounded in-memory map backed by
radix trees. When the client later sends that tool ID back, the prompt
renderer uses the exact DSML bytes the model sampled, not a freshly formatted
approximation. This map can also be saved inside KV cache files, so exact
replay survives server restarts for cached histories.

**Canonicalization is only the backup path.** If the exact DSML block is
missing, or exact replay is disabled with
`--disable-exact-dsml-tool-replay`, the server renders a deterministic DSML
form from the JSON tool object, then reconciles the live checkpoint with the
prompt the next client request will render -- rewriting the checkpoint or
replaying a suffix from an older disk KV snapshot when needed.

During generation, the server treats DSML syntax differently from payload:
protocol structure (tags, parameter headers, JSON punctuation, closing
markers) is decoded greedily so the tool call stays parseable, while argument
payloads -- file contents, edit text, JSON string values -- use the request's
normal sampling settings.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`ns4-server` works with local coding agents that speak OpenAI-compatible chat
completions, the Responses API, or the Anthropic API. Start the server first,
and set the client context limit no higher than the server's `--ctx`.

Size `--ctx` to your VRAM: KV/indexer context buffers cost about 18 KiB per
token of allocated context with the 2-bit model -- about 7.6 GiB at the
launcher's 436k setting. Once the KV cache reaches 8 GiB (about 538k tokens at
prefill chunk 2048) session creation prints `using managed KV cache` and the
cache moves to managed memory: ~1% slower today, but the margin to real paging
is unmeasured, so stay below it (a full 1M-token allocation would need ~17
GiB). On a single 96 GB card the launcher's 436k keeps ~3 GiB of headroom and
`NS4_CTX=532000` is measured-good when the workload needs the room; its header
documents the free-VRAM floor in between. The `384000` output limit below
avoids client-side token caps, since the model can generate very long replies;
the server still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ns4": {
      "name": "ns4 (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ns4 local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ns4": {
      "description": "DeepSeek V4 Flash served by local ns4-server",
      "model": "ns4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ns4": {
      "name": "ns4 local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ns4 local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "ns4",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.ns4]
name = "NS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=ns4
```

For **Claude Code**, use the Anthropic-compatible endpoint with a wrapper:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ns4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ns4 local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Agent clients may send a large initial prompt, often around 25k tokens,
before they start doing useful work. Keep `--kv-disk-dir` enabled: after the
first expensive prefill, the disk KV cache lets later continuations or
restarted sessions reuse the saved prefix instead of processing the whole
prompt again.

### Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `ns4-server` first tries the cheap exact
token-prefix check, then falls back to comparing rendered prompt bytes with
decoded checkpoint bytes. The live in-memory checkpoint covers the current
session; the disk KV cache makes useful prefixes survive session switches and
server restarts. Memory cache handles the active session; disk cache is the
resume mechanism for different sessions.

Enable it with:

```sh
./ns4-server --kv-disk-dir /tmp/ns4-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The NS4 payload still stores the exact token IDs and graph state
for that prefix, so a rendered byte-prefix hit can reuse the checkpoint and
tokenize only the new suffix even when BPE boundaries shift. Files are
written with ordinary `read`/`write` I/O, not `mmap`, so restoring cache
entries does not add more VM mappings to a process that already maps the
model.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
NS4 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    NS4 session payload byte count
```

The optional tool-id map section (present when extension bit 0 is set) stores
unguessable API tool call IDs back to the exact DSML block the model sampled,
so restarted servers can render later client history byte-for-byte:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

The NS4 session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Then it stores the checkpoint token IDs, the `float32[vocab_size]` logits for
the next token (so a loaded snapshot can continue without one extra decode
step), per-layer row counts, and the live raw, compressed, and indexer KV
rows plus frontier tensors, written in logical position order. Distributed
coordinator sessions use the same payload: worker-owned layer tensors are
pulled during save and split back over the registered route during load. The
tensor payload is NS4-specific KV/session state, portable only across
compatible builds for this model layout.

The cache stores checkpoints at four moments: `cold` (after a long first
prompt reaches a stable prefix), `continued` (when prefill or generation
reaches the next absolute aligned frontier, roughly every 10k tokens with the
defaults), `evict` (before an unrelated request replaces the live in-memory
session), and `shutdown`. Cold saves trim a small token suffix and align down
to a prefill chunk boundary to avoid BPE boundary retokenization misses. The
knobs:

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

By default, checkpoints may be reused across the 2-bit and 4-bit
routed-expert variants if the rendered prefix matches; use
`--kv-cache-reject-different-quant` for strict same-quant reuse. The cache
directory is disposable: if behavior looks suspicious, stop the server and
remove it. Cache files contain the verbatim cached prompt, so `hexdump` is
enough to inspect what is stored.

## Running Models Larger Than RAM

The normal path makes the model resident in GPU-addressable memory; that is
the fastest path and should remain the default when the model fits. ns4 also
has an **SSD streaming** capacity mode: non-routed model weights stay
resident, while routed MoE experts are loaded from the GGUF file on demand
through the per-token selected-expert cache.

```sh
./ns4 -m ./ns4flash.gguf --ssd-streaming --ctx 32768 --nothink
```

Streaming needs no cache-size tuning on CUDA; there is no user-sized resident
expert cache on this backend. Non-routed weights, KV cache,
graph scratch, and activations still need resident memory. Long prefills can
still be fast; generation is more sensitive to cache misses because every new
token routes through experts again. A fast NVMe SSD makes misses tolerable.

## Distributed Inference With Pipeline Parallelism (Experimental)

Pipeline parallelism lets ns4 **run a model too large for one machine** by
splitting transformer layers across machines, and can also **speed up long
prefills** by processing different micro-batches at different layers like an
assembly line. Generation is strictly autoregressive -- each token pays at
least one cross-machine activation hop -- so distributed decode is slower
than a single local process. Use it for capacity and long prefills, not to
make decode faster. This mode is experimental in this build: it needs a
multi-machine setup that is not part of the verified hardware, and has not
been re-validated here.

The mental model:

1. Every machine has the GGUF, but each one maps only its slice: `--layers`
   controls which tensors are loaded (`20:output` skips the earlier layers).
2. Layer ranges are inclusive: `10:20` means layers 10..20; `N:output` means
   layer `N` through the final layer plus the output head.
3. One machine is the `coordinator`; the others are `workers` that connect to
   it and register their layer ranges.
4. Each worker keeps its slice of the KV cache.
5. Communication is worker-to-worker: activations flow
   `A -> B -> C -> back to A` without relaying through the coordinator.

Minimal two-host configuration:

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, layers 0..19.
./ns4 \
  -m ./ns4flash.gguf \
  --role coordinator \
  --layers 0:19 \
  --listen 10.0.0.1 1234

# Machine B: worker, connects to A and owns layers 20..output.
./ns4 \
  -m ./ns4flash.gguf \
  --role worker \
  --layers 20:output \
  --coordinator 10.0.0.1 1234
```

The final worker should normally own the output head (`--layers 20:output`)
so it can produce logits directly. On very slow or metered links,
`--layers 20:42` is also supported: the coordinator loads the output head and
computes logits locally, trading extra coordinator work for smaller per-token
replies.

The transport is plain TCP. Fast Ethernet or better is strongly recommended:
high link latency hurts generation directly, while lower bandwidth also pulls
down long-prefill speed. Prefill is pipelined -- the coordinator can run its
slice on chunk N+1 while a worker runs chunk N -- which is why multi-machine
prefill can beat a single machine on long prompts.

Use the coordinator exactly like normal `./ns4`: interactive chat, `/read`,
and ordinary generation go through the same high-level session API. The same
distributed options are wired into `ns4-eval` and `ns4-bench`; for
benchmarks, workers should already be running and `ns4-bench` waits until a
complete route is available.

Useful tuning and diagnostics:

- `--debug` on the coordinator prints route formation and per-hop telemetry:
  layer range, token span, local evaluation time, downstream wait time,
  socket send time, and byte counts. This is the profiling tool for deciding
  whether a split is balanced.
- `--dist-prefill-window N` bounds how many prefill chunks may be in flight;
  `--dist-prefill-chunk N` exists for experiments (the default -- the session
  prefill cap, normally 4096 -- is canonical).
- `--dist-activation-bits 16|8` reduces activation traffic (16 is the first
  thing to try on slower links; 8 is approximate/experimental).

**If a worker disconnects, the coordinator removes it from the active
route.** The request in flight can fail, and later calls report an incomplete
route until a compatible worker reconnects. The coordinator keeps the token
history and can rebuild worker KV state by replaying the prefix. Workers also
validate a rolling 64-bit token-prefix hash on every work item, so a
restarted worker cannot silently accept work for the wrong position. Ctrl+C
is cooperative: ns4 drains the current distributed token or prefill chunk
before returning control. Saved server sessions use the same KV file format
as single-machine sessions.

At the protocol level, workers keep a control TCP connection to the
coordinator (`HELLO` with model ID, quant profile, layer slice, context
capacity, data port), and work moves over low-latency TCP data connections
(`WORK` frames with session ID, token positions, prefix hashes, route info,
and hidden-state payload; `RESULT` frames echo the request ID and post-span
hash). Middle workers forward directly to the next worker; the final worker
returns logits, or ACKs for non-final prefill chunks so the pipeline stays
full. The protocol has no encryption or authentication and is not
release-stable: build coordinator and workers from the same commit and use
trusted machines and networks only.

## Tensor Parallelism Across CUDA GPUs (Experimental)

On a single CUDA server, `--cuda-tensor-parallel` splits DeepSeek V4 Flash
tensor and routed-expert work across an **even number of GPUs**. It does not
use `--role` or the distributed layer pipeline. GPU placement and memory
budgets use the normal `--gpu-devices` and `--gpu-vram` options. This mode
is experimental in this build: it needs a multi-GPU host that is not part of
the verified hardware, and has not been re-validated here.

The device order is significant. With `N` devices, the first `N/2` logical
tiers are contiguous layer-pipeline homes and the second `N/2` tiers are
their tensor-parallel partners: specify all homes first, then all partners,
with the closest P2P pair at matching positions (for physical pairs `(0,1)`,
`(2,3)`, `(4,5)`, `(6,7)` the ordered list is `0,2,4,6,1,3,5,7`). Each pair
stores a 50/50 split of the routed experts, and the vocabulary head is
row-sharded across the participating output tiers; dense attention, router,
and shared expert weights are replicated within each pair.

The imatrix Q4 model has native grouped multi-session kernels for its routed
`Q4_K` layout and gives the best aggregate serving throughput; the Q2 model
is the lower-memory choice, with unsupported grouped shapes falling back to
the exact ordered path. For serving, keep multiple KV sessions resident so
decode rows can be grouped across requests:

```sh
./ns4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf \
  --ctx 100000 \
  --batched-session 16 \
  --host 0.0.0.0
```

The managed launcher `./scripts/run-nvidia-tp-server.sh` runs this topology
with the native MXFP4 GGUF by default (set `NS4_MODEL` to serve the imatrix Q4
file above instead) and also enables the on-disk KV cache. Reduce the session
count or context size if the requested resident KV caches do not fit after
model loading. Without an explicit `--prefill-chunk`, this mode uses
2048-token chunks so a 16-session, 100k-context layout retains enough VRAM for
resident KV caches.

Any even card count that can hold the selected model and graph scratch is a
valid topology; for a four-card P2P-paired subset such as physical GPUs
`0,1,4,5`, the ordered list is `0,4,1,5`. This mode requires DeepSeek V4
Flash and an even multi-GPU placement. DGX Spark is a single-GPU target and
must not be started with `--cuda-tensor-parallel`.

## Speculative Decoding (Experimental)

DSpark is an auxiliary draft model released by DeepSeek for DeepSeek V4
Flash. It reads hidden states from the main model and proposes up to five
future tokens; ns4 verifies those proposals with the main model and commits
only the accepted prefix, so one verification pass can advance the stream by
several tokens. It does not accelerate prefill, and draft plus verification
work is not free: predictable continuations (especially code) benefit most,
while low-yield prompts can be no faster or even slower. DSpark is
explicitly opt-in.

The DSpark checkpoint for Flash 0731 is packaged as a separate support GGUF
of about 6 GB. It is checkpoint-specific: use it only with the 0731 Flash
models.

```sh
./scripts/download_model.sh ds4f-dspark

./ns4 -m ns4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0
```

`--mtp` supplies the support GGUF; `--dspark` selects the DSpark runtime.
The default confidence threshold is `0.7` (`--dspark-confidence 0` forces
fixed five-token blocks, for diagnostics). Sampled decoding does not use
DSpark proposals; `--quality` and `--dspark-strict` keep target-only
decoding for comparisons. DSpark replaces the legacy one-stage MTP support
model (`--mtp MTP.gguf --mtp-draft 2`) for that run rather than stacking
with it. MTP speculative decoding is disabled while native session batching
is active. The main model may be resident or use `--ssd-streaming`; the
support model adds its own weights and runtime state to the memory
requirement.

## Benchmarking

`ns4-bench` measures instantaneous prefill and generation throughput at
context frontiers instead of one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token
interval. After each frontier it saves the live KV state to memory,
generates a fixed greedy non-EOS probe, restores the snapshot, and continues
prefill.

```sh
./ns4-bench \
  -m ns4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334):
<https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for linear spacing or `--step-mul F` for exponential
sweeps. Output is CSV with one row per frontier: latest prefill interval
tokens/sec, generation tokens/sec at that frontier, and `kvcache_bytes`.
Sessions prefill long prompts in 4096-token chunks by default; use
`--prefill-chunk 2048` to match the strict official-vector checkpoint path,
and compare chunk sizes as explicit run configurations.

## Capability Evaluation

`ns4-eval` is a small real-model integration benchmark. It is not a
leaderboard runner and should not be reported as an official GPQA, SuperGPQA,
AIME, or security benchmark score: the questions are an embedded 92-item
subset chosen to make local regression testing useful and visually
inspectable. The program loads the real GGUF, renders DeepSeek chat prompts,
streams sampled tokens in a split-screen TUI, grades the final answer, and
prints a per-question report.

```sh
./ns4-eval -m ns4flash.gguf --trace /tmp/ns4-eval.txt
```

The default run uses `--tokens 16000`, thinking mode, and a soft/hard
`</think>` budget cutoff. Press `p` to pause, `q` to exit and print the
report, Up/Down to inspect or select a question, Enter to run it next;
`--plain` disables the TUI. Use `--regrade-trace /path/to/trace.txt` to
replay the answer extractor and scorer against a prior `--trace` file
without loading the model.

The first 75 embedded questions interleave 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 problems; the final 17 are an audited COMPSEC
subset of single-function C/C++ vulnerability-localization questions (the
task is to identify the flawed source line, or `0` for a safe function --
these are defensive code-review items, not exploit prompts). The set is a
hard capability regression suite, not a pass/fail unit test: a perfect 92/92
run is not expected. It answers a more useful engineering question: after a
kernel, quantization, prompt-rendering, KV-cache, or tool-streaming change,
does the model still solve a representative mix of hard problems through the
same inference path users run?

For inference changes that can affect generation drift, keep this
deterministic q1..q4 token-count gate in the test plan:

```sh
./ns4-eval \
  -m ns4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

The generated-token counts must stay aligned with the baseline:

| Question | Expected state | Expected generated tokens | Expected given/correct |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

## Steering

ns4 supports steering with single-vector activation directions; see
[tools/dir-steering/README.md](tools/dir-steering/README.md). This follows
the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, steer it away
from off-topic domains, and so forth, much faster than fine-tuning. It is
also useful for security researchers who want to reduce a model's
willingness to provide dual-use or offensive guidance.

## Reducing Heat, Power Usage And Fan Noise

If you care more about heat, fan noise, or thermal stress than maximum
throughput, use `--power N`. `--power 100` is the default; lower values
target that percentage of GPU usage by measuring GPU work time and inserting
small sleeps between work units (between layers during prefill, between
tokens during generation). This reduces sustained load without changing
model output. The option is available on the CLI, server, eval, and
benchmark tools:

```sh
./ns4 --power 50
./ns4-server --power 40 --ctx 100000
```

## Test Vectors And Regression Tests

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API (greedy, thinking disabled,
maximum `top_logprobs`). Local vectors are generated with
`./ns4 --dump-logprobs` and compared by token bytes, so tokenizer, template,
and attention regressions show up before they become long generation
failures. The C runner pins a 2048-token prefill chunk for this strict
comparison.

```sh
make test                  # ./ns4_test plus the extractor self-test and unit tests
./ns4_test --logprob-vectors
./ns4_test --server
```

The model-backed batching tests run on the matching GPU backend:

```sh
NS4_TEST_MODEL=/path/to/model.gguf make test-cuda-session-batch
NS4_TEST_MODEL=/path/to/model.gguf make test-cuda-mixed-batch
```

The official 100-case quality scorers, the CUDA matrix, and manual checks
are release gates rather than quick local tests; follow
[docs/QA_BEFORE_RELEASES.md](docs/QA_BEFORE_RELEASES.md).

## Debugging Notes

When a generation looks wrong, three small tools are usually enough for a
first answer:

```sh
./ns4 --dump-tokens -p "..."
./ns4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ns4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt
./ns4-server --trace /tmp/ns4-trace.txt ...
```

- `--dump-tokens` tokenizes the prompt exactly as written, recognizes NS4
  protocol specials, and exits before inference starts.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, separating sampling choices from logit/model
  issues.
- `ns4-server --trace` writes the rendered prompts, cache decisions,
  generated text, and tool-parser events for a whole agent session.

## Status

The software is fast changing; consider it beta quality. Before each release
the QA matrix in [docs/QA_BEFORE_RELEASES.md](docs/QA_BEFORE_RELEASES.md) is
executed, but instabilities are possible.

## Acknowledgements

ns4 is a rework of the **ds4 project** by Salvatore Sanfilippo (antirez).

ns4 does not link against GGML, but it exists thanks to the path opened by
**llama.cpp and GGML**: GGUF quant layouts and tables, CPU quant/dot logic,
and certain CUDA kernels are retained or adapted here under the MIT license
(see [src/cuda/mmq/VENDOR.md](src/cuda/mmq/VENDOR.md)). The upstream
copyright notices are preserved in [LICENSE](LICENSE).

This software is developed with strong assistance from coding models, with
humans leading the ideas, testing, and debugging.

## Logo

The NEUTRON STAR logo was made by hand.
