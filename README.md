# friend.cpp

A local inference engine for trying things -- adaptive speculative decoding,
per-request adapter hot-swap, a prompt cache that actually remembers, and whatever else
earns its place under honest measurement.

By [Solenopsisbot](https://github.com/Solenopsisbot).

---

## What this is

An independent downstream fork of [koboldcpp](docs/KOBOLDCPP.md), itself built on
llama.cpp and ggml, with [PrismML](https://github.com/PrismML-Eng/llama.cpp)'s llama.cpp fork
merged in. Everything koboldcpp already does -- its bundled KoboldAI Lite UI,
KoboldCpp/OpenAI/Ollama-compatible APIs, image generation, speech, all of it --
still works unchanged.

friend.cpp is a testbed for engine-level features: caching strategies, speculative
decoding, adapter management, sampling, steering, and GPU kernels. The ethos is
build, measure honestly, keep what the numbers justify. Fixed-length speculative
drafting nearly doubles speed on predictable text and loses 25% on chat, so the
default adapts per round instead -- and the tables below show both sides. Features
that don't pan out under measurement get documented as what they are, not sold as
what they aren't.

No release binaries yet. Build from source, details below.

## Highlights

| Feature | What you get |
|---|---|
| [Tiered prompt cache](#tiered-prompt-cache) | RAM + optional disk. Returning to a previous conversation restores 1057/1074 tokens; prefill drops from 0.35 s to 0.06 s. On by default. |
| [Per-request adapters](#per-request-adapter-profiles) | LoRA, steering vectors, and LM heads selected per request. Head swaps keep the KV cache valid -- nearly free. |
| [Adaptive speculative decoding](#adaptive-speculative-decoding) | Draft length chosen per-round from live measurements. Fixed drafting loses ~25% on chat; adaptive stayed at or above the no-drafter baseline on every workload tested. |
| [Steering vectors built in](#steering-vectors-built-in) | Build control vectors from contrastive prompts on the running model. 1.3 seconds from 12 pairs on a 1.7B model. |
| [Wider continuous batching](#continuous-batching) | Grammar, DRY, XTC, top-n-sigma, mirostat, and dynamic temperature all batch across concurrent requests. |
| Decode-first scheduling | Ready decodes run before prompt work; `--prefill-tokens` bounds prompt work per round so long prompts do not freeze other requests. |
| Priority scheduling and preemption | Per-request `priority` values, fair tie breaking, and bounded host snapshots let urgent work displace lower-priority sequences when all slots are full. |
| Suffix speculation | `--suffix-draft N` proposes continuations from repeated prompt/history suffixes; it takes precedence over the fixed n-gram drafter. |
| Batch logprobs | Concurrent requests can ask for per-token top alternatives with `logprobs`; results stay isolated per request. |
| Serving metrics | Prometheus text at `/metrics`: queue time, TTFT, ITL, decode time, cache reuse, and batch counters. |
| [Blue-noise sampling](#blue-noise-sampling) | Anti-correlated random rolls: fewer streaks at either end of the distribution, which the model would otherwise amplify into loops or derailing. Unbiased. |
| [Low-bit models and kernels](#low-bit-models-and-kernels) | Q1_0, PQ2_0, PTQ1_0 with CPU/Metal/CUDA/Vulkan kernels. Metal batch-of-8 on 27B Bonsai Q1_0: 190.6 ms down to 100.2 ms. Maxwell GPUs without dp4a go from 25 to 63+ tok/s on Bonsai-8B Q1_0. |

## Quick start

**macOS (Metal):**

```sh
make -j8 LLAMA_METAL=1 koboldcpp_default
python3 koboldcpp.py --model model.gguf --gpulayers 99 --contextsize 8192
```

**Linux (CUDA):**

```sh
make -j8 LLAMA_CUBLAS=1 koboldcpp_cublas
python3 koboldcpp.py --model model.gguf --usecuda --gpulayers 99 --contextsize 8192
```

For older NVIDIA GPUs (Maxwell, sm_5x), you need a CUDA 12.x toolkit -- CUDA 13
dropped support for those architectures. Point the build at it with `CUDA_PATH` if
it isn't the system default.

The API listens on port 5001 by default. Open `http://localhost:5001` for the
KoboldAI Lite UI, or point any OpenAI-compatible client at it.

See [FRIEND.md -- Quick start](FRIEND.md#quick-start-a-persona-server) for more
detail.

---

## Feature tour

### Tiered prompt cache

On by default. Snapshots of processed context (KV + recurrent state) are reused by
longest common prefix -- any later prompt starting with the same system prompt
picks up where the last one left off. RAM budget via `--cache-ram` (default
2048 MiB), optional disk tier via `--cache-dir` that survives restarts.

Hybrid and recurrent models (e.g. Qwen3.5) get checkpoints at conversation-turn
boundaries so they can resume from a shared prefix. Snapshots are taken in idle
time (`--cache-idle-ms`, default 300 ms), so switching conversations after any pause
doesn't pay for them.
Snapshots are zstd-compressed losslessly (~1.2x). Outputs are identical to running
without the cache in every tested scenario.

API: `GET /api/extra/cache`, `POST /api/extra/cache/warm` (prefill + pin a
prompt), `/pin`, `/clear`.

More: [FRIEND.md -- prompt cache](FRIEND.md#tiered-prompt-cache).

### Per-request adapter profiles

Preload pools at startup:

```
--lora-pool NAME=PATH ...
--cvec-pool NAME=PATH ...
--head-pool NAME=PATH ...
```

Each request selects what it needs:

```json
{ "lora": {"name": 0.8}, "steer": {"name": 1.2}, "head": "name" }
```

LM-head hot-swap is new engine code. Swapping a head only changes logits -- the KV
cache stays valid, so per-request head switching is nearly free. LoRA and steering
changes are keyed into the cache so KV is never reused across incompatible adapter
sets. Concurrent requests with different adapters are grouped per decode step.

LoRAs work on Bonsai models (train against the unpacked weights).
`tools/friend-heads/extract_head.py` extracts a head from any GGUF.

More: [FRIEND.md -- adapter profiles](FRIEND.md#per-request-adapter-profiles).

### Steering vectors built in

`POST /api/extra/steer/build` takes contrastive example prompts (positive vs
negative), builds a control vector from the model's own hidden states using mean
difference or PCA, auto-selects the layer band where the examples separate best,
and scales it so strength 1.0 is approximately the observed shift. On a 1.7B
model: built in 1.3 seconds from 12 pairs. Strength 1 / 2 / 4 produces
noticeably / clearly / over-the-top shifted output while staying coherent.

Vectors save to `--cvec-dir` and reload at startup.

### Adaptive speculative decoding

`--draftamount` is now a maximum, not a fixed count. Each round, the engine picks
the draft length (including 0, meaning "don't draft this round") that maximises
expected tokens per millisecond, calculated from live measurements of acceptance
rate, drafter time, and verify time.

27B Bonsai Q1_0 with its DSpark drafter on Apple M5, tokens/s:

| Workload | No drafter | Fixed draft of 3 | Adaptive |
|---|---|---|---|
| Counting | 21.7 | 39.0 | 36.6 |
| Code | 21.6 | 36.6 | 35.7 |
| Surreal haiku | 16.4 | 14.7 | 16.3 |
| Chat | 21.5 | 15.8 | 22.0 |

Fixed drafting wins big on predictable sequences and loses ~25% on chat. Adaptive
stayed at or above the no-drafter baseline on every workload tested. `--draft-fixed` restores the old
behaviour if you want it.

PrismML's standalone DSpark drafters work via `--draftmodel`
(Qwen3.5/3.6-family targets).

More: [FRIEND.md -- DSpark](FRIEND.md#speculative-decoding-with-dspark-drafters).

### Continuous batching

`--parallelrequests N` enables concurrent request handling. Finished requests keep
their KV for prefix reuse; new requests take the best-overlapping slot or share
another sequence's cells. friend.cpp batches more request types than stock
koboldcpp -- grammar, DRY, XTC, top-n-sigma, mirostat, and dynamic temperature all
work across concurrent requests.

Scheduling is decode-first, then chunked prefill. `--prefill-tokens N` limits how
many prompt tokens can be admitted beside one decode round; `0` uses the normal
backend batch size. Smaller values protect inter-token latency under long prompts,
while larger values improve time to first token and throughput. `/metrics` exposes
Prometheus-compatible queue, TTFT, inter-token, decode, batching, and cache-reuse
counters for tuning this tradeoff.

Requests may include `cache_salt` to isolate prefix reuse between tenants. The salt
is hashed before it reaches the native cache key, and adapter/cache identities use
SHA-256. Model/LoRA file identities still incorporate path, size and modification
time rather than hashing entire weight files.

Batch requests may include `priority`; lower values are scheduled first, while
equal-priority requests retain fair round-robin ordering.

`--schedule-tokens N` sets the unified token budget for each batch round. Decode
tokens are admitted first, then prompt and speculative tokens spend the remainder;
`0` keeps the backend batch size. This is useful when a large `--batchsize` would
otherwise make long prefills harm inter-token latency.

`--max-queued-requests N` bounds live continuous-batching requests, including
waiting, running and paused work. A full queue returns a retryable overload error
instead of silently falling back to the legacy generator; `0` leaves the cap off.
`friend_batch_requests_rejected_total` counts those admissions.

`--kv-watermark F` reserves fraction `F` of the estimated sequence-token capacity
for currently active decodes before admitting another prompt. It reduces KV
thrash while the physical allocator is under pressure; `0` disables it.
`friend_batch_kv_watermark_stalls_total` counts deferred admissions.

When every sequence slot is occupied, a waiting request with a strictly higher
priority can evict one lower-priority sequence into the bounded compressed host
snapshot tier. The request resumes with its sampler state and KV contents intact;
equal-priority traffic is never churned. `/metrics` exposes
`friend_batch_preemptions_total`.

For prompt/history-driven workloads, `--suffix-draft N` uses repeated suffixes and
frequency-weighted continuations to generate adaptive draft proposals. It is
compatible with continuous batching and falls back to the existing n-gram drafter
when unset. Requests can pass `"logprobs": K` (up to 20) to retain top-K token
alternatives in the batch result without sharing the legacy process-wide buffer.

Native callers that need concurrent personas can bind an LM head to an individual
`llama_context` with `llama_set_adapter_head(ctx, head)` and inspect it with
`llama_get_adapter_head`. The binding is included in graph reuse keys, so contexts
sharing one model can decode in parallel without mutating the model's global head.
Currently the context-local graph path covers LLaMA, Qwen3/Qwen3-MoE, and
Qwen3.5/Qwen3.5-MoE. Freeing a head unbinds it from its contexts. Profile application uses this path where
supported. The existing `llama_model_set_head` API remains the model-wide
compatibility path for other architectures.

For in-process parallel persona execution, `--profile-lanes N` creates N native
contexts over the same model weights. Each lane owns independent KV, compute
buffers, adapter bindings, and sampler state; requests are affinity-routed by
LoRA/vector/head profile and then scheduled fairly within a lane. CPU thread
counts are divided across lanes, while the backend weights remain shared. This
mode currently requires ordinary continuous batching without a separate draft
model, MTP, or CFG guidance context; use the process router below when those
features or model-wide adapter architectures are required.

Each lane prepares its next batch under the scheduler lock, releases that lock
while `llama_decode` owns the lane-local buffers, and applies pause or abort at
the next completed round. This gives genuine cross-lane overlap today; a
backend-specific prefetch queue with completion fences remains future work.

For real model-replica parallelism, run the small process router alongside several
full workers:

```sh
python tools/friend-bench/data_parallel_router.py --model model.gguf \
  --replicas 2 --port 5001 --replica-arg=--gpulayers --replica-arg=99
```

Each worker owns its own context and device placement. Requests with the same
`cache_salt` and adapter profile stay on one replica for prefix-cache locality;
other requests go to the least-busy worker. LoRA, steering, and head profiles are
therefore isolated in real worker lanes instead of being mutated mid-decode.
The router namespaces request IDs as `router-epoch:worker:local-id`, aggregates
request listings across healthy workers, retries only connection failures before
the request is sent, and forwards SSE/NDJSON chunks as they arrive. Pause/resume
controls use the namespaced ID, so they cannot target a request on another worker.
The router also samples each worker's running/waiting requests and KV-block count;
fresh samples influence least-load selection, while samples older than two seconds
expire back to the local in-flight counter.
Probe latency is folded into that score with an EWMA, and a worker can be marked
draining so existing request owners finish while new work moves elsewhere.
Connection failures put a replica on a short exponential health cooldown, while
successful responses restore it immediately; if every replica is cooling down the
router still probes one so a recovered worker can rejoin.
This is the safe way to scale across devices without sharing mutable adapter or
KV state.

History-based speculation is available with `--ngram-draft N`. It proposes repeated
continuations from the request's own token history and verifies them in the same
target batch; it is opportunistic, so a request with no repeated suffix simply falls
back to ordinary decoding. `/api/extra/requests` lists active batch requests, while
`--suffix-draft N` searches both the prompt and generated history for repeated
suffixes, chooses frequency-weighted continuations, and adapts the proposed length
per request. It takes precedence over `--ngram-draft`.
`POST /api/extra/requests/pause` and `/resume` suspend or continue a request at a
worker boundary, while `/cancel` asks the native scheduler to abort it and
release its KV ownership at the next safe boundary. Paused KV is bounded by a
512 MiB host-memory budget.
Preempted and explicitly paused snapshots use the same lossless byte-plane/zstd
codec as the prompt cache when compression saves space; `/metrics` exposes the
current stored byte count as `friend_batch_offloaded_bytes`.
With native profile lanes enabled, `friend_batch_lane_running{lane="N"}` shows
the bounded live request count for each lane, making affinity imbalance visible.
The scheduler's physical page table reports `friend_batch_kv_pages_used`,
`friend_batch_kv_pages_capacity`, page queries/hits and evictions for tuning cache
pressure. The current llama backend still owns the actual KV tensors; this table
is the ownership layer used while the backend migration to true paged storage is
completed.

`tools/friend-bench/lane_benchmark.py` measures end-to-end throughput for one or
more native profile-lane counts and verifies every result is non-empty.

OpenAI structured output requests use the same cached native grammar path for
JSON Schema and JSON object responses. vLLM-style `structured_outputs: {"json":
...}` / `{"json_schema": ...}` and `{"choice": [...]}` forms, plus
`guided_json` and `guided_choice`, are accepted as aliases.

OpenAI completion and chat requests may set `n` up to 16 for parallel sampling.
Each sample gets an independent RNG/sampler state and indexed choice; native
continuous batching shares complete prompt KV blocks between siblings and
aggregates usage across the returned choices. Streaming `n` uses indexed SSE
fan-in, drains completion races, and preserves split UTF-8 tokens.

The scheduler-owned page table also has a versioned metadata connector in
`friend/kv_connector.hpp`. Peers discover model/layout/dtype/profile/namespace
compatibility before importing, receive export/import/invalidation events, and
cannot invalidate pages that still have live references. The current llama
backend has no device-KV transport hook, so the connector carries ownership
metadata and leaves tensor payload transfer to the future paged backend.

Multimodal requests retain up to eight preprocessed mtmd encoder results. The
cache key includes media bytes, encoder identity, audio/image mode and the
vision resize limit; replacing the encoder drops all entries. Prometheus exposes
`friend_multimodal_encoder_cache_hits_total`, misses, evictions, current entry
count, and encoded embedding bytes. Opaque mtmd metadata remains outside the
byte gauge because the public API does not expose its storage size.
### Blue-noise sampling

`"blue_noise": true` in a request enables anti-correlated random rolls -- each
individual roll stays uniform (unbiased), but streaks of rolls at the same end of the
distribution get rarer. That matters because the model conditions on its own output:
a run of top picks makes the next top pick likelier until the text locks into a loop
(context collapse), and a run of bottom picks derails it. You can also set
`"rng_type": "mt19937" | "lowbias32"` to pick the underlying generator.

Measured: tail-pick streaks 20--27% shorter with mean surprisal unchanged. On a base
model (Qwen3-1.7B-Base, 768-token raw completions, 192 paired runs) blue noise cut
collapse from 24.5% to 19.8% at T=0.7, with less repetitive text overall (paired
t = 2.3); the effect is smaller at T=1.0 with min_p. kaetemi, whose sampler this is,
reports a large reduction in collapse on base models and about 1% better
reasoning-model output in preliminary benchmarks. Short chat replies from an instruct
model showed no judged preference either way -- too short to collapse.

More: [FRIEND.md -- blue-noise sampling](FRIEND.md#blue-noise-sampling).

### Low-bit models and kernels

GGUF types Q1_0 (upstream llama.cpp), PQ2_0, and PTQ1_0 (PrismML-specific, used
by models like `prism-ml/Ternary-Bonsai-*-gguf`) with kernels for CPU, Metal,
CUDA, and Vulkan.

friend.cpp-specific kernel work falls in two places:

**Metal small-batch kernels.** Decoding a batch of n tokens on the 27B Bonsai Q1_0
model (milliseconds per batch, Apple M5):

| Batch size | Stock | friend.cpp |
|---|---|---|
| 1 | 42.5 | 42.4 |
| 2 | 57.1 | 52.2 |
| 4 | 101.3 | 65.9 |
| 8 | 190.6 | 100.2 |
| 16 | 377.6 | 179.2 |

Single-token decode and prefill are unchanged. The wins matter for speculative
decoding verification and concurrent-request batches.

**Maxwell dp4a-free path.** Old NVIDIA GPUs that lack dp4a (compute capability
< 6.1) get a multiply-free bit-plane dot product for Q1_0 and PQ2_0. On a GTX 970
running Bonsai-8B, generation speed: Q1_0 jumps from 25--29 to 63--72 tok/s,
PQ2_0 from 24--26 to 46--48 tok/s. Output is bit-identical to the standard path.
Auto-selected at runtime; you do not need to do anything.

More: [FRIEND.md -- 1-bit / ternary](FRIEND.md#1-bit--ternary-decode-on-gpus-without-dp4a)
and [FRIEND.md -- small-batch Bonsai kernels](FRIEND.md#small-batch-bonsai-kernels-metal).

---

## Experimental features

Most of friend.cpp's additions can be turned off individually for A/B comparisons
or if something misbehaves:

- **Prompt cache:** `--cache-ram 0` disables it entirely.
- **Adaptive drafting:** `--draft-fixed` restores fixed-length drafting.
- **Blue-noise sampling:** opt-in per request (`"blue_noise": true`); off by
  default.
- **Metal small-batch kernels:** `GGML_METAL_BONSAI_SB_DISABLE=1` restores the
  stock kernels.
- **CUDA bit-plane path:** `FRIEND_CUDA_NO_BITPLANES=1` forces the stock dot
  product.

Things may move between releases. The knobs exist so you can isolate what changed.

---

## Status and tested hardware

This is a build-from-source project with no release binaries.

- **Apple Silicon (Metal + CPU):** most tested. The numbers in this README are from
  an M5.
- **CUDA (Maxwell / GTX 970):** built and run. The 1-bit dp4a-free kernels are
  verified here. The CUDA code also compiles for sm_61 through sm_120, but only
  the 970 (sm_52) has been run on real hardware so far.
- **Vulkan:** run on a GTX 970 (Maxwell): Q1_0 and PQ2_0 about
  32 tok/s generation on Bonsai-8B, PTQ1_0 about 14; test-backend-ops
  passes, and PQ2_0 greedy output matches CUDA. CUDA is still faster on
  that card for Q1_0 and PQ2_0; Vulkan wins for PTQ1_0.
- **HIP (AMD):** not built yet.

See [FRIEND.md -- limits](FRIEND.md#limits-and-not-yet-verified) for the honest
list of what has and has not been tested.

## Everything from koboldcpp still works

friend.cpp is a fork, not a replacement. The full koboldcpp feature set -- KoboldAI
Lite UI, KoboldCpp API, OpenAI-compatible API, Ollama-compatible API, image
generation, speech, story writing, all of it -- is intact and documented in
[docs/KOBOLDCPP.md](docs/KOBOLDCPP.md).

## Tools

- **`tools/friend-sync/sync.sh`** -- one-command merge of upstream koboldcpp +
  PrismML with conflict notes. Never pushes.
- **`tools/friend-sync/smoke.py`** -- 24 end-to-end checks in about a minute:
  coherence, determinism, cache-on output must equal cache-off output, head swap,
  steering, batching, GPU vs CPU.
- **`tools/friend-eval/sampler_lab.py`**, **`judge_lab.py`** and
  **`collapse_eval.py`** -- sampler measurement, judged comparison, and
  long-generation context collapse.
- **`tools/friend-bench/`** -- kernel benchmarks.
- **`tools/friend-heads/extract_head.py`** -- extract an LM head from any GGUF.

More: [FRIEND.md -- tools](FRIEND.md#tools-for-maintaining-the-fork).

## Credits

- [koboldcpp](https://github.com/LostRuins/koboldcpp) by LostRuins (Concedo) and
  contributors
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and
  [ggml](https://github.com/ggml-org/ggml) by Georgi Gerganov and contributors
- [PrismML](https://github.com/PrismML-Eng/llama.cpp) for the Bonsai 1-bit/ternary models,
  kernels, and DSpark drafters
- Blue-noise sampler idea adapted from
  [kaetemi's llama.cpp branch](https://github.com/kaetemi/llama.cpp)

friend.cpp-specific code is marked `friend.cpp:` in the source. New modules live
in `friend/`.

## License

AGPL-3.0 (inherited from koboldcpp). See [LICENSE.md](LICENSE.md).

The ggml, llama.cpp, and stable-diffusion.cpp portions remain MIT-licensed; see
[MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md).
