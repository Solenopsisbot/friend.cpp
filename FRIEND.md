# friend.cpp

A fully independent downstream fork of [koboldcpp](https://github.com/LostRuins/koboldcpp) (v1.120+), which is itself built on llama.cpp. friend.cpp also merges [PrismML's llama.cpp fork](https://github.com/prism-ml), bringing in Bonsai 1-bit and ternary GGUF types alongside the full koboldcpp feature set.

The project exists to power **computer friends** -- social AI chatbot personas (Kiko, Amelia, June, and others) that run as Discord bots and through a web portal. That's the design pressure behind every feature here: persona servers need fast context switching between characters, cheap per-character customisation, and sampling that doesn't produce unlucky streaks of garbage in the middle of a conversation. friend.cpp should also be a perfectly good general-purpose local LLM server for anyone who wants one.

Maintained by Viola (she/her). Everything friend.cpp adds to the upstream koboldcpp codebase is marked with `friend.cpp:` comments in the source, and new modules live under `friend/` (notably `adapters.hpp` and `prompt_cache.hpp`).


## What's in the fork

From **koboldcpp upstream**: the full feature set -- all GGUF model support, image/video/audio generation, vision, the bundled KoboldAI Lite UI, all the compatible API endpoints, samplers, everything in the upstream README.

From **PrismML's llama.cpp**: Bonsai 1-bit/ternary GGUF quantisation types. Q1_0 is upstream llama.cpp; PQ2_0 and PTQ1_0 are Prism-only types (you'll find models like `prism-ml/Ternary-Bonsai-*-gguf` on HuggingFace). These come with CPU, Metal, CUDA, and Vulkan compute kernels, plus Hadamard weight folding and Qwen3.5 decode speedups.

**Verified on**: Apple Silicon (Metal + CPU); CUDA on a GTX 970 (Maxwell) with the Bonsai models.
**Present but not yet compiled/tested**: HIP. CUDA for newer Nvidia architectures compiles but hasn't been run.
**Shaders generate but untested on actual GPU hardware**: Vulkan.

friend.cpp-specific features, described below:

1. **Blue-noise sampling** -- anti-correlated randomness that smooths out generation quality
2. **Per-request adapter profiles** -- LoRA mix, steering vectors, and LM head swaps, per request
3. **Tiered prompt cache** -- RAM + disk caching of KV state with automatic prefix reuse
4. **DSpark speculative drafters** -- PrismML's standalone drafters wired into `--draftmodel`, cache-aware
5. **Fast 1-bit / ternary decode on old NVIDIA GPUs** -- a bit-plane mat-vec path for Q1_0 / PQ2_0 on GPUs without `dp4a` (Maxwell)


---


## Quick start: a persona server

This walks through a realistic setup: two character personas sharing a base model, each with their own fine-tuned LM head and a steering vector, with disk-backed prompt caching so restarts are cheap.

### Launch

```bash
python koboldcpp.py --model model.gguf \
  --head-pool kiko=heads/kiko.head.gguf amelia=heads/amelia.head.gguf \
  --cvec-pool happy=cvecs/happy.gguf anxious=cvecs/anxious.gguf \
  --cache-dir ./cache \
  --cache-ram 2048 \
  --cache-disk 20480 \
  --parallelrequests 4 --noshift \
  --port 5001
```

### Warm the persona cards

Before any users connect, pre-fill and pin each persona's system prompt so the first real request is instant. Warm with the **same LoRA/steering set** the persona will chat with: those change the KV cache, so the cache keys on them (the head doesn't matter -- it never touches the KV):

```bash
curl -s http://localhost:5001/api/extra/cache/warm -d '{
  "prompt": "<|im_start|>system\nYou are Kiko. [... full persona card ...]\n<|im_end|>",
  "head": "kiko",
  "steer": {"happy": 3.0},
  "cache_pin": "kiko-persona"
}'

curl -s http://localhost:5001/api/extra/cache/warm -d '{
  "prompt": "<|im_start|>system\nYou are Amelia. [... full persona card ...]\n<|im_end|>",
  "head": "amelia",
  "cache_pin": "amelia-persona"
}'
```

### Chat

```bash
curl http://localhost:5001/v1/chat/completions -d '{
  "model": "whatever",
  "messages": [
    {"role": "system", "content": "You are Kiko. [... persona card ...]"},
    {"role": "user", "content": "hey kiko, how are you?"}
  ],
  "head": "kiko",
  "steer": {"happy": 3.0},
  "temperature": 1.0,
  "blue_noise": true
}'
```

The server will match the system prompt against the pinned cache entry, skip prefilling those ~1000 tokens, apply Kiko's LM head (cache stays valid -- head swaps are nearly free), steer the hidden states toward "happy," and sample with blue-noise anti-correlation. The whole thing takes a fraction of the time a cold start would.


---


## Blue-noise sampling

### The problem it solves

Standard sampling draws each token independently. That's mathematically fine on average, but in practice you sometimes get unlucky streaks -- several consecutive low-probability picks that send the generation off a cliff. The averages are correct; the variance is the problem.

### How it works

Every sampled token consumes a uniform random roll. Blue-noise sampling keeps each individual roll uniform (so sampling remains unbiased -- the distribution of any single token is unchanged) but **anti-correlates consecutive rolls**. If the last roll was low, the next one is nudged away from also being low. The result is fewer clusters of tail-probability picks in a row, without changing what the model thinks is likely.

### Request fields

Works on `/api/v1/generate`, the OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`), and in the web UI settings panel.

| Field | Type | Default | Notes |
|---|---|---|---|
| `blue_noise` | bool | `false` | Enable blue-noise anti-correlation. |
| `rng_type` | string | `"mt19937"` | Base RNG. Options: `"mt19937"`, `"lowbias32"`. |

### Measured results

Evaluated with `tools/friend-eval/blue_noise_eval.py` using Ternary-Bonsai-1.7B, 8 chat prompts, 10 seeds each, T=1.0, min_p=0.02:

| Metric | Baseline | Blue noise | Change |
|---|---|---|---|
| Roll lag-1 autocorrelation | +0.01 | -0.18 | t = -16.8 |
| Longest run of tail picks (p < 0.1) | 2.34 | 1.88 | t = -5.0 |
| Windowed surprisal variance | 0.111 | 0.075 | -33%, t = -5.1 |
| Mean surprisal | 0.971 | 0.987 | unchanged, t = +0.8 |
| Cross-seed diversity | -- | -- | slightly better, t = -2 |
| Repetition | -- | -- | not significantly changed |

The important line: mean surprisal is unchanged (the model's calibration is preserved), but surprisal variance drops by a third (generation is smoother).

### Running the eval

```bash
python tools/friend-eval/blue_noise_eval.py \
  --url http://localhost:5001 \
  --seeds 12 \
  --max-tokens 160 \
  --temperature 1.0 \
  --out results.json
```

Stdlib only -- no extra dependencies. Needs a running server with any chat model loaded.


---


## Per-request adapter profiles

The core idea: preload named pools of LoRA adapters, steering vectors, and LM heads at startup, then mix and match them per request. A persona server can share one base model and switch character "on top" without reloading anything.

### Startup flags

| Flag | What it loads | Behaviour |
|---|---|---|
| `--lora-pool NAME=PATH [NAME=PATH ...]` | LoRA adapters (GGUF) | Off by default; enabled per request. |
| `--cvec-pool NAME=PATH [...]` | Control/steering vectors (llama.cpp control-vector GGUF: F32 tensors `direction.<layer>`) | Off by default; enabled per request. |
| `--head-pool NAME=PATH [...]` | LM heads (GGUF containing `output.weight`, optionally `output_norm.weight`, `output.bias`) | Off by default; selected per request. |

Names must match `[A-Za-z0-9_.-]+`.

**Legacy `--lora FILE [FILE ...]`** still works. Those files join the LoRA pool as always-on at `--loramult`, named after their file stem. All listed files now apply (not just the first).

### Request fields

These work on any text generation endpoint, including OpenAI chat completions.

**LoRA** -- `"lora"`:
```jsonc
// object form: name -> scale
{"lora": {"kiko": 0.8, "sleepy": 0.3}}

// array of names (scale defaults to 1.0)
{"lora": ["kiko"]}

// array of objects
{"lora": [{"name": "kiko", "scale": 0.8}]}

// single string
{"lora": "kiko"}

// empty object: turns OFF legacy --lora adapters for this request
{"lora": {}}
```

If the `lora` field is present, it **replaces** the default-on adapters for that request.

**Steering vectors** -- `"steer"` (alias `"cvec"`):
```json
{"steer": {"happy": 4.0, "anxious": -2.0}}
```
Strengths are multipliers on each stored direction vector. Multiple vectors are summed.

**LM head** -- `"head"`:
```json
{"head": "kiko"}
```
Swaps the final projection layer for this request.

An unknown name in any field fails the request with `finish_reason: "error"`.

### Listing available adapters

```bash
curl http://localhost:5001/api/extra/adapters
```

Returns:

```json
{
  "lora": [{"name": "kiko", "default_scale": 1.0}, ...],
  "cvec": [{"name": "happy"}, ...],
  "head": [{"name": "kiko"}, ...]
}
```

### Cache behaviour

This matters for performance, so pay attention to it:

- **LoRA or steering vector changes invalidate the KV cache.** The prompt cache keys on the adapter set, so each unique combination gets its own cached prefixes. Switching adapters means reprocessing the prompt (or restoring a cache snapshot for that adapter set).
- **Head swaps do NOT invalidate the KV cache.** The head only affects the final projection after the transformer stack, so switching heads per request is nearly free. This is why head-only fine-tunes are the recommended approach for per-persona customisation.

### Building steering vectors

You don't need an external tool: friend.cpp builds control vectors from contrastive examples on the model it already has loaded. Give it prompts where the model *is* the thing (an excited Kiko) and prompts where it isn't (a flat Kiko), ideally in pairs that differ only in that one respect:

```bash
curl -s http://localhost:5001/api/extra/steer/build -d '{
  "name": "excited",
  "positive": ["<|im_start|>system\nYou are wildly excited about everything.<|im_end|>\n<|im_start|>user\nTell me about my cat.<|im_end|>\n<|im_start|>assistant\nOh", "..."],
  "negative": ["<|im_start|>system\nYou are bored and flat about everything.<|im_end|>\n<|im_start|>user\nTell me about my cat.<|im_end|>\n<|im_start|>assistant\nOh", "..."],
  "method": "pca"
}'
```

It reads the residual stream at the end of every layer, takes the difference between the two sets (`"method": "mean"`, the default, or `"pca"` -- the dominant direction across pairs, needs equal-length lists), and applies it across the band of layers where the two sets separate best. The vector joins the live steering pool immediately (`"steer": {"excited": 1.5}`), and with `--cvec-dir DIR` it is saved as `DIR/excited.gguf` and loaded again at every startup.

- **Strength scale:** 1.0 is roughly "shift by the difference actually observed between your two sets". On Ternary-Bonsai-1.7B with 12 excited/bored pairs: 1 is noticeably warmer, 2 clearly excited, 4 over the top but still coherent. Negative strengths push the other way.
- **Other fields:** `"pool": "last"` (default, the final token of each prompt) or `"mean"` (all tokens); `"layers": [start, end]` to override the automatic band; `"normalize": true` for plain unit vectors; `"save": false` to keep it in memory only.
- **Response:** includes `layers` (the band used), `best_layer`, per-layer `separation` (how cleanly the sets split -- a quick check that your examples actually contrast), and `seconds` (about 1.3 s for 24 short prompts on a 1.7B model).
- Rebuilding a vector under the same name is safe: the prompt cache keys on the vector's content, never on its name alone.
- The endpoint respects `--password`, and it is serialized with generation.

### Extracting head files

```bash
uv run --with numpy --with pyyaml --with tqdm \
  python tools/friend-heads/extract_head.py model.gguf out.head.gguf
```

Extracts `output.weight` (or the tied `token_embd.weight`) plus `output_norm.weight` from any GGUF model file. Pass `--no-norm` to skip the norm layer.

The intended workflow: take a base model, fine-tune *only* the LM head (and optionally the final norm) for each persona, extract the result into a tiny GGUF, and load it with `--head-pool`. The head must match the base model's shape: `[n_embd, n_vocab]`.

### Continuous batching

With `--parallelrequests N` (kobold only batches with `--noshift`), requests are grouped by adapter profile per decode step. The worker sticks with a profile while it has work queued for it, then switches to a starved profile after 8 decode rounds.

### What's been verified

- Each adapter kind (LoRA, steering, head) changes greedy output and returns exactly to base output when removed.
- A copy of the model's own head reproduces base output bit-for-bit.
- Head-only switches don't trigger prompt reprocessing.
- Concurrent mixed-profile requests match serial results.


---


## Tiered prompt cache

On by default. Replaces koboldcpp's SmartCache automatic slot switching. (The admin save/load state endpoints still work.)

### Why it exists

Persona servers constantly rotate between characters. Without caching, every time you switch from Kiko to Amelia and back, you re-prefill the entire persona card -- easily 1000+ tokens. The tiered cache snapshots KV state to RAM and disk so that switching back is nearly instant, and the disk tier survives server restarts.

### Startup flags

| Flag | Default | What it does |
|---|---|---|
| `--cache-ram MB` | 2048 | RAM budget for cached snapshots. Set to 0 to disable (falls back to old `--smartcache` behaviour). |
| `--cache-dir PATH` | (none) | Directory for the disk tier. Entries are written through asynchronously. Survives restarts, reloaded lazily on demand. |
| `--cache-disk MB` | 20480 | Disk budget. |
| `--cache-min-tokens N` | 64 | Don't cache or reuse prefixes shorter than this. |
| `--cache-capture-tokens N` | 512 | Snapshot after prefilling at least this many new tokens. |

Requires fast-forward to be enabled (it is by default).

### How reuse works

Snapshots store one sequence's KV state plus next-token logits, keyed by the exact token sequence, adapter set, and any media inputs.

**Attention models** (most transformers): the cache reuses the **longest common prefix** of any snapshot. One snapshot of "system prompt + persona card + chat history" serves every later prompt that starts with the same persona card, even if the conversation after it differs.

**Recurrent/hybrid models** (e.g. Qwen3.5): these can only resume from a snapshot that is a *complete* prefix of the current prompt (you can't skip ahead in a recurrent state). To compensate, friend.cpp takes checkpoints at **conversation-turn boundaries** -- specifically, right before chat-template control tokens (like `<|im_start|>`) that follow a line break. This captures the end of the system/persona block and the start of the final turns.

### When snapshots are taken

- When the live context is about to be discarded (switching to a different conversation).
- After a large prefill (at least `--cache-capture-tokens` new tokens processed).
- At turn-boundary checkpoints (recurrent models).
- When explicitly pinned via the API.

Snapshots that are a prefix of a longer snapshot are deduplicated. Eviction is LRU within each tier's budget. Pinned entries are never evicted.

### Disk tier safety

The disk tier is scoped by a fingerprint of the model file(s) plus the KV cache layout (quantised KV type, flash attention, sliding-window attention mode, MTP). If you change any of those settings, the server won't load incompatible snapshots -- it just starts fresh.

### API

**Inspect the cache:**

```bash
curl http://localhost:5001/api/extra/cache
```

```json
{
  "enabled": true,
  "recurrent": false,
  "entries": 3,
  "ram_bytes": 41943040,
  "disk_bytes": 0,
  "items": [
    {
      "id": 1,
      "tokens": 1057,
      "bytes": 13893632,
      "resident": true,
      "on_disk": true,
      "pinned": true,
      "label": "kiko-persona",
      "adapters": {},
      "idle": 12.5
    }
  ]
}
```

**Warm and pin a prompt:**

```bash
curl http://localhost:5001/api/extra/cache/warm -d '{
  "prompt": "... the full prompt to prefill ...",
  "head": "kiko",
  "cache_pin": "kiko-persona"
}'
```

Body format is the same as `/api/v1/generate` (supports `prompt` and all adapter fields), plus `"cache_pin": "label"` (or just `"label"` as a shorthand) to pin the resulting snapshot.

**Pin or unpin an existing entry:**

```bash
curl http://localhost:5001/api/extra/cache/pin -d '{"id": 12, "pinned": true}'
```

**Clear the cache:**

```bash
# Remove unpinned entries only
curl http://localhost:5001/api/extra/cache/clear -d '{"pinned": false}'

# Remove everything, including pinned entries
curl http://localhost:5001/api/extra/cache/clear -d '{"pinned": true}'
```

The management endpoints (`clear`, `pin`) and `cache_pin` on generation requests respect the server `--password` (via Bearer token).

Any generation request can also carry `"cache_pin": "label"` to pin its processed prompt after generation completes.

### Continuous batching integration

The cache plays well with `--parallelrequests`:

- Finished requests keep their KV state in their slot (retained, not immediately discarded).
- A new request takes the slot with the most token overlap, or an empty slot rather than wiping a valuable retained context.
- On attention models, a new request can share another live or retained sequence's KV cells.
- On any model, it can restore from the prompt cache if no slot has good overlap.
- Retained slots are saved into the prompt cache before a single-user request runs, and released under KV memory pressure.

### Measured performance

On Apple M5, with persona prompts around 1,050-1,100 tokens:

- Returning to a persona after visiting another: restores 1057 of 1074 tokens, prefill drops from 0.35s to 0.06s.
- After a full server restart with `--cache-dir`: snapshot restored from disk, same speed.
- Batched follow-up requests: reuse 1057-1060 tokens from the first request's cached state.
- Qwen3.5-0.8B (recurrent): restores the 1094-token persona checkpoint correctly.
- Outputs are identical to a cache-disabled run in every tested scenario.


---



## Speculative decoding with DSpark drafters

PrismML ships standalone DSpark drafters next to their Bonsai models (e.g. `prism-ml/Bonsai-27B-gguf` has `Bonsai-27B-dspark-Q4_1.gguf`). Load one like any draft model:

```bash
python koboldcpp.py --model Bonsai-27B-Q1_0.gguf \
  --draftmodel Bonsai-27B-dspark-Q4_1.gguf --draftamount 3
```

- The drafter reads hidden states from specific target layers, which currently only the Qwen3.5/3.6 family (`qwen35` arch) can provide. Other targets fail at load with a clear error rather than mid-generation.
- `--draftamount 3` measured best on an M5 (about 1.25x over the target alone at temp 0 and 0.7; 4 barely helped).
- `LLAMA_DSPARK_SHARED_HEAD=1` makes the drafter borrow the target's LM head (saves ~700 MB).
- It survives prompt reuse: fast-forward, context shifts, SmartCache slots and the prompt cache all keep the drafter's captured features in step with the target KV, so drafting keeps working on later turns and after switching conversations. Outputs match the target alone.

## 1-bit / ternary decode on GPUs without dp4a

Maxwell cards (GTX 9xx, sm_5x) and GP100 have no `__dp4a`, so ggml's int8 dot products are emulated byte by byte. For Bonsai Q1_0 and PQ2_0 that made token generation ALU-bound: on a GTX 970 the mat-vec kernels read weights at 22-40 GB/s out of ~190, and the matmuls were essentially the whole token time.

1-bit and ternary weights are just select masks, so the dot product doesn't need multiplies. On those GPUs the activation vector is quantised exactly as for q8_1 (same scale, same int8 values) but stored as 8 bit planes per 32 values plus the exact block sum, and each 32-weight chunk becomes 8 (Q1_0) or 16 (PQ2_0) AND+POPC pairs. Single-token decode then runs in a dedicated kernel where each warp owns 4 rows and reuses the activation planes across them. The integer arithmetic and the float summation order are the same as the stock path, so outputs are **bit-identical**; it's purely a speed change.

It's picked automatically for NVIDIA compute capability < 6.1 on plain (non-MoE) matmuls; newer GPUs keep the stock path. `FRIEND_CUDA_NO_BITPLANES=1` forces the stock path for A/B comparisons. Code: `quantize_q8_1_bitplanes` (quantize.cu), `vec_dot_*_q8_1_bitplanes` (vecdotq.cuh), `mul_mat_vec_q_bitplanes` (mmvq.cu).

Measured on a GTX 970 (Bonsai-8B, greedy, 128 tokens):

| | Q1_0 before | Q1_0 after | PQ2_0 before | PQ2_0 after |
|---|---|---|---|---|
| generation, short context | 25-29 t/s | 63-72 t/s | 24-26 t/s | 46-48 t/s |
| generation, ~2k context | 23-26 t/s | 56-60 t/s | 23-24 t/s | 42-43 t/s |
| prefill (~1.9k tokens, wall clock) | 164 t/s | 159 t/s | 159 t/s | 159 t/s |

Prefill goes through MMQ, which this doesn't touch (the difference is noise). Verification batches for speculative decoding (2-8 columns) also use the bit-plane dot product through the generic kernel; their matmuls got 1.1-1.7x faster in a mat-vec microbenchmark.

## Limits and not-yet-verified

A few things to be honest about:

- **CUDA and HIP**: CUDA has been built and run on a GTX 970 (Maxwell) with the Bonsai Q1_0 / PQ2_0 models; newer Nvidia architectures compile but haven't been run, and nobody has built or run HIP on AMD GPUs yet. Metal + CPU on Apple Silicon is the most tested path.
- **Vulkan**: shaders generate, but haven't been tested on real GPU hardware.
- **Recurrent models in batched mode**: turn-boundary checkpoints (the trick that makes cross-conversation reuse work for recurrent models) are not yet taken during continuous batching. They work fine in single-request mode. In batched mode, recurrent models only snapshot whole prompts, so cross-conversation reuse is limited.
- **DSpark drafting** is modest on Apple Silicon: verifying a draft on the 27B Q1_0 target costs ~60% of a normal token, so the ceiling is low, and on text the drafter predicts badly it can be slower than no drafter. Only standalone `arch=dspark` was exercised; MTP / DFlash / DSpark-in-DFlash paths are unchanged but untested here.
