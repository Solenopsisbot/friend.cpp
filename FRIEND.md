# friend.cpp

A fully independent downstream fork of [koboldcpp](https://github.com/LostRuins/koboldcpp) (v1.120+), which is itself built on llama.cpp. friend.cpp also merges [PrismML's llama.cpp fork](https://github.com/prism-ml), bringing in Bonsai 1-bit and ternary GGUF types alongside the full koboldcpp feature set.

It's a general-purpose local LLM server first. A lot of the design pressure comes from chat personas -- servers that rotate between many characters need fast context switching, cheap per-character customisation, and sampling that doesn't derail mid-conversation -- but everything here is useful for ordinary serving too.

Maintained by [Solenopsisbot](https://github.com/Solenopsisbot). Everything friend.cpp adds to the upstream koboldcpp codebase is marked with `friend.cpp:` comments in the source, and new modules live under `friend/` (notably `adapters.hpp` and `prompt_cache.hpp`).


## What's in the fork

From **koboldcpp upstream**: the full feature set -- all GGUF model support, image/video/audio generation, vision, the bundled KoboldAI Lite UI, all the compatible API endpoints, samplers, everything in the upstream README.

From **PrismML's llama.cpp**: Bonsai 1-bit/ternary GGUF quantisation types. Q1_0 is upstream llama.cpp; PQ2_0 and PTQ1_0 are Prism-only types (you'll find models like `prism-ml/Ternary-Bonsai-*-gguf` on HuggingFace). These come with CPU, Metal, CUDA, and Vulkan compute kernels, plus Hadamard weight folding and Qwen3.5 decode speedups.

**Verified on**: Apple Silicon (Metal + CPU); CUDA and Vulkan on a GTX 970 (Maxwell) with the Bonsai models (Q1_0, PQ2_0, PTQ1_0).
**Present but not yet compiled/tested**: HIP. CUDA for newer Nvidia architectures compiles but hasn't been run. Vulkan has only run on that one NVIDIA card, so its coopmat / integer-dot paths are unexercised.

friend.cpp-specific features, described below:

1. **Blue-noise sampling** -- anti-correlated randomness: fewer streaks for the model to amplify into loops (context collapse) or derailing
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
  --head-pool rook=heads/rook.head.gguf mira=heads/mira.head.gguf \
  --cvec-pool happy=cvecs/happy.gguf anxious=cvecs/anxious.gguf \
  --cache-dir ./cache \
  --cache-ram 2048 \
  --cache-disk 20480 \
  --parallelrequests 4 \
  --port 5001
```

### Warm the persona cards

Before any users connect, pre-fill and pin each persona's system prompt so the first real request is instant. Warm with the **same LoRA/steering set** the persona will chat with: those change the KV cache, so the cache keys on them (the head doesn't matter -- it never touches the KV):

```bash
curl -s http://localhost:5001/api/extra/cache/warm -d '{
  "prompt": "<|im_start|>system\nYou are Rook. [... full persona card ...]\n<|im_end|>",
  "head": "rook",
  "steer": {"happy": 3.0},
  "cache_pin": "rook-persona"
}'

curl -s http://localhost:5001/api/extra/cache/warm -d '{
  "prompt": "<|im_start|>system\nYou are Mira. [... full persona card ...]\n<|im_end|>",
  "head": "mira",
  "cache_pin": "mira-persona"
}'
```

### Chat

```bash
curl http://localhost:5001/v1/chat/completions -d '{
  "model": "whatever",
  "messages": [
    {"role": "system", "content": "You are Rook. [... persona card ...]"},
    {"role": "user", "content": "hey rook, how are you?"}
  ],
  "head": "rook",
  "steer": {"happy": 3.0},
  "temperature": 1.0,
  "blue_noise": true
}'
```

The server will match the system prompt against the pinned cache entry, skip prefilling those ~1000 tokens, apply Rook's LM head (cache stays valid -- head swaps are nearly free), steer the hidden states toward "happy," and sample with blue-noise anti-correlation. The whole thing takes a fraction of the time a cold start would.


---


## Blue-noise sampling

### The problem it solves

Standard sampling draws each token independently. That's mathematically fine on average, but by chance you get streaks: several rolls in a row landing at the same end of the distribution. A streak doesn't stay local, because the model conditions on its own output and amplifies it. A run of max-probability picks makes the text more predictable, which makes the next max-probability pick likelier still, until the generation locks into a loop (context collapse). A run of lowest-probability picks derails it the same way in the other direction. The averages are correct; the streaks are the problem.

The effect is strongest where nothing pulls the model back on track: base models and long generations. kaetemi, whose branch this sampler comes from, reports a large reduction in context collapse on base models with blue noise, and about a 1% improvement on reasoning-model output in preliminary benchmarks.

### How it works

Every sampled token consumes a uniform random roll. Blue-noise sampling keeps each individual roll uniform (so sampling remains unbiased -- the distribution of any single token is unchanged) but **anti-correlates consecutive rolls**. If the last roll was low, the next one is nudged away from also being low, and the same for high. The result is fewer streaks at either end, without changing what the model thinks is likely.

### Request fields

Works on `/api/v1/generate`, the OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`), and in the web UI settings panel.

| Field | Type | Default | Notes |
|---|---|---|---|
| `blue_noise` | bool | `false` | Enable blue-noise anti-correlation. |
| `rng_type` | string | `"mt19937"` | Base RNG. Options: `"mt19937"`, `"lowbias32"`. |

### Measured results

**The mechanism** -- `tools/friend-eval/blue_noise_eval.py`, Ternary-Bonsai-1.7B (instruct), 8 chat prompts, 10 seeds each, 160 tokens, T=1.0, min_p=0.02:

| Metric | Baseline | Blue noise | Change |
|---|---|---|---|
| Roll lag-1 autocorrelation | +0.01 | -0.18 | t = -16.8 |
| Longest run of tail picks (p < 0.1) | 2.34 | 1.88 | t = -5.0 |
| Windowed surprisal variance | 0.111 | 0.075 | -33%, t = -5.1 |
| Mean surprisal | 0.971 | 0.987 | unchanged, t = +0.8 |
| Cross-seed diversity | -- | -- | slightly better, t = -2 |
| Repetition | -- | -- | not significantly changed |

The important line: mean surprisal is unchanged (the model's calibration is preserved), but surprisal variance drops by a third (generation is smoother). Short replies from an instruct model are the wrong place to look for collapse, though: they rarely run long enough to lock in. That's what the next test is for.

**Context collapse** -- `tools/friend-eval/collapse_eval.py`, Qwen3-1.7B-Base (Q4_K_M), 12 raw document openings (fiction, encyclopedia, forum post, code, recipe, news, ...) x 16 seeds, 768 tokens, EOS banned, no repetition penalty. The same (prompt, seed) runs with white and with blue noise, 192 pairs per setting. A generation counts as collapsed when it locks into repetition and never recovers (every 400-character window from some point on repeats more than half its 20-character spans); the continuous measures are the final window's repeated fraction and the zlib compression ratio of the whole text (higher = less repetitive).

| | T=0.7 white | T=0.7 blue | T=1.0, min_p 0.05 white | blue |
|---|---|---|---|---|
| Collapsed | 24.5% | 19.8% | 12.0% | 10.9% |
| Mean onset of collapse (chars) | 1468 | 1589 | 1596 | 1919 |
| Final-window repeated fraction | 0.269 | 0.211 (paired t = -2.1) | 0.157 | 0.148 (t = -0.4) |
| Compression ratio | 0.251 | 0.273 (paired t = +2.3) | 0.296 | 0.309 (t = +1.7) |

Every measure moves in blue noise's favour in both settings. At T=0.7 -- where this base model loops most -- the continuous measures are significant on their own, and about a fifth fewer generations collapse (31 pairs where only white collapsed vs 22 where only blue did; sign test p = 0.27, so the binary count alone isn't conclusive at n = 192). The effect is smaller with min_p truncation at T=1.0. This is the direction kaetemi reports, at a more modest size than "large" on this model and length; longer generations, where collapse has more room to compound, are the obvious next test.

```bash
# base model, server with --parallelrequests 8 and enough context for 8 x (prompt + 768)
python tools/friend-eval/collapse_eval.py --url http://localhost:5001 \
  --seeds 16 --max-tokens 768 --temperature 0.7 --workers 8 --out collapse.json
```

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
{"lora": {"rook": 0.8, "sleepy": 0.3}}

// array of names (scale defaults to 1.0)
{"lora": ["rook"]}

// array of objects
{"lora": [{"name": "rook", "scale": 0.8}]}

// single string
{"lora": "rook"}

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
{"head": "rook"}
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
  "lora": [{"name": "rook", "default_scale": 1.0}, ...],
  "cvec": [{"name": "happy"}, ...],
  "head": [{"name": "rook"}, ...]
}
```

### Cache behaviour

This matters for performance, so pay attention to it:

- **LoRA or steering vector changes invalidate the KV cache.** The prompt cache keys on the adapter set, so each unique combination gets its own cached prefixes. Switching adapters means reprocessing the prompt (or restoring a cache snapshot for that adapter set).
- **Head swaps do NOT invalidate the KV cache.** The head only affects the final projection after the transformer stack, so switching heads per request is nearly free. This is why head-only fine-tunes are the recommended approach for per-persona customisation.

### Building steering vectors

You don't need an external tool: friend.cpp builds control vectors from contrastive examples on the model it already has loaded. Give it prompts where the model *is* the thing (an excited Rook) and prompts where it isn't (a flat Rook), ideally in pairs that differ only in that one respect:

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

With `--parallelrequests N` (which turns context shifting off automatically), requests are grouped by adapter profile per decode step. The worker sticks with a profile while it has work queued for it, then switches to a starved profile after 8 decode rounds.

### What's been verified

- Each adapter kind (LoRA, steering, head) changes greedy output and returns exactly to base output when removed.
- A copy of the model's own head reproduces base output bit-for-bit.
- Head-only switches don't trigger prompt reprocessing.
- Concurrent mixed-profile requests match serial results.

### LoRAs on Bonsai models

LoRAs load in any type the file carries (f32, f16, bf16, q8_0) on the 1-bit and ternary Bonsai models, including the Hadamard-folded Ternary-Bonsai-2. A LoRA is applied to the model's *unrotated* activation, so on a folded model `W_folded * H * x + B * A * x = (W + B * A) * x` -- the same result as the LoRA on the unfolded model. Checked on Ternary-Bonsai-2-27B PQ2_0 with an fp16 adapter on `attn_qkv`, `attn_q`, `ffn_gate` and the two folded-with-extras tensors, `ssm_out` (head permutation) and `ffn_down`: it loads, scale 0 and "no LoRA" give byte-identical output, larger scales move the output progressively, and dropping it returns exactly to base.

Train against unrotated weights and convert with `convert_lora_to_gguf.py --base <that checkpoint>`. For Bonsai 1, PrismML publishes them (`prism-ml/Bonsai-27B-unpacked`, `prism-ml/Ternary-Bonsai-27B-unpacked`). For Ternary-Bonsai-2 there's no unpacked release yet, and its F16 GGUF is folded too (`prism.hadamard.*` metadata), so it isn't a training base as-is; the rotation, signs and head permutation it records would have to be undone first.


---


## Tiered prompt cache

On by default. Replaces koboldcpp's SmartCache automatic slot switching. (The admin save/load state endpoints still work.)

### Why it exists

Persona servers constantly rotate between characters. Without caching, every time you switch from Rook to Mira and back, you re-prefill the entire persona card -- easily 1000+ tokens. The tiered cache snapshots KV state to RAM and disk so that switching back is nearly instant, and the disk tier survives server restarts.

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

- **In idle time** (default): `--cache-idle-ms` (300) after a generation finishes, if no new request has started, a background thread snapshots the live context. So when you do switch conversations, the old one is already stored and the switch pays nothing for it. It never competes with batched work (it skips if anything batched is live). `--cache-idle-ms 0` turns it off.
- When the live context is about to be discarded (switching to a different conversation) and wasn't already captured in idle time.
- After a large prefill (at least `--cache-capture-tokens` new tokens), but only when idle snapshots are off -- otherwise the idle snapshot of prompt + response subsumes it, and skipping it keeps the copy off the request's critical path.
- At turn-boundary checkpoints (recurrent models, single-user and batched).
- When explicitly pinned via the API.

Snapshots that are a prefix of a longer snapshot are deduplicated. Eviction is LRU within each tier's budget. Pinned entries are never evicted.

### Compression

Snapshots are compressed losslessly (`friend/state_codec.hpp`): each 4-byte group is split into four byte planes, which lines up the float exponent bytes, then each 4 MiB chunk goes through zstd level 1 with a checksum. A background worker does this after capture, so capturing never waits for it; until the worker gets to a snapshot it sits in RAM uncompressed and isn't charged against the RAM budget. Disk always stores the compressed form. RAM keeps it compressed too when that saves at least 5%. Both budgets count compressed bytes. A restore decompresses on up to 8 threads and is bit-exact. A corrupt chunk fails its checksum, the entry is dropped, and the prompt gets reprocessed.

Measured on M5 (single-thread pack, 8-thread unpack):

| Snapshot | Raw | Stored | Ratio | Pack | Unpack |
|---|---|---|---|---|---|
| Bonsai-1.7B, 1522 tokens, f16 KV | 166.5 MiB | 139.0 MiB | 1.20x | ~2.0 GB/s (84 ms) | ~16 GB/s (11 ms) |
| Qwen3.5-0.8B hybrid, 1525 tokens | 37.2 MiB | 31.6 MiB | 1.18x | ~1.9 GB/s (20 ms) | ~17 GB/s (2 ms) |

Plain zstd without the plane split only reaches 1.07-1.10x. LZ4 gains nothing (<1.01x) because KV bytes have skewed statistics, not repeats. zstd level 3 adds <0.5% ratio at half the speed. f16 KV tops out around 1.25x even in theory (the mantissa bytes carry ~7.8 bits of entropy each), so expect about 17% more snapshots per budget, not 2x. A server-side restore of the Bonsai snapshot takes ~25-35 ms from RAM (8 ms uncompressed) and ~50-120 ms from disk. It saves 0.35-0.45 s of prefill on that model, and far more on bigger ones.

zstd 1.5.7 is vendored as a single file in `vendor/zstd/` (BSD license, `LICENSE` alongside). It was built with `build/single_file_libs/combine.py` from `zstd-in.c`, with the dictionary builder and `ZSTD_MULTITHREAD` removed.

### Disk tier safety

The disk tier is scoped by a fingerprint of the model file(s) plus the KV cache layout (quantised KV type, flash attention, sliding-window attention mode, MTP). If you change any of those settings, the server won't load incompatible snapshots -- it just starts fresh.

On-disk format v2 stores compressed snapshots. v1 directories (uncompressed) still load and restore as they are, and their entries age out under the disk budget like any others. The loader skips a `.kv` file whose size doesn't match its `.meta`.

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
      "ram_bytes": 11583410,
      "disk_bytes": 11583410,
      "resident": true,
      "on_disk": true,
      "pinned": true,
      "label": "rook-persona",
      "adapters": {},
      "idle": 12.5
    }
  ]
}
```

Per item, `bytes` is the uncompressed state size. `ram_bytes` and `disk_bytes` are what the entry actually takes in each tier (compressed, or 0 if it isn't in that tier).

**Warm and pin a prompt:**

```bash
curl http://localhost:5001/api/extra/cache/warm -d '{
  "prompt": "... the full prompt to prefill ...",
  "head": "rook",
  "cache_pin": "rook-persona"
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
- On recurrent/hybrid models, batched prefills stop at each planned turn boundary and snapshot it, exactly like single-user mode, so a batched request can resume from another conversation's persona checkpoint.

More requests are batchable than in stock koboldcpp: grammar (except `grammar_retain_state`), DRY, XTC, top-nσ, mirostat 1/2 and dynamic temperature now run in batch mode through llama's samplers. Still single-request only: banned strings (they need kobold's rewind), top-a, TFS, smoothing, adaptive-P, custom sampler orders, reasoning budgets, media, guidance and draft models.

### Measured performance

On Apple M5, with persona prompts around 1,050-1,100 tokens:

- Returning to a persona after visiting another: restores 1057 of 1074 tokens, prefill drops from 0.35s to 0.06s.
- After a full server restart with `--cache-dir`: snapshot restored from disk, same speed.
- Batched follow-up requests: reuse 1057-1060 tokens from the first request's cached state.
- Qwen3.5-0.8B (recurrent): restores the 1094-token persona checkpoint correctly, in single-user and batched mode.
- Outputs are identical to a cache-disabled run in every tested scenario.


---



## Speculative decoding with DSpark drafters

PrismML ships standalone DSpark drafters next to their Bonsai models (e.g. `prism-ml/Bonsai-27B-gguf` has `Bonsai-27B-dspark-Q4_1.gguf`). Load one like any draft model:

```bash
python koboldcpp.py --model Bonsai-27B-Q1_0.gguf \
  --draftmodel Bonsai-27B-dspark-Q4_1.gguf --draftamount 6
```

- The drafter reads hidden states from specific target layers, which currently only the Qwen3.5/3.6 family (`qwen35` arch) can provide. Other targets fail at load with a clear error rather than mid-generation.
- **The draft length adapts on its own.** `--draftamount` is now the *maximum*: every round friend.cpp picks the length (0 = don't draft) that maximises expected tokens per millisecond, from live measurements of acceptance rate, drafter time and verify time. Fixed drafting is a trap on unpredictable text. On an M5 with the 27B Q1_0 target and the small-batch kernels below (quiet GPU, tokens/s, no drafter / fixed 3 / adaptive): counting 21.7 / 39.0 / 36.6, code 21.6 / 36.6 / 35.7, surreal haiku 16.4 / 14.7 / 16.3, persona chat 21.5 / 15.8 / 22.0. Adaptive gives up ~7% on highly predictable text (exploration) and never falls below no-drafter; fixed loses ~25% on chat. `--draft-fixed` restores always-draft-`--draftamount` behaviour; `FRIEND_DEBUG_SPEC=1` prints each round's decision.
- `LLAMA_DSPARK_SHARED_HEAD=1` makes the drafter borrow the target's LM head (saves ~700 MB).
- It survives prompt reuse: fast-forward, context shifts, SmartCache slots and the prompt cache all keep the drafter's captured features in step with the target KV, so drafting keeps working on later turns and after switching conversations. Outputs match the target alone.

### Small-batch Bonsai kernels (Metal)

Verify batches (draft + 1 tokens) and continuous batching run 2-32 tokens through every weight matrix at once. On Apple GPUs the stock Q1_0 / PQ2_0 / PTQ1_0 kernels were ALU-bound at those sizes, so every extra token cost 60-80% of a whole single-token pass. friend.cpp adds a lookup-table mat-vec for Q1_0 and PQ2_0 and a decode-once multi-column kernel for PTQ1_0 (details in the friend.cpp section of `ggml/src/ggml-metal/kernels/mul_mv.metal`). Bonsai-27B Q1_0 on an M5, one batch of n tokens:

| n | 1 | 2 | 3 | 4 | 5 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| stock (ms) | 42.5 | 57.1 | 87.6 | 101.3 | 144.0 | 190.6 | 377.6 |
| friend.cpp (ms) | 42.4 | 52.2 | 65.2 | 65.9 | 82.2 | 100.2 | 179.2 |

Single-token decode and prefill are unchanged. Batched logits agree with token-by-token decode to f32 rounding (argmax identical; greedy output identical with and without a drafter). `GGML_METAL_BONSAI_SB_DISABLE=1` restores the stock kernels for A/B runs.

### Latency-bound decode on big Apple GPUs (Metal)

On an M3 Ultra, Ternary-Bonsai-2-27B PQ2_0 decoded at ~45 tok/s, streaming 7.2 GB per token at ~300 GB/s on a chip with ~800. The weight matmuls weren't the problem: timed on their own they run at 640-710 GB/s, about 10 ms of the 22 ms token. The rest was the graph's dependency chain. A token was ~1350 memory barriers, and on that GPU every dependent step costs a few microseconds of drain and refill however little work it does (measured in the real graph by dropping op types: 4-9 us each for norms, scales, swiglu, copies). Batching doesn't hide it at batch 1, so the fix is fewer dependent steps:

- **Hadamard inputs in one kernel.** Every Hadamard-folded matmul input was 2-4 launches: `[ADD] -> RMS_NORM -> weight -> sign -> FWHT` on the input side, `[per-head gated RMS_NORM] -> SWIGLU -> [head permutation] -> sign -> FWHT` on the output side of the FFN and GDN. Two kernels (`kernel_norm_fwht`, `kernel_glu_fwht`) do each chain in one launch, one threadgroup per 1024-wide block with every load issued up front. They read a whole row while writing a block, so the graph lists the chain's inputs as extra sources of the transform's matmul (`llama_hadamard_keepalive`) to stop the allocator placing the output over them.
- **Decode conv step in one kernel**: concat + state write-back + conv + silu.
- **GDN reads its state from the cache** for single-sequence batches (before, every layer gathered its full 3 MB state first), and **l2-normalises q/k itself** (`ggml_gated_delta_net_set_qk_l2`, CPU and Metal).
- **An upstream off-by-one** in `ggml_mem_ranges_check` treated back-to-back buffers as overlapping, planting false barriers in every Metal graph (e.g. `ffn_up` serialised behind `ffn_gate`).

Result: 1772 -> 1076 launches and 1348 -> 793 barriers per token. Logits within 1.1e-4 relative of the unfused path, argmax identical, sequential and batched. The test GPU was shared, so the numbers are best-of samples: up to 55 tok/s in quiet windows (was 45), and +15% interleaved under the same background load (37.4 vs 32.6). Prefill unchanged.

Knobs (each restores the old path): `GGML_METAL_NORM_FWHT_DISABLE`, `GGML_METAL_GLU_FWHT_DISABLE`, `LLAMA_HADAMARD_KEEPALIVE_DISABLE`, `GGML_METAL_CONV_STEP_DISABLE`, `GGML_GDN_STATE_GATHER`, `GGML_GDN_QK_L2_UNFOLD`. For finding the next barrier: `GGML_METAL_TIMING=1` logs encode vs GPU time and a histogram of launched kernels; `=2` adds the encode order with each barrier tagged by the source line that placed it; `=3` logs which buffer forced each barrier. `GGML_METAL_SKIP_OPS=OP,...` drops ops (garbage output) to measure what they cost in the real graph.

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

Prefill doesn't use this path (the difference is noise; see below for what it does use). Verification batches for speculative decoding (2-8 columns) also use the bit-plane dot product through the generic kernel; their matmuls got 1.1-1.7x faster in a mat-vec microbenchmark.

### Prompt processing on Maxwell

It's tempting to apply the same trick to prompt processing, but on a GTX 970 prefill never touches MMQ. ggml doesn't pick MMQ for dense matmuls on pre-Pascal GPUs, and with no fast fp16 the batched path is "dequantize the weight matrix to f32, then `cublasSgemm`". Profiling one 512-token ubatch of Bonsai-8B (`tools/friend-bench/op-profile.cpp`): matmuls are 86-91% of the time, attention 5% on an empty cache and 14% after 1.5k tokens, everything else ~3%. The SGEMM runs at ~3.6 TFLOP/s, about 77% of the card's FP32 peak at the 1.4 GHz it holds when cool.

That's why multiply-free doesn't carry over. Maxwell does 128 FP32 FMAs per SM per clock but only 32 POPCs, and a bit-plane dot product costs 8 POPCs per 32 weights. So even a perfect bit-plane GEMM would only tie the FMA roof cuBLAS already reaches 77% of, and it would also switch the activations to q8_1 (MMQ numerics instead of f32). A lookup-table GEMM (Four Russians) is capped by shared-memory bandwidth at about 1.6x the FMA roof before its table-building cost, so it isn't worth the complexity either. Forcing MMQ (`FRIEND_CUDA_FORCE_MMQ=1`, a measurement knob) confirms the order: it's ~1.6x slower than cuBLAS for both types.

What did help was the dequant pass itself. The generic kernel wrote f32 at ~100 GB/s and cost ~9% of every prefill matmul. Dedicated Q1_0 / PQ2_0 kernels with float4 stores get the matmuls 2-4% faster, with bit-identical outputs (`FRIEND_CUDA_NO_FAST_DEQUANT=1` restores the generic kernel). The ~31 GB of f32 written per ubatch is the floor that's left.

Matmul time per 512-token ubatch (`tools/friend-bench/cuda-mm-bench.cpp`, GTX 970):

| | f32 weights (pure SGEMM) | stock dequant + cuBLAS | fast dequant + cuBLAS | forced MMQ |
|---|---|---|---|---|
| Q1_0 | 2102 ms | 2299-2340 ms | 2246-2261 ms | 3496 ms |
| PQ2_0 | 2102 ms | 2284-2286 ms | 2228-2233 ms | 3632 ms |

Whole 512-token ubatch with the GPU at its sustained ~1.3 GHz (`op-profile`, best of 3, before -> after):

| | empty cache | after 1536 cached tokens |
|---|---|---|
| Q1_0 | 180.3 -> 184.1 t/s | 161.7 -> 164.4 t/s |
| PQ2_0 | 180.5 -> 184.7 t/s | 161.7 -> 165.3 t/s |

A cold card does ~205-210 t/s on the first ubatch before it heats up to ~80 C and drops its clock. Greedy output is identical before and after. KoboldCpp's reported "process speed" overstates the first request after a long prompt: the last partial ubatch gets counted as generation time, so on a GTX 970 a fresh 1.9k-token prompt shows ~250 t/s prefill next to ~26 t/s generation, against ~62 t/s generation on the next request. Use `op-profile` for prefill numbers.

Still slow on this card: **PTQ1_0 decode through CUDA** (3.6 t/s on Ternary-Bonsai-8B vs 72 for Q1_0). Its mat-vec hands one 128-weight block to each thread, and on sm_52 that's latency-bound: 4096-wide rows run 3x slower per weight than 12288-wide ones. It needs a bit-plane or split-block kernel. On pre-dp4a GPUs, PQ2_0 is the ternary format to use for now.

## Vulkan

First run on real hardware: GTX 970, NVIDIA 580 driver (no fp16, no integer dot, no cooperative matrices, so every Bonsai matmul takes the scalar shaders). Build with `make LLAMA_VULKAN=1 koboldcpp_vulkan`. The Makefile uses the bundled `glslc-linux` unless a system `glslc` prints "glslang" in its version string, so the Arch shaderc package's `glslc` is skipped (the bundled one works). Shader changes only rebuild if you delete `vulkan-shaders-gen` and `ggml/src/ggml-vulkan-shaders.cpp`, because the generated file only depends on the generator's source.

What happened on the first run:

- **PQ2_0 had no Vulkan kernels at all**, so Ternary-Bonsai models ran every weight matrix on the CPU. It's now a standalone per-type shader (the Q2_0 codec at group 128, same as PTQ1_0's setup): mat-vec, matmul, dequant, get_rows.
- **PTQ1_0 was correct but 17x slower than Q1_0** in mat-vec (2.2 t/s generation), because each element did its own byte fetch, a divergent three-way branch and a loop. It now decodes four trits per 32-bit load in closed form and without branches: mat-vec 30 -> 223 GFLOPS, 512-column matmul 555 -> 907 GFLOPS, generation 2.2 -> 14 t/s.

`test-backend-ops` (MUL_MAT 309 cases, MUL_MAT_ID 225, GET_ROWS) passes against the CPU for q1_0 / pq2_0 / ptq1_0. Bonsai-8B family, fully offloaded (37/37 layers), 128 greedy tokens per prompt:

| | Q1_0 | PQ2_0 | PTQ1_0 |
|---|---|---|---|
| generation, short context (Vulkan / CUDA) | 32 / 72 t/s | 31.5 / 50 t/s | 14 / 3.6 t/s |
| prefill, 512-token ubatch (Vulkan / CUDA) | 71 / 184 t/s | ~70 / 185 t/s | -- |
| greedy output vs CPU | first ~60 tokens identical on the long prompt, all 128 on the short one | identical | identical |

PTQ1_0 was produced losslessly from the PQ2_0 file (`quantize_gguf --allow-requantize ... PTQ1_0`) and gives the same text as PQ2_0 on every backend. Q1_0's small divergence is float summation order. CUDA diverges from the CPU at about the same point, just on the other prompt. Vulkan prefill runs at a third of CUDA's because its generic scalar matmul shader reaches ~1 TFLOP/s on this card, against cuBLAS's ~3.6. Tuning that shader for pre-Turing NVIDIA is open work.

## Running under llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) starts and stops model servers on demand behind one OpenAI-compatible endpoint. friend.cpp works as a backend like llama-server, with one difference: its health endpoint is `/ping`, not `/health`, so set `checkEndpoint` (otherwise llama-swap waits out `healthCheckTimeout` and gives up).

```yaml
healthCheckTimeout: 300

models:
  "bonsai2-27b":
    cmd: |
      python3 /path/to/friend.cpp/koboldcpp.py
        --model /models/Ternary-Bonsai-2-27B-PQ2_0.gguf
        --gpulayers 99 --contextsize 32768 --flashattention
        --host 127.0.0.1 --port ${PORT}
        --skiplauncher --quiet
        --cache-dir /var/cache/friendcpp/bonsai2
    checkEndpoint: /ping
    ttl: 600

  "qwen3.5-0.8b":
    cmd: |
      python3 /path/to/friend.cpp/koboldcpp.py
        --model /models/Qwen3.5-0.8B-Q4_0.gguf
        --gpulayers 99 --host 127.0.0.1 --port ${PORT}
        --skiplauncher --quiet
    checkEndpoint: /ping
```

- The server answers only once the model is loaded, so `/ping` returning 200 means ready.
- It serves whichever model it loaded and ignores the request's `model` field; llama-swap's routing is what picks the model.
- A swap throws away the RAM tier of the prompt cache. `--cache-dir` (one directory per model) keeps a disk tier that survives the restart, so a swapped-back model resumes long system prompts instead of re-prefilling them.
- `--parallelrequests N` needs `--contextsize` large enough for N concurrent requests: the context is split between the slots.
- koboldcpp's own `--admin --routermode` also hot-swaps models from `.kcpps` configs, if you'd rather not run a proxy.

## Tools for maintaining the fork

### Upstream sync

`tools/friend-sync/sync.sh` merges koboldcpp and PrismML into a dated `sync/<date>` branch, applies the mechanical conflict rules learned from past syncs (keep kobold's deletions of llama.cpp tests/CI/CMake, drop Prism-only tests/tools), stops with notes at the first real conflict (`--continue` after resolving, `--abort` to bail), then checks unity-build coverage, builds, and runs the smoke tests. It never pushes. See `tools/friend-sync/README.md`.

### Smoke tests

`python tools/friend-sync/smoke.py` starts the server against each available test model (Bonsai-1.7B Q1_0, Ternary-Bonsai-1.7B PQ2_0, Qwen3.5-0.8B hybrid) and checks coherence, determinism, blue noise, head swap, steering build, prompt-cache reuse (output must equal a cache-disabled run), GPU-vs-CPU agreement and continuous batching (concurrent must equal serial, through the real batch worker). About a minute; non-zero exit on any failure.

### Sampler lab

`tools/friend-eval/sampler_lab.py --configs FILE.json` runs any number of sampler configurations (a JSON object of `{name: {request fields}}`) on the same prompts and seeds, interleaved, and reports derailment (tail picks, streaks, surprisal burstiness), repetition, diversity and length against a baseline with Welch t-values. `tools/friend-eval/configs/chat_presets.json` is a starting set.

First run (Ternary-Bonsai-1.7B, 8 persona prompts x 6 seeds, baseline T=1.0 + min_p 0.05; Welch t in brackets):

| config | longest tail streak | tail picks | 4-gram repetition | distinct bigrams |
|---|---|---|---|---|
| baseline | 2.00 | 0.102 | 0.0035 | 0.946 |
| blue noise | 1.46 (-4.1) | 0.097 | 0.0023 | 0.955 |
| blue noise, T=1.2 | 1.58 (-3.1) | 0.094 | 0.0048 | 0.954 |
| dynatemp 0.5 | 1.88 (-0.8) | 0.106 | 0.0028 | 0.951 |
| XTC 0.1/0.5 | 1.77 (-1.6) | 0.083 (-3.4) | 0.0017 | 0.959 (+2.5) |
| DRY 0.8 | 2.08 (+0.5) | 0.103 | 0.0020 | 0.953 |
| blue + XTC + DRY | 1.31 (-5.3) | 0.072 (-4.9) | 0.0005 (-2.6) | 0.967 (+3.9) |

What it supports: blue noise cuts derailing streaks by about a quarter, and even at T=1.2 it stays below plain sampling at T=1.0 -- roughly 0.2 of extra temperature (livelier text) for free. **Caveat:** the tail/surprisal metrics are measured on the *post-sampler* distribution, so samplers that reshape it (XTC removes top choices, which mechanically lowers measured surprisal) move those columns partly by construction. Blue noise only changes the roll, so its rows are clean; the XTC/DRY rows need a judged quality eval before anyone treats them as proof.

### Judged comparison

`tools/friend-eval/judge_lab.py` asks a second model which of two replies is better -- blind, pairwise against the baseline, order randomised, with the judge's probability read from its logprobs. First run: Ternary-Bonsai-1.7B generating, Bonsai-27B Q1_0 judging, 8 persona prompts x 4 seeds (32 judgments per config):

| config | preferred over baseline | 95% CI |
|---|---|---|
| blue noise | 0.41 | [0.25, 0.56] |
| blue noise, T=1.2 | 0.47 | [0.31, 0.63] |
| dynatemp 0.5 | 0.47 | [0.31, 0.63] |
| XTC 0.1/0.5 | 0.38 | [0.22, 0.56] |
| DRY 0.8 | 0.50 | [0.31, 0.66] |
| blue + XTC + DRY | 0.38 | [0.22, 0.56] |

No configuration beats plain T=1.0 + min_p 0.05 here: every interval includes 0.5. So on 160-token persona replies from an instruct model, the mechanical gains above (fewer derailing streaks, less repetition) don't show up as replies this judge prefers -- n=32 is small and a 1-bit judge is weak. This setup can't see context collapse, which needs long generations and shows up most on base models; see [blue-noise sampling](#blue-noise-sampling) for that test.

## Limits and not-yet-verified

A few things to be honest about:

- **CUDA**: built and run on a GTX 970 (Maxwell, CUDA 12.9) with the Bonsai Q1_0 / PQ2_0 models; the CUDA code also compiles for sm_61 through sm_120 but has only *run* on the 970. **HIP** has never been compiled. Metal + CPU on Apple Silicon is the most tested path.
- **Vulkan**: shaders generate, but haven't been tested on real GPU hardware.
- **DSpark drafting** is modest on Apple Silicon: verifying a 3-token draft on the 27B Q1_0 target costs ~1.5x a normal token (2.4x before the small-batch kernels above), so the ceiling is about 1.8x on predictable text; adaptive drafting keeps it from being slower than no drafter elsewhere. Only standalone `arch=dspark` was exercised; MTP / DFlash / DSpark-in-DFlash paths are unchanged but untested here.
