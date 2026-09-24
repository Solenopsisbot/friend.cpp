# vLLM-inspired serving implementation goal

Status: active. Requested 2026-09-24. Commit coherent implementation checkpoints
as work proceeds. All 16 items below are in scope; custom logits processors remain
excluded under the earlier instruction. Item 8 in this list is beam search and is
included.

## Baseline and constraints

The starting checkpoint is `86e3bb7c8`. Existing features include native profile
lanes, continuous batching, decode-first chunked prefill, priority preemption,
compressed host snapshots, tiered prompt caching, logical KV block hashes,
sequence-cell prefix sharing, adapter profiles, suffix/ngram speculation,
request-local logprobs, aggregate metrics and per-lane occupancy gauges.

These are foundations, not evidence that the new scope is complete. Audit the
llama/ggml memory and execution implementations before replacing abstractions:
logical hashes are not a new physical allocator, separate lane threads are not
proof of CPU/device overlap, and profile isolation is not mixed-LoRA batching.
Existing model-based speculation must also be audited before adding another path.

Performance gains require measurements. Start with local CPU/Metal validation;
record CUDA, multi-device and cross-host validation separately when the required
hardware is available. Do not assume ongoing access to borrowed machines. An
unimplemented backend must fail clearly rather than silently use a stub.

## Deliverables and completion evidence

All entries start unchecked pending an implementation audit and the acceptance
checks below. Update each entry with commit IDs, executed tests, measurements,
supported backends and remaining limitations as it progresses.

| Done | Item | Required implementation and evidence |
|---|---|---|
| [~] | 1. Admission control and KV watermarks | Queue cap, retryable overload response, rejection counter, estimated token watermark and stall counter are implemented. Physical-capacity accounting plus cancellation/recovery/starvation coverage remain. |
| [~] | 2. Physical paged KV management | Refcounted scheduler-owned physical page identities, LRU eviction metadata, live/retained ownership and page metrics are integrated. Backend tensor paging, true copy-on-write pages and recomputation under physical exhaustion remain. |
| [~] | 3. State-aware data-parallel routing | Router health cooldowns plus fresh running/waiting/KV samples and EWMA probe latency now influence routing; draining excludes a worker from new assignments while preserving owners, with unit/integration coverage. Owner controls under state races and no-replay-after-submit coverage remain. |
| [~] | 4. Asynchronous scheduling | Native lanes already own independent llama batches/contexts, release the scheduler mutex during device decode, and apply pause/abort controls at completed-round boundaries; the two-lane live regression exercises overlap. A prefetch queue with explicit backend completion fences and device-idle measurements remains. |
| [~] | 5. Configurable stream batching | `stream_interval` buffering, final/error flushes and live SSE coverage are implemented. UTF-8/reasoning/tool/logprob interaction and streamed-vs-unbatched text equivalence remain. |
| [~] | 6. Multi-LoRA batching | Profile-grouped execution now runs concurrently across native context lanes, with `--max-lora-profiles` bounded residency/admission and a rejection metric. True per-sequence adapter selection inside one `llama_decode` remains unavailable because llama's public adapter API is context-wide; mixed-batch equivalence and throughput measurements remain. |
| [~] | 7. Parallel sampling | OpenAI `n` now fans out bounded independent child requests into separate executor threads, so all children enter native continuous batching concurrently; nested request state is isolated, fixed seeds are offset, indexed choices and usage are preserved, native rejection cannot fall back to the singleton generator, failed children abort admitted siblings, and OpenAI streaming now fans in indexed native token cursors. Focused overlap/rejection/streaming tests and the live regression cover the path. A native branch object remains. |
| [ ] | 8. Beam search | Deliberately not approximated with text replay: requests now fail clearly instead of silently becoming ordinary sampling (`2e08e37fb`). llama's public API does not expose a clonable sampler/sequence branch with grammar/EOS state, so a correct shared-KV implementation still requires native branch ownership. |
| [~] | 9. Stronger speculative decoding | The legacy/native path already integrates separate draft models, built-in MTP, DFlash and DSpark with rollback checks; `friend/spec_tuner.hpp` adapts draft length from acceptance and measured draft/verify cost, and the live regression verifies suffix speculation/output equivalence. Batch-native model-draft scheduling, EAGLE-specific validation and per-request draft budgets remain. |
| [~] | 10. Disaggregated prefill/decode | `--disaggregated-prefill` now evaluates a request on lane 0, samples its first pending token, serializes the live llama sequence state, and restores it on a decode lane with ownership transfer and cleanup. The connector carries bounded serialized payloads. Independently configured processes, device KV tensor attachment, network transport and transfer-overhead measurements remain. |
| [~] | 11. Expanded structured outputs | Cached JSON Schema/object handling accepts vLLM-style structured/guided JSON, choices and a bounded guided-regex subset, with live choice and regex constraint coverage. Backend selection and concurrent/speculative/streaming interaction coverage remain. |
| [~] | 12. Per-request timing data | Batch results now expose queue, prefill, TTFT, decode and total timing fields, streaming metadata, and pause duration/count plus preemption count; live pause/resume coverage checks lifecycle fields. Cancellation and cross-worker timing correlation remain. |
| [~] | 13. Richer cache metrics | Per-lane occupancy plus physical page capacity/usage, query/hit/eviction counters and watermark/rejection counters are exposed and live-checked. Recompute/transfer metrics and controlled pressure accounting remain. |
| [~] | 14. Multimodal encoder caching | Bounded eight-entry reuse of deep-copied mtmd image/audio chunks keyed by content, encoder identity and preprocessing size, with teardown invalidation and Prometheus hit/miss/eviction gauges. A live vision/audio model test and byte-accurate memory accounting remain. |
| [~] | 15. KV connector/event APIs | `friend/kv_connector.hpp` now provides versioned discovery, compatibility-checked export/import, bounded serialized payload transfer, corruption/conflict rejection, refcount-safe invalidation and event callbacks over scheduler-owned pages, with focused tests. A live network connector and backend tensor integration remain. |
| [ ] | 16. Expert and context parallelism | Real expert placement/routing and context/KV sharding with explicit device groups and collective communication. Validate single-device equivalence, cross-device correctness and communication cost; do not count layer splitting or replicated workers as these features. |

## Implementation order

1. Admission control, timing/cache telemetry, stream batching and state-aware
   routing (1, 12, 13, 5, 3). Audit the existing circuit-breaker success path and
   strengthen failure/recovery tests before relying on its state.
2. Physical KV ownership and connector/event contracts (2, 15). Keep contracts
   consistent with backend layouts and existing quantized/hybrid cache behavior.
3. Async scheduling and mixed-LoRA execution (4, 6), with benchmarks covering
   short and long prompts, concurrent personas and uneven request lengths.
4. Shared-KV branching, beam search, model-based speculation and structured
   outputs (7, 8, 9, 11), checking interactions rather than isolated happy paths.
5. Multimodal encoder reuse, prefill/decode separation, expert parallelism and
   context parallelism (14, 10, 16). Use the connector and ownership foundations;
   maintain a backend/hardware support matrix with actual execution evidence.

## Validation and reporting

Each checkpoint should include focused correctness checks, relevant builds and
live serving tests where available. Performance changes need reproducible
concurrency sweeps reporting throughput, TTFT/ITL percentiles and memory use;
do not infer performance from nonempty output or thread count. Preserve seeded
behavior where promised and use appropriate distribution checks for stochastic
speculation rather than assuming bitwise equality.

Keep the goal active until all deliverables are integrated and required validation
is complete. Document partial work and unavailable hardware explicitly; a passing
unit test for a standalone abstraction is not completion of its serving feature.
