# vLLM-inspired serving implementation goal

Status: active. Requested 2026-09-24. Commit coherent implementation checkpoints
as work proceeds. All listed items except item 8 are in scope; custom logits
processors and beam search remain excluded under the earlier instruction.

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
| [~] | 1. Admission control and KV watermarks | Queue cap, retryable overload response, rejection counter, estimated token watermark and stall counter are implemented. Shared-lane admission now ages requests after eight scheduler rounds so priority traffic cannot starve older work; `/api/extra/requests/cancel` exposes safe native abort/recovery with cancellation timing and live regression coverage. Physical page pressure now stalls or fails admission instead of running without scheduler ownership. |
| [~] | 2. Physical paged KV management | Refcounted scheduler-owned physical page identities, LRU eviction metadata, live/retained ownership and page metrics are integrated; page acquisition failures now surface as `friend_batch_kv_page_stalls_total` and never enter backend execution without ownership. Backend tensor paging, true copy-on-write pages and recomputation under physical exhaustion remain. |
| [~] | 3. State-aware data-parallel routing | Router health cooldowns plus fresh running/waiting/KV samples and EWMA probe latency now influence routing; draining excludes a worker from new assignments while preserving owners, successful responses clear cooldowns, and health probes run concurrently across replicas. Unit/integration coverage includes owner controls, all-draining rejection and no-replay-after-submit. |
| [~] | 4. Asynchronous scheduling | Native lanes already own independent llama batches/contexts, release the scheduler mutex during device decode, and apply pause/abort controls at completed-round boundaries; the two-lane live regression exercises overlap. A prefetch queue with explicit backend completion fences and device-idle measurements remains. |
| [~] | 5. Configurable stream batching | `stream_interval` buffering, final/error flushes and live SSE coverage are implemented. Parallel native fan-in now drains completion races and preserves split UTF-8 tokens. Live regression covers constrained streaming with logprobs at multiple intervals; reasoning/tool interaction remains. |
| [~] | 6. Multi-LoRA batching | Profile-grouped execution now runs concurrently across native context lanes, with `--max-lora-profiles` bounded residency/admission and a rejection metric. True per-sequence adapter selection inside one `llama_decode` remains unavailable because llama's public adapter API is context-wide; mixed-batch equivalence and throughput measurements remain. |
| [~] | 7. Parallel sampling | OpenAI `n` now fans out bounded independent child requests into separate executor threads, so all children enter native continuous batching concurrently; nested request state is isolated, fixed seeds are offset, indexed choices and usage are preserved, native rejection cannot fall back to the singleton generator, failed children abort admitted siblings, and OpenAI streaming now fans in indexed native token cursors with disconnect cancellation, completion-tail draining and split-UTF-8 handling. Focused overlap/rejection/streaming tests and the live regression cover the path. A native branch object remains. |
| [ ] | 8. Beam search | Explicitly excluded per request. Requests fail clearly instead of silently becoming ordinary sampling (`2e08e37fb`). llama's public API does not expose a clonable sampler/sequence branch with grammar/EOS state, so a correct shared-KV implementation still requires native branch ownership. |
| [~] | 9. Stronger speculative decoding | The legacy/native path already integrates separate draft models, built-in MTP, DFlash and DSpark with rollback checks; `friend/spec_tuner.hpp` adapts draft length from acceptance and measured draft/verify cost, and the live regression verifies suffix speculation/output equivalence. Requests can now cap or disable history-based drafting with `speculative_tokens`; batch-native model-draft scheduling and EAGLE-specific validation remain. |
| [~] | 10. Disaggregated prefill/decode | `--disaggregated-prefill` now evaluates a request on lane 0, samples its first pending token, serializes the live llama sequence state, and restores it on a decode lane with ownership transfer and cleanup; in-process transfer bytes/count/failure metrics are exposed. The connector carries bounded serialized payloads, and `friend/kv_socket_connector.hpp` moves those snapshots over connected POSIX sockets using bounded checksummed framing with transfer counters. Independently configured serving processes and device KV tensor attachment remain. |
| [~] | 11. Expanded structured outputs | Cached JSON Schema/object handling accepts vLLM-style structured/guided JSON, choices and a bounded guided-regex subset, with live choice and regex constraint coverage. Backend selection and concurrent/speculative/streaming interaction coverage remain. |
| [~] | 12. Per-request timing data | Batch results now expose queue, prefill, TTFT, decode, end-to-end and total timing fields, streaming metadata, lane/request identity, cancellation latency, and pause duration/count plus preemption count; live pause/resume/cancel coverage checks lifecycle fields. The data-parallel router adds epoch/worker and queue/upstream-header latency headers for cross-worker correlation. |
| [~] | 13. Richer cache metrics | Per-lane occupancy plus physical page capacity/usage, query/hit/eviction counters, watermark/rejection counters, and in-process KV transfer bytes/count/failures are exposed and live-checked. Recompute and controlled pressure accounting remain. |
| [~] | 14. Multimodal encoder caching | Bounded eight-entry reuse now retains deep-copied mtmd chunks plus immutable encoded embeddings keyed by content, encoder identity and preprocessing size; cache hits skip the projector, teardown invalidates entries, and Prometheus exposes hit/miss/eviction/entry/encoded-byte gauges. A live vision/audio model test and accounting for opaque mtmd metadata remain. |
| [~] | 15. KV connector/event APIs | `friend/kv_connector.hpp` now provides versioned discovery, compatibility-checked export/import, bounded serialized payload transfer, corruption/conflict rejection, refcount-safe invalidation and event callbacks over scheduler-owned pages; `friend/kv_socket_connector.hpp` provides live POSIX-socket export/import over bounded checksummed frames and transfer statistics. Focused tests pass. Serving-process orchestration and backend tensor integration remain. |
| [ ] | 16. Expert and context parallelism | `/api/extra/capabilities` now reports these as unsupported with explicit reasons. Real expert placement/routing and context/KV sharding with explicit device groups and collective communication still require backend APIs; do not count layer splitting or replicated workers as these features. |

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
