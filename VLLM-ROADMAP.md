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
| [~] | 1. Admission control and KV watermarks | Queue cap, retryable overload response, rejection counter and saturation regression are implemented. KV headroom/watermark admission and cancellation/recovery/starvation coverage remain. |
| [ ] | 2. Physical paged KV management | Backend-integrated physical allocation, block references, shared-prefix ownership, copy-on-write, eviction and recomputation. Reuse existing llama primitives where suitable; verify divergent sequences, reclamation, memory bounds and output equivalence under pressure. |
| [ ] | 3. State-aware data-parallel routing | Balance measured worker backlog, KV pressure, latency and cache/profile affinity using bounded, freshness-aware telemetry. Verify failure cooldown/reset, recovery, stale stats, owner controls and no replay after submission. |
| [ ] | 4. Asynchronous scheduling | Prepare future work while execution proceeds, with explicit buffer ownership and completion fences. Validate cancellation and adapter changes under overlap, and measure device idle time and latency. |
| [ ] | 5. Configurable stream batching | Configurable token interval with prompt final/error flushes and correct UTF-8, reasoning/tool parsing, usage and logprob delivery. Compare streamed text with the unbatched reference. |
| [ ] | 6. Multi-LoRA batching | Per-sequence adapter selection within a single model batch, with bounded adapter residency and a max-LoRA policy. Compare mixed batches against isolated reference runs and measure throughput/memory. |
| [ ] | 7. Parallel sampling | Multiple independent sampled completions from shared prompt KV, separate RNG/sampler state, indexed streaming choices and correct usage/cancellation. Verify divergent branches do not corrupt their siblings. |
| [ ] | 8. Beam search | Explicit beam scoring, pruning, finish/length handling and shared KV branch ownership. Validate against an exhaustive small reference and verify memory reclamation. |
| [ ] | 9. Stronger speculative decoding | Integrate supported model-based draft/MTP/EAGLE-style paths with batch scheduling, adaptive draft lengths, acceptance/cost feedback and request budgets. Document model/backend requirements and validate rollback and sampling correctness. |
| [ ] | 10. Disaggregated prefill/decode | Independently configured prefill and decode workers with actual KV handoff, compatibility checks, ownership/lifetime handling and cancellation/failure cleanup. Verify output equivalence and report transfer overhead, TTFT and ITL. |
| [ ] | 11. Expanded structured outputs | Consistent JSON Schema, regex, choices and grammar constraints, reusable compilation, backend selection and explicit unsupported-feature errors. Verify constraints during concurrent, speculative and streaming generation. |
| [ ] | 12. Per-request timing data | Expose well-defined queue, prefill, first-token, decode and inter-token timings with request correlation, including streaming completion metadata. Check lifecycle accounting across pause/preemption. |
| [ ] | 13. Richer cache metrics | Capacity, physical occupancy, cache lookup/hit counters, eviction, recomputation and transfer metrics with bounded labels. Check counters against controlled sharing/pressure workloads. |
| [ ] | 14. Multimodal encoder caching | Bounded reuse of image/audio encoder results keyed by input content, preprocessing and encoder identity. Verify invalidation, concurrent reuse, memory bounds and output equivalence. |
| [ ] | 15. KV connector/event APIs | Versioned export/import, discovery, invalidation and event interfaces with model/layout/dtype/profile/cache-namespace compatibility and safe transfer lifetimes. Supply a working connector and exercise cancellation, corruption and incompatible peers. |
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
