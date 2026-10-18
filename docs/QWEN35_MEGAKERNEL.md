# Qwen3.5-0.8B fused CUDA backend

`optimizations/megakernel` carries the Lucebox fused Qwen3.5-0.8B forward
pass. It contains the single-dispatch decode kernel, chunk-parallel prefill,
and the Blackwell NVFP4 variants. The backend is intentionally optional: the
normal friend.cpp build and GGUF loader are unchanged.

## Build

Use a Python environment containing PyTorch with CUDA and Transformers, then
build the extension from this checkout:

```bash
cd optimizations/megakernel
python -m pip install -e . --no-build-isolation
```

`setup.py` detects the active CUDA device and compiles BF16 for Ampere and
newer cards, FP16 for Turing, and NVFP4 support for Blackwell. Override the
detected architecture and launch sizing with `MEGAKERNEL_CUDA_ARCH`,
`MEGAKERNEL_NUM_BLOCKS`, `MEGAKERNEL_BLOCK_SIZE`, `MEGAKERNEL_LM_NUM_BLOCKS`,
and `MEGAKERNEL_LM_BLOCK_SIZE` when reproducing a benchmark on a different
card. SM 7.5 is the minimum supported architecture because the implementation
uses cooperative grid barriers.

The default decode block is 1024 threads. On the RTX 3090 this gives each
full-attention query head twice as many warps for its long-context KV scan. A
512-thread build remains available with `MEGAKERNEL_BLOCK_SIZE=512` when that
geometry is preferable.

For the fastest quality-preserving decode on an Ampere RTX 3090, build the
mixed int8 path. It keeps prefill and recurrent constants in BF16, while
packing every decode projection (including the vocabulary head) as aligned
per-row int8 with a float scale:

```bash
MEGAKERNEL_LM_INT8=1 MEGAKERNEL_LAYER_INT8=1 \
MEGAKERNEL_LM_NUM_BLOCKS=8192 \
python setup.py build_ext --inplace
```

The int8 tables are created from the original Hugging Face weights at load
time; no ternary checkpoint is substituted and the prefill path still consumes
the unmodified BF16 tensors.

The prefill cuBLAS calls explicitly request `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for
the BF16 projection, QK, and value GEMMs. On Elysium this measures about
19.1k tok/s at pp302 and 29.1k tok/s at pp4096 while preserving the exact
handoff output.

On Ampere and newer GPUs, DeltaNet prefill uses the WMMA d/Q·state kernels by
default while keeping the recurrent state update in FP32. Set
`MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME=0` to force the scalar reference path when
debugging numerical differences. The FP32 state update uses a four-column
blocked loop by default; set `MEGAKERNEL_DN_PHASE2_VECTOR_UPDATE=0` to select
the scalar reference loop.
The phase-2 WMMA launch uses 512 threads by default on Ampere. On Elysium this
measured 5.70 ms for a 512-token prefill, versus 7.55 ms with 128 threads; a
1024-thread launch regressed because it cannot meet the kernel's two-block
occupancy target.
The causal-attention cuBLAS path uses 2048-token query tiles by default. On
Elysium this improved warmed prefill from 27.6k to 29.1k tok/s at 4k context
and from 17.6k to 18.0k tok/s at 32k. Override it with
`MEGAKERNEL_PREFILL_ATTN_BLOCK` when tuning another GPU.

`MEGAKERNEL_PREFILL_ATTN_BATCH=1` enables an experimental GQA-batched variant
that computes the four query heads sharing each KV head with strided batched
cuBLAS calls. It passes the handoff check, but the RTX 3090 measurements were
effectively neutral at 4k–16k context and the larger 4096-token tile regressed,
so the extra scratch allocation and path remain opt-in.

## Run

The fused backend loads the original Hugging Face checkpoint:

```bash
python optimizations/megakernel/run.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "The capital of France is" \
  --max-tokens 128
```

Pass `--max-seq-len 8192` when the prompt plus generation exceeds the default
2048-token cache.

Set `MEGAKERNEL_PREFILL_GRAPH=1` to capture and replay the BF16 prefill for
each prompt length. Captures are cached by prompt length. On Elysium this
raised a repeated pp302 prefill from 19.4k to 20.1k tok/s (about 3.6%); at
pp4096 the gain was smaller, 30.0k to 30.2k tok/s. The first call for a new
prompt length pays the graph-capture cost, so eager execution remains the
default for workloads with mostly unique prompt lengths.

For long contexts, build the split-KV attention variant:

```bash
MEGAKERNEL_LM_NUM_BLOCKS=8192 MEGAKERNEL_LM_INT8=1 \
MEGAKERNEL_LAYER_INT8=1 MEGAKERNEL_DECODE_SPLITKV=1 \
MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT=512 \
python setup.py build_ext --inplace
```

`MEGAKERNEL_DECODE_SPLITKV=1` makes every cooperative decode block scan a
disjoint portion of the KV cache, then merges attention with a stable one-pass
online-softmax reduction. The 512-token threshold keeps the original warp
schedule for short contexts, then enables the split scan once there is enough
KV work to amortize the cooperative reduction. On Elysium's RTX 3090 it
measured 534 tok/s at 4k, 482 tok/s at 16k, and 423 tok/s at 32k. A
50-prompt, 32-token corpus matched a freshly rebuilt non-split reference
exactly (50/50).

The extension exports its compile-time launch and packing configuration. The
Python loader checks it against the runtime `MEGAKERNEL_*` environment before
allocating buffers and raises a clear error instead of launching with
incompatible LM-head or layer packing.

For contexts beyond the default 2048-token cache, construct `Decoder` with a
larger `max_seq_len`; the KV cache grows with that value while the prefill
scratch workspace grows with the actual prompt. On Elysium, warmed direct
measurements reach about 26.5k pp4096, 21.3k pp16384, and 16.7k pp32768 tok/s.
The decline at long context is from the causal-attention layers scanning their
KV cache, rather than from the DeltaNet recurrence.

Quantised GGUF files use friend.cpp's production loader, which supports the
full sampler, prompt cache, recurrent checkpoints, batching, and CPU/CUDA/
Vulkan backends:

```bash
python optimizations/megakernel/run.py \
  --model /models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --backend quantized \
  --prompt "The capital of France is" \
  --max-tokens 128
```

GGUF execution stays on the existing friend.cpp path, preserving its sampler,
cache, recurrent-state, batching, and backend features. The megakernel int8
mode is a separate runtime packing step for the HF checkpoint; it does not
silently reinterpret a GGUF tensor or replace the production loader.

The Qwen3.5 QAT `TQ2_0` files are currently a CPU fallback on CUDA because
the CUDA MMQ backend does not yet implement that ternary format. Use the
BF16 megakernel for GPU throughput measurements; adding a native TQ2 CUDA
mat-vec is a separate optimization.

## Validation

`bench_pp_tg.py` checks prefill/decode handoff against a token-by-token
reference before timing. `final_bench.py` performs warmed prefill and decode
measurements. The benchmark scripts accept `--backend auto|bf16|nvfp4`.

On Elysium's RTX 3090, the final mixed BF16/INT8 build passes the handoff
check and measures 19.1k tok/s prefill and about 650 tok/s decode at
pp302/tg128.

Set `MEGAKERNEL_DECODE_PROFILE=1` to print the fused layer-kernel and
vocabulary-head timings for the first few decode steps. The final split build
decodes at about 534 tok/s at 4096 context, 482 tok/s at 16384, and 423 tok/s
at 32768 with the default 1024-thread block, vectorized BF16 pair-load path,
and four-way attention-loop unroll. The long-context gain comes from giving
each attention head more cooperative KV-scan work.

The remaining single-token bottleneck is the projection bandwidth: at short
context the Elysium profile is about 1.22 ms for the 24-layer fused kernel and
0.31 ms for the INT8 vocabulary head; at 4096 context the layer portion grows
to about 1.48 ms while the vocabulary head is unchanged. A decode-only
weight-only INT4 prototype was slower (about 614 versus 650 tok/s) and gave
poor generations because nibble unpacking outweighed the bandwidth reduction,
so it is not enabled. The quality-preserving Elysium build remains the mixed
BF16/INT8 configuration above.

The Elysium build selects 64 decode blocks automatically from 512 through
8192 tokens, then returns to the full 82-block grid for longer contexts. This
reduced the 4096-token layer phase from about 1.47 to 1.44 ms (roughly 1.5%
end-to-end); set `MEGAKERNEL_DECODE_CONTEXT_BLOCKS` and
`MEGAKERNEL_DECODE_CONTEXT_MAX` to override the policy. The layer INT8
projection uses BF16 pair loads on Ampere, which is a small additional win and
keeps the exact handoff result.

The Python decoder's LM reduction buffers are sized from the same 8192-block
default as the CUDA vocabulary kernel. The old 1024-slot fallback let the
8192-block kernel write out of bounds and could produce invalid token IDs at
long context even when short-context tests passed.

Larger experiments were measured and rejected. Splitting each DeltaNet head
across multiple decode blocks added grid barriers and slowed the layer phase;
changing the prefill recurrence geometry from 512 threads × four state slices
was slower at 256/1024 threads, eight slices, and 8-token chunks. An all-block
GQA KV-tile prototype was rejected after its staged shared-memory path failed
long-context correctness, so the validated split-KV path stays enabled. The
optional batched-GQA prefill attention path remains available for experiments
but is not the default because it is neutral at 4k and slower at 16k.

For long greedy generations, `Decoder.step_many(token_id, count)` now chains
token IDs on the CUDA stream and returns a device tensor, avoiding a host
round-trip for every token. It produces the same tokens as repeated `step()`;
on the RTX 3090 the current 128-token test is about 672 tok/s versus 669 tok/s
for the host-synchronized loop. This is stream batching of one sequence, not
multi-sequence continuous batching.

`Decoder.generate()` uses that path in 32-token chunks by default, so the
normal CLI is less sensitive to Elysium's CPU load. Set
`MEGAKERNEL_GENERATE_CHUNK=1` for token-at-a-time streaming behavior; larger
chunks reduce host wakeups but may compute a few tokens past an early EOS.

For equal-length prompt batches, `BatchDecoder(batch_size).prefill_batch(ids)`
copies each prompt through the validated prefill path and then uses one
batch-strided decode launch per step. On Elysium this reached 1,209 aggregate
tok/s at batch 4 with a 32-token prompt, 1,127 tok/s at batch 4 with a
4096-token prompt, and 895 tok/s at batch 4 with a 16384-token prompt.
State remains isolated per sequence. The batch grid has fewer cooperative
blocks per sequence than a standalone decode, so a borderline top-1 logit can
occasionally differ; use the 512-thread build for strict standalone-token
identity comparisons. Variable-length prompt bucketing and batched prefill
remain future work.

`MEGAKERNEL_DECODE_GQA_REUSE=1` contains an experimental tiled implementation
of that reuse. It passes the handoff check, but on the RTX 3090 its extra tile
barriers make it slower than the default at both 4k and 16k contexts, so the
flag defaults to zero.

`MEGAKERNEL_DECODE_ATTENTION_VEC2=1` is enabled by default. It loads adjacent
BF16/FP16 K/V values as pairs in the decode attention loop, preserving the
scalar fallback while reducing conversion and load instructions. Combined
with the default 1024-thread block, it is part of the 482 tok/s at 16k and
423 tok/s at 32k measurements above; the short-context handoff check remains
exact. The prefill body is BF16; INT8 is used only for decode projections and
the vocabulary head. The experimental
`MEGAKERNEL_LM_DP4A=1` path quantizes the final hidden state to int8 for
`dp4a`; it is disabled by default because it fails the exact handoff check.
`MEGAKERNEL_DN_PHASE2_STATE_WMMA=1` path uses BF16 operands for the recurrent
state update and is not quality-preserving; a fresh Elysium build also
triggered an illegal-access failure during prefill. It remains disabled, and
the default keeps the recurrent state update in FP32.

The source in this directory is distributed under the Apache License 2.0 from
Lucebox; see [`optimizations/megakernel/LICENSE`](../optimizations/megakernel/LICENSE).

On the local GTX 970 the fused BF16 extension is unavailable because it
requires SM 7.5 or newer. The production quantized path was tested with the
official `Qwen3.5-0.8B-Q4_0.gguf` over Vulkan: it produced the expected
“Paris” continuation and measured about 331 prompt tok/s and 33.4 generated
tok/s at a 2016-token prompt. The CPU fallback measured 82 prompt tok/s and
0.26 generated tok/s with 16 threads.

The locally available Prism artifacts were also smoke-tested on that Vulkan
backend. `Qwen3.5-0.8B-Q8_0.gguf` produced the expected “Paris” answer and
measured 528 prompt tok/s and 63.9 generated tok/s at a 2016-token prompt;
`Qwen3.5-0.8B-F16.gguf` was coherent but, without FP16 hardware, decoded at
only 1.9 tok/s. `Qwen3.5-0.8B-Q4_K_M.gguf` loaded and passed a simple arithmetic
check at 412 prompt and 76.7 generated tok/s, although its low-bit output is
less reliable than Q8. The QAT/TQ2 and ternary files load as unknown GGUF
quantisation types but produce invalid text (and a TQ2 benchmark can crash in
the Vulkan path); they are not currently usable production models. Their
custom quant kernels need to be implemented before advertising them as
fallbacks.
