# Benchmark Results

The published pp/tg table is batch size 1 and single-stream. The BF16 path also
has an equal-length `BatchDecoder` decode mode with independent recurrent/KV
state; its prefill currently falls back to one validated prefill per sequence.

## Hardware

| Machine | GPU/Chip | Memory |
|---------|----------|--------|
| Lucebox | NVIDIA RTX 3090 | 24GB VRAM |
| MacBook Pro | Apple M5 Max | 36GB Unified |

## RTX 3090: pp520 tg128

| Method | pp520 (tok/s) | tg128 (tok/s) |
|--------|:---:|:---:|
| **Megakernel** | **21,347** | **413** |
| llama.cpp BF16 | 11,247 | 267 |
| PyTorch HF | 7,578 | 108 |

### Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Decode (tg128)** | **1.55x** | **3.8x** |

## Apple M5 Max

| Method | tok/s |
|--------|:---:|
| LM Studio (llama.cpp) BF16 | 229 |

## Power Efficiency (DVFS)

| Power Limit | Clock | Draw | tok/s | tok/J | vs Stock |
|---|---|---|---|---|---|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

Sweet spot: 220W, 1.87 tok/J.

## Methodology

- **Precision:** BF16 weights and activations, FP32 accumulation. No quantization. All baselines (llama.cpp, PyTorch HF) also run BF16 for apples-to-apples comparison.
- **Power measurement:** Accelerator power only via NVML energy counters (NVIDIA) and `powermetrics` (Apple Silicon), consistent with [Hazy Research's Intelligence Per Watt methodology](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles). Total system draw is higher for both platforms.
- **Correctness:** `bench_pp_tg.py` includes an end-to-end correctness check, comparing megakernel output (prefill + decode) against a token-by-token reference decode path. Both must produce identical token sequences.
- **Warm-up:** One warm-up run before timed measurements. Timing uses `torch.cuda.synchronize()` barriers with `time.perf_counter()`.
- **llama.cpp version:** Latest release at time of testing, BF16 mode, default settings.

## Batched decode

On Elysium's RTX 3090, with the mixed INT8 decode weights, vectorized BF16
attention loads, and the default 1024-thread decode block, aggregate decode
throughput was:

| Batch | Context | Aggregate tok/s |
|---:|---:|---:|
| 2 | 32 | 1,000 |
| 4 | 32 | 1,201 |
| 8 | 32 | 1,307 |
| 2 | 4,096 | 343 |
| 2 | 16,384 | 504 |
| 2 | 32,768 | 332 |

The B=2 short-context serial baseline was 682 tok/s aggregate. Batch prompts
must currently have equal lengths. The batch kernel keeps state isolated per
sequence; because it uses fewer cooperative blocks per sequence, borderline
top-1 logits can differ from a standalone 82-block decode. Use the 512-thread
build when exact standalone token identity is required for such comparisons.

## What this doesn't measure

- Quantized model performance (INT4/INT8)
- Models larger than 0.8B parameters
- Multi-GPU or tensor-parallel setups
- Total system power (CPU, RAM, PSU losses)

## NVIDIA DGX Spark (GB10, sm_121a)

First results for the megakernel on NVIDIA's DGX Spark (GB10). Decode runs
through a persistent NVFP4 megakernel; prefill uses a bf16 body with an
NVFP4 LM head (the default "hybrid" prefill mode).

**Hardware:** NVIDIA GB10 Grace Blackwell Superchip (DGX Spark), compute cap
12.1 (`sm_121a`), driver 580.126.09, CUDA 13.2, PyTorch 2.13.0a (cu13.2).

| Method | Prefill pp520 (tok/s) | Decode tg128 (tok/s) |
|---|:---:|:---:|
| **Megakernel (NVFP4 decode, hybrid prefill)** | **20,639** | **181** |
| PyTorch HuggingFace (bf16) | 5,649 | 65 |

**3.6× PyTorch HF on prefill, 2.8× on decode.** Output tokens match the
PyTorch reference on the pp520 prompt from `final_bench.py`.

### Run

```bash
# Auto-dispatches to the NVFP4 path on Blackwell
python optimizations/megakernel/final_bench.py

# Or force a backend
python optimizations/megakernel/final_bench.py --backend nvfp4
python optimizations/megakernel/final_bench.py --backend bf16

# Switch prefill mode (default is "hybrid"; "raw" uses prefill_megakernel_nvfp4)
MEGAKERNEL_PREFILL_MODE=raw python optimizations/megakernel/final_bench.py --backend nvfp4
```
