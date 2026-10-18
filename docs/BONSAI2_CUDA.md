# Bonsai 2 CUDA backend

Bonsai 2 27B uses the Qwen3.5 hybrid layout: 64 decoder layers, with 48
Gated DeltaNet recurrent layers and 16 full-attention layers. Its published
GGUF files use Prism's `PQ2_0` or `PTQ1_0` types and signed Hadamard metadata.
The CUDA path therefore has to keep the Prism type registrations, dequantizers,
Hadamard transforms, and Qwen3.5 recurrent state handling together.

## Build

For CUDA 12 or newer, the Makefile CUDA targets compile CUDA Graph support:

```bash
make -j LLAMA_CUBLAS=1 LLAMA_ARCHES_CU12=1 koboldcpp_cublas
```

Run with the graph optimizer enabled:

```bash
GGML_CUDA_GRAPH_OPT=1 python koboldcpp.py \
  --model /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --gpulayers 99 --contextsize 2048
```

The first stable decode shape is captured and then replayed by CUDA Graphs,
while the existing fused GDN, norm, Hadamard, attention, and PQ2/PTQ1 kernels
remain responsible for the work inside the graph. Change in prompt length,
batch shape, or context size can require a new capture. Set
`GGML_CUDA_DISABLE_GRAPHS=1` to force the uncaptured path for comparison.

This is the practical fused backend currently implemented for Bonsai. A single
persistent kernel that decodes all 64 layers would require moving the full
quantized weight stream, recurrent state, KV cache, Hadamard transforms, and
the 248,320-token output projection into one custom scheduler; it is a separate
kernel project rather than a safe widening of the Qwen3.5 0.8B BF16 megakernel.
