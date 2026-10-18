import os
import re

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_arch():
    arch = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if arch:
        return arch
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 12 and minor in (0, 1):
                return f"sm_{major}{minor}a"
            return f"sm_{major}{minor}"
    except Exception:
        pass
    # The fused kernel relies on cooperative-grid synchronization and the
    # warp primitives available from Turing onward.  Do not silently build a
    # binary for an older device: it either cannot launch or deadlocks at the
    # first grid barrier.  Quantised GGUF inference remains available through
    # friend.cpp's normal backend on those devices.
    return "sm_75"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
is_blackwell = arch.startswith("sm_12")

# Extract numeric SM level (e.g. "sm_75" → 75, "sm_120a" → 120) for compile-time guards.
_sm_match = re.search(r"sm_(\d+)", arch)
target_sm = int(_sm_match.group(1)) if _sm_match else 75

if target_sm < 75:
    raise RuntimeError(
        f"Qwen3.5 megakernel requires NVIDIA SM 7.5 or newer (got {arch}). "
        "Use friend.cpp/koboldcpp.py for quantised GGUF inference on this GPU."
    )

num_blocks = _int_env("MEGAKERNEL_NUM_BLOCKS", 82)
# 1024 decode threads materially improve long-context attention on Ampere.
block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 1024)
lm_num_blocks = _int_env("MEGAKERNEL_LM_NUM_BLOCKS", 8192)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)
dn_phase2_wmma = _int_env("MEGAKERNEL_DN_PHASE2_WMMA", 0)
dn_phase2_splits = _int_env("MEGAKERNEL_DN_PHASE2_J_SPLITS", 4)
dn_chunk_c = _int_env("MEGAKERNEL_DN_CHUNK_C", 16)
# 512 threads reduce the per-chunk loop and barrier overhead on Ampere while
# retaining two resident blocks per SM for the WMMA phase-2 kernel.
dn_phase2_block = _int_env("MEGAKERNEL_DN_PHASE2_BLOCK", 512)
dn_phase2_state_wmma = _int_env("MEGAKERNEL_DN_PHASE2_STATE_WMMA", 0)
dn_phase2_vector_update = _int_env("MEGAKERNEL_DN_PHASE2_VECTOR_UPDATE", 1)
dn_head_split = _int_env("MEGAKERNEL_DN_HEAD_SPLIT", 1)
lm_int8 = _int_env("MEGAKERNEL_LM_INT8", 0)
layer_int8 = _int_env("MEGAKERNEL_LAYER_INT8", 0)
layer_dp4a = _int_env("MEGAKERNEL_LAYER_DP4A", 0)
decode_gqa_reuse = _int_env("MEGAKERNEL_DECODE_GQA_REUSE", 0)
decode_attention_vec2 = _int_env("MEGAKERNEL_DECODE_ATTENTION_VEC2", 1)
attention_exp_approx = _int_env("MEGAKERNEL_ATTENTION_EXP_APPROX", 0)
# Optional deliberately lossy decode approximation: full-attention layers only
# scan the most recent N KV entries.  Zero keeps the exact full-context path.
approx_attn_window = _int_env("MEGAKERNEL_APPROX_ATTN_WINDOW", 0)
# Optional lossy MLP width cap.  Zero keeps all 3584 intermediate channels;
# positive values compute only the first N channels in every decode MLP.
approx_mlp_width = _int_env("MEGAKERNEL_APPROX_MLP_WIDTH", 0)
# Skip every decode MLP block entirely (extreme quality/speed experiment).
approx_skip_mlp = _int_env("MEGAKERNEL_APPROX_SKIP_MLP", 0)
# Or skip one MLP periodically; e.g. 4 skips layers 0,4,8,... and keeps the
# other three layer outputs intact. Zero disables this approximation.
approx_skip_mlp_every = _int_env("MEGAKERNEL_APPROX_SKIP_MLP_EVERY", 0)
lm_dp4a = _int_env("MEGAKERNEL_LM_DP4A", 0)
decode_splitkv = _int_env("MEGAKERNEL_DECODE_SPLITKV", 0)
decode_splitkv_min_context = _int_env("MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT", 512)
maxrregcount = int(_int_env("MEGAKERNEL_MAXRREGCOUNT", 0))
ptxas_verbose = int(_int_env("MEGAKERNEL_PTXAS_VERBOSE", 0))

sources = [
    "torch_bindings.cpp",
    "kernel.cu",
    "prefill.cu",
]
libraries = ["cublas"]
cxx_args = ["-O3"]
nvcc_args = [
    "-O3",
    f"-arch={arch}",
    "--use_fast_math",
    "-std=c++17",
    f"-DNUM_BLOCKS={num_blocks}",
    f"-DBLOCK_SIZE={block_size}",
    f"-DLM_NUM_BLOCKS={lm_num_blocks}",
    f"-DLM_BLOCK_SIZE={lm_block_size}",
    f"-DMEGAKERNEL_DN_PHASE2_WMMA={dn_phase2_wmma}",
    f"-DMEGAKERNEL_DN_PHASE2_J_SPLITS={dn_phase2_splits}",
    f"-DMEGAKERNEL_DN_CHUNK_C={dn_chunk_c}",
    f"-DMEGAKERNEL_DN_PHASE2_BLOCK={dn_phase2_block}",
    f"-DMEGAKERNEL_DN_PHASE2_STATE_WMMA={dn_phase2_state_wmma}",
    f"-DMEGAKERNEL_DN_PHASE2_VECTOR_UPDATE={dn_phase2_vector_update}",
    f"-DMEGAKERNEL_DN_HEAD_SPLIT={dn_head_split}",
    f"-DMEGAKERNEL_LM_INT8={lm_int8}",
    f"-DMEGAKERNEL_LAYER_INT8={layer_int8}",
    f"-DMEGAKERNEL_LAYER_DP4A={layer_dp4a}",
    f"-DMEGAKERNEL_DECODE_GQA_REUSE={decode_gqa_reuse}",
    f"-DMEGAKERNEL_DECODE_ATTENTION_VEC2={decode_attention_vec2}",
    f"-DMEGAKERNEL_ATTENTION_EXP_APPROX={attention_exp_approx}",
    f"-DMEGAKERNEL_APPROX_ATTN_WINDOW={approx_attn_window}",
    f"-DMEGAKERNEL_APPROX_MLP_WIDTH={approx_mlp_width}",
    f"-DMEGAKERNEL_APPROX_SKIP_MLP={approx_skip_mlp}",
    f"-DMEGAKERNEL_APPROX_SKIP_MLP_EVERY={approx_skip_mlp_every}",
    f"-DMEGAKERNEL_LM_DP4A={lm_dp4a}",
    f"-DMEGAKERNEL_DECODE_SPLITKV={decode_splitkv}",
    f"-DMEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT={decode_splitkv_min_context}",
    f"-DTARGET_SM={target_sm}",
]
if maxrregcount > 0:
    nvcc_args.append(f"-maxrregcount={maxrregcount}")
if ptxas_verbose:
    nvcc_args.append("-Xptxas=-v")
# Expose to host compiler (torch_bindings.cpp, prefill.cu host-side) so it
# can also branch on the flag without needing a separate nvcc pass.
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_WMMA={dn_phase2_wmma}")
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_J_SPLITS={dn_phase2_splits}")
cxx_args.append(f"-DMEGAKERNEL_DN_CHUNK_C={dn_chunk_c}")
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_BLOCK={dn_phase2_block}")
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_STATE_WMMA={dn_phase2_state_wmma}")
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_VECTOR_UPDATE={dn_phase2_vector_update}")
cxx_args.append(f"-DMEGAKERNEL_DN_HEAD_SPLIT={dn_head_split}")
cxx_args.append(f"-DMEGAKERNEL_LM_INT8={lm_int8}")
cxx_args.append(f"-DMEGAKERNEL_LAYER_INT8={layer_int8}")
cxx_args.append(f"-DMEGAKERNEL_LAYER_DP4A={layer_dp4a}")
cxx_args.append(f"-DMEGAKERNEL_DECODE_GQA_REUSE={decode_gqa_reuse}")
cxx_args.append(f"-DMEGAKERNEL_DECODE_ATTENTION_VEC2={decode_attention_vec2}")
cxx_args.append(f"-DMEGAKERNEL_ATTENTION_EXP_APPROX={attention_exp_approx}")
cxx_args.append(f"-DMEGAKERNEL_APPROX_ATTN_WINDOW={approx_attn_window}")
cxx_args.append(f"-DMEGAKERNEL_APPROX_MLP_WIDTH={approx_mlp_width}")
cxx_args.append(f"-DMEGAKERNEL_APPROX_SKIP_MLP={approx_skip_mlp}")
cxx_args.append(f"-DMEGAKERNEL_APPROX_SKIP_MLP_EVERY={approx_skip_mlp_every}")
cxx_args.append(f"-DMEGAKERNEL_LM_DP4A={lm_dp4a}")
cxx_args.append(f"-DMEGAKERNEL_DECODE_SPLITKV={decode_splitkv}")
cxx_args.append(f"-DMEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT={decode_splitkv_min_context}")
cxx_args.append(f"-DTARGET_SM={target_sm}")

if is_blackwell:
    sources.append("kernel_gb10_nvfp4.cu")
    sources.append("prefill_megakernel.cu")
    sources.append("prefill_bw.cu")
    # Exposed to both nvcc (for the Blackwell .cu files) and the host
    # compiler (so torch_bindings.cpp registers the NVFP4 ops).
    cxx_args.append("-DMEGAKERNEL_HAS_NVFP4")
    nvcc_args.append("-DMEGAKERNEL_HAS_NVFP4")
    nvcc_args.append("-DMEGAKERNEL_HAS_PREFILL_MEGA")
    libraries.append("cublasLt")

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=sources,
            depends=["half_type.h", "rope.h"],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            libraries=libraries,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
