"""Optional fused Qwen3.5-0.8B backend.

The regular friend.cpp loaders remain the supported path for GGUF models.  This
package exposes the architecture-specific CUDA megakernel for an HF
Qwen3.5-0.8B checkpoint on NVIDIA Turing and newer GPUs.
"""

__all__ = [
    "Decoder",
    "load_weights",
    "max_safe_decode_blocks",
    "set_decode_blocks",
]


def __getattr__(name):
    # Keep the main friend.cpp Python tooling importable on CPU-only systems;
    # torch is an optional dependency of this backend, not of the repository.
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
