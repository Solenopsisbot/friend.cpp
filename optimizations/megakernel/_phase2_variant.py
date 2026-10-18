"""Runtime marker for DN phase2 kernel variant.

Reads MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME. On Ampere+ the default is WMMA;
older devices default to scalar FP32.

This module is imported for its side-effect (the print) so the user can
confirm which variant is active without inspecting build flags.
"""
import os as _os

_val = _os.environ.get("MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME")
if _val is None:
    try:
        import torch as _torch
        _val = "1" if _torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 8 else "0"
    except Exception:
        _val = "0"
DN_PHASE2_VARIANT = "wmma" if _val.strip() == "1" else "scalar"
print(f"[megakernel] DN phase2 variant = {DN_PHASE2_VARIANT}")
