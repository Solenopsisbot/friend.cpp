#!/usr/bin/env python3
"""Run Qwen3.5-0.8B through the fused backend or friend.cpp's GGUF path.

The fused backend consumes the original Hugging Face BF16 checkpoint.  GGUF
files (including Q4/Q5/Q8 models) are delegated to koboldcpp.py so quantised
inference keeps the same CLI and sampler behaviour as the rest of friend.cpp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _is_gguf(model: str) -> bool:
    return model.lower().endswith(".gguf") or any(
        p.lower().endswith(".gguf") for p in model.split(",")
    )


def _run_quantized(model: str, prompt: str, max_tokens: int, args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    launcher = root / "koboldcpp.py"
    cmd = [
        sys.executable,
        str(launcher),
        "--model",
        model,
        "--prompt",
        prompt,
        "--genlimit",
        str(max_tokens),
        "--skiplauncher",
        "--quiet",
    ]
    if args.gpulayers is not None:
        cmd.extend(["--gpulayers", str(args.gpulayers)])
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id/directory or a GGUF file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--backend", choices=("auto", "megakernel", "quantized"), default="auto")
    parser.add_argument("--decode-blocks", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="megakernel KV-cache length (default: 2048)")
    parser.add_argument("--gpulayers", type=int, default=None, help="GGUF path: layers to offload")
    args = parser.parse_args()

    backend = "quantized" if args.backend == "auto" and _is_gguf(args.model) else args.backend
    if backend == "quantized":
        return _run_quantized(args.model, args.prompt, args.max_tokens, args)

    if _is_gguf(args.model):
        parser.error("the megakernel consumes an HF checkpoint; use --backend quantized for GGUF")

    try:
        import torch
        from transformers import AutoTokenizer
        try:
            from .model import Decoder, set_decode_blocks
        except ImportError:
            # Direct ``python optimizations/megakernel/run.py`` execution.
            from model import Decoder, set_decode_blocks
    except ImportError as exc:
        parser.error(
            "the megakernel needs torch and transformers; install this directory's "
            "pyproject dependencies before using --backend megakernel (" + str(exc) + ")"
        )

    if not torch.cuda.is_available():
        parser.error("the megakernel requires a CUDA device; use --backend quantized for CPU/GGUF inference")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 5):
        parser.error(
            f"the megakernel requires NVIDIA SM 7.5 or newer (detected SM {major}.{minor}); "
            "use --backend quantized"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    decoder_kwargs = {"model_name": args.model, "tokenizer": tokenizer, "verbose": True}
    if args.max_seq_len is not None:
        decoder_kwargs["max_seq_len"] = args.max_seq_len
    decoder = Decoder(**decoder_kwargs)
    if args.decode_blocks is not None:
        set_decode_blocks(args.decode_blocks)
    print(decoder.generate(args.prompt, max_tokens=args.max_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
