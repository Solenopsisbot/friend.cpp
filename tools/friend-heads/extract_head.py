#!/usr/bin/env python3
"""
friend.cpp: extract a swappable LM head from a GGUF model.

A head file holds just the final projection -- `output.weight` -- plus, optionally,
`output_norm.weight` (and `output.bias` / NVFP4 scales if the source has them). Load
heads with `--head-pool NAME=head.gguf` and pick one per request with `"head": "NAME"`.

Why you'd want this: a head-only finetune (train just lm_head, and maybe the final
norm, on a persona's chat logs) is tiny and cheap, and swapping it never invalidates
the KV cache, unlike a LoRA. Export the finetuned model to GGUF, then run this to pull
out the head.

Tied-embedding models (no `output.weight`) use `token_embd.weight` as their head; that
tensor is copied and renamed, since the head file must say `output.weight`.

Usage:
  python tools/friend-heads/extract_head.py finetuned.gguf persona.head.gguf [--no-norm]

Needs gguf-py (in this repo) and numpy:
  uv run --with numpy --with pyyaml --with tqdm python tools/friend-heads/extract_head.py ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


HEAD_TENSORS = ["output.weight", "output_norm.weight", "output.bias", "output.scale", "output.input_scale"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source model GGUF")
    ap.add_argument("dst", help="output head GGUF")
    ap.add_argument("--no-norm", action="store_true",
                    help="don't include output_norm.weight (the model's own final norm is then kept at swap time)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.src)
    tensors = {t.name: t for t in reader.tensors}
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else None

    picks: list[tuple[str, gguf.ReaderTensor]] = []
    if "output.weight" in tensors:
        picks.append(("output.weight", tensors["output.weight"]))
    elif "token_embd.weight" in tensors:
        print("note: tied embeddings, using token_embd.weight as the head", file=sys.stderr)
        picks.append(("output.weight", tensors["token_embd.weight"]))
    else:
        print("error: source has neither output.weight nor token_embd.weight", file=sys.stderr)
        return 1
    for name in HEAD_TENSORS[1:]:
        if name == "output_norm.weight" and args.no_norm:
            continue
        if name in tensors:
            picks.append((name, tensors[name]))

    writer = gguf.GGUFWriter(args.dst, arch=arch or "llama")
    writer.add_type("head")
    writer.add_string("general.name", Path(args.src).stem + " head")
    for name, t in picks:
        # copy raw (possibly quantized) bytes; the writer derives the element shape from the
        # reader's byte-shaped array for quantized types
        writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
        print(f"  {name:<22} {t.tensor_type.name:<8} {list(int(x) for x in t.shape)}", file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.dst} ({os.path.getsize(args.dst) / 2**20:.1f} MiB, arch={arch})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
