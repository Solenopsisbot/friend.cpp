"""Benchmark equal-length BF16 batch decode on the fused Qwen3.5 backend."""

import argparse
import time

import torch
from transformers import AutoTokenizer

from model import BatchDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BatchDecoder requires CUDA")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    seed = tok.encode(
        "Explain the history of artificial intelligence in detail. ",
        add_special_tokens=False,
    )
    ids = (seed * ((args.prompt_tokens + len(seed) - 1) // len(seed)))[:args.prompt_tokens]
    token_ids = torch.tensor([ids] * args.batch_size, dtype=torch.int32, device="cuda")
    dec = BatchDecoder(args.batch_size, verbose=True, max_seq_len=args.max_seq_len)

    first = dec.prefill_batch(token_ids)
    dec.step(first)
    torch.cuda.synchronize()
    first = dec.prefill_batch(token_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    current = first
    for _ in range(args.gen_tokens):
        current = dec.step(current)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total = args.batch_size * args.gen_tokens
    print(
        f"B={args.batch_size} prompt={args.prompt_tokens} gen={args.gen_tokens}: "
        f"{total / elapsed:.1f} aggregate tok/s ({elapsed * 1000:.1f} ms total)"
    )


if __name__ == "__main__":
    main()
