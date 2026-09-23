#!/usr/bin/env python3
"""
friend.cpp context-collapse evaluation: does blue-noise sampling keep long generations from
getting stuck?

The mechanism (as kaetemi describes it): a model conditions on its own output, so a streak of
unusual rolls doesn't stay local. A run of max-probability picks makes the text more
predictable, which makes the next max-probability pick likelier still -- the generation locks
into a loop. A run of lowest-probability picks derails it the same way in the other
direction. White noise produces such streaks by chance; blue noise anti-correlates
consecutive rolls, so they are rarer. The effect should be strongest on base models (no
instruction tuning pulling them back on track) and on long generations.

blue_noise_eval.py measures the roll statistics on short chat replies. This one measures the
outcome: raw document-style prompts, long generations, EOS banned, no repetition penalty.
Per generation:

  collapsed   the text locks into repetition and never recovers: from some window onward,
              every window of --window characters has a repeated-20-gram fraction above
              --loop (characters, so it works for any language and for code)
  onset       character position where that happens
  zratio      zlib compression ratio of the text (lower = more repetitive)
  top_streak  longest run of consecutive top-probability picks
  tail_streak longest run of consecutive picks with p < --tail-p

Blue and white noise run the same (prompt, seed) pairs, so collapse is compared paired: a sign
test over the pairs where exactly one of the two collapsed.

  python tools/friend-eval/collapse_eval.py --url http://localhost:5001 \\
      --seeds 8 --max-tokens 768 --temperature 0.7 --workers 8 --out collapse.json

Start the server with --parallelrequests >= --workers so the requests batch. The server keeps one
set of "last logprobs", so the streak columns are only measured with --workers 1 (collapse is
measured from the text and is safe either way). Stdlib only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import statistics
import sys
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blue_noise_eval import per_token  # noqa: E402

# Document openings, the way a base model sees text. Mixed genres so no single style dominates.
DEFAULT_PROMPTS = [
    "The lighthouse keeper had not spoken to another person in eleven years when the boat appeared.",
    "Photosynthesis is the process by which green plants and certain other organisms",
    "Re: Anyone else's sourdough starter smell like nail polish remover?\n\nPosted by breadhead_92\n\n",
    "def parse_config(path):\n    \"\"\"Load a TOML config file and return a dict of settings.\"\"\"\n",
    "Ingredients:\n- 2 cups all-purpose flour\n- 1 tsp baking soda\n",
    "BREAKING: City council votes to replace all downtown parking meters with",
    "Chapter 1\n\nMy grandmother kept a jar of buttons on the kitchen windowsill, and",
    "The history of the Byzantine Empire can be divided into three periods.",
    "Dear Hiring Manager,\n\nI am writing to apply for the position of",
    "Q: What's the difference between a list and a tuple in Python?\nA:",
    "It was the last day of summer, and the whole town had gathered at the lake",
    "Abstract. We study the problem of",
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/api/v1/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def generate(args, prompt: str, seed: int, blue: bool) -> dict:
    body = {
        "prompt": prompt,
        "max_length": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "rep_pen": 1.0,
        "rep_pen_range": 0,
        "sampler_seed": seed,
        "ban_eos_token": True,
        "blue_noise": blue,
        "rng_type": args.rng_type,
        "logprobs": args.workers == 1,   # the server's last-logprobs slot is global
    }
    r = post(args.url, body)["results"][0]
    # a failed decode (e.g. the KV cache too small for --parallelrequests x the generation)
    # comes back as finish_reason "error" with empty text -- never score that as a generation
    if r.get("finish_reason") == "error" or not r.get("text"):
        raise RuntimeError(f"generation failed (finish_reason={r.get('finish_reason')!r}, "
                           f"{len(r.get('text') or '')} chars): check the server log and its --contextsize")
    toks = per_token({"choices": [{"logprobs": r.get("logprobs") or {}}]})
    return {"text": r["text"], "toks": toks}


def longest_run(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def loop_frac(text: str, n: int = 20) -> float:
    """Fraction of character n-grams in the window that repeat an earlier one in the same window.
    Ordinary prose almost never repeats a 20-character span within a few hundred characters;
    a generation stuck in a loop repeats nearly all of them."""
    g = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(g)) / len(g) if g else 0.0


def metrics(gen: dict, args) -> dict:
    text = gen["text"]
    W, step = args.window, max(1, args.window // 4)
    scores = [(i, loop_frac(text[i:i + W])) for i in range(0, max(1, len(text) - W + 1), step)]
    # collapse = locked in: every window from the onset to the end is loopy
    onset = None
    for k, (i, _) in enumerate(scores):
        if all(sc > args.loop for _, sc in scores[k:]):
            onset = i
            break
    raw = gen["text"].encode()
    toks = gen["toks"]
    return {
        "n_chars": len(text),
        "collapsed": onset is not None,
        "onset": onset,
        "final_loop": scores[-1][1] if scores else 0.0,
        "zratio": len(zlib.compress(raw, 9)) / max(1, len(raw)),
        "top_streak": longest_run(t["rank"] == 0 for t in toks) if toks else None,
        "tail_streak": longest_run(t["p"] < args.tail_p for t in toks) if toks else None,
    }


def sign_test_p(a: int, b: int) -> float:
    """Two-sided exact binomial test of a vs b discordant pairs (H0: p = 0.5)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:5001")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--rng-type", default="mt19937", choices=["mt19937", "lowbias32"])
    ap.add_argument("--window", type=int, default=400, help="characters per loop-detection window")
    ap.add_argument("--loop", type=float, default=0.5, help="repeated-20-gram fraction that counts as a loop")
    ap.add_argument("--tail-p", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (server --parallelrequests)")
    ap.add_argument("--prompts", help="JSON file: [prompt, ...]")
    ap.add_argument("--out")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts)) if args.prompts else DEFAULT_PROMPTS
    jobs = [(pi, seed, blue) for pi in range(len(prompts)) for seed in range(5000, 5000 + args.seeds)
            for blue in (False, True)]
    res: dict = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(generate, args, prompts[pi], seed, blue): (pi, seed, blue) for pi, seed, blue in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            key = futs[fut]
            gen = fut.result()
            res[key] = {**metrics(gen, args), "text": gen["text"]}
            print(f"\r{n}/{len(jobs)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    white = [res[(pi, s, False)] for pi, s, _ in jobs if not _]
    blue = [res[(pi, s, True)] for pi, s, b in jobs if b]

    # paired collapse: same (prompt, seed)
    only_white = sum(1 for w, b in zip(white, blue) if w["collapsed"] and not b["collapsed"])
    only_blue = sum(1 for w, b in zip(white, blue) if b["collapsed"] and not w["collapsed"])

    def rate(xs):
        return sum(x["collapsed"] for x in xs) / len(xs)

    def mean(xs, k):
        v = [x[k] for x in xs if x[k] is not None]
        return statistics.fmean(v) if v else float("nan")

    print(f"\n{len(white)} pairs, T={args.temperature} top_p={args.top_p} top_k={args.top_k} min_p={args.min_p}, "
          f"{args.max_tokens} tokens, EOS banned, no repetition penalty")
    print(f"{'':22}{'white':>10}{'blue':>10}")
    print(f"{'collapsed':22}{rate(white):>10.1%}{rate(blue):>10.1%}")
    print(f"{'  onset (chars, mean)':22}{mean(white, 'onset'):>10.0f}{mean(blue, 'onset'):>10.0f}")
    for k in ("final_loop", "zratio", "top_streak", "tail_streak"):
        print(f"{k:22}{mean(white, k):>10.3f}{mean(blue, k):>10.3f}")
    print(f"\npaired: collapsed with white only {only_white}, with blue only {only_blue}, "
          f"sign test p = {sign_test_p(only_white, only_blue):.3f}")

    if args.out:
        out = {"args": vars(args), "prompts": prompts,
               "runs": [{"prompt": pi, "seed": s, "blue": b, **res[(pi, s, b)]} for pi, s, b in jobs]}
        json.dump(out, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
