#!/usr/bin/env python3
"""
friend.cpp sampler lab: compare any number of sampler configurations on the same prompts
and seeds, with the metrics from blue_noise_eval.py plus a few more, and significance
against a baseline.

A config is just a dict of request fields merged into every /v1/chat/completions call
(temperature, min_p, blue_noise, dry_multiplier, xtc_probability, dynatemp_range, ...).

  python tools/friend-eval/sampler_lab.py --url http://localhost:5001 \
      --configs tools/friend-eval/configs/chat_presets.json --seeds 6 --out lab.json

configs file: {"baseline": {...}, "name2": {...}, ...}; the first entry is the baseline
(or pass --baseline NAME).

Metrics (per generation, then averaged; "lower is better" unless noted):
  tail_frac, longest_tail_run, surprisal_window_var   -- derailment risk (tail = p_pick < 0.1,
                                                          measured on the post-sampler distribution)
  mean_surprisal      -- how adventurous the picks are (not better/worse; context for the rest)
  repeat_4gram_frac   -- within-generation repetition
  cross_seed_jaccard3 -- similarity between seeds of the same prompt (lower = more diverse)
  distinct2           -- distinct word bigrams / total (higher = more varied wording)
  n_tokens            -- length (a sampler that just stops early can look "clean")
Welch t is reported against the baseline; |t| > ~2 is where it starts to mean something.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blue_noise_eval import DEFAULT_PROMPTS, jaccard3, metrics_for, welch_t  # noqa: E402

import urllib.request  # noqa: E402

KEYS = ["tail_frac", "longest_tail_run", "surprisal_window_var", "mean_surprisal",
        "repeat_4gram_frac", "cross_seed_jaccard3", "distinct2", "n_tokens"]


def chat(url: str, system: str, user: str, seed: int, cfg: dict, max_tokens: int) -> dict:
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "seed": seed, "logprobs": True, "top_logprobs": 10}
    body.update(cfg)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:5001")
    ap.add_argument("--configs", required=True, help="JSON file {name: {request fields}}")
    ap.add_argument("--baseline", help="config name to compare against (default: first)")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--tail-p", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--prompts", help="JSON file [[system, user], ...] (default: built-in chat prompts)")
    ap.add_argument("--out", help="write raw results here")
    args = ap.parse_args()

    configs: dict = json.load(open(args.configs))
    names = list(configs)
    base = args.baseline or names[0]
    prompts = json.load(open(args.prompts)) if args.prompts else DEFAULT_PROMPTS

    results = {n: [] for n in names}
    total = len(names) * len(prompts) * args.seeds
    done = 0
    # interleave configs per (prompt, seed) so drift (thermal, other load) hits all equally
    for pi, (system, user) in enumerate(prompts):
        for seed in range(2000, 2000 + args.seeds):
            for n in names:
                m = metrics_for(chat(args.url, system, user, seed, configs[n], args.max_tokens), args)
                words = m.pop("_words")
                m["distinct2"] = len(set(zip(words, words[1:]))) / max(1, len(words) - 1)
                m.update(prompt=pi, seed=seed, _words=words)
                results[n].append(m)
                done += 1
                print(f"\r{done}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    for n in names:
        by_prompt: dict = {}
        for m in results[n]:
            by_prompt.setdefault(m["prompt"], []).append(m["_words"])
        for pi, gens in by_prompt.items():
            sims = [jaccard3(a, b) for a, b in itertools.combinations(gens, 2)]
            for m in results[n]:
                if m["prompt"] == pi:
                    m["cross_seed_jaccard3"] = statistics.fmean(sims) if sims else None

    col = max(12, max(len(n) for n in names) + 2)
    print(f"\n{'metric':<22}" + "".join(f"{n:>{col}}" for n in names))
    for k in KEYS:
        row = f"{k:<22}"
        b = [m[k] for m in results[base] if m.get(k) is not None]
        for n in names:
            v = [m[k] for m in results[n] if m.get(k) is not None]
            if not v:
                row += f"{'-':>{col}}"
                continue
            mean = statistics.fmean(v)
            if n == base:
                row += f"{mean:>{col}.4f}"
            else:
                t = welch_t(b, v)
                row += f"{mean:>{col - 7}.4f}({(t or 0):+5.1f})"
        print(row)
    print(f"\n(t) = Welch t vs '{base}'. tail/streak/variance/repetition/jaccard: lower is better; distinct2: higher.")

    if args.out:
        for n in names:
            for m in results[n]:
                m.pop("_words", None)
                m.pop("_text", None)
        json.dump({"configs": configs, "baseline": base, "results": results}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
