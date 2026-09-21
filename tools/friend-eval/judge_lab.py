#!/usr/bin/env python3
"""
friend.cpp judged sampler comparison: which sampler config produces *better replies*,
according to a (bigger) judge model, blind and pairwise.

sampler_lab.py measures mechanics (tail picks, streaks, repetition) on the post-sampler
distribution -- samplers that reshape that distribution (XTC, DRY) move those numbers
partly by construction. This asks a different question: shown two replies to the same
persona + message, which one would a reader prefer?

  1. a generator server (any model) produces one reply per (prompt, seed, config)
  2. a judge server compares each candidate config's reply against the baseline's reply
     for the same (prompt, seed), in random order (cancels position bias), with a grammar
     forcing the answer to "1" or "2"; the judge's probability of preferring the candidate
     is read from its logprobs, so near-ties count as ~0.5 instead of a coin flip
  3. report the mean preference per config with a bootstrap 95% interval; 0.5 = no
     difference, > 0.5 = the candidate's replies are preferred

  python tools/friend-eval/judge_lab.py --gen-url http://localhost:5099 \
      --judge-url http://localhost:5098 --configs tools/friend-eval/configs/chat_presets.json \
      --seeds 4 --out judge.json

The judge prompt uses ChatML; pass --judge-template to adapt it. A judge is only as good as
its model: treat small-judge results as a hint, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blue_noise_eval import DEFAULT_PROMPTS  # noqa: E402

JUDGE_TEMPLATE = (
    "<|im_start|>system\nYou are a strict, fair judge of roleplay chat replies. You compare two replies "
    "written by the same character and pick the better one: more in character, more engaging and natural "
    "to talk to, coherent, not repetitive, no nonsense. Length alone is not quality.<|im_end|>\n"
    "<|im_start|>user\nCharacter description:\n{system}\n\nThe user wrote:\n{user}\n\n"
    "Reply 1:\n{a}\n\nReply 2:\n{b}\n\nWhich reply is better? Answer with just the number 1 or 2.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def generate(url, system, user, seed, cfg, max_tokens):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "seed": seed}
    body.update(cfg)
    return post(url, "/v1/chat/completions", body)["choices"][0]["message"]["content"] or ""


def judge_prob_first(url, template, system, user, a, b) -> float:
    """P(judge prefers reply 1), from logprobs of the grammar-constrained single-token answer."""
    body = {"prompt": template.format(system=system, user=user, a=a.strip(), b=b.strip()),
            "max_length": 1, "temperature": 0, "top_k": 1, "grammar": 'root ::= "1" | "2"', "logprobs": True}
    r = post(url, "/api/v1/generate", body)["results"][0]
    lp = r.get("logprobs") or {}
    tops = (lp.get("top_logprobs") or [{}])[0]
    p1 = math.exp(tops["1"]) if "1" in tops else 0.0
    p2 = math.exp(tops["2"]) if "2" in tops else 0.0
    if p1 + p2 == 0.0:  # no logprobs: fall back to the hard answer
        return 1.0 if r["text"].strip().startswith("1") else 0.0
    return p1 / (p1 + p2)


def bootstrap_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-url", required=True)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--prompts")
    ap.add_argument("--judge-template", help="file with a template using {system} {user} {a} {b}")
    ap.add_argument("--out")
    args = ap.parse_args()

    configs = json.load(open(args.configs))
    names = list(configs)
    base = args.baseline or names[0]
    prompts = json.load(open(args.prompts)) if args.prompts else DEFAULT_PROMPTS
    template = open(args.judge_template).read() if args.judge_template else JUDGE_TEMPLATE
    rng = random.Random(1234)

    replies = {}
    total = len(prompts) * args.seeds * len(names)
    done = 0
    for pi, (system, user) in enumerate(prompts):
        for seed in range(3000, 3000 + args.seeds):
            for n in names:
                replies[(pi, seed, n)] = generate(args.gen_url, system, user, seed, configs[n], args.max_tokens)
                done += 1
                print(f"\rgenerating {done}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    prefs = {n: [] for n in names if n != base}
    records = []
    total = len(prompts) * args.seeds * len(prefs)
    done = 0
    for pi, (system, user) in enumerate(prompts):
        for seed in range(3000, 3000 + args.seeds):
            b_reply = replies[(pi, seed, base)]
            for n in prefs:
                c_reply = replies[(pi, seed, n)]
                cand_first = rng.random() < 0.5
                a, b = (c_reply, b_reply) if cand_first else (b_reply, c_reply)
                p_first = judge_prob_first(args.judge_url, template, system, user, a, b)
                p_cand = p_first if cand_first else 1.0 - p_first
                prefs[n].append(p_cand)
                records.append({"prompt": pi, "seed": seed, "config": n, "candidate_first": cand_first,
                                "p_candidate": p_cand, "candidate": c_reply, "baseline": b_reply})
                done += 1
                print(f"\rjudging {done}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    print(f"\npreference for each config over '{base}' (0.5 = no difference; judged blind, order randomized)")
    print(f"{'config':<16}{'mean':>8}{'95% CI':>18}{'wins':>8}")
    for n, xs in prefs.items():
        lo, hi = bootstrap_ci(xs)
        wins = sum(1 for x in xs if x > 0.5)
        print(f"{n:<16}{statistics.fmean(xs):>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}{f'{wins}/{len(xs)}':>8}")
    if args.out:
        json.dump({"configs": configs, "baseline": base, "records": records}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
