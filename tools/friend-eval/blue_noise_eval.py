#!/usr/bin/env python3
"""
friend.cpp blue-noise sampling evaluation.

Question this answers: does `blue_noise` actually change anything we care about,
or is it a cute RNG that makes no measurable difference?

How blue noise is *supposed* to work
------------------------------------
Every sampled token consumes one uniform "roll" u in [0,1) and picks the token whose
CDF interval contains u. With white noise (mt19937) the rolls are i.i.d., so by chance
you get clumps: several high rolls in a row => several tail tokens in a row, which is
how a generation derails. Blue noise keeps the *marginal* distribution of each roll
uniform (so per-token sampling is still unbiased) but anti-correlates consecutive
rolls, suppressing low-frequency variance. In theory: fewer unlucky streaks, same
average behaviour.

What we measure
---------------
1. Mechanism check -- is the RNG really blue?
   We reconstruct each roll from the logprobs the server returns: the pick's
   probability-integral-transform u_t = (sum of p over higher-ranked candidates) +
   p_pick/2. (Only approximate: it's the midpoint of the pick's interval, and it's
   only computable when the pick is inside the returned top-10.) Blue noise should
   show negative lag-1 autocorrelation and lower variance of windowed means than
   white noise.

2. Outcome metrics -- does it matter?
   - mean surprisal of picks (should be ~equal: blue noise must not bias sampling)
   - tail picks (p_pick < --tail-p) and the longest run of consecutive tail picks
   - windowed surprisal variance (streakiness of "weird token" bursts)
   - repetition: fraction of repeated 4-grams within a generation
   - cross-seed diversity: mean pairwise Jaccard similarity of 3-gram sets between
     generations of the same prompt (lower = more diverse)

Runs every (prompt, seed) pair with blue_noise off and on, same seeds, then reports
means, the difference, and a Welch t statistic per metric. |t| > ~2 is where it
starts being worth believing.

Usage
-----
  python tools/friend-eval/blue_noise_eval.py --url http://localhost:5001 \
      --seeds 12 --max-tokens 160 --temperature 1.0 --out results.json

Needs a friend.cpp server running any chat model. Stdlib only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import urllib.request

# Character-chat style prompts: the use case friend.cpp exists for. Open-ended on
# purpose -- a question with one right answer gives the sampler nothing to do.
DEFAULT_PROMPTS = [
    ("You are Kiko, a chaotic, affectionate gremlin of a friend who loves bugs and bad puns.",
     "ok i finally finished my exams. entertain me"),
    ("You are Amelia, a calm, dry-witted librarian who secretly writes fantasy novels.",
     "What's the story you're working on right now about?"),
    ("You are June, an upbeat night-owl who is way too into synthesizers.",
     "I can't sleep. Talk to me about something."),
    ("You are a friendly companion who remembers the user's day and asks follow-ups.",
     "My cat knocked my coffee onto my keyboard this morning and I've been sulking since."),
    ("You are a playful friend who likes to invent little games.",
     "I'm bored on a train for two hours. Got anything?"),
    ("You are a thoughtful friend who gives honest opinions.",
     "Be real with me: is it weird to still sleep with a stuffed animal at 18?"),
    ("You are a sarcastic but kind friend.",
     "I just spent three hours debugging and the fix was a missing semicolon."),
    ("You are a curious friend who loves tangents.",
     "Why do you think people like rainy days?"),
]


def chat(url: str, system: str, user: str, *, seed: int, blue: bool, args) -> dict:
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "top_k": args.top_k,
        "seed": seed,
        "blue_noise": blue,
        "rng_type": args.rng_type,
        "logprobs": True,
        "top_logprobs": 10,
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def per_token(resp: dict) -> list[dict]:
    """Extract [{p, rank, u}] for each generated token from an OpenAI-style logprobs block."""
    lp = resp["choices"][0].get("logprobs") or {}
    out = []
    for item in lp.get("content", []):
        p_pick = math.exp(item["logprob"])
        tops = item.get("top_logprobs", [])
        rank, cdf_before = None, 0.0
        for r, t in enumerate(tops):
            if t.get("token_id") == item.get("token_id"):
                rank = r
                break
            cdf_before += math.exp(t["logprob"])
        # PIT midpoint; None when the pick fell outside the returned top list
        u = min(cdf_before + p_pick / 2, 1.0) if rank is not None else None
        out.append({"p": p_pick, "rank": rank, "u": u})
    return out


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def lag1_autocorr(xs: list[float]) -> float | None:
    if len(xs) < 4:
        return None
    m = statistics.fmean(xs)
    den = sum((x - m) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((xs[i] - m) * (xs[i + 1] - m) for i in range(len(xs) - 1)) / den


def window_var(xs: list[float], w: int) -> float | None:
    """Variance of non-overlapping window means: low-frequency energy of the sequence."""
    means = [statistics.fmean(xs[i:i + w]) for i in range(0, len(xs) - w + 1, w)]
    return statistics.pvariance(means) if len(means) >= 2 else None


def metrics_for(resp: dict, args) -> dict:
    toks = per_token(resp)
    text = resp["choices"][0]["message"]["content"] or ""
    surpr = [-math.log(max(t["p"], 1e-9)) for t in toks]
    tail = [t["p"] < args.tail_p for t in toks]
    longest = cur = 0
    for is_tail in tail:
        cur = cur + 1 if is_tail else 0
        longest = max(longest, cur)
    words = text.split()
    g4 = ngrams(words, 4)
    us = [t["u"] for t in toks if t["u"] is not None]
    return {
        "n_tokens": len(toks),
        "mean_surprisal": statistics.fmean(surpr) if surpr else None,
        "tail_frac": sum(tail) / len(tail) if tail else None,
        "longest_tail_run": longest,
        "surprisal_window_var": window_var(surpr, args.window),
        "repeat_4gram_frac": (1 - len(set(g4)) / len(g4)) if g4 else None,
        "roll_lag1_autocorr": lag1_autocorr(us),
        "roll_window_var": window_var(us, args.window),
        "_words": words,
        "_text": text,
    }


def jaccard3(a: list[str], b: list[str]) -> float:
    A, B = set(ngrams(a, 3)), set(ngrams(b, 3))
    return len(A & B) / len(A | B) if (A or B) else 0.0


def welch_t(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (statistics.fmean(b) - statistics.fmean(a)) / se if se > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:5001")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per prompt (per mode)")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--rng-type", default="mt19937", choices=["mt19937", "lowbias32"])
    ap.add_argument("--tail-p", type=float, default=0.10, help="a pick below this prob counts as a tail pick")
    ap.add_argument("--window", type=int, default=16, help="window size for low-frequency variance")
    ap.add_argument("--prompts", help="JSON file: [[system, user], ...] (default: built-in chat prompts)")
    ap.add_argument("--out", help="write raw per-generation metrics + texts here")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts)) if args.prompts else DEFAULT_PROMPTS
    results = {False: [], True: []}
    per_prompt_words = {False: {}, True: {}}

    total = len(prompts) * args.seeds * 2
    done = 0
    for pi, (system, user) in enumerate(prompts):
        for seed in range(1000, 1000 + args.seeds):
            for blue in (False, True):
                m = metrics_for(chat(args.url, system, user, seed=seed, blue=blue, args=args), args)
                m.update(prompt=pi, seed=seed, blue=blue)
                results[blue].append(m)
                per_prompt_words[blue].setdefault(pi, []).append(m["_words"])
                done += 1
                print(f"\r{done}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    # cross-seed diversity per prompt, per mode
    for blue in (False, True):
        for pi, gens in per_prompt_words[blue].items():
            sims = [jaccard3(a, b) for a, b in itertools.combinations(gens, 2)]
            for m in results[blue]:
                if m["prompt"] == pi:
                    m["cross_seed_jaccard3"] = statistics.fmean(sims) if sims else None

    keys = ["mean_surprisal", "tail_frac", "longest_tail_run", "surprisal_window_var",
            "repeat_4gram_frac", "cross_seed_jaccard3", "roll_lag1_autocorr", "roll_window_var", "n_tokens"]
    print(f"\n{'metric':<24}{'white':>12}{'blue':>12}{'diff':>12}{'welch t':>10}")
    for k in keys:
        a = [m[k] for m in results[False] if m.get(k) is not None]
        b = [m[k] for m in results[True] if m.get(k) is not None]
        if not a or not b:
            continue
        t = welch_t(a, b)
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        print(f"{k:<24}{ma:>12.4f}{mb:>12.4f}{mb - ma:>+12.4f}{(f'{t:+.2f}' if t is not None else '-'):>10}")
    print("\nroll_* rows check the mechanism (blue should be lower / more negative);"
          "\nmean_surprisal should NOT move much (blue noise must stay unbiased).")

    if args.out:
        for blue in (False, True):
            for m in results[blue]:
                m.pop("_words", None)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "white": results[False], "blue": results[True]}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
