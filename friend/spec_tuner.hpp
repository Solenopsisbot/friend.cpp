#pragma once
// friend.cpp: adaptive speculative draft length.
//
// A fixed --draftamount is wrong most of the time: when the text is predictable, longer
// drafts pay; when it isn't, every rejected draft token is wasted verify compute, and on
// hardware where verifying k+1 tokens costs much more than decoding 1 (e.g. 1-bit models on
// Metal: +60% per extra token) drafting can end up *slower* than not drafting at all.
//
// The tuner learns, online and per run, everything the decision needs:
//   p            per-token acceptance probability (a draft token is accepted given all
//                earlier ones were), from a decayed success/failure count so it tracks the
//                text as it changes
//   verify_ms[n] time to decode a verify batch of n tokens (EMA per n)
//   draft_ms[k]  time the drafter takes to propose k tokens (EMA per k)
//   base_ms      time of a plain single-token decode
// and each round picks k in [0, k_max] maximising expected tokens per millisecond:
//   rate(k) = E[k] / (draft_ms[k] + verify_ms[k+1]),   E[k] = (1 - p^(k+1)) / (1 - p)
//   rate(0) = 1 / base_ms
// k = 0 means "don't draft this round". To keep estimates honest it occasionally probes:
// a plain decode now and then to refresh base_ms, and a small draft now and then while
// drafting is switched off, so it notices when the text becomes predictable again.

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>

namespace friend_spec {

class tuner {
public:
    void reset(int k_max_) {
        k_max = std::max(1, k_max_);
        verify_ms.assign(k_max + 2, 0.0);
        draft_ms.assign(k_max + 1, 0.0);
        base_ms = 0.0;
        succ = 0.0;
        fail = 0.0;
        round = 0;
        last_base_round = 0;
        last_draft_round = 0;
        last_k = k_max;
        probe_interval = 8;
        probing = false;
    }

    // Draft length for the next round (0 = plain decode).
    int choose() {
        ++round;
        if (k_max <= 0) {
            return 0;
        }
        // warm-up: a few mid-length drafts (real acceptance evidence), then a plain decode
        // for the baseline, before trusting the model
        if (base_ms == 0.0) {
            return (round <= 3) ? std::min(3, k_max) : 0;
        }
        // keep the plain-decode cost fresh (it drifts with context length and load)
        if (round - last_base_round > 64) {
            return 0;
        }
        const double p = acceptance();
        double best_rate = 1.0 / base_ms;
        int best_k = 0;
        for (int k = 1; k <= k_max; ++k) {
            const double e = (p >= 0.999) ? (k + 1.0) : (1.0 - std::pow(p, k + 1)) / (1.0 - p);
            const double cost = est_draft(k) + est_verify(k + 1);
            const double rate = e / cost;
            if (rate > best_rate) {
                best_rate = rate;
                best_k = k;
            }
        }
        // drafting off: probe every 8 rounds with the length that would win if acceptance
        // were high -- a 1-token probe can never show that the text got predictable again
        if (best_k == 0 && round - last_draft_round >= probe_interval) {
            best_k = optimistic_k();
            probing = true;
        }
        // light exploration of the neighbours so their costs get measured, not guessed
        if (best_k > 0 && (round % 13) == 0) {
            best_k = std::clamp(best_k + ((round / 13) % 2 ? 1 : -1), 1, k_max);
        }
        last_k = best_k;
        if (debug) {
            fprintf(stderr, "[spec-tuner] round %llu p=%.2f base=%.1fms ->k=%d |", (unsigned long long) round, p, base_ms, best_k);
            for (int k = 1; k <= k_max; ++k) fprintf(stderr, " k%d:d%.0f+v%.0f", k, est_draft(k), est_verify(k + 1));
            fprintf(stderr, "\n");
        }
        return best_k;
    }
    bool debug = false;

    // A drafted round: k proposed, `accepted` of them matched, with measured timings.
    void record_draft(int k, int accepted, double d_ms, double v_ms) {
        if (k <= 0) {
            return;
        }
        k = std::min(k, k_max);
        accepted = std::clamp(accepted, 0, k);
        // geometric model: each accepted token is a success; the first rejection a failure
        succ = decay * succ + accepted;
        fail = decay * fail + (accepted < k ? 1.0 : 0.0);
        if (probing) {
            // back off while probes keep failing; snap back as soon as one pays off
            probe_interval = (2 * accepted >= k) ? 8 : std::min<uint64_t>(probe_interval * 2, 64);
            probing = false;
        }
        ema(draft_ms[k], d_ms);
        ema(verify_ms[std::min(k + 1, (int) verify_ms.size() - 1)], v_ms);
        last_draft_round = round;
    }

    // A plain single-token decode.
    void record_plain(double ms) {
        ema(base_ms, ms);
        ema(verify_ms[1], ms);
        last_base_round = round;
    }

    double acceptance() const { return (succ + 1.0) / (succ + fail + 2.0); } // Laplace prior

    // A new request may be a different kind of text: keep the cost measurements (hardware),
    // but shrink the acceptance evidence so it re-learns within a few rounds.
    void new_request() {
        succ *= 0.3;
        fail *= 0.3;
        probe_interval = 8;
    }
    int last_choice() const { return last_k; }

private:
    static constexpr double decay = 0.95; // ~20 rounds of memory for acceptance
    int k_max = 0;
    std::vector<double> verify_ms, draft_ms;
    double base_ms = 0.0;
    double succ = 0.0, fail = 0.0;
    uint64_t round = 0, last_base_round = 0, last_draft_round = 0;
    int last_k = 0;
    uint64_t probe_interval = 8;
    bool probing = false;

    static void ema(double & slot, double v) {
        slot = slot == 0.0 ? v : 0.7 * slot + 0.3 * v;
    }

    // Unmeasured verify sizes: interpolate/extrapolate linearly from measured ones,
    // anchored at base_ms for n = 1. Never assume a batch is cheaper than one token.
    double est_verify(int n) const {
        if (n < (int) verify_ms.size() && verify_ms[n] > 0.0) {
            return verify_ms[n];
        }
        int lo = 1, hi = -1;
        for (int i = 1; i < (int) verify_ms.size(); ++i) {
            if (verify_ms[i] <= 0.0) continue;
            if (i < n) lo = i;
            if (i > n && hi < 0) hi = i;
        }
        const double vlo = lo == 1 ? base_ms : verify_ms[lo];
        if (hi > 0) {
            return vlo + (verify_ms[hi] - vlo) * (double) (n - lo) / (double) (hi - lo);
        }
        // extrapolate with the slope seen so far, or a pessimistic 60%/token prior
        double slope = 0.6 * base_ms;
        if (lo > 1) {
            slope = std::max(0.0, (verify_ms[lo] - base_ms) / (double) (lo - 1));
        }
        return std::max(base_ms, vlo + slope * (double) (n - lo));
    }

    // Unmeasured draft sizes: the nearest measured size, unscaled. Block drafters (DSpark,
    // DFlash) cost the same for any k because they always compute a whole block; MTP grows
    // with k, but it gets measured at the sizes actually used within a few rounds anyway.
    double est_draft(int k) const {
        if (draft_ms[k] > 0.0) {
            return draft_ms[k];
        }
        int best = -1;
        for (int i = 1; i < (int) draft_ms.size(); ++i) {
            if (draft_ms[i] > 0.0 && (best < 0 || std::abs(i - k) < std::abs(best - k))) best = i;
        }
        return best < 0 ? 0.3 * base_ms : draft_ms[best];
    }

    // The draft length that would be best if acceptance were high (p = 0.9); used for probes.
    int optimistic_k() const {
        const double p = 0.9;
        double best_rate = 0.0;
        int best_k = 1;
        for (int k = 1; k <= k_max; ++k) {
            const double rate = ((1.0 - std::pow(p, k + 1)) / (1.0 - p)) / (est_draft(k) + est_verify(k + 1));
            if (rate > best_rate) { best_rate = rate; best_k = k; }
        }
        return best_k;
    }
};

} // namespace friend_spec
