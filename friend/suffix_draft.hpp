#pragma once

#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace friend_spec {

// Search the prompt and generated history for the longest recent suffix, choose
// its most frequent continuation, then repeat for the proposed continuation.
inline std::vector<llama_token> suffix_draft(
        const std::vector<llama_token> & history,
        llama_token pending,
        int max_tokens) {
    if (max_tokens <= 0 || history.size() < 2) return {};
    std::vector<llama_token> corpus = history;
    corpus.push_back(pending);
    std::vector<llama_token> out;
    out.reserve(max_tokens);

    for (int step = 0; step < max_tokens; ++step) {
        if (corpus.size() < 2) break;
        const size_t suffix_max = std::min<size_t>(16, corpus.size() - 1);
        llama_token best = 0;
        int best_count = 0;
        size_t best_pos = 0;
        bool found = false;
        for (size_t width = suffix_max; width >= 1; --width) {
            const size_t suffix_start = corpus.size() - width;
            std::unordered_map<llama_token, int> counts;
            std::unordered_map<llama_token, size_t> latest;
            for (size_t pos = 0; pos + width < suffix_start; ++pos) {
                if (!std::equal(corpus.begin() + pos, corpus.begin() + pos + width,
                                corpus.begin() + suffix_start)) continue;
                const llama_token candidate = corpus[pos + width];
                const int count = ++counts[candidate];
                latest[candidate] = pos;
                if (count > best_count || (count == best_count && latest[candidate] > best_pos)) {
                    best = candidate;
                    best_count = count;
                    best_pos = latest[candidate];
                    found = true;
                }
            }
            if (found || width == 1) break;
        }
        if (!found) break;
        out.push_back(best);
        corpus.push_back(best);
    }
    return out;
}

} // namespace friend_spec
