#pragma once

// Logical KV blocks used by the serving scheduler. llama.cpp still owns the
// device allocator, but keeping the content-addressed block table here gives
// the scheduler the same unit vLLM uses for prefix reuse and eviction. It also
// means a future paged backend can replace the state snapshot without changing
// request scheduling or cache keys.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace friend_kv {

struct block_table {
    static constexpr size_t block_tokens = 16;

    std::vector<uint64_t> hashes;
    size_t token_count = 0;

    static uint64_t mix(uint64_t h, int32_t token) {
        // SplitMix-style avalanche; deterministic across processes and cheap
        // enough to rebuild when a sequence is extended or rewound.
        uint64_t x = h ^ (uint32_t) token + UINT64_C(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }

    void clear() {
        hashes.clear();
        token_count = 0;
    }

    template <typename Tokens>
    void rebuild(const Tokens & tokens) {
        clear();
        token_count = tokens.size();
        const size_t blocks = (token_count + block_tokens - 1) / block_tokens;
        hashes.reserve(blocks);
        uint64_t parent = UINT64_C(0x243f6a8885a308d3);
        for(size_t begin = 0; begin < token_count; begin += block_tokens) {
            const size_t end = std::min(token_count, begin + block_tokens);
            uint64_t h = mix(parent, (int32_t) (end - begin));
            for(size_t i = begin; i < end; ++i) h = mix(h, (int32_t) tokens[i]);
            hashes.push_back(h);
            parent = h;
        }
    }

    size_t common_blocks(const block_table & other) const {
        const size_t n = std::min(hashes.size(), other.hashes.size());
        size_t i = 0;
        while(i < n && hashes[i] == other.hashes[i]) ++i;
        return i;
    }

    size_t bytes() const { return hashes.size() * sizeof(uint64_t); }
};

} // namespace friend_kv
