#pragma once
// friend.cpp: lossless compression for prompt-cache snapshots (llama_state_seq_get_data bytes).
//
// What the bytes are: mostly f16 KV rows (attention layers) and, on hybrid models, f32
// recurrent state, with a sprinkling of small integer headers. Floats compress badly as a
// byte stream -- the sign/exponent byte is fairly predictable (~5 bits of entropy in f16
// KV) but it is interleaved with mantissa bytes that are close to random, so an LZ/entropy
// coder sees noise. Splitting every 4-byte group into 4 byte planes (byte 0 of every group,
// then byte 1, ...) puts the exponent bytes next to each other (plane 1 and 3 for f16,
// plane 3 for f32), where zstd's Huffman stage can squeeze them. Measured on real
// snapshots (see FRIEND.md) that roughly doubles the gain over plain zstd: ~1.20x on f16
// KV, ~1.18x on a hybrid model, versus ~1.07-1.10x unsplit. LZ4 gains nothing (<1.01x):
// there is almost no repetition to find, only skewed byte statistics.
//
// Format ("FCZ1"), little-endian:
//   u32 magic, u32 chunk_size, u64 raw_size, u32 n_chunks, u32 comp_size[n_chunks], chunks...
// Each chunk holds `chunk_size` raw bytes (the last one may be shorter), byte-plane split
// and then compressed as an independent zstd frame (with a content checksum), so chunks
// can be (de)compressed in parallel and a corrupted file is detected, not restored.
//
// Restores are bit-exact: the transform and zstd are both lossless, and unpack() verifies
// every size and checksum before reporting success.

#include <zstd/zstd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace friend_cache {
namespace codec {

static constexpr uint32_t MAGIC      = 0x315a4346; // "FCZ1"
static constexpr uint32_t CHUNK      = 4u << 20;   // 4 MiB: dozens of chunks per snapshot to spread over threads
static constexpr int      ZSTD_LEVEL = 1;          // level 3 buys <0.5% more ratio at half the speed; negative levels skip Huffman and gain ~nothing
static constexpr size_t   HEADER     = 20;

// Byte-plane split of n bytes (n4 = n / 4 groups; the <4-byte tail is copied as is).
// Written as plain strided loops so clang/gcc turn them into ld4/st4-style shuffles.
inline void split4(const uint8_t * in, uint8_t * out, size_t n) {
    const size_t n4 = n / 4;
    uint8_t * p0 = out, * p1 = out + n4, * p2 = out + 2 * n4, * p3 = out + 3 * n4;
    for (size_t i = 0; i < n4; ++i) {
        p0[i] = in[4 * i + 0];
        p1[i] = in[4 * i + 1];
        p2[i] = in[4 * i + 2];
        p3[i] = in[4 * i + 3];
    }
    memcpy(out + 4 * n4, in + 4 * n4, n - 4 * n4);
}

inline void unsplit4(const uint8_t * in, uint8_t * out, size_t n) {
    const size_t n4 = n / 4;
    const uint8_t * p0 = in, * p1 = in + n4, * p2 = in + 2 * n4, * p3 = in + 3 * n4;
    for (size_t i = 0; i < n4; ++i) {
        out[4 * i + 0] = p0[i];
        out[4 * i + 1] = p1[i];
        out[4 * i + 2] = p2[i];
        out[4 * i + 3] = p3[i];
    }
    memcpy(out + 4 * n4, in + 4 * n4, n - 4 * n4);
}

// Runs fn(i) for i in [0, n) on up to `threads` threads (the caller's thread included).
template <typename F>
inline void parallel_for(size_t n, unsigned threads, F && fn) {
    threads = (unsigned) std::min<size_t>(std::max(1u, threads), n);
    if (threads <= 1) {
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < n;) fn(i);
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto & t : pool) t.join();
}

inline unsigned default_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return std::max(1u, std::min(hw ? hw : 4u, 8u));
}

inline uint32_t rd32(const uint8_t * p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint64_t rd64(const uint8_t * p) { uint64_t v; memcpy(&v, p, 8); return v; }

// Compress `n` bytes. Never fails for sane input (zstd stores incompressible data raw).
inline std::vector<uint8_t> pack(const uint8_t * src, size_t n, unsigned threads = 1) {
    const size_t n_chunks = (n + CHUNK - 1) / CHUNK;
    std::vector<std::vector<uint8_t>> parts(n_chunks);
    bool ok = true;
    parallel_for(n_chunks, threads, [&](size_t i) {
        const size_t off = i * CHUNK, len = std::min<size_t>(CHUNK, n - off);
        std::vector<uint8_t> planes(len);
        split4(src + off, planes.data(), len);
        ZSTD_CCtx * cctx = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, ZSTD_LEVEL);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);
        parts[i].resize(ZSTD_compressBound(len));
        const size_t got = ZSTD_compress2(cctx, parts[i].data(), parts[i].size(), planes.data(), len);
        ZSTD_freeCCtx(cctx);
        if (ZSTD_isError(got)) {
            ok = false;
            parts[i].clear();
        } else {
            parts[i].resize(got);
        }
    });
    if (!ok) {
        return {};
    }
    size_t total = HEADER + 4 * n_chunks;
    for (auto & p : parts) total += p.size();
    std::vector<uint8_t> out(total);
    uint8_t * w = out.data();
    auto w32 = [&](uint32_t v) { memcpy(w, &v, 4); w += 4; };
    const uint64_t raw = n;
    w32(MAGIC); w32(CHUNK); memcpy(w, &raw, 8); w += 8; w32((uint32_t) n_chunks);
    for (auto & p : parts) w32((uint32_t) p.size());
    for (auto & p : parts) { memcpy(w, p.data(), p.size()); w += p.size(); }
    return out;
}

// Uncompressed size recorded in a packed blob, or 0 if it isn't one.
inline size_t raw_size(const uint8_t * src, size_t n) {
    return n >= HEADER && rd32(src) == MAGIC ? (size_t) rd64(src + 8) : 0;
}

// Decompress into `out` (resized to the raw size). False on any corruption/mismatch.
inline bool unpack(const uint8_t * src, size_t n, std::vector<uint8_t> & out, unsigned threads = default_threads()) {
    if (n < HEADER || rd32(src) != MAGIC) {
        return false;
    }
    const uint32_t chunk = rd32(src + 4);
    const uint64_t raw = rd64(src + 8);
    const uint32_t n_chunks = rd32(src + 16);
    if (chunk == 0 || n_chunks != (raw + chunk - 1) / chunk || HEADER + 4ull * n_chunks > n) {
        return false;
    }
    std::vector<size_t> offs(n_chunks + 1);
    offs[0] = HEADER + 4ull * n_chunks;
    for (uint32_t i = 0; i < n_chunks; ++i) {
        offs[i + 1] = offs[i] + rd32(src + HEADER + 4ull * i);
    }
    if (offs[n_chunks] != n) {
        return false;
    }
    out.resize(raw);
    std::atomic<bool> ok{true};
    parallel_for(n_chunks, threads, [&](size_t i) {
        const size_t off = i * (size_t) chunk, len = std::min<size_t>(chunk, raw - off);
        std::vector<uint8_t> planes(len);
        const size_t got = ZSTD_decompress(planes.data(), len, src + offs[i], offs[i + 1] - offs[i]);
        if (ZSTD_isError(got) || got != len) {
            ok = false;
            return;
        }
        unsplit4(planes.data(), out.data() + off, len);
    });
    if (!ok) {
        out.clear();
    }
    return ok;
}

} // namespace codec
} // namespace friend_cache
