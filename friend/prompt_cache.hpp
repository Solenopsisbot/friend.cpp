#pragma once
// friend.cpp: tiered, content-addressed prompt (KV) cache.
//
// The store holds *snapshots* of one sequence's state (llama_state_seq_get_data): the
// exact tokens that are in the KV, the KV/recurrent state bytes, and the logits for the
// next position. A new prompt reuses the snapshot sharing the longest prefix with it.
//
//   - normal (attention-only) models: any entry is usable up to its longest common prefix
//     with the prompt, because KV can be truncated. So one entry holding
//     "system prompt + persona card + chat" serves every later prompt that starts with the
//     same system prompt + card, regardless of what follows.
//   - recurrent/hybrid models (Mamba, Qwen3.5 GDN, ...): state can't be rewound, so an
//     entry is usable only if it is a *full* prefix of the prompt ("exact" entries).
//
// Tiers: resident entries live in RAM up to a byte budget. With a disk directory set,
// every entry is also written through to disk asynchronously; RAM eviction then just
// drops the bytes, and entries survive restarts (reloaded lazily on a hit). Disk has its
// own byte budget. Pinned entries are never evicted from either tier.
//
// Keys that must match for an entry to be usable: the model/KV-layout fingerprint (baked
// into the disk directory name), the adapter identity (LoRA+steering, see adapters.hpp)
// and, for entries containing media placeholder tokens, the media hash.
//
// Thread-safety: all public methods lock the store. Capturing/restoring llama state is
// done by the caller (gpttype_adapter.cpp); the store only deals in bytes.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace friend_cache {

namespace fs = std::filesystem;

struct entry {
    uint64_t id = 0;
    std::vector<int32_t> tokens;     // exactly what is in the state (no trailing un-decoded token)
    std::string kv_key;              // adapter identity (adapters.hpp profile::kv_key)
    std::string media_hash;          // "" if the entry contains no media placeholder tokens
    bool exact_only = false;         // recurrent/hybrid snapshot: usable only as a full prefix
    bool pinned = false;
    std::string label;

    std::vector<uint8_t> state;      // main context seq state; empty when not resident
    std::vector<uint8_t> draft_state;// same-model draft context (MTP) seq state, optional
    std::vector<float> logits;       // next-token logits after `tokens`
    std::string logits_head_key;     // LM head that produced `logits`

    size_t state_bytes = 0;          // sizes, valid even when the bytes are only on disk
    size_t draft_bytes = 0;
    uint64_t last_used = 0;          // store clock
    bool on_disk = false;
    bool resident() const { return state_bytes == 0 || !state.empty(); }
    size_t ram_bytes() const { return state.size() + draft_state.size() + logits.size() * sizeof(float) + tokens.size() * sizeof(int32_t); }
    size_t disk_bytes() const { return state_bytes + draft_bytes + logits.size() * sizeof(float) + tokens.size() * sizeof(int32_t) + 256; }
};
using entry_ptr = std::shared_ptr<entry>;

struct match {
    entry_ptr e;
    size_t usable = 0; // tokens of the prompt this entry covers
};

// Longest common prefix of two token sequences.
inline size_t lcp(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

inline bool is_prefix(const std::vector<int32_t> & p, const std::vector<int32_t> & of) {
    return p.size() <= of.size() && std::equal(p.begin(), p.end(), of.begin());
}

class store {
public:
    struct config {
        std::string fingerprint;       // model + KV layout identity (hex); scopes the disk dir
        size_t ram_budget  = 0;        // bytes; 0 disables the cache entirely
        std::string disk_dir;          // "" = RAM only
        size_t disk_budget = 0;        // bytes
        size_t min_tokens  = 64;       // don't bother caching/reusing less than this
    };

    ~store() { shutdown(); }

    void configure(const config & c) {
        shutdown();
        std::lock_guard<std::mutex> lk(mu);
        cfg = c;
        entries.clear();
        clock = 0;
        next_id = 1;
        stop = false;
        if (!enabled_locked()) {
            return;
        }
        if (!cfg.disk_dir.empty()) {
            std::error_code ec;
            dir = fs::path(cfg.disk_dir) / cfg.fingerprint;
            fs::create_directories(dir, ec);
            if (ec) {
                fprintf(stderr, "friend-cache: cannot create %s (%s), disk tier disabled\n", dir.string().c_str(), ec.message().c_str());
                dir.clear();
            } else {
                load_index_locked();
                writer = std::thread([this] { writer_loop(); });
            }
        }
        std::string disk_desc;
        if (!dir.empty()) {
            disk_desc = ", disk " + dir.string() + " (budget " + std::to_string(cfg.disk_budget >> 20) + " MiB, " +
                        std::to_string(entries.size()) + " entries restored)";
        }
        printf("\nfriend-cache: RAM budget %.0f MiB%s\n", cfg.ram_budget / 1048576.0, disk_desc.c_str());
    }

    bool enabled() const {
        std::lock_guard<std::mutex> lk(mu);
        return enabled_locked();
    }
    size_t min_tokens() const { return cfg.min_tokens; }

    // Best entry for a prompt. `recurrent` restricts to full-prefix matches.
    match find(const std::vector<int32_t> & prompt, const std::string & kv_key, const std::string & media_hash, bool recurrent) {
        std::lock_guard<std::mutex> lk(mu);
        match best;
        for (auto & e : entries) {
            if (e->kv_key != kv_key) {
                continue;
            }
            // every entry under this fingerprint came from the same model, so whether the
            // state can be truncated is a property of the model, not of the entry
            size_t usable = recurrent ? (is_prefix(e->tokens, prompt) ? e->tokens.size() : 0)
                                      : lcp(e->tokens, prompt);
            if (!e->media_hash.empty() && e->media_hash != media_hash) {
                // media placeholders are identical ids across different media; stop before the first
                usable = std::min(usable, first_media_index(e->tokens));
                if (recurrent && usable < e->tokens.size()) {
                    usable = 0;
                }
            }
            if (usable > best.usable || (usable == best.usable && best.e && e->last_used > best.e->last_used)) {
                best.e = e;
                best.usable = usable;
            }
        }
        if (best.usable < cfg.min_tokens) {
            return {};
        }
        return best;
    }

    // Insert a snapshot. Deduplicates against entries it subsumes / is subsumed by.
    // Returns false if it was redundant (and was not stored).
    bool insert(entry_ptr n) {
        std::unique_lock<std::mutex> lk(mu);
        if (!enabled_locked() || n->tokens.size() < cfg.min_tokens) {
            return false;
        }
        n->state_bytes = n->state.size();
        n->draft_bytes = n->draft_state.size();
        for (auto it = entries.begin(); it != entries.end();) {
            entry & e = **it;
            const bool same_keys = e.kv_key == n->kv_key && e.media_hash == n->media_hash;
            if (!same_keys) {
                ++it;
                continue;
            }
            if (e.tokens == n->tokens) {
                // same content: refresh, keep the stronger flags
                e.pinned = e.pinned || n->pinned;
                if (!n->label.empty()) e.label = n->label;
                e.last_used = ++clock;
                return false;
            }
            // truncatable entries: a longer entry with the same prefix makes the shorter one redundant
            const bool truncatable = !e.exact_only && !n->exact_only;
            if (truncatable && is_prefix(n->tokens, e.tokens)) {
                if (n->pinned && !e.pinned) {
                    // keep coverage, but the pin belongs to the prefix the user asked for
                    ++it;
                    continue;
                }
                e.last_used = ++clock;
                return false;
            }
            if (truncatable && !e.pinned && is_prefix(e.tokens, n->tokens)) {
                drop_locked(*it);
                it = entries.erase(it);
                continue;
            }
            // exact snapshots on one lineage: drop near-duplicates (the newest wins within 48 tokens).
            // Kept small on purpose: turn-boundary checkpoints (persona end vs. final turn) can
            // legitimately sit close together and each serves different future prompts.
            // Turn-boundary snapshots (label "turn") are exempt: they mark where future prompts
            // diverge, which a nearby generic checkpoint does not.
            if (e.exact_only && n->exact_only && !e.pinned && e.label != "turn" && n->label != "turn" &&
                (is_prefix(e.tokens, n->tokens) || is_prefix(n->tokens, e.tokens)) &&
                (e.tokens.size() > n->tokens.size() ? e.tokens.size() - n->tokens.size() : n->tokens.size() - e.tokens.size()) <= 48) {
                drop_locked(*it);
                it = entries.erase(it);
                continue;
            }
            ++it;
        }
        n->id = next_id++;
        n->last_used = ++clock;
        entries.push_back(n);
        if (!dir.empty()) {
            write_queue.push_back(n);
            cv.notify_one();
        }
        enforce_budgets_locked();
        return true;
    }

    // Make sure the state bytes are in RAM (reads them back from disk if needed).
    bool ensure_resident(const entry_ptr & e) {
        std::lock_guard<std::mutex> lk(mu);
        e->last_used = ++clock;
        if (e->resident()) {
            return true;
        }
        if (dir.empty() || !e->on_disk) {
            return false;
        }
        std::ifstream f(state_path(*e), std::ios::binary);
        if (!f) {
            return false;
        }
        e->state.resize(e->state_bytes);
        e->draft_state.resize(e->draft_bytes);
        f.read((char *) e->state.data(), e->state_bytes);
        f.read((char *) e->draft_state.data(), e->draft_bytes);
        if (!f) {
            e->state.clear();
            e->draft_state.clear();
            return false;
        }
        enforce_budgets_locked(e.get());
        return true;
    }

    void forget(const entry_ptr & e) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = std::find(entries.begin(), entries.end(), e);
        if (it != entries.end()) {
            drop_locked(*it, true);
            entries.erase(it);
        }
    }

    struct info {
        uint64_t id; size_t n_tokens; size_t bytes; bool resident; bool on_disk; bool pinned; bool exact_only;
        std::string label; std::string kv_key; uint64_t age;
    };
    std::vector<info> list() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<info> out;
        for (auto & e : entries) {
            out.push_back({ e->id, e->tokens.size(), e->state_bytes + e->draft_bytes, e->resident(), e->on_disk, e->pinned,
                            e->exact_only, e->label, e->kv_key, clock - e->last_used });
        }
        return out;
    }

    // Removes entries; pinned ones only if include_pinned. Returns how many were removed.
    size_t clear(bool include_pinned) {
        std::lock_guard<std::mutex> lk(mu);
        size_t n = 0;
        for (auto it = entries.begin(); it != entries.end();) {
            if ((*it)->pinned && !include_pinned) {
                ++it;
                continue;
            }
            drop_locked(*it, true);
            it = entries.erase(it);
            ++n;
        }
        return n;
    }

    bool set_pinned(uint64_t id, bool pinned) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & e : entries) {
            if (e->id == id) {
                e->pinned = pinned;
                if (!dir.empty()) {
                    write_queue.push_back(e); // persist the flag
                    cv.notify_one();
                }
                return true;
            }
        }
        return false;
    }

    void stats(size_t & n, size_t & ram, size_t & disk) {
        std::lock_guard<std::mutex> lk(mu);
        n = entries.size();
        ram = ram_used_locked();
        disk = disk_used_locked();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (writer.joinable()) {
            writer.join();
        }
    }

private:
    static constexpr uint32_t MAGIC = 0x46434b56; // "FCKV"
    static constexpr uint32_t VERSION = 1;

    mutable std::mutex mu;
    std::condition_variable cv;
    config cfg;
    fs::path dir;
    std::vector<entry_ptr> entries;
    std::deque<entry_ptr> write_queue;
    std::thread writer;
    bool stop = false;
    uint64_t clock = 0;
    uint64_t next_id = 1;

    bool enabled_locked() const { return cfg.ram_budget > 0; }

    static size_t first_media_index(const std::vector<int32_t> & t) {
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] < 0) {
                return i; // kcpp media placeholders are negative ids
            }
        }
        return t.size();
    }

    fs::path meta_path(const entry & e) const { return dir / (std::to_string(e.id) + ".meta"); }
    fs::path state_path(const entry & e) const { return dir / (std::to_string(e.id) + ".kv"); }

    size_t ram_used_locked() const {
        size_t s = 0;
        for (auto & e : entries) s += e->ram_bytes();
        return s;
    }
    size_t disk_used_locked() const {
        size_t s = 0;
        for (auto & e : entries) if (e->on_disk) s += e->disk_bytes();
        return s;
    }

    void drop_locked(const entry_ptr & e, bool delete_files = true) {
        if (delete_files && !dir.empty()) {
            std::error_code ec;
            fs::remove(meta_path(*e), ec);
            fs::remove(state_path(*e), ec);
        }
        e->on_disk = false;
    }

    // Evict LRU unpinned entries until both budgets hold. `keep` is never evicted from RAM.
    void enforce_budgets_locked(const entry * keep = nullptr) {
        auto lru = [&](auto pred) -> entry_ptr {
            entry_ptr victim;
            for (auto & e : entries) {
                if (e->pinned || e.get() == keep || !pred(*e)) continue;
                if (!victim || e->last_used < victim->last_used) victim = e;
            }
            return victim;
        };
        // RAM: entries already safe on disk just drop their bytes; others are removed outright.
        // Entries still queued for writing are skipped (their bytes are needed by the writer).
        while (ram_used_locked() > cfg.ram_budget) {
            entry_ptr v = lru([&](const entry & e) {
                return !e.state.empty() && std::find(write_queue.begin(), write_queue.end(), entries_ptr(e)) == write_queue.end();
            });
            if (!v) break;
            if (v->on_disk) {
                std::vector<uint8_t>().swap(v->state);
                std::vector<uint8_t>().swap(v->draft_state);
            } else {
                drop_locked(v, false);
                entries.erase(std::find(entries.begin(), entries.end(), v));
            }
        }
        if (!dir.empty()) {
            while (disk_used_locked() > cfg.disk_budget) {
                entry_ptr v = lru([](const entry & e) { return e.on_disk; });
                if (!v) break;
                const bool resident = !v->state.empty();
                drop_locked(v, true);
                if (!resident) {
                    entries.erase(std::find(entries.begin(), entries.end(), v));
                }
            }
        }
    }

    entry_ptr entries_ptr(const entry & e) const {
        for (auto & p : entries) if (p.get() == &e) return p;
        return nullptr;
    }

    // --- disk format: <id>.meta (small, read at startup) + <id>.kv (state bytes) ---
    void writer_loop() {
        std::unique_lock<std::mutex> lk(mu);
        while (true) {
            cv.wait(lk, [&] { return stop || !write_queue.empty(); });
            if (write_queue.empty() && stop) {
                return;
            }
            entry_ptr e = write_queue.front();
            // snapshot what we need while holding the lock; the bytes themselves are only
            // freed by enforce_budgets_locked, which skips queued entries
            const bool still_listed = std::find(entries.begin(), entries.end(), e) != entries.end();
            lk.unlock();
            bool ok = false;
            if (still_listed && e->on_disk) {
                ok = write_meta(*e); // already persisted: only flags (pin) changed
            } else if (still_listed && !e->state.empty()) {
                ok = write_entry(*e);
            }
            lk.lock();
            write_queue.pop_front();
            const bool listed_now = std::find(entries.begin(), entries.end(), e) != entries.end();
            if (ok && listed_now) {
                e->on_disk = true;
                enforce_budgets_locked();
            } else if (ok && !listed_now) {
                // deduped/cleared while we were writing: don't leave it to be resurrected at startup
                std::error_code ec;
                fs::remove(meta_path(*e), ec);
                fs::remove(state_path(*e), ec);
            } else if (!ok && still_listed) {
                fprintf(stderr, "friend-cache: failed to write entry %llu to disk\n", (unsigned long long) e->id);
            }
        }
    }

    bool write_meta(const entry & e) {
        const fs::path tmp = meta_path(e).string() + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            auto w32 = [&](uint32_t v) { f.write((const char *) &v, 4); };
            auto w64 = [&](uint64_t v) { f.write((const char *) &v, 8); };
            auto wstr = [&](const std::string & s) { w32((uint32_t) s.size()); f.write(s.data(), s.size()); };
            w32(MAGIC); w32(VERSION);
            w64(e.id);
            w32((e.exact_only ? 1u : 0u) | (e.pinned ? 2u : 0u));
            wstr(e.kv_key); wstr(e.media_hash); wstr(e.label); wstr(e.logits_head_key);
            w64(e.state_bytes); w64(e.draft_bytes);
            w64(e.tokens.size()); f.write((const char *) e.tokens.data(), e.tokens.size() * 4);
            w64(e.logits.size()); f.write((const char *) e.logits.data(), e.logits.size() * 4);
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, meta_path(e), ec);
        return !ec;
    }

    bool write_entry(const entry & e) {
        const fs::path tmp = state_path(e).string() + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            f.write((const char *) e.state.data(), e.state.size());
            f.write((const char *) e.draft_state.data(), e.draft_state.size());
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, state_path(e), ec);
        return !ec && write_meta(e); // meta last: an entry is only listed once its bytes exist
    }

    void load_index_locked() {
        std::error_code ec;
        for (auto & de : fs::directory_iterator(dir, ec)) {
            if (de.path().extension() != ".meta") {
                if (de.path().extension() == ".tmp") fs::remove(de.path(), ec); // interrupted write
                continue;
            }
            std::ifstream f(de.path(), std::ios::binary);
            auto r32 = [&]() { uint32_t v = 0; f.read((char *) &v, 4); return v; };
            auto r64 = [&]() { uint64_t v = 0; f.read((char *) &v, 8); return v; };
            auto rstr = [&]() { std::string s(r32(), '\0'); f.read(s.data(), s.size()); return s; };
            if (r32() != MAGIC || r32() != VERSION) continue;
            auto e = std::make_shared<entry>();
            e->id = r64();
            const uint32_t flags = r32();
            e->exact_only = flags & 1u;
            e->pinned = flags & 2u;
            e->kv_key = rstr(); e->media_hash = rstr(); e->label = rstr(); e->logits_head_key = rstr();
            e->state_bytes = r64(); e->draft_bytes = r64();
            const uint64_t nt = r64();
            if (!f || nt > (1u << 24)) continue;
            e->tokens.resize(nt);
            f.read((char *) e->tokens.data(), nt * 4);
            const uint64_t nl = r64();
            if (!f || nl > (1u << 24)) continue;
            e->logits.resize(nl);
            f.read((char *) e->logits.data(), nl * 4);
            if (!f || !fs::exists(state_path(*e), ec)) continue;
            e->on_disk = true;
            e->last_used = 0;
            next_id = std::max(next_id, e->id + 1);
            entries.push_back(e);
        }
    }
};

inline store & global() {
    static store s;
    return s;
}

// FNV-1a, for fingerprints and media hashes (stable across runs, unlike std::hash)
inline std::string hash_hex(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

} // namespace friend_cache
