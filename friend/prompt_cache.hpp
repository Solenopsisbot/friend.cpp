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
// Compression (state_codec.hpp): a background worker thread packs every new snapshot
// losslessly (byte-plane split + zstd, ~1.2x on f16 KV). Until the worker gets to an
// entry it sits in RAM uncompressed, so a capture never waits for compression. The disk
// tier always stores packed bytes; RAM keeps them packed too when that saves enough to be
// worth a decompress per restore (worth_packing), which costs a few ms per 100 MiB on
// several threads. Restores are bit-exact. Both budgets count the bytes actually stored.
//
// Keys that must match for an entry to be usable: the model/KV-layout fingerprint (baked
// into the disk directory name), the adapter identity (LoRA+steering, see adapters.hpp)
// and, for entries containing media placeholder tokens, the media hash.
//
// Thread-safety: all public methods lock the store. Capturing/restoring llama state is
// done by the caller (gpttype_adapter.cpp); the store only deals in bytes. Byte buffers
// are immutable and shared (bytes_ptr), so whoever holds one -- a restore in progress, the
// worker compressing or writing -- keeps it alive even if eviction or the worker swaps
// the entry's copy out meanwhile. Compression, decompression and file I/O all run with
// the store unlocked.

#include "state_codec.hpp"

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

using bytes = std::vector<uint8_t>;
using bytes_ptr = std::shared_ptr<const bytes>;

inline bytes_ptr share(bytes && b) { return std::make_shared<const bytes>(std::move(b)); }

// One serialized llama state (main or draft context), in whichever forms are at hand.
struct blob {
    size_t size = 0;        // uncompressed bytes; valid even when nothing is in RAM
    size_t packed_size = 0; // compressed bytes once packed (0 = not yet / raw on disk)
    bytes_ptr raw;          // uncompressed bytes in RAM (fresh capture, or not worth packing)
    bytes_ptr packed;       // compressed bytes in RAM
    bool in_ram() const { return size == 0 || raw || packed; }
    size_t ram() const { return (raw ? raw->size() : 0) + (packed ? packed->size() : 0); }
    size_t disk() const { return packed_size ? packed_size : size; } // bytes in the .kv file
    void release() { raw.reset(); packed.reset(); }
};

// Keep a blob packed in RAM only if that saves at least ~5%. Below that (quantised KV,
// say) every restore would pay for a decompress in exchange for next to no RAM.
inline bool worth_packing(size_t raw, size_t packed) { return packed + raw / 20 <= raw; }

struct entry {
    uint64_t id = 0;
    std::vector<int32_t> tokens;     // exactly what is in the state (no trailing un-decoded token)
    std::string kv_key;              // adapter identity (adapters.hpp profile::kv_key)
    std::string media_hash;          // "" if the entry contains no media placeholder tokens
    bool exact_only = false;         // recurrent/hybrid snapshot: usable only as a full prefix
    bool pinned = false;
    std::string label;

    blob state;                      // main context seq state
    blob draft;                      // same-model draft context (MTP / DSpark) seq state, optional
    std::vector<float> logits;       // next-token logits after `tokens`
    std::string logits_head_key;     // LM head that produced `logits`

    uint64_t last_used = 0;          // store clock
    bool on_disk = false;
    bool kv_raw = false;             // the .kv file holds raw bytes (written by disk format v1)
    bool packing = false;            // inserted, not yet through the worker: RAM holds it uncompressed
    bool resident() const { return state.in_ram() && draft.in_ram(); }
    size_t payload_ram() const { return state.ram() + draft.ram(); }
    size_t ram_bytes() const { return payload_ram() + logits.size() * sizeof(float) + tokens.size() * sizeof(int32_t); }
    size_t disk_bytes() const { return state.disk() + draft.disk() + logits.size() * sizeof(float) + tokens.size() * sizeof(int32_t) + 256; }
};
using entry_ptr = std::shared_ptr<entry>;

struct match {
    entry_ptr e;
    size_t usable = 0; // tokens of the prompt this entry covers
};

// Uncompressed state bytes of an entry, ready for llama_state_seq_set_data.
struct payload {
    bytes_ptr state;
    bytes_ptr draft; // empty if the entry has no draft state
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
        work_queue.clear();
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
            }
        }
        // the worker compresses new entries even without a disk tier
        worker = std::thread([this] { worker_loop(); });
        std::string disk_desc;
        if (!dir.empty()) {
            disk_desc = ", disk " + dir.string() + " (budget " + std::to_string(cfg.disk_budget >> 20) + " MiB, " +
                        std::to_string(entries.size()) + " entries restored)";
        }
        printf("\nfriend-cache: RAM budget %.0f MiB%s, compressed snapshots\n", cfg.ram_budget / 1048576.0, disk_desc.c_str());
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

    // Is this context already represented? Identical tokens, or -- for truncatable
    // (attention) models -- a prefix of an existing entry, under the same adapter/media keys.
    // Lets callers skip the device->host state copy for a snapshot insert() would discard
    // as redundant anyway. Touches the covering entry so it stays warm.
    bool covers(const std::vector<int32_t> & tokens, const std::string & kv_key, const std::string & media_hash, bool recurrent) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & e : entries) {
            if (e->kv_key != kv_key || e->media_hash != media_hash) {
                continue;
            }
            if (e->tokens == tokens || (!recurrent && !e->exact_only && is_prefix(tokens, e->tokens))) {
                e->last_used = ++clock;
                return true;
            }
        }
        return false;
    }

    // Insert a snapshot (state.raw / draft.raw set, uncompressed). Deduplicates against
    // entries it subsumes / is subsumed by. Returns false if it was redundant (and was not stored).
    bool insert(entry_ptr n) {
        std::unique_lock<std::mutex> lk(mu);
        if (!enabled_locked() || n->tokens.size() < cfg.min_tokens) {
            return false;
        }
        n->state.size = n->state.raw ? n->state.raw->size() : 0;
        n->draft.size = n->draft.raw ? n->draft.raw->size() : 0;
        n->state.packed_size = n->draft.packed_size = 0;
        n->packing = true;
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
        work_queue.push_back(n); // compress (+ write through to disk)
        cv.notify_one();
        enforce_budgets_locked();
        return true;
    }

    // Uncompressed state bytes of an entry, from RAM or (reading it back) from disk.
    // Decompression and file reads happen with the store unlocked. False if the bytes
    // are gone or corrupt; the caller should then forget() the entry.
    bool load(const entry_ptr & e, payload & out) {
        std::unique_lock<std::mutex> lk(mu);
        e->last_used = ++clock;
        blob s = e->state, d = e->draft; // copies of the shared pointers + sizes
        const bool in_ram = e->resident();
        const bool kv_raw = e->kv_raw;
        fs::path path;
        if (!in_ram) {
            if (dir.empty() || !e->on_disk) {
                return false;
            }
            path = state_path(*e);
        }
        lk.unlock();

        if (!in_ram) {
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                return false;
            }
            auto rd = [&](size_t n) {
                bytes b(n);
                f.read((char *) b.data(), n);
                return share(std::move(b));
            };
            if (kv_raw) {
                s.raw = rd(s.size);
                d.raw = rd(d.size);
            } else {
                s.packed = rd(s.packed_size);
                d.packed = d.packed_size ? rd(d.packed_size) : nullptr;
            }
            if (!f) {
                return false;
            }
        }
        auto materialize = [](blob & b) {
            if (b.raw || b.size == 0) {
                return true;
            }
            bytes raw;
            if (!b.packed || !codec::unpack(b.packed->data(), b.packed->size(), raw) || raw.size() != b.size) {
                return false;
            }
            b.raw = share(std::move(raw));
            return true;
        };
        if (!materialize(s) || !materialize(d)) {
            fprintf(stderr, "friend-cache: entry %llu is corrupt, dropping it\n", (unsigned long long) e->id);
            return false;
        }

        if (!in_ram) {
            // keep it resident in the same form a fresh entry would settle into
            lk.lock();
            if (is_listed_locked(e) && !e->resident()) {
                auto install = [](blob & dst, const blob & src) {
                    if (src.packed && worth_packing(src.size, src.packed->size())) {
                        dst.packed = src.packed;
                    } else {
                        dst.raw = src.raw;
                    }
                };
                install(e->state, s);
                install(e->draft, d);
                enforce_budgets_locked(e.get());
            }
            lk.unlock();
        }
        out.state = s.raw ? s.raw : std::make_shared<const bytes>();
        out.draft = d.raw ? d.raw : std::make_shared<const bytes>();
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
        uint64_t id; size_t n_tokens; size_t bytes; size_t ram; size_t disk; bool resident; bool on_disk; bool pinned; bool exact_only;
        std::string label; std::string kv_key; uint64_t age;
    };
    std::vector<info> list() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<info> out;
        for (auto & e : entries) {
            out.push_back({ e->id, e->tokens.size(), e->state.size + e->draft.size, e->payload_ram(),
                            e->on_disk ? e->state.disk() + e->draft.disk() : 0, e->resident(), e->on_disk, e->pinned,
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
                    work_queue.push_back(e); // persist the flag
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
        if (worker.joinable()) {
            worker.join();
        }
    }

private:
    static constexpr uint32_t MAGIC = 0x46434b56; // "FCKV"
    // v1: .kv = raw state bytes then raw draft bytes.
    // v2: .kv = packed state blob then packed draft blob (codec::pack); meta records both
    //     packed sizes, plus flag 4 for a v1 entry whose .kv is still raw (pin rewrote its meta).
    static constexpr uint32_t VERSION = 2;

    mutable std::mutex mu;
    std::condition_variable cv;
    config cfg;
    fs::path dir;
    std::vector<entry_ptr> entries;
    std::deque<entry_ptr> work_queue; // new entries to compress (+ write), and pin changes to persist
    std::thread worker;
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

    bool is_listed_locked(const entry_ptr & e) const { return std::find(entries.begin(), entries.end(), e) != entries.end(); }

    // `settled`: leave out the payload of entries still waiting to be packed. Budget
    // enforcement uses that: those bytes can't be evicted anyway (the worker needs them),
    // and charging a fresh capture at its uncompressed size would evict older entries to
    // make room the entry won't need once packed. The worker re-enforces after packing.
    size_t ram_used_locked(bool settled = false) const {
        size_t s = 0;
        for (auto & e : entries) s += e->ram_bytes() - (settled && e->packing ? e->payload_ram() : 0);
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
        // Entries still queued for the worker are skipped: they are about to shrink (packing)
        // and/or be persisted, and dropping them now would lose them for good.
        while (ram_used_locked(true) > cfg.ram_budget) {
            entry_ptr v = lru([&](const entry & e) {
                return e.payload_ram() > 0 && std::find(work_queue.begin(), work_queue.end(), entries_ptr(e)) == work_queue.end();
            });
            if (!v) break;
            if (v->on_disk) {
                v->state.release();
                v->draft.release();
            } else {
                drop_locked(v, false);
                entries.erase(std::find(entries.begin(), entries.end(), v));
            }
        }
        if (!dir.empty()) {
            while (disk_used_locked() > cfg.disk_budget) {
                entry_ptr v = lru([](const entry & e) { return e.on_disk; });
                if (!v) break;
                const bool resident = v->payload_ram() > 0;
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

    // Background worker: pack new entries, swap the packed bytes in for the raw ones, and
    // write entries through to disk. Each step holds the lock only to read or publish
    // shared pointers; packing and file I/O run unlocked.
    void worker_loop() {
        std::unique_lock<std::mutex> lk(mu);
        while (true) {
            cv.wait(lk, [&] { return stop || !work_queue.empty(); });
            // on shutdown, finish pending disk writes (they survive the restart); RAM-only
            // compression would be wasted work
            if (stop && (work_queue.empty() || dir.empty())) {
                return;
            }
            entry_ptr e = work_queue.front();
            const bool listed = is_listed_locked(e);
            const bool meta_only = listed && e->on_disk; // already persisted: only flags (pin) changed
            blob s = e->state, d = e->draft;
            lk.unlock();

            bool pack_ok = true;
            if (listed && !meta_only) {
                auto pack = [&](blob & b) {
                    if (b.packed || !b.raw || b.size == 0) {
                        return;
                    }
                    bytes p = codec::pack(b.raw->data(), b.raw->size());
                    if (p.empty()) {
                        pack_ok = false;
                        return;
                    }
                    b.packed = share(std::move(p));
                    b.packed_size = b.packed->size();
                };
                pack(s);
                pack(d);
            }

            lk.lock();
            bytes meta;
            const bool listed_now = is_listed_locked(e);
            if (listed_now && !meta_only && pack_ok) {
                // publish: RAM keeps whichever form is worth keeping (see worth_packing)
                auto publish = [](blob & dst, const blob & src) {
                    if (!src.packed || dst.packed) {
                        return;
                    }
                    dst.packed_size = src.packed_size;
                    if (!dst.raw) {
                        return; // evicted meanwhile (can't happen while queued, but be safe)
                    }
                    if (worth_packing(src.size, src.packed_size)) {
                        dst.packed = src.packed;
                        dst.raw.reset();
                    }
                };
                publish(e->state, s);
                publish(e->draft, d);
            }
            if (!meta_only && e->packing) {
                e->packing = false; // settled (packed, or left raw if packing failed): charge it in full now
                enforce_budgets_locked();
            }
            const bool write = !dir.empty() && listed_now && (meta_only || pack_ok);
            if (write) {
                meta = encode_meta_locked(*e); // snapshot under the lock: pin/label may change
            }
            lk.unlock();

            bool ok = false;
            if (write && meta_only) {
                ok = write_file(meta_path(*e), { &meta });
            } else if (write) {
                // .kv first, meta last: an entry is only listed at startup once its bytes exist
                ok = write_file(state_path(*e), { s.packed.get(), d.packed.get() }) && write_file(meta_path(*e), { &meta });
            }

            lk.lock();
            work_queue.pop_front();
            const bool listed_after = is_listed_locked(e);
            if (ok && listed_after) {
                e->on_disk = true;
                if (!meta_only) {
                    e->kv_raw = false;
                }
            } else if (ok && !listed_after) {
                // deduped/cleared while we were writing: don't leave it to be resurrected at startup
                std::error_code ec;
                fs::remove(meta_path(*e), ec);
                fs::remove(state_path(*e), ec);
            } else if (write && listed_after) {
                fprintf(stderr, "friend-cache: failed to write entry %llu to disk\n", (unsigned long long) e->id);
            } else if (!pack_ok) {
                fprintf(stderr, "friend-cache: failed to compress entry %llu, keeping it uncompressed in RAM\n", (unsigned long long) e->id);
            }
            enforce_budgets_locked(); // sizes changed (packed) and/or the entry now counts on disk
        }
    }

    // --- disk format: <id>.meta (small, read at startup) + <id>.kv (state bytes) ---
    bytes encode_meta_locked(const entry & e) const {
        bytes out;
        auto put = [&](const void * p, size_t n) { out.insert(out.end(), (const uint8_t *) p, (const uint8_t *) p + n); };
        auto w32 = [&](uint32_t v) { put(&v, 4); };
        auto w64 = [&](uint64_t v) { put(&v, 8); };
        auto wstr = [&](const std::string & s) { w32((uint32_t) s.size()); put(s.data(), s.size()); };
        w32(MAGIC); w32(VERSION);
        w64(e.id);
        w32((e.exact_only ? 1u : 0u) | (e.pinned ? 2u : 0u) | (e.kv_raw ? 4u : 0u));
        wstr(e.kv_key); wstr(e.media_hash); wstr(e.label); wstr(e.logits_head_key);
        w64(e.state.size); w64(e.draft.size);
        w64(e.state.packed_size); w64(e.draft.packed_size);
        w64(e.tokens.size()); put(e.tokens.data(), e.tokens.size() * 4);
        w64(e.logits.size()); put(e.logits.data(), e.logits.size() * 4);
        return out;
    }

    // Write buffers (null = skipped) to `path` atomically via a .tmp rename.
    static bool write_file(const fs::path & path, std::initializer_list<const bytes *> parts) {
        const fs::path tmp = path.string() + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            for (const bytes * p : parts) {
                if (p) f.write((const char *) p->data(), p->size());
            }
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, path, ec);
        return !ec;
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
            if (r32() != MAGIC) continue;
            const uint32_t version = r32();
            if (version != 1 && version != 2) continue;
            auto e = std::make_shared<entry>();
            e->id = r64();
            const uint32_t flags = r32();
            e->exact_only = flags & 1u;
            e->pinned = flags & 2u;
            e->kv_raw = version == 1 || (flags & 4u); // v1 files hold uncompressed state
            e->kv_key = rstr(); e->media_hash = rstr(); e->label = rstr(); e->logits_head_key = rstr();
            e->state.size = r64(); e->draft.size = r64();
            if (version >= 2) {
                e->state.packed_size = r64(); e->draft.packed_size = r64();
            }
            if (e->kv_raw) {
                e->state.packed_size = e->draft.packed_size = 0;
            }
            const uint64_t nt = r64();
            if (!f || nt > (1u << 24)) continue;
            e->tokens.resize(nt);
            f.read((char *) e->tokens.data(), nt * 4);
            const uint64_t nl = r64();
            if (!f || nl > (1u << 24)) continue;
            e->logits.resize(nl);
            f.read((char *) e->logits.data(), nl * 4);
            // the .kv must exist and have exactly the recorded size (catches truncated files)
            if (!f || fs::file_size(state_path(*e), ec) != e->state.disk() + e->draft.disk() || ec) continue;
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
