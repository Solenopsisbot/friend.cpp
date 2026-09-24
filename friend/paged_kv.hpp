#pragma once

// A small physical page table for the serving scheduler. llama.cpp still owns
// the backend KV tensors, but this allocator owns the scheduler's page identity,
// references and eviction policy. Keeping that contract separate lets the native
// backend move from sequence-cell sharing to true paged KV without changing request
// admission or prefix-cache keys.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace friend_kv {

class paged_allocator {
public:
    using page_id = uint32_t;
    static constexpr page_id invalid_page = std::numeric_limits<page_id>::max();

    explicit paged_allocator(size_t capacity = 0) { reset(capacity); }

    void reset(size_t capacity) {
        pages_.assign(capacity, page{});
        by_key_.clear();
        tick_ = queries_ = hits_ = evictions_ = 0;
    }

    page_id acquire(uint64_t key) {
        ++queries_;
        auto found = by_key_.find(key);
        if(found != by_key_.end()) {
            page & p = pages_[found->second];
            ++p.references;
            p.last_touch = ++tick_;
            ++hits_;
            return found->second;
        }
        page_id id = find_free();
        if(id == invalid_page) id = evict_one();
        if(id == invalid_page) return invalid_page;
        page & p = pages_[id];
        p.key = key;
        p.references = 1;
        p.last_touch = ++tick_;
        p.occupied = true;
        by_key_[key] = id;
        return id;
    }

    void retain(page_id id) {
        if(id >= pages_.size() || !pages_[id].occupied) return;
        ++pages_[id].references;
        pages_[id].last_touch = ++tick_;
    }

    void release(page_id id) {
        if(id >= pages_.size() || !pages_[id].occupied) return;
        page & p = pages_[id];
        if(p.references > 0) --p.references;
        p.last_touch = ++tick_;
    }

    bool alive(page_id id) const {
        return id < pages_.size() && pages_[id].occupied;
    }

    size_t capacity() const { return pages_.size(); }
    size_t used() const {
        size_t n = 0;
        for(const page & p : pages_) if(p.occupied) ++n;
        return n;
    }
    size_t referenced() const {
        size_t n = 0;
        for(const page & p : pages_) if(p.occupied && p.references > 0) ++n;
        return n;
    }
    uint64_t queries() const { return queries_; }
    uint64_t hits() const { return hits_; }
    uint64_t evictions() const { return evictions_; }

    struct entry {
        uint64_t key = 0;
        uint32_t references = 0;
        page_id id = invalid_page;
    };

    std::vector<entry> entries() const {
        std::vector<entry> result;
        result.reserve(by_key_.size());
        for(const auto & item : by_key_) {
            const page_id id = item.second;
            if(id < pages_.size() && pages_[id].occupied)
                result.push_back({item.first, pages_[id].references, id});
        }
        std::sort(result.begin(), result.end(), [](const entry & a, const entry & b) {
            return a.key < b.key;
        });
        return result;
    }

    bool invalidate(uint64_t key) {
        auto found = by_key_.find(key);
        if(found == by_key_.end()) return false;
        const page_id id = found->second;
        if(id >= pages_.size() || !pages_[id].occupied || pages_[id].references != 0) return false;
        by_key_.erase(found);
        pages_[id] = page{};
        return true;
    }

private:
    struct page {
        uint64_t key = 0;
        uint64_t last_touch = 0;
        uint32_t references = 0;
        bool occupied = false;
    };

    page_id find_free() const {
        for(page_id i = 0; i < pages_.size(); ++i)
            if(!pages_[i].occupied) return i;
        return invalid_page;
    }

    page_id evict_one() {
        page_id victim = invalid_page;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for(page_id i = 0; i < pages_.size(); ++i) {
            const page & p = pages_[i];
            if(p.occupied && p.references == 0 && p.last_touch < oldest) {
                oldest = p.last_touch;
                victim = i;
            }
        }
        if(victim == invalid_page) return victim;
        by_key_.erase(pages_[victim].key);
        pages_[victim] = page{};
        ++evictions_;
        return victim;
    }

    std::vector<page> pages_;
    std::unordered_map<uint64_t, page_id> by_key_;
    uint64_t tick_ = 0;
    uint64_t queries_ = 0;
    uint64_t hits_ = 0;
    uint64_t evictions_ = 0;
};

} // namespace friend_kv
