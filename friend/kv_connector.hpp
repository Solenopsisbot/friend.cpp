#pragma once

// Versioned metadata transport for scheduler-owned KV pages.  The current
// llama.cpp backend still owns the tensor bytes, so this connector deliberately
// exports page identities and ownership rather than pretending it can transfer
// device memory.  A future backend can attach a payload to the same wire
// contract without changing discovery, compatibility, or lifetime rules.

#include "friend/paged_kv.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace friend_kv {

struct connector_descriptor {
    uint64_t model = 0;
    uint64_t layout = 0;
    uint32_t dtype = 0;
    std::string profile;
    std::string cache_namespace;

    bool operator==(const connector_descriptor & other) const {
        return model == other.model && layout == other.layout && dtype == other.dtype &&
               profile == other.profile && cache_namespace == other.cache_namespace;
    }
};

enum class connector_event_type : uint8_t {
    exported = 1,
    imported = 2,
    invalidated = 3,
    rejected = 4,
};

struct connector_event {
    connector_event_type type = connector_event_type::rejected;
    uint64_t key = 0;
    size_t entries = 0;
    const char * reason = nullptr;
};

struct connector_entry {
    uint64_t key = 0;
    uint32_t references = 0;
    std::vector<uint8_t> payload;
};

struct connector_snapshot {
    connector_descriptor descriptor;
    std::vector<connector_entry> entries;
};

class kv_connector {
public:
    static constexpr uint32_t protocol_version = 2;
    static constexpr size_t max_string = 4096;
    static constexpr size_t max_entries = 1u << 20;
    static constexpr size_t max_payload = size_t(64) << 20;
    static constexpr size_t max_total_payload = size_t(256) << 20;
    using event_callback = std::function<void(const connector_event &)>;

    kv_connector(paged_allocator & allocator, connector_descriptor descriptor,
                 event_callback callback = {})
        : allocator_(allocator), descriptor_(std::move(descriptor)), callback_(std::move(callback)) {}

    const connector_descriptor & descriptor() const { return descriptor_; }
    uint32_t version() const { return protocol_version; }

    // Backend integrations attach serialized KV bytes after allocating a page.
    // The scheduler still owns references; payloads are bounded and never make
    // an otherwise-dead page appear live.
    bool set_payload(uint64_t key, std::vector<uint8_t> payload) {
        if(!allocator_.alive(allocator_.find(key)) || payload.size() > max_payload) return false;
        size_t total = payload.size();
        for(const auto & item : payloads_)
            if(item.first != key) total += item.second.size();
        if(total > max_total_payload) return false;
        payloads_[key] = std::move(payload);
        return true;
    }

    const std::vector<uint8_t> * payload(uint64_t key) const {
        auto found = payloads_.find(key);
        return found == payloads_.end() ? nullptr : &found->second;
    }

    // Discovery is intentionally small and immutable: peers can compare it
    // before attempting an import, avoiding partial ownership transfers.
    connector_descriptor discover() const { return descriptor_; }

    connector_snapshot export_snapshot() const {
        connector_snapshot snapshot;
        snapshot.descriptor = descriptor_;
        for(const auto & entry : allocator_.entries()) {
            connector_entry item{entry.key, entry.references, {}};
            auto found = payloads_.find(entry.key);
            if(found != payloads_.end()) item.payload = found->second;
            snapshot.entries.push_back(std::move(item));
        }
        emit({connector_event_type::exported, 0, snapshot.entries.size(), nullptr});
        return snapshot;
    }

    bool import_snapshot(const connector_snapshot & snapshot) {
        if(!compatible(snapshot.descriptor) || snapshot.entries.size() > max_entries) {
            emit({connector_event_type::rejected, 0, snapshot.entries.size(), "incompatible snapshot"});
            return false;
        }
        size_t total_payload = 0;
        std::unordered_set<uint64_t> keys;
        for(const auto & item : snapshot.entries) {
            if(!keys.insert(item.key).second) {
                emit({connector_event_type::rejected, item.key, 0, "duplicate key"});
                return false;
            }
            if(item.payload.size() > max_payload ||
               item.payload.size() > max_total_payload - std::min(total_payload, max_total_payload)) {
                emit({connector_event_type::rejected, item.key, 0, "payload too large"});
                return false;
            }
            total_payload += item.payload.size();
            auto found = payloads_.find(item.key);
            if(found != payloads_.end() && found->second != item.payload) {
                emit({connector_event_type::rejected, item.key, 0, "payload conflict"});
                return false;
            }
        }
        // Acquire all entries first and roll back on capacity failure. Existing
        // keys are reference counted by acquire, so importing is idempotent for
        // a peer that reconnects after a lost acknowledgement.
        std::vector<paged_allocator::page_id> acquired;
        auto rollback = [&]() {
            for(auto id = acquired.rbegin(); id != acquired.rend(); ++id)
                allocator_.release(*id);
            acquired.clear();
        };
        for(const auto & item : snapshot.entries) {
            if(item.references > (1u << 20)) {
                rollback();
                emit({connector_event_type::rejected, item.key, 0, "invalid references"});
                return false;
            }
            if(item.references == 0) continue;
            for(uint32_t i = 0; i < item.references; ++i) {
                const paged_allocator::page_id id = allocator_.acquire(item.key);
                if(id == paged_allocator::invalid_page) {
                    const size_t acquired_count = acquired.size();
                    rollback();
                    emit({connector_event_type::rejected, item.key, acquired_count, "capacity"});
                    return false;
                }
                acquired.push_back(id);
            }
        }
        for(const auto & item : snapshot.entries)
            if(item.references > 0 && !item.payload.empty()) payloads_[item.key] = item.payload;
        emit({connector_event_type::imported, 0, snapshot.entries.size(), nullptr});
        return true;
    }

    std::vector<uint8_t> export_wire() const {
        const connector_snapshot snapshot = export_snapshot();
        std::vector<uint8_t> wire;
        wire.insert(wire.end(), {'F', 'K', 'V', 'C'});
        put_u32(wire, protocol_version);
        put_u64(wire, snapshot.descriptor.model);
        put_u64(wire, snapshot.descriptor.layout);
        put_u32(wire, snapshot.descriptor.dtype);
        put_string(wire, snapshot.descriptor.profile);
        put_string(wire, snapshot.descriptor.cache_namespace);
        put_u32(wire, (uint32_t) snapshot.entries.size());
        for(const auto & item : snapshot.entries) {
            put_u64(wire, item.key);
            put_u32(wire, item.references);
            put_u64(wire, item.payload.size());
            wire.insert(wire.end(), item.payload.begin(), item.payload.end());
        }
        return wire;
    }

    bool import_wire(const std::vector<uint8_t> & wire) {
        connector_snapshot snapshot;
        size_t offset = 0;
        uint32_t version = 0;
        if(wire.size() < 4 || std::memcmp(wire.data(), "FKVC", 4) != 0 || !get_bytes(wire, offset, 4, nullptr) ||
           !get_u32(wire, offset, version) || version != protocol_version ||
           !get_u64(wire, offset, snapshot.descriptor.model) ||
           !get_u64(wire, offset, snapshot.descriptor.layout) ||
           !get_u32(wire, offset, snapshot.descriptor.dtype) ||
           !get_string(wire, offset, snapshot.descriptor.profile) ||
           !get_string(wire, offset, snapshot.descriptor.cache_namespace)) {
            emit({connector_event_type::rejected, 0, 0, "corrupt header"});
            return false;
        }
        uint32_t count = 0;
        if(!get_u32(wire, offset, count) || count > max_entries) {
            emit({connector_event_type::rejected, 0, count, "corrupt entries"});
            return false;
        }
        snapshot.entries.reserve(count);
        for(uint32_t i = 0; i < count; ++i) {
            connector_entry item;
            uint64_t payload_size = 0;
            if(!get_u64(wire, offset, item.key) || !get_u32(wire, offset, item.references) ||
               !get_u64(wire, offset, payload_size) || payload_size > max_payload ||
               payload_size > wire.size() - std::min(offset, wire.size())) {
                emit({connector_event_type::rejected, item.key, i, "truncated entries"});
                return false;
            }
            item.payload.resize((size_t) payload_size);
            if(payload_size && !get_bytes(wire, offset, payload_size, item.payload.data())) {
                emit({connector_event_type::rejected, item.key, i, "truncated payload"});
                return false;
            }
            snapshot.entries.push_back(item);
        }
        if(offset != wire.size()) {
            emit({connector_event_type::rejected, 0, count, "trailing bytes"});
            return false;
        }
        return import_snapshot(snapshot);
    }

    bool invalidate(uint64_t key) {
        const bool ok = allocator_.invalidate(key);
        if(ok) payloads_.erase(key);
        emit({ok ? connector_event_type::invalidated : connector_event_type::rejected,
              key, ok ? 1u : 0u, ok ? nullptr : "page is still referenced"});
        return ok;
    }

private:
    bool compatible(const connector_descriptor & peer) const { return peer == descriptor_; }

    void emit(connector_event event) const { if(callback_) callback_(event); }

    static void put_u32(std::vector<uint8_t> & out, uint32_t value) {
        for(int i = 0; i < 4; ++i) out.push_back((uint8_t) (value >> (i * 8)));
    }
    static void put_u64(std::vector<uint8_t> & out, uint64_t value) {
        for(int i = 0; i < 8; ++i) out.push_back((uint8_t) (value >> (i * 8)));
    }
    static void put_string(std::vector<uint8_t> & out, const std::string & value) {
        const size_t size = std::min(max_string, value.size());
        put_u32(out, (uint32_t) size);
        out.insert(out.end(), value.begin(), value.begin() + size);
    }
    static bool get_bytes(const std::vector<uint8_t> & in, size_t & offset, size_t size, uint8_t * out) {
        if(size > in.size() - std::min(offset, in.size())) return false;
        if(out) std::memcpy(out, in.data() + offset, size);
        offset += size;
        return true;
    }
    static bool get_u32(const std::vector<uint8_t> & in, size_t & offset, uint32_t & value) {
        uint8_t bytes[4]; if(!get_bytes(in, offset, sizeof(bytes), bytes)) return false;
        value = (uint32_t) bytes[0] | ((uint32_t) bytes[1] << 8) | ((uint32_t) bytes[2] << 16) | ((uint32_t) bytes[3] << 24);
        return true;
    }
    static bool get_u64(const std::vector<uint8_t> & in, size_t & offset, uint64_t & value) {
        uint8_t bytes[8]; if(!get_bytes(in, offset, sizeof(bytes), bytes)) return false;
        value = 0; for(int i = 0; i < 8; ++i) value |= (uint64_t) bytes[i] << (i * 8);
        return true;
    }
    static bool get_string(const std::vector<uint8_t> & in, size_t & offset, std::string & value) {
        uint32_t size = 0;
        if(!get_u32(in, offset, size) || size > max_string || size > in.size() - std::min(offset, in.size())) return false;
        value.assign((const char *) in.data() + offset, size);
        offset += size;
        return true;
    }

    paged_allocator & allocator_;
    connector_descriptor descriptor_;
    event_callback callback_;
    std::unordered_map<uint64_t, std::vector<uint8_t>> payloads_;
};

} // namespace friend_kv
