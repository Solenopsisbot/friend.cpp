#pragma once

// Network-neutral framing for kv_connector::export_wire().  Keeping framing
// separate from sockets lets the serving process use TLS, Unix sockets, or an
// RPC transport without changing the ownership and compatibility contract.

#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <istream>
#include <limits>
#include <ostream>
#include <vector>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace friend_kv {

// Counters are deliberately transport-local.  A serving process can fold
// these into its own metrics without making the framing layer depend on a
// particular metrics registry or request lifetime.
struct transport_stats {
    uint64_t frames_sent = 0;
    uint64_t frames_received = 0;
    uint64_t payload_bytes_sent = 0;
    uint64_t payload_bytes_received = 0;
    uint64_t wire_bytes_sent = 0;
    uint64_t wire_bytes_received = 0;
    uint64_t send_failures = 0;
    uint64_t receive_failures = 0;
};

class transport_frame {
public:
    static constexpr uint32_t protocol_version = 1;
    static constexpr size_t max_payload = size_t(256) << 20;

    static std::vector<uint8_t> encode(const std::vector<uint8_t> & payload) {
        if(payload.size() > max_payload) return {};
        std::vector<uint8_t> frame;
        frame.reserve(20 + payload.size());
        frame.insert(frame.end(), {'F', 'K', 'V', 'T'});
        put_u32(frame, protocol_version);
        put_u64(frame, payload.size());
        put_u32(frame, checksum(payload));
        frame.insert(frame.end(), payload.begin(), payload.end());
        return frame;
    }

    static bool decode(const std::vector<uint8_t> & frame, std::vector<uint8_t> & payload) {
        size_t offset = 0;
        uint32_t version = 0, expected_checksum = 0;
        uint64_t size = 0;
        if(frame.size() < 20 || !get_bytes(frame, offset, 4, nullptr) ||
           frame[0] != 'F' || frame[1] != 'K' || frame[2] != 'V' || frame[3] != 'T' ||
           !get_u32(frame, offset, version) || version != protocol_version ||
           !get_u64(frame, offset, size) || size > max_payload ||
           size > frame.size() - offset || !get_u32(frame, offset, expected_checksum) ||
           size != frame.size() - offset) return false;
        payload.assign(frame.begin() + (ptrdiff_t) offset, frame.end());
        return checksum(payload) == expected_checksum;
    }

    static bool write(std::ostream & stream, const std::vector<uint8_t> & payload,
                      transport_stats * stats = nullptr) {
        const auto frame = encode(payload);
        if(frame.empty() && !payload.empty()) {
            if(stats) ++stats->send_failures;
            return false;
        }
        stream.write((const char *) frame.data(), (std::streamsize) frame.size());
        if(!stream.good()) {
            if(stats) ++stats->send_failures;
            return false;
        }
        if(stats) {
            ++stats->frames_sent;
            stats->payload_bytes_sent += payload.size();
            stats->wire_bytes_sent += frame.size();
        }
        return true;
    }

    static bool read(std::istream & stream, std::vector<uint8_t> & payload,
                     transport_stats * stats = nullptr) {
        uint8_t header[20] = {};
        stream.read((char *) header, sizeof(header));
        if(stream.gcount() != (std::streamsize) sizeof(header) ||
           header[0] != 'F' || header[1] != 'K' || header[2] != 'V' || header[3] != 'T') {
            if(stats) ++stats->receive_failures;
            return false;
        }
        const uint32_t version = read_u32(header + 4);
        const uint64_t size = read_u64(header + 8);
        const uint32_t expected_checksum = read_u32(header + 16);
        if(version != protocol_version || size > max_payload ||
           size > (uint64_t) std::numeric_limits<std::streamsize>::max()) {
            if(stats) ++stats->receive_failures;
            return false;
        }
        payload.resize((size_t) size);
        if(size) {
            stream.read((char *) payload.data(), (std::streamsize) size);
            if(stream.gcount() != (std::streamsize) size) {
                payload.clear();
                if(stats) ++stats->receive_failures;
                return false;
            }
        }
        if(checksum(payload) != expected_checksum) {
            payload.clear();
            if(stats) ++stats->receive_failures;
            return false;
        }
        if(stats) {
            ++stats->frames_received;
            stats->payload_bytes_received += payload.size();
            stats->wire_bytes_received += sizeof(header) + payload.size();
        }
        return true;
    }

#if !defined(_WIN32)
    // The descriptor helpers intentionally operate on an already-connected
    // socket. Callers can use TCP, Unix sockets, or an encrypted wrapper while
    // retaining the same bounded frame and checksum validation.
    static bool send_socket(int fd, const std::vector<uint8_t> & payload,
                            transport_stats * stats = nullptr) {
        const auto frame = encode(payload);
        if(frame.empty() && !payload.empty()) {
            if(stats) ++stats->send_failures;
            return false;
        }
        size_t offset = 0;
        while(offset < frame.size()) {
            const ssize_t sent = ::send(fd, frame.data() + offset, frame.size() - offset, MSG_NOSIGNAL);
            if(sent <= 0) {
                if(stats) ++stats->send_failures;
                return false;
            }
            offset += (size_t) sent;
        }
        if(stats) {
            ++stats->frames_sent;
            stats->payload_bytes_sent += payload.size();
            stats->wire_bytes_sent += frame.size();
        }
        return true;
    }

    static bool receive_socket(int fd, std::vector<uint8_t> & payload,
                               transport_stats * stats = nullptr) {
        uint8_t header[20] = {};
        if(!receive_all(fd, header, sizeof(header)) || header[0] != 'F' || header[1] != 'K' ||
           header[2] != 'V' || header[3] != 'T') {
            if(stats) ++stats->receive_failures;
            return false;
        }
        const uint32_t version = read_u32(header + 4);
        const uint64_t size = read_u64(header + 8);
        const uint32_t expected_checksum = read_u32(header + 16);
        if(version != protocol_version || size > max_payload) {
            if(stats) ++stats->receive_failures;
            return false;
        }
        payload.resize((size_t) size);
        if(size && !receive_all(fd, payload.data(), (size_t) size)) {
            payload.clear();
            if(stats) ++stats->receive_failures;
            return false;
        }
        if(checksum(payload) != expected_checksum) {
            payload.clear();
            if(stats) ++stats->receive_failures;
            return false;
        }
        if(stats) {
            ++stats->frames_received;
            stats->payload_bytes_received += payload.size();
            stats->wire_bytes_received += sizeof(header) + payload.size();
        }
        return true;
    }
#endif

private:
    static uint32_t checksum(const std::vector<uint8_t> & bytes) {
        uint32_t hash = UINT32_C(2166136261);
        for(uint8_t byte : bytes) {
            hash ^= byte;
            hash *= UINT32_C(16777619);
        }
        return hash;
    }
    static void put_u32(std::vector<uint8_t> & out, uint32_t value) {
        for(int i = 0; i < 4; ++i) out.push_back((uint8_t) (value >> (i * 8)));
    }
    static void put_u64(std::vector<uint8_t> & out, uint64_t value) {
        for(int i = 0; i < 8; ++i) out.push_back((uint8_t) (value >> (i * 8)));
    }
    static bool get_bytes(const std::vector<uint8_t> & in, size_t & offset, size_t size, uint8_t * out) {
        if(offset > in.size() || size > in.size() - offset) return false;
        if(out) std::copy(in.begin() + (ptrdiff_t) offset,
                          in.begin() + (ptrdiff_t) (offset + size), out);
        offset += size;
        return true;
    }
    static bool get_u32(const std::vector<uint8_t> & in, size_t & offset, uint32_t & value) {
        uint8_t bytes[4];
        if(!get_bytes(in, offset, sizeof(bytes), bytes)) return false;
        value = read_u32(bytes);
        return true;
    }
    static bool get_u64(const std::vector<uint8_t> & in, size_t & offset, uint64_t & value) {
        uint8_t bytes[8];
        if(!get_bytes(in, offset, sizeof(bytes), bytes)) return false;
        value = read_u64(bytes);
        return true;
    }
    static uint32_t read_u32(const uint8_t * bytes) {
        return (uint32_t) bytes[0] | ((uint32_t) bytes[1] << 8) |
               ((uint32_t) bytes[2] << 16) | ((uint32_t) bytes[3] << 24);
    }
    static uint64_t read_u64(const uint8_t * bytes) {
        uint64_t value = 0;
        for(int i = 0; i < 8; ++i) value |= (uint64_t) bytes[i] << (i * 8);
        return value;
    }
#if !defined(_WIN32)
    static bool receive_all(int fd, uint8_t * data, size_t size) {
        size_t offset = 0;
        while(offset < size) {
            const ssize_t received = ::recv(fd, data + offset, size - offset, 0);
            if(received <= 0) return false;
            offset += (size_t) received;
        }
        return true;
    }
#endif
};

} // namespace friend_kv
