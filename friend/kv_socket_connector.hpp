#pragma once

#include "friend/kv_connector.hpp"
#include "friend/kv_transport.hpp"

namespace friend_kv {

// A thin live transport around kv_connector. Ownership and compatibility stay
// in kv_connector; this class only moves one bounded snapshot per call over an
// already-authenticated connected socket.
class socket_connector {
public:
    explicit socket_connector(kv_connector & connector) : connector_(connector) {}

    const transport_stats & stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

#if !defined(_WIN32)
    bool export_to(int fd) const {
        return transport_frame::send_socket(fd, connector_.export_wire(), &stats_);
    }

    bool import_from(int fd) {
        std::vector<uint8_t> wire;
        return transport_frame::receive_socket(fd, wire, &stats_) && connector_.import_wire(wire);
    }
#endif

private:
    kv_connector & connector_;
    mutable transport_stats stats_;
};

} // namespace friend_kv
