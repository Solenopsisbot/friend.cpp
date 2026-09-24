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

#if !defined(_WIN32)
    bool export_to(int fd) const {
        return transport_frame::send_socket(fd, connector_.export_wire());
    }

    bool import_from(int fd) {
        std::vector<uint8_t> wire;
        return transport_frame::receive_socket(fd, wire) && connector_.import_wire(wire);
    }
#endif

private:
    kv_connector & connector_;
};

} // namespace friend_kv
