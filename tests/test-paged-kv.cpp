#include "friend/paged_kv.hpp"
#include "friend/kv_connector.hpp"
#include "friend/kv_transport.hpp"
#include "friend/kv_socket_connector.hpp"
#include "friend/serving_metrics.hpp"

#include <cassert>
#include <sstream>
#if !defined(_WIN32)
#include <sys/socket.h>
#include <unistd.h>
#endif

int main() {
    friend_kv::paged_allocator pages(2);
    const auto a = pages.acquire(11);
    const auto b = pages.acquire(22);
    assert(a != friend_kv::paged_allocator::invalid_page);
    assert(b != friend_kv::paged_allocator::invalid_page && b != a);
    assert(pages.used() == 2);
    pages.release(a);
    const auto a_hit = pages.acquire(11);
    assert(a_hit == a && pages.hits() == 1);
    pages.release(a_hit);
    pages.release(b);
    const auto c = pages.acquire(33);
    assert(c != friend_kv::paged_allocator::invalid_page);
    assert(pages.evictions() == 1);
    assert(c == a || c == b);
    pages.release(c);
    pages.reset(1);
    const auto held = pages.acquire(44);
    assert(pages.acquire(55) == friend_kv::paged_allocator::invalid_page);
    pages.release(held);
    assert(pages.acquire(55) != friend_kv::paged_allocator::invalid_page);

    friend_kv::paged_allocator source(4), destination(4);
    const auto source_page = source.acquire(101);
    source.retain(source_page);
    std::vector<friend_kv::connector_event_type> events;
    friend_kv::kv_connector source_connector(source, {7, 8, 9, "profile", "namespace"},
        [&](const friend_kv::connector_event & event) { events.push_back(event.type); });
    assert(source_connector.set_payload(101, {1, 2, 3, 5, 8}));
    assert(source_connector.payload(101) && source_connector.payload(101)->size() == 5);
    friend_kv::kv_connector destination_connector(destination, source_connector.discover());
    const auto wire = source_connector.export_wire();
    const auto framed = friend_kv::transport_frame::encode(wire);
    std::vector<uint8_t> unframed;
    assert(friend_kv::transport_frame::decode(framed, unframed) && unframed == wire);
    std::stringstream stream;
    friend_kv::transport_stats stream_stats;
    assert(friend_kv::transport_frame::write(stream, wire, &stream_stats));
    assert(stream_stats.frames_sent == 1 && stream_stats.payload_bytes_sent == wire.size());
    unframed.clear();
    assert(friend_kv::transport_frame::read(stream, unframed, &stream_stats) && unframed == wire);
    assert(stream_stats.frames_received == 1 && stream_stats.payload_bytes_received == wire.size());
    auto bad_frame = framed;
    bad_frame.back() ^= 1;
    assert(!friend_kv::transport_frame::decode(bad_frame, unframed));
    friend_kv::transport_stats bad_stats;
    std::stringstream corrupt_stream(std::string((const char *) bad_frame.data(), bad_frame.size()));
    assert(!friend_kv::transport_frame::read(corrupt_stream, unframed, &bad_stats));
    assert(bad_stats.receive_failures == 1 && bad_stats.frames_received == 0);
    auto short_frame = framed;
    short_frame.pop_back();
    assert(!friend_kv::transport_frame::decode(short_frame, unframed));
#if !defined(_WIN32)
    int sockets[2] = {-1, -1};
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(friend_kv::transport_frame::send_socket(sockets[0], wire));
    unframed.clear();
    assert(friend_kv::transport_frame::receive_socket(sockets[1], unframed) && unframed == wire);
    ::close(sockets[0]);
    ::close(sockets[1]);

    int connector_sockets[2] = {-1, -1};
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, connector_sockets) == 0);
    friend_kv::paged_allocator network_destination(4);
    friend_kv::kv_connector network_connector(network_destination, source_connector.discover());
    friend_kv::socket_connector sender(source_connector), receiver(network_connector);
    assert(sender.export_to(connector_sockets[0]));
    assert(receiver.import_from(connector_sockets[1]));
    assert(sender.stats().frames_sent == 1 && sender.stats().payload_bytes_sent == wire.size());
    assert(receiver.stats().frames_received == 1 && receiver.stats().payload_bytes_received == wire.size());
    assert(network_destination.entries().size() == 1);
    ::close(connector_sockets[0]);
    ::close(connector_sockets[1]);
#endif
    assert(destination_connector.import_wire(wire));
    assert(destination.entries().size() == 1 && destination.entries()[0].key == 101);
    assert(destination_connector.payload(101) && (*destination_connector.payload(101))[2] == 3);
    assert(events.size() >= 1 && events[0] == friend_kv::connector_event_type::exported);
    assert(!destination_connector.invalidate(101));
    destination.release(destination.entries()[0].id);
    destination.release(destination.entries()[0].id);
    assert(destination_connector.invalidate(101));
    auto corrupt = wire;
    corrupt[0] ^= 0xff;
    assert(!destination_connector.import_wire(corrupt));
    auto truncated = wire;
    truncated.resize(truncated.size() - 1);
    assert(!destination_connector.import_wire(truncated));
    friend_kv::kv_connector incompatible(destination, {8, 8, 9, "profile", "namespace"});
    assert(!incompatible.import_wire(wire));
    friend_serving::metrics metrics;
    metrics.kv_page_stalls = 2;
    assert(metrics.render(0, 0).find("friend_batch_kv_page_stalls_total 2") != std::string::npos);
    return 0;
}
