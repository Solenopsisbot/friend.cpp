#include "friend/paged_kv.hpp"
#include "friend/kv_connector.hpp"

#include <cassert>

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
    friend_kv::kv_connector destination_connector(destination, source_connector.discover());
    const auto wire = source_connector.export_wire();
    assert(destination_connector.import_wire(wire));
    assert(destination.entries().size() == 1 && destination.entries()[0].key == 101);
    assert(events.size() == 1 && events[0] == friend_kv::connector_event_type::exported);
    assert(!destination_connector.invalidate(101));
    destination.release(destination.entries()[0].id);
    destination.release(destination.entries()[0].id);
    assert(destination_connector.invalidate(101));
    auto corrupt = wire;
    corrupt.back() ^= 0xff;
    assert(!destination_connector.import_wire(corrupt));
    friend_kv::kv_connector incompatible(destination, {8, 8, 9, "profile", "namespace"});
    assert(!incompatible.import_wire(wire));
    return 0;
}
