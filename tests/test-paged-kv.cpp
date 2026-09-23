#include "friend/paged_kv.hpp"

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
    return 0;
}
