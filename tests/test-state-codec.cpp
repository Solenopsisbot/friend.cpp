#include "friend/state_codec.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    std::vector<uint8_t> input(1 << 20);
    for(size_t i = 0; i < input.size(); ++i) input[i] = (uint8_t) ((i * 17 + (i >> 8)) & 0xff);
    const auto packed = friend_cache::codec::pack(input.data(), input.size(), 2);
    assert(!packed.empty());
    std::vector<uint8_t> output;
    assert(friend_cache::codec::unpack(packed.data(), packed.size(), output, 2));
    assert(output == input);
    auto corrupt = packed;
    corrupt.back() ^= 0x80;
    output.clear();
    assert(!friend_cache::codec::unpack(corrupt.data(), corrupt.size(), output, 2));
    return 0;
}
