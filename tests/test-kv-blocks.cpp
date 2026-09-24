#include "friend/kv_blocks.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<int> a(33);
    for(int i = 0; i < (int) a.size(); ++i) a[i] = i;
    std::vector<int> b = a;
    b.push_back(99);
    friend_kv::block_table ta, tb;
    ta.rebuild(a);
    tb.rebuild(b);
    assert(ta.hashes.size() == 3);
    assert(tb.hashes.size() == 3);
    assert(ta.common_blocks(tb) == 2);
    b[16] = -1;
    tb.rebuild(b);
    assert(ta.common_blocks(tb) == 1);
    return 0;
}
