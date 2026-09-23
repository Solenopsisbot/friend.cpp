#include "friend/suffix_draft.hpp"

#include <cassert>
#include <vector>

int main() {
    // The final two-token suffix repeats twice in the history; the most
    // frequent continuation should be proposed first and then extended from
    // the newly proposed token when the pattern repeats again.
    const std::vector<llama_token> history = {
        1, 2, 9,
        1, 2, 9,
        1, 2,
    };
    const auto draft = friend_spec::suffix_draft(history, 2, 2);
    assert(draft.size() == 2);
    assert(draft[0] == 9);
    assert(draft[1] == 1);

    assert(friend_spec::suffix_draft(history, 2, 0).empty());
    return 0;
}
