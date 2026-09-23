#pragma once
// Serving counters are updated with batch_mutex held. Histograms retain constant
// space; request IDs, prompts and cache salts never become metric labels.
#include <array>
#include <cstdint>
#include <sstream>
#include <string>

namespace friend_serving {
struct histogram {
    static constexpr std::array<double, 12> bounds = {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5, 30, 120};
    std::array<uint64_t, bounds.size()> buckets{};
    uint64_t count = 0;
    double sum = 0;
    void observe(double seconds) {
        ++count;
        sum += seconds;
        for(size_t i = 0; i < bounds.size(); ++i)
            if(seconds <= bounds[i]) ++buckets[i];
    }
    void write(std::ostream & out, const char * name) const {
        out << "# TYPE " << name << " histogram\n";
        for(size_t i = 0; i < bounds.size(); ++i)
            out << name << "_bucket{le=\"" << bounds[i] << "\"} " << buckets[i] << '\n';
        out << name << "_bucket{le=\"+Inf\"} " << count << '\n';
        out << name << "_sum " << sum << '\n' << name << "_count " << count << '\n';
    }
};
struct metrics {
    uint64_t submitted = 0, completed = 0, failed = 0, cancelled = 0;
    uint64_t prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
    uint64_t rounds = 0, batch_tokens = 0, preemptions = 0, kv_block_shares = 0;
    uint64_t draft_proposed = 0, draft_accepted = 0;
    histogram queue, first_token, inter_token, decode;
    std::string render(size_t waiting, size_t running, size_t offloaded_bytes = 0) const {
        std::ostringstream out;
        auto value = [&](const char * name, const char * type, uint64_t n) {
            out << "# TYPE friend_batch_" << name << ' ' << type << '\n';
            out << "friend_batch_" << name << ' ' << n << '\n';
        };
        value("requests_submitted_total", "counter", submitted);
        value("requests_completed_total", "counter", completed);
        value("requests_failed_total", "counter", failed);
        value("requests_cancelled_total", "counter", cancelled);
        value("requests_waiting", "gauge", waiting);
        value("requests_running", "gauge", running);
        value("offloaded_bytes", "gauge", offloaded_bytes);
        value("prefill_tokens_total", "counter", prompt_tokens);
        value("reused_tokens_total", "counter", reused_tokens);
        value("generated_tokens_total", "counter", generated_tokens);
        value("draft_proposed_tokens_total", "counter", draft_proposed);
        value("draft_accepted_tokens_total", "counter", draft_accepted);
        value("rounds_total", "counter", rounds);
        value("scheduled_tokens_total", "counter", batch_tokens);
        value("preemptions_total", "counter", preemptions);
        value("kv_block_shares_total", "counter", kv_block_shares);
        queue.write(out, "friend_batch_queue_seconds");
        first_token.write(out, "friend_batch_time_to_first_token_seconds");
        inter_token.write(out, "friend_batch_inter_token_seconds");
        decode.write(out, "friend_batch_decode_seconds");
        return out.str();
    }
};
} // namespace friend_serving
