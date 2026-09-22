// friend.cpp: per-op time breakdown of one prefill ubatch, for finding where prompt processing goes.
//
//   op-profile <model.gguf> [n_tokens=512] [flash_attn=0|1] [reps=3] [n_past=0]
//
// Times one n_tokens ubatch decoded on top of n_past already-cached tokens (the cache is filled,
// untimed, in n_tokens-sized ubatches first, so attention sees a realistic KV length).
// First times plain llama_decode of that ubatch (the real number, CUDA graphs and all),
// then re-runs it with an eval callback that observes every node. The scheduler synchronizes
// the backend before each observation, so the wall time between two callbacks is the time of
// that one node (plus a launch + sync, ~tens of us, which is why the plain run is reported too).
// Nodes are grouped by op, and MUL_MAT / MUL_MAT_ID additionally by weight type and shape.
//
// Links against a koboldcpp_*.so (which exports the llama API), e.g. on the CUDA box:
//   g++ -O2 -std=c++17 -Iinclude -Iggml/include tools/friend-bench/op-profile.cpp \
//       ./koboldcpp_cublas.so -Wl,-rpath,'$ORIGIN' -o op-profile
// Not part of any shipped target.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

struct prof_state {
    clk::time_point last;
    bool active = false;
    std::map<std::string, std::pair<double, int>> by_key; // key -> (us, count)
    double total_us = 0.0;
};

static std::string node_key(const ggml_tensor * t) {
    std::string k = ggml_op_desc(t);
    if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[0] && t->src[1]) {
        char buf[160];
        snprintf(buf, sizeof(buf), " %s %lldx%lld x %s n=%lld", ggml_type_name(t->src[0]->type),
                 (long long) t->src[0]->ne[0], (long long) t->src[0]->ne[1], ggml_type_name(t->src[1]->type),
                 (long long) t->src[1]->ne[1]);
        k += buf;
    }
    return k;
}

static bool prof_cb(struct ggml_tensor * t, bool ask, void * ud) {
    prof_state * st = (prof_state *) ud;
    if (ask) {
        return st->active; // observe every node while profiling
    }
    const auto now = clk::now();
    const double us = std::chrono::duration<double, std::micro>(now - st->last).count();
    st->last = now;
    auto & e = st->by_key[node_key(t)];
    e.first += us;
    e.second += 1;
    st->total_us += us;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_tokens=512] [flash_attn=0|1] [reps=3] [n_past=0]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int n_tok = argc > 2 ? atoi(argv[2]) : 512;
    const bool fa   = argc > 3 ? atoi(argv[3]) != 0 : false;
    const int reps  = argc > 4 ? atoi(argv[4]) : 3;
    const int n_past = argc > 5 ? atoi(argv[5]) : 0;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    prof_state st;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = std::max(1024, n_past + n_tok + 16);
    cp.n_batch         = n_tok;
    cp.n_ubatch        = n_tok;
    cp.n_seq_max       = 1;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval         = prof_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // deterministic pseudo-random prompt tokens: content does not matter for timing
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    // fills batch with n tokens at positions [p0, p0 + n)
    auto fill = [&](int p0, int n) {
        batch.n_tokens = n;
        uint32_t s = 12345 + (uint32_t) p0;
        for (int i = 0; i < n; i++) {
            s = s * 1664525u + 1013904223u;
            batch.token[i] = (llama_token) (1000 + (s >> 8) % (uint32_t) (n_vocab - 2000));
            batch.pos[i] = p0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1; // prefill only needs the last token's logits
        }
    };

    // clears the cache and prefills n_past tokens (untimed), then stages the measured ubatch
    auto prepare = [&]() {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p0 = 0; p0 < n_past; p0 += n_tok) {
            fill(p0, std::min(n_tok, n_past - p0));
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "prefill failed\n");
                exit(1);
            }
        }
        llama_synchronize(ctx);
        fill(n_past, n_tok);
    };

    auto decode_once = [&]() {
        prepare();
        const auto t0 = clk::now();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
        llama_synchronize(ctx);
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    decode_once(); // warmup
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        const double ms = decode_once();
        best = std::min(best, ms);
        printf("plain decode of %d tokens after %d: %.1f ms (%.1f tok/s)\n", n_tok, n_past, ms, n_tok / ms * 1e3);
    }

    prepare();
    st.active = true;
    st.last = clk::now();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    llama_synchronize(ctx);
    st.active = false;

    std::vector<std::pair<std::string, std::pair<double, int>>> rows(st.by_key.begin(), st.by_key.end());
    std::sort(rows.begin(), rows.end(), [](auto & a, auto & b) { return a.second.first > b.second.first; });
    printf("\nprofiled decode: %.1f ms total over %zu op kinds (best plain: %.1f ms, fa=%d)\n",
           st.total_us / 1e3, rows.size(), best, (int) fa);
    for (auto & r : rows) {
        printf("%9.2f ms %5.1f%%  x%-4d %s\n", r.second.first / 1e3, 100.0 * r.second.first / st.total_us,
               r.second.second, r.first.c_str());
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
