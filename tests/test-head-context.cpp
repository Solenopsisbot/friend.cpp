// Native regression for sharing one model between contexts with different LM heads.
// Build against koboldcpp_default.so and pass a model plus an extracted head GGUF.
#include "llama.h"
#include "llama-ext.h"
#include "llama-adapter.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        const auto * base_output = model->output;

        std::unique_ptr<llama_adapter_head, decltype(&llama_adapter_head_free)> head(
            llama_adapter_head_init(model.get(), argv[2]), llama_adapter_head_free);
        require(head != nullptr && head->output_norm != nullptr, "head with output norm required");
        // A zero final norm makes the alternate output observably different while
        // keeping its tensor type and allocation identical to a normal loaded head.
        ggml_backend_tensor_memset(head->output_norm, 0, 0, ggml_nbytes(head->output_norm));

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = 32;
        cp.n_ubatch = 32;
        cp.n_seq_max = 1;
        std::unique_ptr<llama_context, decltype(&llama_free)> base(
            llama_init_from_model(model.get(), cp), llama_free);
        std::unique_ptr<llama_context, decltype(&llama_free)> alternate(
            llama_init_from_model(model.get(), cp), llama_free);
        require(base != nullptr && alternate != nullptr, "context creation failed");

        const char * prompt = "The capital of France is";
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> tokens(32);
        const int n = llama_tokenize(vocab, prompt, std::strlen(prompt), tokens.data(), tokens.size(), true, false);
        require(n > 0, "tokenization failed");
        tokens.resize(n);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        auto decode = [&](llama_context * ctx) {
            llama_memory_clear(llama_get_memory(ctx), true);
            require(llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) == 0, "decode failed");
            const float * logits = llama_get_logits(ctx);
            require(logits != nullptr, "missing logits");
            return std::vector<float>(logits, logits + n_vocab);
        };

        const auto expected = decode(base.get());
        require(llama_set_adapter_head(alternate.get(), head.get()) == 0, "binding alternate head failed");
        require(llama_get_adapter_head(alternate.get()) == head.get(), "head getter disagrees");
        require(llama_get_adapter_head(base.get()) == nullptr, "base context was changed");
        require(model->output == base_output && llama_model_get_head(model.get()) == nullptr,
                "binding a context mutated the shared model");

        for (int round = 0; round < 3; ++round) {
            auto a = std::async(std::launch::async, decode, base.get());
            auto b = std::async(std::launch::async, decode, alternate.get());
            const auto base_logits = a.get();
            const auto alternate_logits = b.get();
            require(base_logits == expected, "concurrent alternate decode changed base logits");
            require(alternate_logits != expected, "alternate head had no effect");
            require(std::all_of(alternate_logits.begin(), alternate_logits.end(), [](float x) {
                return std::isfinite(x);
            }), "alternate logits are non-finite");
        }

        require(llama_set_adapter_head(alternate.get(), nullptr) == 0, "unbind failed");
        require(decode(alternate.get()) == expected, "unbound context did not return to base logits");
        alternate.reset();
        base.reset();
        head.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & err) {
        fprintf(stderr, "head context regression: %s\n", err.what());
        return 1;
    }
}
