#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;

//
// llama_adapter_head (friend.cpp)
//
// A hot-swappable LM head: `output.weight` plus optional `output_norm.weight`,
// `output.bias`, `output.scale` and `output.input_scale`, loaded from a small GGUF
// and allocated on the same device as the base head. Swapping a head changes only
// the final projection, so KV caches computed with a different head stay valid --
// that is what makes per-request head selection cheap compared to LoRA.
//
// Heads are always treated as un-rotated: Hadamard folding (prism) is keyed by
// tensor pointer, so a swapped-in head never picks up the base head's rotation.
struct llama_adapter_head {
    llama_model * model = nullptr;

    ggml_tensor * output      = nullptr; // required, [n_embd, n_vocab]
    ggml_tensor * output_norm = nullptr; // optional, [n_embd]; base norm is kept when absent
    ggml_tensor * output_b    = nullptr; // optional, [n_vocab]
    ggml_tensor * output_s    = nullptr; // optional NVFP4 scale2
    ggml_tensor * output_in_s = nullptr; // optional NVFP4 input scale

    // Contexts are unbound before this head's buffers are released. The caller
    // still serializes binding/freeing with inference on those contexts.
    std::unordered_set<llama_context *> bound_contexts;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    explicit llama_adapter_head(llama_model * model) : model(model) {}
};
