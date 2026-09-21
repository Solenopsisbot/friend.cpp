#pragma once
// friend.cpp: per-request adapter profiles.
//
// A *profile* is what a single request asks the model to "be": a mix of LoRA adapters
// with scales, a mix of control (steering) vectors with strengths, and optionally a
// swapped LM head. Everything is preloaded at startup into named pools; switching
// between profiles per request is then just pointer/scale changes, no file IO.
//
// Cost model, so callers can make sane scheduling decisions:
//   - LoRA set change   -> llama re-reserves its scheduler on the next decode (ms-scale)
//                          and INVALIDATES KV computed under a different set
//   - control vectors   -> same as LoRA (re-reserve + KV invalidation)
//   - head swap         -> graph rebuild only; KV stays valid (only logits change)
// So `profile::kv_key` covers LoRA + cvec only, and caches must key on it; the head is
// tracked separately in `profile::head_key`.
//
// Wire format (built by koboldcpp.py, which validates names first):
//   pools : one entry per line, fields separated by '\t'
//           lora: "name\tpath\tdefault_scale"   (default_scale 0 = off unless requested)
//           cvec: "name\tpath"
//           head: "name\tpath"
//   profile spec: entries separated by ';', fields by ' ' (names are [A-Za-z0-9_.-]+)
//           "lora_explicit"      the request specified its LoRA set (even if empty),
//                                replacing the default-on adapters
//           "L <name> <scale>"   LoRA
//           "C <name> <strength>" control vector
//           "H <name>"           head

#include "llama.h"
#include "llama-ext.h"
#include "common/common.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace friend_adapters {

struct lora_entry {
    std::string name;
    std::string path;
    llama_adapter_lora * adapter = nullptr;
    float default_scale = 0.0f;
};

struct cvec_entry {
    std::string name;
    std::string path;
    std::vector<float> data; // n_embd * n_layer_data, layer 1 at offset 0 (common_control_vector format)
};

struct head_entry {
    std::string name;
    std::string path;
    llama_adapter_head * head = nullptr;
};

struct profile {
    std::vector<std::pair<int, float>> loras; // (pool index, scale), sorted, non-zero only
    std::vector<std::pair<int, float>> cvecs; // (pool index, strength), sorted, non-zero only
    int head = -1;                            // pool index, -1 = model's own head

    std::string kv_key;   // canonical LoRA+cvec identity; "" = plain base model
    std::string head_key; // "" = base head

    // human-readable for logs
    std::string describe() const {
        std::string s = kv_key.empty() ? "base" : kv_key;
        if (!head_key.empty()) {
            s += " head=" + head_key;
        }
        return s;
    }
};

struct registry {
    llama_model * model = nullptr;
    int32_t n_embd = 0;
    int32_t n_layer = 0;

    std::vector<lora_entry> loras;
    std::vector<cvec_entry> cvecs;
    std::vector<head_entry> heads;

    // what is currently applied to each context, so switching to the same profile is free
    std::map<const llama_context *, std::string> applied_lora;
    std::map<const llama_context *, std::string> applied_cvec;
};

inline registry & reg() {
    static registry r;
    return r;
}

inline std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep)) {
        if (!cur.empty() && cur.back() == '\r') {
            cur.pop_back();
        }
        if (!cur.empty()) {
            out.push_back(cur);
        }
    }
    return out;
}

inline std::string fmt_scale(float v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.4g", v);
    return buf;
}

// Load every pool. Returns false (after printing why) on any failure: a missing adapter
// is a configuration error the user must see at startup, not a silent no-op later.
inline bool load_pools(llama_model * model, const char * lora_pool, const char * cvec_pool, const char * head_pool) {
    auto & r = reg();
    r.model   = model;
    r.n_embd  = llama_model_n_embd(model);
    r.n_layer = llama_model_n_layer(model);

    for (const auto & line : split(lora_pool ? lora_pool : "", '\n')) {
        const auto f = split(line, '\t');
        if (f.size() < 2) {
            fprintf(stderr, "friend: malformed lora pool entry '%s'\n", line.c_str());
            return false;
        }
        lora_entry e;
        e.name = f[0];
        e.path = f[1];
        e.default_scale = f.size() > 2 ? std::stof(f[2]) : 0.0f;
        printf("\nfriend: loading LoRA '%s' from %s (default scale %s)\n", e.name.c_str(), e.path.c_str(), fmt_scale(e.default_scale).c_str());
        e.adapter = llama_adapter_lora_init(model, e.path.c_str());
        if (!e.adapter) {
            fprintf(stderr, "friend: failed to load LoRA '%s'\n", e.name.c_str());
            return false;
        }
        r.loras.push_back(std::move(e));
    }

    for (const auto & line : split(cvec_pool ? cvec_pool : "", '\n')) {
        const auto f = split(line, '\t');
        if (f.size() < 2) {
            fprintf(stderr, "friend: malformed cvec pool entry '%s'\n", line.c_str());
            return false;
        }
        cvec_entry e;
        e.name = f[0];
        e.path = f[1];
        printf("\nfriend: loading control vector '%s' from %s\n", e.name.c_str(), e.path.c_str());
        common_control_vector_data cv = common_control_vector_load({ { 1.0f, e.path } });
        if (cv.n_embd == -1) {
            fprintf(stderr, "friend: failed to load control vector '%s'\n", e.name.c_str());
            return false;
        }
        if (cv.n_embd != r.n_embd) {
            fprintf(stderr, "friend: control vector '%s' has n_embd %d, model has %d\n", e.name.c_str(), cv.n_embd, r.n_embd);
            return false;
        }
        e.data = std::move(cv.data);
        r.cvecs.push_back(std::move(e));
    }

    for (const auto & line : split(head_pool ? head_pool : "", '\n')) {
        const auto f = split(line, '\t');
        if (f.size() < 2) {
            fprintf(stderr, "friend: malformed head pool entry '%s'\n", line.c_str());
            return false;
        }
        head_entry e;
        e.name = f[0];
        e.path = f[1];
        printf("\nfriend: loading LM head '%s' from %s\n", e.name.c_str(), e.path.c_str());
        e.head = llama_adapter_head_init(model, e.path.c_str());
        if (!e.head) {
            fprintf(stderr, "friend: failed to load head '%s'\n", e.name.c_str());
            return false;
        }
        r.heads.push_back(std::move(e));
    }

    return true;
}

inline void free_pools() {
    auto & r = reg();
    if (r.model) {
        llama_model_set_head(r.model, nullptr);
    }
    for (auto & h : r.heads) {
        llama_adapter_head_free(h.head);
    }
    // LoRA adapters are owned by the model and freed with it
    r = registry();
}

template <typename T>
static int find_by_name(const std::vector<T> & pool, const std::string & name) {
    for (size_t i = 0; i < pool.size(); ++i) {
        if (pool[i].name == name) {
            return (int) i;
        }
    }
    return -1;
}

// Canonicalise: drop zero weights, sort, compute keys.
inline void finalize(profile & p) {
    auto & r = reg();
    auto canon = [](std::vector<std::pair<int, float>> & v) {
        v.erase(std::remove_if(v.begin(), v.end(), [](const auto & e) { return e.second == 0.0f; }), v.end());
        std::sort(v.begin(), v.end());
        // merge duplicates by summing (a request may list the same adapter twice)
        std::vector<std::pair<int, float>> merged;
        for (const auto & e : v) {
            if (!merged.empty() && merged.back().first == e.first) {
                merged.back().second += e.second;
            } else {
                merged.push_back(e);
            }
        }
        v.swap(merged);
    };
    canon(p.loras);
    canon(p.cvecs);

    std::string k;
    for (const auto & [i, s] : p.loras) {
        k += "L:" + r.loras[i].name + "@" + fmt_scale(s) + ";";
    }
    for (const auto & [i, s] : p.cvecs) {
        k += "C:" + r.cvecs[i].name + "@" + fmt_scale(s) + ";";
    }
    p.kv_key   = k;
    p.head_key = p.head >= 0 ? r.heads[p.head].name : "";
}

inline profile default_profile() {
    auto & r = reg();
    profile p;
    for (size_t i = 0; i < r.loras.size(); ++i) {
        if (r.loras[i].default_scale != 0.0f) {
            p.loras.emplace_back((int) i, r.loras[i].default_scale);
        }
    }
    finalize(p);
    return p;
}

// spec == nullptr or "" means "defaults". Unknown names are an error.
inline bool parse_profile_impl(const char * spec, profile & out, std::string & err) {
    auto & r = reg();
    out = profile();
    bool explicit_lora = false;
    std::vector<std::pair<int, float>> loras;

    for (const auto & entry : split(spec ? spec : "", ';')) {
        const auto f = split(entry, ' ');
        if (f.empty()) {
            continue;
        }
        if (f[0] == "lora_explicit") {
            explicit_lora = true;
        } else if (f[0] == "L" && f.size() == 3) {
            const int i = find_by_name(r.loras, f[1]);
            if (i < 0) { err = "unknown lora '" + f[1] + "'"; return false; }
            loras.emplace_back(i, std::stof(f[2]));
        } else if (f[0] == "C" && f.size() == 3) {
            const int i = find_by_name(r.cvecs, f[1]);
            if (i < 0) { err = "unknown control vector '" + f[1] + "'"; return false; }
            out.cvecs.emplace_back(i, std::stof(f[2]));
        } else if (f[0] == "H" && f.size() == 2) {
            const int i = find_by_name(r.heads, f[1]);
            if (i < 0) { err = "unknown head '" + f[1] + "'"; return false; }
            out.head = i;
        } else {
            err = "malformed adapter profile entry '" + entry + "'";
            return false;
        }
    }

    if (explicit_lora) {
        out.loras = loras;
    } else {
        out.loras = default_profile().loras;
    }
    finalize(out);
    return true;
}

inline bool parse_profile(const char * spec, profile & out, std::string & err) {
    try {
        return parse_profile_impl(spec, out, err);
    } catch (const std::exception & e) { // std::stof on a malformed number
        err = std::string("malformed adapter profile: ") + e.what();
        return false;
    }
}

// Composite steering vector for a profile (sum of strength * direction).
inline std::vector<float> compose_cvec(const profile & p) {
    auto & r = reg();
    std::vector<float> data;
    for (const auto & [i, s] : p.cvecs) {
        const auto & src = r.cvecs[i].data;
        if (data.size() < src.size()) {
            data.resize(src.size(), 0.0f);
        }
        for (size_t j = 0; j < src.size(); ++j) {
            data[j] += s * src[j];
        }
    }
    return data;
}

// Apply a profile to every context that runs on the registry's model (main, guidance,
// MTP draft...). Contexts on *other* models (a separate draft model) must not be passed.
// Must be called between decodes. Returns false if llama rejected something.
inline bool apply(const profile & p, const std::vector<llama_context *> & ctxs) {
    auto & r = reg();
    if (!r.model) {
        return true; // no GGUF model / pools never loaded
    }

    // LoRA set and cvec composition; each applied per context only when it changed,
    // since both force a scheduler re-reserve inside llama
    std::string lora_key, cvec_key;
    for (const auto & [i, s] : p.loras) {
        lora_key += r.loras[i].name + "@" + fmt_scale(s) + ";";
    }
    for (const auto & [i, s] : p.cvecs) {
        cvec_key += r.cvecs[i].name + "@" + fmt_scale(s) + ";";
    }

    std::vector<float> cvec_data;
    bool cvec_composed = false;

    for (llama_context * ctx : ctxs) {
        if (!ctx) {
            continue;
        }
        auto it_l = r.applied_lora.find(ctx);
        if (it_l == r.applied_lora.end() || it_l->second != lora_key) {
            std::vector<llama_adapter_lora *> adapters;
            std::vector<float> scales;
            for (const auto & [i, s] : p.loras) {
                adapters.push_back(r.loras[i].adapter);
                scales.push_back(s);
            }
            if (llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), scales.data()) != 0) {
                fprintf(stderr, "friend: failed to apply LoRA set '%s'\n", lora_key.c_str());
                return false;
            }
            r.applied_lora[ctx] = lora_key;
        }

        auto it_c = r.applied_cvec.find(ctx);
        // contexts never touched start with no control vector, which equals the empty key
        const std::string & have = it_c == r.applied_cvec.end() ? std::string() : it_c->second;
        if (have != cvec_key) {
            int32_t rc;
            if (p.cvecs.empty()) {
                rc = llama_set_adapter_cvec(ctx, nullptr, 0, r.n_embd, -1, -1);
            } else {
                if (!cvec_composed) {
                    cvec_data = compose_cvec(p);
                    cvec_composed = true;
                }
                rc = llama_set_adapter_cvec(ctx, cvec_data.data(), cvec_data.size(), r.n_embd, 1, r.n_layer);
            }
            if (rc != 0) {
                fprintf(stderr, "friend: failed to apply control vectors '%s'\n", cvec_key.c_str());
                return false;
            }
            r.applied_cvec[ctx] = cvec_key;
        }
    }

    // the head is model-wide
    if (llama_model_set_head(r.model, p.head >= 0 ? r.heads[p.head].head : nullptr) != 0) {
        fprintf(stderr, "friend: failed to apply head '%s'\n", p.head_key.c_str());
        return false;
    }
    return true;
}

// Forget per-context bookkeeping (call when a context is freed/recreated).
inline void forget_context(const llama_context * ctx) {
    reg().applied_lora.erase(ctx);
    reg().applied_cvec.erase(ctx);
}

} // namespace friend_adapters
