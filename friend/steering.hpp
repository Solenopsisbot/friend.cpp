#pragma once
// friend.cpp: build steering (control) vectors from contrastive examples.
//
// Give it prompts where the model "is" something (positive: an excited Rook) and prompts
// where it isn't (negative: a flat Rook). We run every prompt through the base model,
// read the residual stream at the end of each layer (the `l_out-N` graph tensors -- the
// exact point where llama adds a control vector back in), and per layer take:
//
//   mean  : mean(positive) - mean(negative)            (pairs not required)
//   pca   : first principal component of the per-pair differences, signed and scaled by
//           its projection of the mean difference     (needs equal-length pos/neg lists)
//
// Steering is applied only across a band of layers -- by default the contiguous band around
// the layer where the two sets separate best (separation >= 60% of the peak) -- and each
// layer's vector is divided by the band width. Vectors added at successive layers accumulate
// in the residual stream, so this makes strength 1.0 mean roughly "shift the hidden state by
// the difference actually observed between the two sets", rather than band-width times that.
// `normalize` gives plain unit vectors instead (no division), for manual tuning.
//
// Capture runs in a short-lived side context on the already-loaded model with an eval
// callback, so the main context, its KV cache and any active adapters are untouched and
// the measurement is always against the plain base model.
//
// Output format is llama.cpp's control-vector GGUF (arch "controlvector", F32 tensors
// "direction.<layer>", layer >= 1), loadable by --cvec-pool / common_control_vector_load.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace friend_steering {

struct spec {
    std::vector<std::vector<llama_token>> positive;
    std::vector<std::vector<llama_token>> negative;
    std::string method = "mean";   // "mean" | "pca"
    std::string pool   = "last";   // "last" token of each prompt | "mean" over all tokens
    int layer_start = -1;          // inclusive; -1 = automatic band (see build)
    int layer_end   = -1;          // inclusive; -1 = last layer (or automatic band)
    bool normalize  = false;       // unit vectors per layer instead of observed-shift scale
    int n_threads   = 4;
};

struct result {
    bool ok = false;
    std::string error;
    int32_t n_embd  = 0;
    int32_t n_layer = 0;
    std::vector<float> data;        // (n_layer - 1) * n_embd, layer 1 at offset 0
    std::vector<float> layer_norms; // per layer 1..n_layer-1, for diagnostics
    std::vector<float> separation;  // per layer: |mean diff| / pooled std of the projection
    int band_start = 0;             // layers the vector is actually applied to (inclusive)
    int band_end   = 0;
};

// eval-callback state: one pooled [n_embd] vector per layer for the prompt being decoded
struct capture {
    int n_layer = 0;
    int n_embd  = 0;
    bool mean_pool = false;
    std::vector<std::vector<float>> per_layer;
    std::vector<uint8_t> got;
    std::vector<float> scratch;
};

inline bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (capture *) ud;
    if (strncmp(t->name, "l_out-", 6) != 0) {
        return !ask; // don't observe other tensors, never abort the graph
    }
    if (ask) {
        return true;
    }
    const int il = atoi(t->name + 6);
    if (il < 0 || il >= st->n_layer || t->type != GGML_TYPE_F32 || t->ne[0] != st->n_embd || !ggml_is_contiguous(t)) {
        return true;
    }
    const int64_t rows = ggml_nrows(t);
    if (rows <= 0) {
        return true;
    }
    auto & out = st->per_layer[il];
    out.assign(st->n_embd, 0.0f);
    if (st->mean_pool) {
        st->scratch.resize((size_t) rows * st->n_embd);
        ggml_backend_tensor_get(t, st->scratch.data(), 0, ggml_nbytes(t));
        for (int64_t r = 0; r < rows; ++r) {
            const float * row = st->scratch.data() + r * st->n_embd;
            for (int j = 0; j < st->n_embd; ++j) out[j] += row[j];
        }
        for (float & v : out) v /= (float) rows;
    } else {
        ggml_backend_tensor_get(t, out.data(), (size_t) (rows - 1) * t->nb[1], (size_t) st->n_embd * sizeof(float));
    }
    st->got[il] = 1;
    return true;
}

inline double dot(const float * a, const float * b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double) a[i] * b[i];
    return s;
}

// Run each prompt through a side context and return per-prompt, per-layer pooled states.
inline bool capture_all(llama_model * model, const std::vector<std::vector<llama_token>> & prompts, const spec & sp,
                        std::vector<std::vector<std::vector<float>>> & out, std::string & err) {
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    size_t max_len = 1;
    for (const auto & p : prompts) max_len = std::max(max_len, p.size());

    capture st;
    st.n_layer = n_layer;
    st.n_embd = n_embd;
    st.mean_pool = sp.pool == "mean";

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = (uint32_t) (max_len + 16);
    cp.n_batch    = cp.n_ctx;
    cp.n_ubatch   = cp.n_ctx; // whole prompt in one ubatch -> one callback per layer
    cp.n_seq_max  = 1;
    cp.cb_eval    = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        err = "could not create capture context";
        return false;
    }
    llama_set_n_threads(ctx, sp.n_threads, sp.n_threads);

    out.clear();
    llama_batch batch = llama_batch_init((int32_t) max_len, 0, 1);
    bool ok = true;
    for (const auto & p : prompts) {
        if (p.empty()) { err = "empty prompt"; ok = false; break; }
        llama_memory_clear(llama_get_memory(ctx), true);
        st.per_layer.assign(n_layer, {});
        st.got.assign(n_layer, 0);
        batch.n_tokens = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            batch.token[i]     = p[i];
            batch.pos[i]       = (llama_pos) i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1; // every row is an output, so the last layer keeps all rows
        }
        batch.n_tokens = (int32_t) p.size();
        if (llama_decode(ctx, batch) != 0) {
            err = "decode failed while capturing";
            ok = false;
            break;
        }
        for (int il = 1; il < n_layer; ++il) {
            if (!st.got[il]) {
                err = "layer " + std::to_string(il) + " output (l_out) was not observable for this architecture";
                ok = false;
                break;
            }
        }
        if (!ok) break;
        out.push_back(st.per_layer);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

inline result build(llama_model * model, const spec & sp) {
    result r;
    r.n_embd  = llama_model_n_embd(model);
    r.n_layer = llama_model_n_layer(model);
    const int n_embd = r.n_embd, n_layer = r.n_layer;
    if (sp.positive.empty() || sp.negative.empty()) {
        r.error = "need at least one positive and one negative example";
        return r;
    }
    const bool paired = sp.positive.size() == sp.negative.size();
    if (sp.method == "pca" && (!paired || sp.positive.size() < 2)) {
        r.error = "method 'pca' needs equal-length positive/negative lists with at least 2 pairs";
        return r;
    }
    if (sp.method != "pca" && sp.method != "mean") {
        r.error = "method must be 'mean' or 'pca'";
        return r;
    }

    std::vector<std::vector<std::vector<float>>> pos, neg;
    if (!capture_all(model, sp.positive, sp, pos, r.error) || !capture_all(model, sp.negative, sp, neg, r.error)) {
        return r;
    }

    const int l0 = std::max(1, sp.layer_start);
    const int l1 = sp.layer_end < 0 ? n_layer - 1 : std::min(n_layer - 1, sp.layer_end);
    std::vector<std::vector<float>> dirs(n_layer);
    r.data.assign((size_t) (n_layer - 1) * n_embd, 0.0f);
    r.layer_norms.assign(n_layer - 1, 0.0f);
    r.separation.assign(n_layer - 1, 0.0f);

    std::vector<double> mp(n_embd), mn(n_embd);
    std::vector<float> md(n_embd), dir(n_embd);
    for (int il = 1; il < n_layer; ++il) {
        std::fill(mp.begin(), mp.end(), 0.0);
        std::fill(mn.begin(), mn.end(), 0.0);
        for (auto & s : pos) for (int j = 0; j < n_embd; ++j) mp[j] += s[il][j];
        for (auto & s : neg) for (int j = 0; j < n_embd; ++j) mn[j] += s[il][j];
        for (int j = 0; j < n_embd; ++j) md[j] = (float) (mp[j] / pos.size() - mn[j] / neg.size());
        const double md_norm = std::sqrt(dot(md.data(), md.data(), n_embd));

        if (sp.method == "pca" && md_norm > 0) {
            // power iteration on the (uncentred) covariance of pair differences; the first PC
            // is the dominant consistent direction, robust to one-off pair noise
            const size_t np = pos.size();
            std::vector<std::vector<float>> d(np, std::vector<float>(n_embd));
            for (size_t i = 0; i < np; ++i)
                for (int j = 0; j < n_embd; ++j) d[i][j] = pos[i][il][j] - neg[i][il][j];
            for (int j = 0; j < n_embd; ++j) dir[j] = (float) (md[j] / md_norm);
            std::vector<double> proj(np);
            std::vector<float> next(n_embd);
            for (int it = 0; it < 40; ++it) {
                for (size_t i = 0; i < np; ++i) proj[i] = dot(d[i].data(), dir.data(), n_embd);
                std::fill(next.begin(), next.end(), 0.0f);
                for (size_t i = 0; i < np; ++i)
                    for (int j = 0; j < n_embd; ++j) next[j] += (float) proj[i] * d[i][j];
                const double nn = std::sqrt(dot(next.data(), next.data(), n_embd));
                if (nn == 0) break;
                for (int j = 0; j < n_embd; ++j) dir[j] = (float) (next[j] / nn);
            }
            double s = dot(dir.data(), md.data(), n_embd); // signed so +strength pushes toward positive
            for (int j = 0; j < n_embd; ++j) md[j] = (float) (dir[j] * s);
        }

        // how cleanly the two sets separate along this direction (d' on the projections)
        {
            const double n2 = dot(md.data(), md.data(), n_embd);
            if (n2 > 0) {
                auto stats = [&](const std::vector<std::vector<std::vector<float>>> & set, double & m, double & v) {
                    m = 0; v = 0;
                    std::vector<double> p;
                    for (auto & s : set) p.push_back(dot(s[il].data(), md.data(), n_embd) / std::sqrt(n2));
                    for (double x : p) m += x;
                    m /= p.size();
                    for (double x : p) v += (x - m) * (x - m);
                    v = p.size() > 1 ? v / (p.size() - 1) : 0;
                };
                double m1, v1, m2, v2;
                stats(pos, m1, v1);
                stats(neg, m2, v2);
                const double pooled = std::sqrt(0.5 * (v1 + v2));
                r.separation[il - 1] = pooled > 0 ? (float) ((m1 - m2) / pooled) : 0.0f;
            }
        }

        dirs[il] = md;
    }

    // layer band: explicit, or automatic around the best-separating layer
    int b0 = l0, b1 = l1;
    if (sp.layer_start < 0) {
        int best = 1;
        for (int il = 2; il < n_layer; ++il) if (r.separation[il - 1] > r.separation[best - 1]) best = il;
        const float thr = 0.6f * r.separation[best - 1];
        b0 = best; b1 = best;
        while (b0 > 1 && r.separation[b0 - 2] >= thr) --b0;
        while (b1 < n_layer - 1 && r.separation[b1] >= thr) ++b1;
    }
    r.band_start = b0;
    r.band_end   = b1;
    const float band_scale = 1.0f / (float) (b1 - b0 + 1);
    for (int il = b0; il <= b1; ++il) {
        const auto & v = dirs[il];
        const double norm = std::sqrt(dot(v.data(), v.data(), n_embd));
        const float scale = sp.normalize ? (norm > 0 ? (float) (1.0 / norm) : 0.0f) : band_scale;
        r.layer_norms[il - 1] = (float) norm * scale;
        float * dst = r.data.data() + (size_t) (il - 1) * n_embd;
        for (int j = 0; j < n_embd; ++j) dst[j] = v[j] * scale;
    }
    r.ok = true;
    return r;
}

// Write llama.cpp's control-vector GGUF.
inline bool write_gguf(const std::string & path, const result & r, const std::string & model_hint) {
    const int n_dirs = r.n_layer - 1;
    ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) n_dirs * (ggml_tensor_overhead() + (size_t) r.n_embd * sizeof(float)) + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "controlvector");
    gguf_set_val_str(g, "controlvector.model_hint", model_hint.c_str());
    gguf_set_val_i32(g, "controlvector.layer_count", n_dirs);
    for (int il = 1; il < r.n_layer; ++il) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, r.n_embd);
        ggml_set_name(t, ("direction." + std::to_string(il)).c_str());
        memcpy(t->data, r.data.data() + (size_t) (il - 1) * r.n_embd, (size_t) r.n_embd * sizeof(float));
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

} // namespace friend_steering
