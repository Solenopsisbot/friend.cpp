// friend.cpp: small-batch mat-vec check + timing for the Metal backend.
//
// For a quantized weight matrix A (M x K) and f32 activations B (K x n) this
//   1. computes mul_mat(A, B[:, :n]) in one op (the small-batch / verify-batch path), and
//   2. computes each column on its own with n = 1 (the single-token decode kernel),
// then reports the max abs difference between the two relative to max |out|, and the
// time per batched op. The point: a speculative-decode verify batch must score each
// token the way plain decode would, so the batched kernels should agree with the n = 1
// kernel to f32 rounding, not merely with the CPU reference (which quantizes the
// activations to Q8 for these types and so cannot tell f16 from f32 activations apart).
//
// usage: mv-check [type=q1_0] [M=34816] [K=5120] [nmax=16] [reps=64] [rounds=3]
//
// A/B on a shared GPU: MV_MODES="base;MC=1,MC_R0=2" runs every mode for every n,
// interleaved over `rounds` rounds, keeping each mode's best time. Each mode is a
// comma-separated list of GGML_METAL_<VAR>=<VAL> settings applied with setenv, so it
// only works for knobs the backend re-reads per op (not the static-cached ones).
//
// Not part of any shipped target: `make LLAMA_METAL=1 mv-check`.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static ggml_type parse_type(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * name = ggml_type_name((ggml_type) t);
        if (name && strcmp(name, s) == 0) {
            return (ggml_type) t;
        }
    }
    fprintf(stderr, "unknown type %s\n", s);
    exit(1);
}

typedef std::vector<std::pair<std::string, std::string>> mode_t_;

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static void apply_mode(const mode_t_ & mode, const std::set<std::string> & all_vars) {
    for (const auto & v : all_vars) {
        unsetenv(v.c_str());
    }
    for (const auto & kv : mode) {
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
}

int main(int argc, char ** argv) {
    const ggml_type type = parse_type(argc > 1 ? argv[1] : "q1_0");
    const int64_t M      = argc > 2 ? atoll(argv[2]) : 34816;
    const int64_t K      = argc > 3 ? atoll(argv[3]) : 5120;
    const int     nmax   = argc > 4 ? atoi (argv[4]) : 16;
    const int     reps   = argc > 5 ? atoi (argv[5]) : 64;
    const int     rounds = argc > 6 ? atoi (argv[6]) : 3;
    const int     nmin   = getenv("MV_NMIN") ? atoi(getenv("MV_NMIN")) : 1;

    std::vector<std::string> mode_names = split(getenv("MV_MODES") ? getenv("MV_MODES") : "base", ';');
    std::vector<mode_t_> modes;
    std::set<std::string> all_vars;
    for (const auto & m : mode_names) {
        mode_t_ mode;
        if (m != "base") {
            for (const auto & kv : split(m, ',')) {
                const auto eq = kv.find('=');
                const std::string k = "GGML_METAL_" + kv.substr(0, eq);
                const std::string v = eq == std::string::npos ? "1" : kv.substr(eq + 1);
                mode.push_back({k, v});
                all_vars.insert(k);
            }
        }
        modes.push_back(mode);
    }

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "no metal backend\n");
        return 1;
    }

    // weights + activations live in one backend buffer
    ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_2d(ctx_w, type,          K, M);
    ggml_tensor * B = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, nmax);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);

    std::mt19937 rng(1234);
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> wf(M*K);
        for (auto & v : wf) v = nd(rng);
        std::vector<uint8_t> wq(ggml_nbytes(A));
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, M, K, nullptr);
        ggml_backend_tensor_set(A, wq.data(), 0, wq.size());

        // activation-like: mostly small, a few large outlier channels
        std::vector<float> bf(K*nmax);
        for (int64_t i = 0; i < K*nmax; i++) {
            bf[i] = nd(rng) * ((i % K) % 97 == 0 ? 20.0f : 1.0f);
        }
        ggml_backend_tensor_set(B, bf.data(), 0, bf.size()*sizeof(float));
    }

    printf("type=%s M=%lld K=%lld (%.1f MB of weights), best of %d rounds x 5 x %d ops\n",
            ggml_type_name(type), (long long) M, (long long) K, ggml_nbytes(A)/1e6, rounds, reps);
    printf("%3s", "n");
    for (const auto & m : mode_names) {
        printf(" | %24s: %8s %9s", m.substr(0, 24).c_str(), "us/op", "maxrel");
    }
    printf("\n");

    for (int n = nmin; n <= nmax; n++) {
        std::vector<double> err (modes.size(), 0.0);
        std::vector<double> best(modes.size(), 1e30);

        // --- exactness: batched op vs n separate single-column ops, per mode
        for (size_t im = 0; im < modes.size(); im++) {
            apply_mode(modes[im], all_vars);

            ggml_init_params gp = { (size_t) (4*n + 64)*ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true };
            ggml_context * ctx = ggml_init(gp);

            ggml_tensor * Bn  = ggml_view_2d(ctx, B, K, n, B->nb[1], 0);
            ggml_tensor * out = ggml_mul_mat(ctx, A, Bn);
            std::vector<ggml_tensor *> refs;
            for (int j = 0; j < n; j++) {
                ggml_tensor * bj = ggml_view_2d(ctx, B, K, 1, B->nb[1], j*B->nb[1]);
                refs.push_back(ggml_mul_mat(ctx, A, bj));
            }

            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(gf, out);
            for (auto * r : refs) ggml_build_forward_expand(gf, r);

            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_alloc_graph(galloc, gf);
            ggml_backend_graph_compute(backend, gf);

            std::vector<float> o(M*n), r(M);
            ggml_backend_tensor_get(out, o.data(), 0, o.size()*sizeof(float));
            double maxdiff = 0.0, maxabs = 0.0;
            for (int j = 0; j < n; j++) {
                ggml_backend_tensor_get(refs[j], r.data(), 0, r.size()*sizeof(float));
                for (int64_t i = 0; i < M; i++) {
                    const double diff = fabs((double) o[j*M + i] - (double) r[i]);
                    maxdiff = std::isnan(diff) ? INFINITY : std::max(maxdiff, diff);
                    maxabs  = std::max(maxabs, (double) fabsf(r[i]));
                }
            }
            err[im] = maxabs > 0 ? maxdiff/maxabs : 0.0;
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }

        // --- timing: reps independent batched ops in one graph, modes interleaved
        ggml_init_params tp = { (size_t) (2*reps + 64)*ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true };
        ggml_context * ctx_t = ggml_init(tp);
        ggml_cgraph * gt = ggml_new_graph_custom(ctx_t, 8192, false);
        ggml_tensor * Bt = ggml_view_2d(ctx_t, B, K, n, B->nb[1], 0);
        for (int i = 0; i < reps; i++) {
            ggml_build_forward_expand(gt, ggml_mul_mat(ctx_t, A, Bt));
        }
        ggml_gallocr_t galloc_t = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(galloc_t, gt);

        for (int round = 0; round < rounds; round++) {
            for (size_t im = 0; im < modes.size(); im++) {
                apply_mode(modes[im], all_vars);
                ggml_backend_graph_compute(backend, gt); // warmup (pipeline compile)
                for (int it = 0; it < 5; it++) {
                    const auto t0 = std::chrono::high_resolution_clock::now();
                    ggml_backend_graph_compute(backend, gt);
                    const auto t1 = std::chrono::high_resolution_clock::now();
                    best[im] = std::min(best[im], std::chrono::duration<double, std::micro>(t1 - t0).count() / reps);
                }
            }
        }
        ggml_gallocr_free(galloc_t);
        ggml_free(ctx_t);

        printf("%3d", n);
        for (size_t im = 0; im < modes.size(); im++) {
            printf(" | %24s  %8.1f %9.2e", "", best[im], err[im]);
        }
        printf("\n");
        fflush(stdout);
    }

    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return 0;
}
