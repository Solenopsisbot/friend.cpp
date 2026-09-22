// friend.cpp: standalone matmul bench for Q1_0 / PQ2_0 on the CUDA backend, decode and prefill.
// Links against koboldcpp_cublas.so and uses only the public ggml API, so it times exactly the
// dispatch the server runs (mmvq for ncols <= 8, then MMQ or dequantize + cuBLAS).
//
// usage: cuda-mm-bench <q1|pq2|ptq1|f32> <ncols> <dumpfile|-> [reps]
//   (f32 = unquantized weights: the pure cuBLAS SGEMM cost, to separate out the dequant pass)
//   env FRIEND_BENCH_NOMMQ=1   -> ggml_cuda_set_mul_mat_q(false), i.e. koboldcpp's --nommq
//
// Prints per-shape microseconds per matmul for a Qwen3-8B-shaped model (36 layers, hidden 4096,
// ffn 12288, 8 kv heads) and the implied matmul time per ncols-token ubatch. The output head only
// runs for the last token during prefill, so it is skipped when ncols > 8. The outputs of one
// instance of each shape go to <dumpfile> for exact comparison between builds / paths.
//
// build (on the CUDA box, from the source dir):
//   g++ -O2 -std=c++17 -Iggml/include tools/friend-bench/cuda-mm-bench.cpp ./koboldcpp_cublas.so \
//       -Wl,-rpath,'$ORIGIN' -o cuda-mm-bench
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct shape { const char * name; int K, N, per_ubatch; };

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <q1|pq2|ptq1|f32> <ncols> <dump|-> [reps]\n", argv[0]); return 1; }
    const std::string tname = argv[1];
    const ggml_type type = tname == "q1" ? GGML_TYPE_Q1_0 : tname == "ptq1" ? GGML_TYPE_PTQ1_0 :
                           tname == "f32" ? GGML_TYPE_F32 : GGML_TYPE_PQ2_0;
    const int ncols = atoi(argv[2]);
    const char * dump = argv[3];
    const int reps = argc > 4 ? atoi(argv[4]) : (ncols > 8 ? 5 : 20);

    // per_ubatch = how many times the shape runs per forward pass (36 layers)
    const shape shapes[] = {
        { "q/o 4096x4096",    4096,   4096, 72 },
        { "k/v 4096x1024",    4096,   1024, 72 },
        { "gate/up 4096x12288", 4096, 12288, 72 },
        { "down 12288x4096", 12288,   4096, 36 },
        { "out 4096x151669",  4096, 151669,  1 },
    };

    if (getenv("FRIEND_BENCH_NOMMQ")) {
        ggml_cuda_set_mul_mat_q(false);
    }

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no cuda\n"); return 1; }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    FILE * fd = strcmp(dump, "-") ? fopen(dump, "wb") : nullptr;

    double total_us = 0.0;
    for (const shape & s : shapes) {
        if (ncols > 8 && s.per_ubatch == 1) {
            continue; // prefill only computes logits for the last token
        }
        // enough distinct weight copies that the working set exceeds the 1.75 MB L2
        const size_t wbytes = ggml_row_size(type, s.K) * s.N;
        int ncopy = (int) ((8u << 20) / wbytes) + 1;
        if (ncopy > 8) ncopy = 8;
        // every op gets its own output (no allocator reuse), so keep big batches to a few ops
        const int nops = ncols > 8 ? 4 : 32;

        ggml_init_params ip = { ggml_tensor_overhead() * (ncopy + nops * 2 + 8) + ggml_graph_overhead_custom(1024, false), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        std::vector<ggml_tensor *> W(ncopy);
        for (auto & w : W) w = ggml_new_tensor_2d(ctx, type, s.K, s.N);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, ncols);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        std::vector<ggml_tensor *> outs;
        for (int i = 0; i < nops; ++i) {
            ggml_tensor * o = ggml_mul_mat(ctx, W[i % ncopy], x);
            outs.push_back(o);
            ggml_build_forward_expand(gf, o);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        // weights: quantize random normal rows (per copy, in slices to bound host memory)
        {
            const int rows_per = 4096;
            std::vector<float> f((size_t) s.K * rows_per);
            std::vector<uint8_t> q(ggml_row_size(type, s.K) * rows_per);
            for (int c = 0; c < ncopy; ++c) {
                for (int r0 = 0; r0 < s.N; r0 += rows_per) {
                    const int nr = std::min(rows_per, s.N - r0);
                    for (size_t i = 0; i < (size_t) s.K * nr; ++i) f[i] = nd(rng);
                    const size_t nb = type == GGML_TYPE_F32 ? f.size() * sizeof(float) * nr / rows_per
                                    : ggml_quantize_chunk(type, f.data(), q.data(), 0, nr, s.K, nullptr);
                    ggml_backend_tensor_set(W[c], type == GGML_TYPE_F32 ? (const void *) f.data() : (const void *) q.data(),
                                            ggml_row_size(type, s.K) * r0, nb);
                }
            }
        }
        {
            std::vector<float> xf((size_t) s.K * ncols);
            for (auto & v : xf) v = nd(rng);
            ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
        }

        ggml_backend_graph_compute(be, gf); // warmup (+ CUDA graph capture)
        ggml_backend_synchronize(be);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * nops);
        const double gbs = wbytes / us * 1e-3;
        const double tflops = 2.0 * s.K * (double) s.N * ncols / us * 1e-6;
        printf("%-20s %9.1f us/matmul  %6.1f GB/s weights  %5.2f TFLOP/s  x%d = %8.1f us\n",
               s.name, us, gbs, tflops, s.per_ubatch, us * s.per_ubatch);
        total_us += us * s.per_ubatch;

        if (fd) {
            std::vector<float> out((size_t) s.N * ncols);
            ggml_backend_tensor_get(outs[0], out.data(), 0, out.size() * sizeof(float));
            fwrite(out.data(), sizeof(float), out.size(), fd);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    if (ncols > 8) {
        printf("layer matmul time per %d-token ubatch: %.2f ms  (%.1f tok/s matmul-only)\n",
               ncols, total_us / 1000.0, ncols / (total_us * 1e-6));
    } else {
        printf("estimated matmul time per token (ncols=%d): %.2f ms\n", ncols, total_us / 1000.0);
    }
    if (fd) fclose(fd);
    ggml_backend_free(be);
    return 0;
}
