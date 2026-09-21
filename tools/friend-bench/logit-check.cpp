// friend.cpp: whole-model logit check for small-batch decode paths.
//
//   logit-check run <model.gguf> <out.bin> [nmax=8] [mode=batch|seq]
//       Prefills a fixed prompt (48 tokens, one batch -- above every small-batch
//       threshold, so the prefill path is the same in every configuration), then for
//       n = 1..nmax decodes the next n prompt tokens in ONE batch (mode=batch) or one at
//       a time (mode=seq) and writes the logits of all n positions. The model state is
//       cleared and re-prefilled for every n (hybrid/recurrent models cannot roll back).
//       Kernel selection comes from the environment (e.g. GGML_METAL_BONSAI_SB_DISABLE=1).
//
//   logit-check cmp <a.bin> <b.bin>
//       Per n: max |a - b| over all positions and logits, max |a|, and how many positions
//       have the same argmax.
//
// Not part of any shipped target: `make LLAMA_METAL=1 logit-check`.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * k_prompt =
    "The lighthouse keeper had kept a log for thirty-one years, and in all that time the entries "
    "had been short: wind, weather, ships sighted, lamp trimmed. On the night the fog came in thick "
    "enough to swallow the beam entirely, she opened the book to a fresh page and wrote something else. "
    "She wrote about the gulls that nested on the gallery rail every spring, about the supply boat that "
    "came late in bad years and early in good ones, about the child who had once rowed out alone to ask "
    "whether the light ever got tired. She wrote until the oil in the lamp beside her ran low, and then "
    "she kept writing by the glow of the great lens turning overhead, because some nights a log is not "
    "enough and a letter is the only honest thing left to write.";

static int run(const char * model_path, const char * out_path, int nmax, bool seq) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 512;
    cp.n_batch   = 512;
    cp.n_ubatch  = 512;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(512);
    const int n_tok = llama_tokenize(vocab, k_prompt, (int) strlen(k_prompt), toks.data(), (int) toks.size(), true, false);
    const int n_pre = 48;
    if (n_tok < n_pre + nmax) {
        fprintf(stderr, "prompt too short: %d tokens\n", n_tok);
        return 1;
    }
    toks.resize(n_tok);

    FILE * fo = fopen(out_path, "wb");
    fwrite(&nmax, sizeof(int), 1, fo);
    fwrite(&n_vocab, sizeof(int), 1, fo);

    llama_batch batch = llama_batch_init(512, 0, 1);

    for (int n = 1; n <= nmax; n++) {
        llama_memory_clear(llama_get_memory(ctx), true);

        // prefill
        batch.n_tokens = 0;
        for (int i = 0; i < n_pre; i++) {
            batch.token[i] = toks[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = false;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }

        std::vector<float> logits((size_t) n*n_vocab);
        if (!seq) {
            batch.n_tokens = 0;
            for (int j = 0; j < n; j++) {
                const int i = n_pre + j;
                batch.token[j] = toks[i]; batch.pos[j] = i; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = true;
                batch.n_tokens++;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            for (int j = 0; j < n; j++) {
                memcpy(logits.data() + (size_t) j*n_vocab, llama_get_logits_ith(ctx, j), n_vocab*sizeof(float));
            }
        } else {
            for (int j = 0; j < n; j++) {
                const int i = n_pre + j;
                batch.n_tokens = 1;
                batch.token[0] = toks[i]; batch.pos[0] = i; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                memcpy(logits.data() + (size_t) j*n_vocab, llama_get_logits_ith(ctx, 0), n_vocab*sizeof(float));
            }
        }
        fwrite(logits.data(), sizeof(float), logits.size(), fo);
    }

    fclose(fo);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int cmp(const char * a_path, const char * b_path) {
    FILE * fa = fopen(a_path, "rb");
    FILE * fb = fopen(b_path, "rb");
    if (!fa || !fb) { fprintf(stderr, "cannot open inputs\n"); return 1; }
    int na, va, nb, vb;
    if (fread(&na, sizeof(int), 1, fa) != 1 || fread(&va, sizeof(int), 1, fa) != 1 ||
        fread(&nb, sizeof(int), 1, fb) != 1 || fread(&vb, sizeof(int), 1, fb) != 1 || na != nb || va != vb) {
        fprintf(stderr, "header mismatch\n"); return 1;
    }
    printf("%3s %12s %12s %10s %12s\n", "n", "max|a-b|", "max|a|", "rel", "argmax same");
    bool all_same = true;
    for (int n = 1; n <= na; n++) {
        std::vector<float> a((size_t) n*va), b((size_t) n*va);
        if (fread(a.data(), sizeof(float), a.size(), fa) != a.size() || fread(b.data(), sizeof(float), b.size(), fb) != b.size()) {
            fprintf(stderr, "short read\n"); return 1;
        }
        double md = 0, ma = 0;
        int same = 0;
        for (int j = 0; j < n; j++) {
            int ia = 0, ib = 0;
            for (int t = 0; t < va; t++) {
                const float x = a[(size_t) j*va + t], y = b[(size_t) j*va + t];
                md = std::max(md, (double) fabsf(x - y));
                ma = std::max(ma, (double) fabsf(x));
                if (x > a[(size_t) j*va + ia]) ia = t;
                if (y > b[(size_t) j*va + ib]) ib = t;
            }
            same += ia == ib;
        }
        all_same = all_same && same == n;
        printf("%3d %12.5f %12.3f %10.2e %8d/%d\n", n, md, ma, md/ma, same, n);
    }
    printf("argmax identical at every position: %s\n", all_same ? "yes" : "NO");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && strcmp(argv[1], "run") == 0) {
        const int  nmax = argc > 4 ? atoi(argv[4]) : 8;
        const bool seq  = argc > 5 && strcmp(argv[5], "seq") == 0;
        return run(argv[2], argv[3], nmax, seq);
    }
    if (argc >= 4 && strcmp(argv[1], "cmp") == 0) {
        return cmp(argv[2], argv[3]);
    }
    fprintf(stderr, "usage: %s run <model> <out.bin> [nmax] [batch|seq] | cmp <a.bin> <b.bin>\n", argv[0]);
    return 1;
}
