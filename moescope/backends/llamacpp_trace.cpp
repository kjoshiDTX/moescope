// moescope llama.cpp tracer: runs prefill + greedy decode and records each MoE layer's top-k
// router choices through the llama.cpp eval callback.
//
// Input (--input): a sequence of prompts, each either
//   TOK <id> <n>\n<n space-separated token ids>\n      pre-tokenized
//   TXT <id> <nbytes>\n<nbytes of utf-8 text>\n           tokenized here (chat template with --chat)
// Output (--out): binary records, all little-endian int32/float32:
//   int32 prompt_id, phase (0 prefill, 1 decode), token_pos, layer (raw block index), n_tokens, k
//   int32 ids[n_tokens * k]     highest weight first
//   float weights[n_tokens * k] softmax over the chosen experts' logits
//   float logit_max[n_tokens]
// stdout: one JSON line of model info, then one line per finished prompt ("done <id> <n_tokens>").

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

struct State {
    FILE * out = nullptr;
    int32_t prompt_id = 0;
    int32_t phase = 0;
    int32_t pos0 = 0;
    int32_t batch_n = 0;   // tokens submitted in the current llama_decode call
    int32_t n_layer = 0;
    bool list_names = false;
    std::set<std::string> names;
    std::map<int, std::vector<float>> logits;   // layer -> [n_tokens * n_expert]
    std::map<int, int64_t> logits_cols;          // layer -> n_expert
    std::map<int, int32_t> layer_offset;         // tokens already recorded for this decode call
};

static bool starts_with(const char * s, const char * prefix) { return strncmp(s, prefix, strlen(prefix)) == 0; }

static int layer_of(const char * name) {
    const char * dash = strrchr(name, '-');
    return dash ? atoi(dash + 1) : -1;
}

// Read a 2-D tensor (possibly a strided view) as float or int32 rows.
static std::vector<uint8_t> read_raw(const ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), raw.data(), 0, raw.size());
    return raw;
}

static float as_float(const ggml_tensor * t, const uint8_t * p) {
    switch (t->type) {
        case GGML_TYPE_F32: return *(const float *) p;
        case GGML_TYPE_F16: return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        default: fprintf(stderr, "unsupported logits type %s\n", ggml_type_name(t->type)); exit(3);
    }
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (State *) user_data;
    const bool want = starts_with(t->name, "ffn_moe_logits-") || starts_with(t->name, "ffn_moe_topk-");
    const bool listed = st->list_names && strstr(t->name, "moe") != nullptr;
    if (ask) return want || listed;
    if (listed) st->names.insert(t->name);
    if (!want) return true;
    const int il = layer_of(t->name);
    const int64_t cols = t->ne[0], rows = t->ne[1];
    auto raw = read_raw(t);

    if (starts_with(t->name, "ffn_moe_logits-")) {
        auto & dst = st->logits[il];
        dst.resize(cols * rows);
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) dst[r * cols + c] = as_float(t, raw.data() + r * t->nb[1] + c * t->nb[0]);
        st->logits_cols[il] = cols;
        return true;
    }

    // ffn_moe_topk: int32 [k, n_tokens]
    if (t->type != GGML_TYPE_I32) { fprintf(stderr, "unexpected topk type %s\n", ggml_type_name(t->type)); exit(3); }
    const int64_t k = cols;
    const auto & lg = st->logits[il];
    const int64_t n_expert = st->logits_cols[il];
    if ((int64_t) lg.size() != n_expert * rows || n_expert == 0) {
        fprintf(stderr, "layer %d: topk seen without matching router logits\n", il);
        exit(3);
    }
    std::vector<int32_t> ids(k * rows);
    std::vector<float> weights(k * rows), lmax(rows);
    for (int64_t r = 0; r < rows; r++) {
        const float * row = lg.data() + r * n_expert;
        lmax[r] = *std::max_element(row, row + n_expert);
        std::vector<int32_t> chosen(k);
        for (int64_t j = 0; j < k; j++) chosen[j] = *(const int32_t *) (raw.data() + r * t->nb[1] + j * t->nb[0]);
        std::sort(chosen.begin(), chosen.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
        float top = row[chosen[0]], sum = 0;
        for (int64_t j = 0; j < k; j++) sum += std::exp(row[chosen[j]] - top);
        for (int64_t j = 0; j < k; j++) {
            ids[r * k + j] = chosen[j];
            weights[r * k + j] = std::exp(row[chosen[j]] - top) / sum;
        }
    }
    // llama.cpp computes the last layer only for tokens whose logits are requested (the final
    // prompt token in prefill), so a short last-layer batch belongs at the end of the sequence.
    int32_t pos = st->pos0 + st->layer_offset[il];
    if (il == st->n_layer - 1 && rows < st->batch_n) pos = st->pos0 + st->batch_n - (int32_t) rows;
    int32_t header[6] = {st->prompt_id, st->phase, pos, il, (int32_t) rows, (int32_t) k};
    st->layer_offset[il] += (int32_t) rows;
    fwrite(header, sizeof(int32_t), 6, st->out);
    fwrite(ids.data(), sizeof(int32_t), ids.size(), st->out);
    fwrite(weights.data(), sizeof(float), weights.size(), st->out);
    fwrite(lmax.data(), sizeof(float), lmax.size(), st->out);
    return true;
}

static std::string meta(const llama_model * model, const std::string & key) {
    char buf[512];
    int n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

// Bytes of one routed expert: sum of the first MoE block's *_exps tensors over the expert count.
static int64_t expert_bytes(const char * path, int64_t n_expert) {
    gguf_init_params params = {/*no_alloc*/ true, /*ctx*/ nullptr};
    gguf_context * g = gguf_init_from_file(path, params);
    if (!g) return -1;
    std::map<int, int64_t> per_block;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        std::string name = gguf_get_tensor_name(g, i);
        int blk = -1;
        if (sscanf(name.c_str(), "blk.%d.", &blk) == 1 && name.find("_exps.weight") != std::string::npos)
            per_block[blk] += (int64_t) gguf_get_tensor_size(g, i);
    }
    gguf_free(g);
    if (per_block.empty() || n_expert <= 0) return -1;
    return per_block.begin()->second / n_expert;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n, add_special, true);
    return toks;
}

static std::string apply_chat(const llama_model * model, const std::string & user) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) return user;
    llama_chat_message msg = {"user", user.c_str()};
    std::vector<char> buf(user.size() * 2 + 1024);
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    if (n < 0) return user;  // template not supported by llama.cpp's built-in list
    if (n > (int) buf.size()) {
        buf.resize(n);
        llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    }
    return std::string(buf.data(), n);
}

static void usage() {
    fprintf(stderr, "usage: moescope-llama-trace --model M.gguf --input prompts.txt --out trace.bin "
                    "[--n-predict N] [--chat] [--n-gpu-layers N] [--n-ctx N] [--list-tensors]\n");
    exit(2);
}

int main(int argc, char ** argv) {
    std::string model_path, input_path, out_path;
    int n_predict = 128, n_gpu_layers = 999, n_ctx = 4096;
    bool chat = false, list = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) usage(); return std::string(argv[++i]); };
        if (a == "--model") model_path = next();
        else if (a == "--input") input_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--n-predict") n_predict = std::stoi(next());
        else if (a == "--n-gpu-layers") n_gpu_layers = std::stoi(next());
        else if (a == "--n-ctx") n_ctx = std::stoi(next());
        else if (a == "--chat") chat = true;
        else if (a == "--list-tensors") list = true;
        else usage();
    }
    if (model_path.empty() || input_path.empty() || out_path.empty()) usage();

    llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }, nullptr);
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string arch = meta(model, "general.architecture");
    const int64_t n_expert = atoll(meta(model, arch + ".expert_count").c_str());
    const int64_t n_used = atoll(meta(model, arch + ".expert_used_count").c_str());
    const int64_t n_shared = atoll(meta(model, arch + ".expert_shared_count").c_str());
    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    printf("{\"arch\": \"%s\", \"desc\": \"%s\", \"n_layer\": %d, \"n_expert\": %lld, \"n_expert_used\": %lld, "
           "\"n_expert_shared\": %lld, \"expert_bytes\": %lld, \"has_chat_template\": %s}\n",
           arch.c_str(), desc, llama_model_n_layer(model), (long long) n_expert, (long long) n_used, (long long) n_shared,
           (long long) expert_bytes(model_path.c_str(), n_expert), llama_model_chat_template(model, nullptr) ? "true" : "false");
    fflush(stdout);
    if (n_expert <= 0) { fprintf(stderr, "%s is not a mixture-of-experts model (no %s.expert_count)\n", model_path.c_str(), arch.c_str()); return 4; }

    State st;
    st.list_names = list;
    st.n_layer = llama_model_n_layer(model);
    st.out = fopen(out_path.c_str(), "wb");
    if (!st.out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.cb_eval = eval_cb;
    cparams.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_sampler * smpl = llama_sampler_init_greedy();

    FILE * in = fopen(input_path.c_str(), "rb");
    if (!in) { fprintf(stderr, "cannot read %s\n", input_path.c_str()); return 1; }
    char kind[4];
    int id;
    long long n;
    while (fscanf(in, "%3s %d %lld", kind, &id, &n) == 3) {
        fgetc(in);  // newline
        std::vector<llama_token> toks;
        if (strcmp(kind, "TOK") == 0) {
            toks.resize(n);
            for (long long i = 0; i < n; i++) if (fscanf(in, "%d", &toks[i]) != 1) { fprintf(stderr, "bad token list\n"); return 1; }
            fgetc(in);
        } else {
            std::string text(n, '\0');
            if (fread(text.data(), 1, n, in) != (size_t) n) { fprintf(stderr, "short text\n"); return 1; }
            fgetc(in);
            toks = chat ? tokenize(vocab, apply_chat(model, text), false) : tokenize(vocab, text, true);
        }
        if ((int) toks.size() + n_predict > n_ctx) toks.resize(std::max(1, n_ctx - n_predict));

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);
        st.prompt_id = id;
        st.phase = 0;
        st.pos0 = 0;
        st.batch_n = (int32_t) toks.size();
        st.layer_offset.clear();
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int32_t) toks.size()))) { fprintf(stderr, "decode failed\n"); return 1; }
        int decoded = 0;
        for (int step = 0; step < n_predict; step++) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) break;
            st.phase = 1;
            st.pos0 = (int32_t) toks.size() + step;
            st.batch_n = 1;
            st.layer_offset.clear();
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) { fprintf(stderr, "decode failed\n"); return 1; }
            decoded++;
        }
        fflush(st.out);
        printf("done %d %d\n", id, (int) toks.size() + decoded);
        fflush(stdout);
        if (list) {
            for (const auto & name : st.names) fprintf(stderr, "tensor %s\n", name.c_str());
            list = st.list_names = false;
        }
    }
    fclose(in);
    fclose(st.out);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
