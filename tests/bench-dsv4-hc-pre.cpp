#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct bench_args {
    int64_t n_embd   = 4096;
    int64_t n_hc     = 4;
    int64_t n_tokens = 1;
    int     n_threads = 32;
    int     n_warmup  = 10;
    int     n_iters   = 200;
    int     sinkhorn_iters = 2;
    float   eps       = 1e-6f;
};

static int64_t arg_i64(const char * s) {
    return std::strtoll(s, nullptr, 10);
}

static int arg_i32(const char * s) {
    return std::atoi(s);
}

static float arg_f32(const char * s) {
    return std::strtof(s, nullptr);
}

static bench_args parse_args(int argc, char ** argv) {
    bench_args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char * {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", key.c_str());
                std::exit(2);
            }
            return argv[i];
        };

        if (key == "--n-embd") {
            args.n_embd = arg_i64(next());
        } else if (key == "--n-hc") {
            args.n_hc = arg_i64(next());
        } else if (key == "--tokens") {
            args.n_tokens = arg_i64(next());
        } else if (key == "--threads") {
            args.n_threads = arg_i32(next());
        } else if (key == "--warmup") {
            args.n_warmup = arg_i32(next());
        } else if (key == "--iters") {
            args.n_iters = arg_i32(next());
        } else if (key == "--sinkhorn-iters") {
            args.sinkhorn_iters = arg_i32(next());
        } else if (key == "--eps") {
            args.eps = arg_f32(next());
        } else if (key == "--help") {
            std::printf("usage: %s [--n-embd N] [--n-hc N] [--tokens N] [--threads N] [--warmup N] [--iters N] [--sinkhorn-iters N] [--eps F]\n", argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key.c_str());
            std::exit(2);
        }
    }
    return args;
}

struct graph_case {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * inp = nullptr;
    ggml_tensor  * out = nullptr;
    ggml_cplan     plan = {};
    std::vector<uint8_t> work;

    graph_case() = default;
    graph_case(const graph_case &) = delete;
    graph_case & operator=(const graph_case &) = delete;

    graph_case(graph_case && other) noexcept {
        *this = std::move(other);
    }

    graph_case & operator=(graph_case && other) noexcept {
        if (this != &other) {
            if (ctx) {
                ggml_free(ctx);
            }
            ctx = other.ctx;
            gf = other.gf;
            inp = other.inp;
            out = other.out;
            plan = other.plan;
            work = std::move(other.work);
            if (plan.work_size > 0) {
                plan.work_data = work.data();
            }
            other.ctx = nullptr;
            other.gf = nullptr;
            other.inp = nullptr;
            other.out = nullptr;
            other.plan = {};
        }
        return *this;
    }

    ~graph_case() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static void init_input(ggml_tensor * t) {
    float * data = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        const float a = std::sin(float(i) * 0.001f);
        const float b = std::cos(float(i) * 0.0003f);
        data[i] = 0.25f * a + 0.125f * b;
    }
}

static void init_scaled(ggml_tensor * t, float scale, float bias) {
    float * data = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        const float a = std::sin(float(i) * 0.0017f);
        const float b = std::cos(float(i) * 0.0005f);
        data[i] = bias + scale * (0.5f * a + 0.25f * b);
    }
}

static graph_case make_graph(const bench_args & args, const char * variant) {
    const int64_t hc_dim = args.n_embd * args.n_hc;

    const size_t mem_size =
        64ull * 1024 * 1024 +
        8ull * sizeof(float) * size_t(hc_dim) * size_t(std::max<int64_t>(1, args.n_tokens));

    ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    graph_case c;
    c.ctx = ggml_init(params);
    c.inp = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, args.n_embd, args.n_tokens);
    ggml_set_name(c.inp, "embd");
    init_input(c.inp);

    ggml_tensor * out = nullptr;
    if (std::strcmp(variant, "baseline") == 0 || std::strcmp(variant, "view-only") == 0 || std::strcmp(variant, "cont-only") == 0) {
        ggml_tensor * x = ggml_reshape_3d(c.ctx, c.inp, args.n_embd, 1, args.n_tokens);
        x = ggml_repeat_4d(c.ctx, x, args.n_embd, args.n_hc, args.n_tokens, 1);
        x = ggml_reshape_3d(c.ctx, x, args.n_embd, args.n_hc, args.n_tokens);

        ggml_tensor * flat = ggml_reshape_2d(c.ctx, x, hc_dim, args.n_tokens);
        if (std::strcmp(variant, "view-only") == 0) {
            out = ggml_rms_norm(c.ctx, flat, args.eps);
        } else {
            ggml_tensor * cont = ggml_cont(c.ctx, flat);
            out = std::strcmp(variant, "cont-only") == 0 ? cont : ggml_rms_norm(c.ctx, cont, args.eps);
        }
    } else if (std::strcmp(variant, "rms-contig") == 0) {
        ggml_tensor * flat = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, hc_dim, args.n_tokens);
        ggml_set_name(flat, "flat");
        float * dst = (float *) flat->data;
        const float * src = (const float *) c.inp->data;
        for (int64_t t = 0; t < args.n_tokens; ++t) {
            for (int64_t h = 0; h < args.n_hc; ++h) {
                std::memcpy(dst + t * hc_dim + h * args.n_embd,
                        src + t * args.n_embd,
                        size_t(args.n_embd) * sizeof(float));
            }
        }
        out = ggml_rms_norm(c.ctx, flat, args.eps);
    } else if (std::strcmp(variant, "hc-mix-only") == 0 || std::strcmp(variant, "hc-mix-split-pre") == 0) {
        const int64_t mix_hc = (2 + args.n_hc) * args.n_hc;
        ggml_tensor * flat = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, hc_dim, args.n_tokens);
        ggml_tensor * fn   = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, hc_dim, mix_hc);
        ggml_tensor * scale = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 3);
        ggml_tensor * base  = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, mix_hc);
        ggml_set_name(flat, "flat");
        ggml_set_name(fn, "hc_fn");
        ggml_set_name(scale, "hc_scale");
        ggml_set_name(base, "hc_base");
        init_scaled(flat, 0.1f, 0.0f);
        init_scaled(fn, 0.01f, 0.0f);
        init_scaled(scale, 0.1f, 1.0f);
        init_scaled(base, 0.01f, 0.0f);
        ggml_tensor * mixes = ggml_mul_mat(c.ctx, fn, flat);
        if (std::strcmp(variant, "hc-mix-only") == 0) {
            out = mixes;
        } else {
            ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(c.ctx, mixes, scale, base, args.n_hc, args.sinkhorn_iters, args.eps);
            ggml_tensor * pre = ggml_view_2d(c.ctx, split, args.n_hc, args.n_tokens, split->nb[1], 0);
            out = ggml_cont(c.ctx, pre);
        }
    } else if (std::strcmp(variant, "hc-split-only") == 0 || std::strcmp(variant, "hc-pre-cont-only") == 0) {
        const int64_t mix_hc = (2 + args.n_hc) * args.n_hc;
        ggml_tensor * mixes = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, mix_hc, args.n_tokens);
        ggml_tensor * scale = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 3);
        ggml_tensor * base  = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, mix_hc);
        ggml_set_name(mixes, "mixes");
        ggml_set_name(scale, "hc_scale");
        ggml_set_name(base, "hc_base");
        init_scaled(mixes, 0.1f, 0.0f);
        init_scaled(scale, 0.1f, 1.0f);
        init_scaled(base, 0.01f, 0.0f);
        ggml_tensor * split = std::strcmp(variant, "hc-pre-cont-only") == 0
            ? mixes
            : ggml_dsv4_hc_split_sinkhorn(c.ctx, mixes, scale, base, args.n_hc, args.sinkhorn_iters, args.eps);
        if (std::strcmp(variant, "hc-split-only") == 0) {
            out = split;
        } else {
            ggml_tensor * pre = ggml_view_2d(c.ctx, split, args.n_hc, args.n_tokens, split->nb[1], 0);
            out = ggml_cont(c.ctx, pre);
        }
    } else {
        std::fprintf(stderr, "unknown variant: %s\n", variant);
        std::exit(2);
    }

    ggml_set_name(out, variant);
    c.out = out;
    c.gf = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(c.gf, out);
    c.plan = ggml_graph_plan(c.gf, args.n_threads, nullptr);
    if (c.plan.work_size > 0) {
        c.work.resize(c.plan.work_size);
        c.plan.work_data = c.work.data();
    }
    return c;
}

static void compute(graph_case & c) {
    ggml_graph_compute(c.gf, &c.plan);
}

static double bench_graph(graph_case & c, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        compute(c);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        compute(c);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / double(iters);
}

static void manual_repeat_rms(const float * inp, float * out, const bench_args & args) {
    const int64_t hc_dim = args.n_embd * args.n_hc;
    for (int64_t t = 0; t < args.n_tokens; ++t) {
        const float * x = inp + t * args.n_embd;
        double sum = 0.0;
        for (int64_t i = 0; i < args.n_embd; ++i) {
            sum += double(x[i]) * double(x[i]);
        }
        const float scale = 1.0f / std::sqrt(float(sum / double(args.n_embd)) + args.eps);
        for (int64_t h = 0; h < args.n_hc; ++h) {
            float * y = out + t * hc_dim + h * args.n_embd;
            for (int64_t i = 0; i < args.n_embd; ++i) {
                y[i] = x[i] * scale;
            }
        }
    }
}

static double bench_manual(const float * inp, float * out, const bench_args & args) {
    for (int i = 0; i < args.n_warmup; ++i) {
        manual_repeat_rms(inp, out, args);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < args.n_iters; ++i) {
        manual_repeat_rms(inp, out, args);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / double(args.n_iters);
}

static float max_abs_diff(const float * a, const float * b, int64_t n) {
    float diff = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

int main(int argc, char ** argv) {
    const bench_args args = parse_args(argc, argv);
    const int64_t hc_dim = args.n_embd * args.n_hc;
    const int64_t out_elems = hc_dim * args.n_tokens;

    std::printf("n_embd=%lld n_hc=%lld hc_dim=%lld n_tokens=%lld threads=%d warmup=%d iters=%d sinkhorn_iters=%d eps=%g\n",
            (long long) args.n_embd, (long long) args.n_hc, (long long) hc_dim,
            (long long) args.n_tokens, args.n_threads, args.n_warmup, args.n_iters,
            args.sinkhorn_iters, double(args.eps));
    std::fflush(stdout);

    graph_case baseline = make_graph(args, "baseline");
    graph_case view_only = make_graph(args, "view-only");
    graph_case cont_only = make_graph(args, "cont-only");
    graph_case rms_contig = make_graph(args, "rms-contig");
    graph_case hc_mix_only = make_graph(args, "hc-mix-only");
    graph_case hc_split_only = make_graph(args, "hc-split-only");
    graph_case hc_pre_cont_only = make_graph(args, "hc-pre-cont-only");
    graph_case hc_mix_split_pre = make_graph(args, "hc-mix-split-pre");

    std::vector<float> manual_out(static_cast<size_t>(out_elems));

    const double baseline_us = bench_graph(baseline, args.n_warmup, args.n_iters);
    const double view_us     = bench_graph(view_only, args.n_warmup, args.n_iters);
    const double cont_us     = bench_graph(cont_only, args.n_warmup, args.n_iters);
    const double rms_us      = bench_graph(rms_contig, args.n_warmup, args.n_iters);
    const double manual_us   = bench_manual((const float *) baseline.inp->data, manual_out.data(), args);
    const double hc_mix_us        = bench_graph(hc_mix_only, args.n_warmup, args.n_iters);
    const double hc_split_us      = bench_graph(hc_split_only, args.n_warmup, args.n_iters);
    const double hc_pre_cont_us   = bench_graph(hc_pre_cont_only, args.n_warmup, args.n_iters);
    const double hc_mix_split_us  = bench_graph(hc_mix_split_pre, args.n_warmup, args.n_iters);

    compute(baseline);
    manual_repeat_rms((const float *) baseline.inp->data, manual_out.data(), args);

    const float diff = max_abs_diff((const float *) baseline.out->data, manual_out.data(), out_elems);

    std::printf("| case | us/iter | vs baseline | notes |\n");
    std::printf("| --- | ---: | ---: | --- |\n");
    auto row = [&](const char * name, double us, const char * notes) {
        std::printf("| %s | %.3f | %.3f | %s |\n", name, us, us / baseline_us, notes);
    };
    row("ggml repeat+reshape+cont+rms_norm", baseline_us, "candidate A baseline");
    row("ggml repeat+reshape+rms_norm", view_us, "Step23-style no explicit CONT");
    row("ggml repeat+reshape+cont", cont_us, "CONT component estimate");
    row("ggml rms_norm on contiguous flat", rms_us, "RMS_NORM component estimate");
    row("manual repeat+rms_norm fused", manual_us, "upper-bound fused CPU loop");
    row("ggml hc_fn mul_mat", hc_mix_us, "Candidate B mix projection");
    row("ggml dsv4 split_sinkhorn", hc_split_us, "Candidate B split component");
    row("ggml pre view+cont", hc_pre_cont_us, "Candidate B pre CONT component");
    row("ggml hc_fn+split+pre_cont", hc_mix_split_us, "Candidate B combined");
    std::printf("max_abs_diff_manual_vs_baseline=%.9g\n", double(diff));

    return 0;
}
