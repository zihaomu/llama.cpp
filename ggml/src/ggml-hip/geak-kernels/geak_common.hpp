#pragma once

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../../ggml-common.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define GEAK_HIP_CHECK(expr)                                                     \
    do {                                                                         \
        hipError_t err__ = (expr);                                                \
        if (err__ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,     \
                         hipGetErrorString(err__));                              \
            std::exit(1);                                                         \
        }                                                                        \
    } while (0)

#define GEAK_Q8_1_D(b)    ((b).data.data.d)
#define GEAK_Q8_1_S(b)    ((b).data.data.s)
#define GEAK_Q2_K_D(b)    ((b).data.data.d)
#define GEAK_Q2_K_DMIN(b) ((b).data.data.dmin)

struct geak_options {
    bool correctness = false;
    bool benchmark = false;
    bool profile = false;
    bool full_benchmark = false;
    int iterations = 200;
};

static inline geak_options geak_parse_options(int argc, char ** argv) {
    geak_options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--correctness") {
            opt.correctness = true;
        } else if (arg == "--benchmark") {
            opt.benchmark = true;
        } else if (arg == "--profile") {
            opt.profile = true;
        } else if (arg == "--full-benchmark") {
            opt.full_benchmark = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            opt.iterations = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            std::exit(2);
        }
    }

    if (!opt.correctness && !opt.benchmark && !opt.profile && !opt.full_benchmark) {
        opt.correctness = true;
    }

    const char * env_iters = std::getenv("GEAK_BENCHMARK_ITERATIONS");
    if (env_iters != nullptr && opt.iterations == 200) {
        opt.iterations = std::atoi(env_iters);
    }
    opt.iterations = std::max(1, opt.iterations);
    return opt;
}

static inline std::vector<int> geak_selected_indices(int n, const geak_options & opt) {
    std::vector<int> idx;
    if (opt.profile) {
        idx.push_back(std::max(0, n / 2));
        return idx;
    }
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
}

static inline void geak_print_indices(const std::vector<int> & idx) {
    std::printf("GEAK_SHAPES_USED=[");
    for (size_t i = 0; i < idx.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", idx[i]);
    }
    std::printf("]\n");
}

static inline float geak_half_to_float(ggml_half h) {
    const uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
    uint32_t exp = (uint32_t(h) >> 10) & 0x1fu;
    uint32_t mant = uint32_t(h) & 0x03ffu;

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            exp = exp + (127 - 15);
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        out = sign | (exp << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static inline ggml_half geak_float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return ggml_half(sign);
        }
        mant = (mant | 0x00800000u) >> uint32_t(1 - exp);
        return ggml_half(sign | ((mant + 0x00001000u) >> 13));
    }
    if (exp >= 31) {
        return ggml_half(sign | 0x7c00u);
    }
    return ggml_half(sign | (uint32_t(exp) << 10) | ((mant + 0x00001000u) >> 13));
}

template <typename T>
struct geak_device_buffer {
    T * ptr = nullptr;
    size_t count = 0;

    explicit geak_device_buffer(size_t n) : count(n) {
        GEAK_HIP_CHECK(hipMalloc(&ptr, sizeof(T) * count));
    }

    ~geak_device_buffer() {
        if (ptr != nullptr) {
            hipFree(ptr);
        }
    }

    geak_device_buffer(const geak_device_buffer &) = delete;
    geak_device_buffer & operator=(const geak_device_buffer &) = delete;
};

template <typename T>
static inline void geak_copy_to_device(geak_device_buffer<T> & dst, const std::vector<T> & src) {
    assert(dst.count >= src.size());
    GEAK_HIP_CHECK(hipMemcpy(dst.ptr, src.data(), sizeof(T) * src.size(), hipMemcpyHostToDevice));
}

template <typename T>
static inline void geak_copy_to_host(std::vector<T> & dst, const geak_device_buffer<T> & src) {
    assert(src.count >= dst.size());
    GEAK_HIP_CHECK(hipMemcpy(dst.data(), src.ptr, sizeof(T) * dst.size(), hipMemcpyDeviceToHost));
}

template <typename Fn>
static inline float geak_measure_ms(Fn && fn, int iterations) {
    constexpr int warmup = 50;
    constexpr int samples = 5;
    std::vector<float> times;
    times.reserve(samples);

    for (int s = 0; s < samples; ++s) {
        for (int i = 0; i < warmup; ++i) {
            fn();
        }
        GEAK_HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start;
        hipEvent_t stop;
        GEAK_HIP_CHECK(hipEventCreate(&start));
        GEAK_HIP_CHECK(hipEventCreate(&stop));
        GEAK_HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            fn();
        }
        GEAK_HIP_CHECK(hipEventRecord(stop));
        GEAK_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        GEAK_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        GEAK_HIP_CHECK(hipEventDestroy(start));
        GEAK_HIP_CHECK(hipEventDestroy(stop));
        times.push_back(ms / float(iterations));
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static inline bool geak_check_close(
        const std::vector<float> & got,
        const std::vector<float> & ref,
        float atol,
        float rtol,
        const char * label) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float tol = atol + rtol * std::fabs(ref[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / std::max(1.0e-6f, std::fabs(ref[i])));
        if (!(diff <= tol)) {
            ++bad;
            if (bad <= 4) {
                std::fprintf(stderr, "%s mismatch[%zu]: got=%g ref=%g diff=%g tol=%g\n",
                             label, i, got[i], ref[i], diff, tol);
            }
        }
    }
    std::printf("%s max_abs=%g max_rel=%g bad=%d/%zu\n", label, max_abs, max_rel, bad, got.size());
    return bad == 0;
}

static inline void geak_fill_q8_0(std::vector<block_q8_0> & x, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale_dist(0.004f, 0.04f);
    std::uniform_int_distribution<int> q_dist(-90, 90);
    for (block_q8_0 & b : x) {
        b.d = geak_float_to_half(scale_dist(rng));
        for (int i = 0; i < QK8_0; ++i) {
            b.qs[i] = int8_t(q_dist(rng));
        }
    }
}

static inline void geak_fill_q8_1(std::vector<block_q8_1> & y, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale_dist(0.004f, 0.04f);
    std::uniform_int_distribution<int> q_dist(-90, 90);
    for (block_q8_1 & b : y) {
        const float d = scale_dist(rng);
        int sum = 0;
        for (int i = 0; i < QK8_1; ++i) {
            b.qs[i] = int8_t(q_dist(rng));
            sum += int(b.qs[i]);
        }
        GEAK_Q8_1_D(b) = geak_float_to_half(d);
        GEAK_Q8_1_S(b) = geak_float_to_half(d * float(sum));
    }
}

static inline void geak_fill_q2_k(std::vector<block_q2_K> & x, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale_dist(0.004f, 0.04f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> nibble_dist(0, 15);
    for (block_q2_K & b : x) {
        for (uint8_t & s : b.scales) {
            s = uint8_t(nibble_dist(rng) | (nibble_dist(rng) << 4));
        }
        for (uint8_t & q : b.qs) {
            q = uint8_t(byte_dist(rng));
        }
        GEAK_Q2_K_D(b) = geak_float_to_half(scale_dist(rng));
        GEAK_Q2_K_DMIN(b) = geak_float_to_half(scale_dist(rng));
    }
}

static inline void geak_fill_iq2_xxs(std::vector<block_iq2_xxs> & x, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale_dist(0.004f, 0.04f);
    std::uniform_int_distribution<int> word_dist(0, 65535);
    for (block_iq2_xxs & b : x) {
        b.d = geak_float_to_half(scale_dist(rng));
        for (uint16_t & q : b.qs) {
            q = uint16_t(word_dist(rng));
        }
    }
}

static inline float geak_q8_0_value(const block_q8_0 & b, int i) {
    return geak_half_to_float(b.d) * float(b.qs[i]);
}

static inline float geak_q8_1_value(const block_q8_1 * y, int i) {
    const block_q8_1 & b = y[i / QK8_1];
    return geak_half_to_float(GEAK_Q8_1_D(b)) * float(b.qs[i % QK8_1]);
}

static inline float geak_q2_k_value(const block_q2_K & b, int i) {
    const int super = i / 128;
    const int within = i - 128 * super;
    const int j = within / 32;
    const int half = (within % 32) / 16;
    const int l = within % 16;
    const int scale_idx = super * 8 + j * 2 + half;
    const int q_idx = super * 32 + half * 16 + l;
    const int shift = 2 * j;

    const uint8_t sc = b.scales[scale_idx];
    const float d = geak_half_to_float(GEAK_Q2_K_D(b));
    const float dmin = geak_half_to_float(GEAK_Q2_K_DMIN(b));
    const int q = (b.qs[q_idx] >> shift) & 0x3;
    return d * float(sc & 0x0f) * float(q) - dmin * float(sc >> 4);
}

static inline float geak_iq2_xxs_value(const block_iq2_xxs & b, int i) {
    const int ib32 = i / 32;
    const int group = (i % 32) / 8;
    const int j = i % 8;
    const uint16_t * q = b.qs + 4 * ib32;
    const uint32_t aux0 = uint32_t(q[0]) | (uint32_t(q[1]) << 16);
    const uint32_t aux1 = uint32_t(q[2]) | (uint32_t(q[3]) << 16);
    const uint8_t grid_idx = uint8_t((aux0 >> (8 * group)) & 0xffu);
    const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * group)) & 127u];
    const uint8_t grid = uint8_t((iq2xxs_grid[grid_idx] >> (8 * j)) & 0xffu);
    const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
    const float db = geak_half_to_float(b.d) * (0.5f + float(aux1 >> 28)) * 0.25f;
    return db * float(grid) * sign;
}

static inline void geak_reference_q8_0_mat(
        const std::vector<block_q8_0> & x,
        const std::vector<block_q8_1> & y,
        std::vector<float> & out,
        int rows,
        int cols,
        int k) {
    const int xb = k / QK8_0;
    const int yb = k / QK8_1;
    out.assign(size_t(rows) * cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += geak_q8_0_value(x[r * xb + i / QK8_0], i % QK8_0) *
                       geak_q8_1_value(y.data() + c * yb, i);
            }
            out[size_t(c) * rows + r] = sum;
        }
    }
}

static inline void geak_reference_q2_k_mat(
        const std::vector<block_q2_K> & x,
        const std::vector<block_q8_1> & y,
        std::vector<float> & out,
        int rows,
        int cols,
        int k) {
    const int xb = k / QK_K;
    const int yb = k / QK8_1;
    out.assign(size_t(rows) * cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += geak_q2_k_value(x[r * xb + i / QK_K], i % QK_K) *
                       geak_q8_1_value(y.data() + c * yb, i);
            }
            out[size_t(c) * rows + r] = sum;
        }
    }
}

static inline void geak_reference_iq2_xxs_mat(
        const std::vector<block_iq2_xxs> & x,
        const std::vector<block_q8_1> & y,
        std::vector<float> & out,
        int rows,
        int cols,
        int k) {
    const int xb = k / QK_K;
    const int yb = k / QK8_1;
    out.assign(size_t(rows) * cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += geak_iq2_xxs_value(x[r * xb + i / QK_K], i % QK_K) *
                       geak_q8_1_value(y.data() + c * yb, i);
            }
            out[size_t(c) * rows + r] = sum;
        }
    }
}

using geak_q8_0_matmul_launch_fn = void (*)(
        const block_q8_0 *, const block_q8_1 *, float *,
        int, int, int, hipStream_t);

using geak_q2_k_matmul_launch_fn = void (*)(
        const block_q2_K *, const block_q8_1 *, float *,
        int, int, int, hipStream_t);

using geak_iq2_xxs_matmul_launch_fn = void (*)(
        const block_iq2_xxs *, const block_q8_1 *, float *,
        int, int, int, hipStream_t);

struct geak_mat_shape {
    int rows;
    int cols;
    int k;
};

static inline std::vector<geak_mat_shape> geak_quant_shapes(int cols) {
    if (cols == 1) {
        return {
            { 256, 1, 2048 },
            { 512, 1, 4096 },
            { 768, 1, 8192 },
        };
    }
    return {
        { 128, cols, 2048 },
        { 256, cols, 4096 },
        { 384, cols, 8192 },
    };
}

static inline int geak_run_q8_0_matmul(
        const char * name,
        int cols,
        geak_q8_0_matmul_launch_fn launch,
        int argc,
        char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto shapes = geak_quant_shapes(cols);
    const auto selected = geak_selected_indices(int(shapes.size()), opt);
    bool ok_all = true;

    for (const int si : selected) {
        const geak_mat_shape s = shapes[si];
        std::mt19937 rng(1234 + si);
        std::vector<block_q8_0> x(size_t(s.rows) * (s.k / QK8_0));
        std::vector<block_q8_1> y(size_t(s.cols) * (s.k / QK8_1));
        std::vector<float> ref;
        std::vector<float> got(size_t(s.rows) * s.cols, 0.0f);

        geak_fill_q8_0(x, rng);
        geak_fill_q8_1(y, rng);
        geak_reference_q8_0_mat(x, y, ref, s.rows, s.cols, s.k);

        geak_device_buffer<block_q8_0> dx(x.size());
        geak_device_buffer<block_q8_1> dy(y.size());
        geak_device_buffer<float> dout(got.size());
        geak_copy_to_device(dx, x);
        geak_copy_to_device(dy, y);
        GEAK_HIP_CHECK(hipMemset(dout.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            launch(dx.ptr, dy.ptr, dout.ptr, s.rows, s.cols, s.k, nullptr);
        };

        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, dout);
            ok_all = geak_check_close(got, ref, 5.0e-2f, 2.0e-3f, name) && ok_all;
        }

        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("%s shape=%d rows=%d cols=%d k=%d median_ms=%g\n",
                        name, si, s.rows, s.cols, s.k, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}

static inline int geak_run_q2_k_matmul(
        const char * name,
        int cols,
        geak_q2_k_matmul_launch_fn launch,
        int argc,
        char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto shapes = geak_quant_shapes(cols);
    const auto selected = geak_selected_indices(int(shapes.size()), opt);
    bool ok_all = true;

    for (const int si : selected) {
        const geak_mat_shape s = shapes[si];
        std::mt19937 rng(2234 + si);
        std::vector<block_q2_K> x(size_t(s.rows) * (s.k / QK_K));
        std::vector<block_q8_1> y(size_t(s.cols) * (s.k / QK8_1));
        std::vector<float> ref;
        std::vector<float> got(size_t(s.rows) * s.cols, 0.0f);

        geak_fill_q2_k(x, rng);
        geak_fill_q8_1(y, rng);
        geak_reference_q2_k_mat(x, y, ref, s.rows, s.cols, s.k);

        geak_device_buffer<block_q2_K> dx(x.size());
        geak_device_buffer<block_q8_1> dy(y.size());
        geak_device_buffer<float> dout(got.size());
        geak_copy_to_device(dx, x);
        geak_copy_to_device(dy, y);
        GEAK_HIP_CHECK(hipMemset(dout.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            launch(dx.ptr, dy.ptr, dout.ptr, s.rows, s.cols, s.k, nullptr);
        };

        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, dout);
            ok_all = geak_check_close(got, ref, 6.0e-2f, 2.0e-3f, name) && ok_all;
        }

        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("%s shape=%d rows=%d cols=%d k=%d median_ms=%g\n",
                        name, si, s.rows, s.cols, s.k, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}

static inline int geak_run_iq2_xxs_matmul(
        const char * name,
        int cols,
        geak_iq2_xxs_matmul_launch_fn launch,
        int argc,
        char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto shapes = geak_quant_shapes(cols);
    const auto selected = geak_selected_indices(int(shapes.size()), opt);
    bool ok_all = true;

    for (const int si : selected) {
        const geak_mat_shape s = shapes[si];
        std::mt19937 rng(3234 + si);
        std::vector<block_iq2_xxs> x(size_t(s.rows) * (s.k / QK_K));
        std::vector<block_q8_1> y(size_t(s.cols) * (s.k / QK8_1));
        std::vector<float> ref;
        std::vector<float> got(size_t(s.rows) * s.cols, 0.0f);

        geak_fill_iq2_xxs(x, rng);
        geak_fill_q8_1(y, rng);
        geak_reference_iq2_xxs_mat(x, y, ref, s.rows, s.cols, s.k);

        geak_device_buffer<block_iq2_xxs> dx(x.size());
        geak_device_buffer<block_q8_1> dy(y.size());
        geak_device_buffer<float> dout(got.size());
        geak_copy_to_device(dx, x);
        geak_copy_to_device(dy, y);
        GEAK_HIP_CHECK(hipMemset(dout.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            launch(dx.ptr, dy.ptr, dout.ptr, s.rows, s.cols, s.k, nullptr);
        };

        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, dout);
            ok_all = geak_check_close(got, ref, 6.0e-2f, 2.0e-3f, name) && ok_all;
        }

        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("%s shape=%d rows=%d cols=%d k=%d median_ms=%g\n",
                        name, si, s.rows, s.cols, s.k, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}
