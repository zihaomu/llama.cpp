#include "../geak_common.hpp"

extern "C" void geak_f16_matvec_launch(
        const ggml_half *, const ggml_half *, float *, int, int, hipStream_t);

struct f16_shape {
    int rows;
    int k;
};

static std::vector<f16_shape> shapes() {
    return {
        { 512, 2048 },
        { 1024, 4096 },
        { 1536, 8192 },
    };
}

static void fill_half(std::vector<ggml_half> & v, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (ggml_half & h : v) {
        h = geak_float_to_half(dist(rng));
    }
}

static void reference(
        const std::vector<ggml_half> & x,
        const std::vector<ggml_half> & y,
        std::vector<float> & out,
        int rows,
        int k) {
    out.assign(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += geak_half_to_float(x[size_t(r) * k + i]) * geak_half_to_float(y[i]);
        }
        out[r] = sum;
    }
}

int main(int argc, char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto all_shapes = shapes();
    const auto selected = geak_selected_indices(int(all_shapes.size()), opt);
    bool ok_all = true;

    for (int si : selected) {
        const f16_shape s = all_shapes[si];
        std::mt19937 rng(4234 + si);
        std::vector<ggml_half> x(size_t(s.rows) * s.k);
        std::vector<ggml_half> y(s.k);
        std::vector<float> ref;
        std::vector<float> got(s.rows, 0.0f);

        fill_half(x, rng);
        fill_half(y, rng);
        reference(x, y, ref, s.rows, s.k);

        geak_device_buffer<ggml_half> dx(x.size());
        geak_device_buffer<ggml_half> dy(y.size());
        geak_device_buffer<float> dout(got.size());
        geak_copy_to_device(dx, x);
        geak_copy_to_device(dy, y);
        GEAK_HIP_CHECK(hipMemset(dout.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            geak_f16_matvec_launch(dx.ptr, dy.ptr, dout.ptr, s.rows, s.k, nullptr);
        };

        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, dout);
            ok_all = geak_check_close(got, ref, 5.0e-2f, 2.0e-3f, "f16_matvec") && ok_all;
        }

        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("f16_matvec shape=%d rows=%d k=%d median_ms=%g\n", si, s.rows, s.k, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}
