#include "../geak_common.hpp"

extern "C" void geak_dsv4_fp8_kv_quantize_launch(
        const float *, float *, int, int, int, hipStream_t);

struct kv_shape {
    int rows;
    int ne00;
    int n_rot;
};

static std::vector<kv_shape> shapes() {
    return {
        { 512, 96, 32 },
        { 1024, 128, 64 },
        { 1536, 192, 64 },
    };
}

static float e4m3fn_value(int i) {
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * std::exp2(float(exp - 7));
}

static float e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = std::min(std::fabs(x), 448.0f);
    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = e4m3fn_value(i);
        const float diff = std::fabs(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * e4m3fn_value(best);
}

static void reference(
        const std::vector<float> & src,
        std::vector<float> & dst,
        int rows,
        int ne00,
        int n_rot) {
    const int n_nope = ne00 - n_rot;
    dst.assign(src.size(), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float * src_row = src.data() + size_t(r) * ne00;
        float * dst_row = dst.data() + size_t(r) * ne00;
        for (int off = 0; off < n_nope; off += 64) {
            float amax = 1.0e-4f;
            for (int i = 0; i < 64; ++i) {
                amax = std::max(amax, std::fabs(src_row[off + i]));
            }
            const float scale = std::exp2(std::ceil(std::log2(amax / 448.0f)));
            for (int i = 0; i < 64; ++i) {
                const float v = std::min(std::max(src_row[off + i] / scale, -448.0f), 448.0f);
                dst_row[off + i] = e4m3fn_dequant(v) * scale;
            }
        }
        for (int i = n_nope; i < ne00; ++i) {
            dst_row[i] = src_row[i];
        }
    }
}

int main(int argc, char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto all_shapes = shapes();
    const auto selected = geak_selected_indices(int(all_shapes.size()), opt);
    bool ok_all = true;

    for (int si : selected) {
        const kv_shape s = all_shapes[si];
        std::mt19937 rng(6234 + si);
        std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
        std::vector<float> src(size_t(s.rows) * s.ne00);
        for (float & v : src) {
            v = dist(rng);
        }
        std::vector<float> ref;
        std::vector<float> got(src.size(), 0.0f);
        reference(src, ref, s.rows, s.ne00, s.n_rot);

        geak_device_buffer<float> dsrc(src.size());
        geak_device_buffer<float> ddst(got.size());
        geak_copy_to_device(dsrc, src);
        GEAK_HIP_CHECK(hipMemset(ddst.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            geak_dsv4_fp8_kv_quantize_launch(dsrc.ptr, ddst.ptr, s.rows, s.ne00, s.n_rot, nullptr);
        };
        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, ddst);
            ok_all = geak_check_close(got, ref, 1.0e-6f, 1.0e-6f, "dsv4_fp8_kv_quantize") && ok_all;
        }
        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("dsv4_fp8_kv_quantize shape=%d rows=%d ne00=%d n_rot=%d median_ms=%g\n",
                        si, s.rows, s.ne00, s.n_rot, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}
