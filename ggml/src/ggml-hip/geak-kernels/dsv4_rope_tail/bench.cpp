#include "../geak_common.hpp"

extern "C" void geak_dsv4_rope_tail_launch(
        const float *, const int32_t *, float *,
        int, int, int, int, float, hipStream_t);

struct rope_shape {
    int rows;
    int ne00;
    int n_dims;
    int mode;
};

static std::vector<rope_shape> shapes() {
    return {
        { 512, 80, 32, 0 },
        { 1024, 128, 64, 1 },
        { 1536, 192, 64, 0 },
    };
}

static void reference(
        const std::vector<float> & src,
        const std::vector<int32_t> & pos,
        std::vector<float> & dst,
        int rows,
        int ne00,
        int n_dims,
        int mode,
        float freq_base) {
    const int n_nope = ne00 - n_dims;
    const float theta_scale = std::pow(freq_base, -2.0f / float(n_dims));
    dst.assign(src.size(), 0.0f);

    for (int row = 0; row < rows; ++row) {
        const float * src_row = src.data() + size_t(row) * ne00;
        float * dst_row = dst.data() + size_t(row) * ne00;
        for (int i = 0; i < n_nope; ++i) {
            dst_row[i] = src_row[i];
        }
        const float p = float(pos[row]);
        if (mode == 1) {
            const int n_half = n_dims / 2;
            for (int ic = 0; ic < n_half; ++ic) {
                const int rel = 2 * ic;
                const float theta = p * std::pow(theta_scale, float(rel) * 0.5f);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const int j0 = n_nope + ic;
                const int j1 = n_nope + ic + n_half;
                const float x0 = src_row[j0];
                const float x1 = src_row[j1];
                dst_row[j0] = x0 * c - x1 * s;
                dst_row[j1] = x0 * s + x1 * c;
            }
        } else {
            for (int r = 0; r < n_dims; r += 2) {
                const float theta = p * std::pow(theta_scale, float(r) * 0.5f);
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const int j0 = n_nope + r;
                const int j1 = j0 + 1;
                const float x0 = src_row[j0];
                const float x1 = src_row[j1];
                dst_row[j0] = x0 * c - x1 * s;
                dst_row[j1] = x0 * s + x1 * c;
            }
        }
    }
}

int main(int argc, char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto all_shapes = shapes();
    const auto selected = geak_selected_indices(int(all_shapes.size()), opt);
    constexpr float freq_base = 10000.0f;
    bool ok_all = true;

    for (int si : selected) {
        const rope_shape s = all_shapes[si];
        std::mt19937 rng(7234 + si);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> src(size_t(s.rows) * s.ne00);
        std::vector<int32_t> pos(s.rows);
        for (float & v : src) { v = dist(rng); }
        for (int i = 0; i < s.rows; ++i) { pos[i] = i % 2048; }

        std::vector<float> ref;
        std::vector<float> got(src.size(), 0.0f);
        reference(src, pos, ref, s.rows, s.ne00, s.n_dims, s.mode, freq_base);

        geak_device_buffer<float> dsrc(src.size());
        geak_device_buffer<int32_t> dpos(pos.size());
        geak_device_buffer<float> ddst(got.size());
        geak_copy_to_device(dsrc, src);
        geak_copy_to_device(dpos, pos);
        GEAK_HIP_CHECK(hipMemset(ddst.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            geak_dsv4_rope_tail_launch(
                dsrc.ptr, dpos.ptr, ddst.ptr,
                s.rows, s.ne00, s.n_dims, s.mode, freq_base, nullptr);
        };
        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, ddst);
            ok_all = geak_check_close(got, ref, 8.0e-5f, 2.0e-4f, "dsv4_rope_tail") && ok_all;
        }
        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("dsv4_rope_tail shape=%d rows=%d ne00=%d n_dims=%d mode=%d median_ms=%g\n",
                        si, s.rows, s.ne00, s.n_dims, s.mode, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}
