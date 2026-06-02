#include "../geak_common.hpp"

extern "C" void geak_dsv4_hc_split_sinkhorn_launch(
        const float *, const float *, const float *, float *,
        int, int, int, float, hipStream_t);

struct hc_shape {
    int rows;
    int n_hc;
    int iters;
};

static std::vector<hc_shape> shapes() {
    return {
        { 512, 4, 3 },
        { 1024, 8, 2 },
        { 1536, 16, 2 },
    };
}

static void reference(
        const std::vector<float> & mixes,
        const std::vector<float> & scale,
        const std::vector<float> & base,
        std::vector<float> & dst,
        int rows,
        int n_hc,
        int sinkhorn_iters,
        float eps) {
    const int width = (2 + n_hc) * n_hc;
    dst.assign(size_t(rows) * width, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float * mix = mixes.data() + size_t(r) * width;
        float * out = dst.data() + size_t(r) * width;
        for (int i = 0; i < n_hc; ++i) {
            out[i] = 1.0f / (1.0f + std::exp(-(mix[i] * scale[0] + base[i]))) + eps;
        }
        for (int i = 0; i < n_hc; ++i) {
            const int off = n_hc + i;
            out[off] = 2.0f / (1.0f + std::exp(-(mix[off] * scale[1] + base[off])));
        }

        float c[16 * 16];
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_max = -INFINITY;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                const int off = 2 * n_hc + idx;
                c[idx] = mix[off] * scale[2] + base[off];
                row_max = std::max(row_max, c[idx]);
            }
            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                c[idx] = std::exp(c[idx] - row_max);
                row_sum += c[idx];
            }
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                c[idx] = c[idx] / row_sum + eps;
            }
        }
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] /= (sum + eps);
            }
        }
        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                float sum = 0.0f;
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    c[src_hc + dst_hc * n_hc] /= (sum + eps);
                }
            }
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                float sum = 0.0f;
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    c[src_hc + dst_hc * n_hc] /= (sum + eps);
                }
            }
        }
        for (int i = 0; i < n_hc * n_hc; ++i) {
            out[2 * n_hc + i] = c[i];
        }
    }
}

int main(int argc, char ** argv) {
    const geak_options opt = geak_parse_options(argc, argv);
    const auto all_shapes = shapes();
    const auto selected = geak_selected_indices(int(all_shapes.size()), opt);
    bool ok_all = true;

    for (int si : selected) {
        const hc_shape s = all_shapes[si];
        const int width = (2 + s.n_hc) * s.n_hc;
        const float eps = 1.0e-6f;
        std::mt19937 rng(5234 + si);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> mixes(size_t(s.rows) * width);
        std::vector<float> scale = { 0.75f, 1.25f, 0.5f };
        std::vector<float> base(width);
        for (float & v : mixes) { v = dist(rng); }
        for (float & v : base) { v = dist(rng); }
        std::vector<float> ref;
        std::vector<float> got(size_t(s.rows) * width, 0.0f);
        reference(mixes, scale, base, ref, s.rows, s.n_hc, s.iters, eps);

        geak_device_buffer<float> dmixes(mixes.size());
        geak_device_buffer<float> dscale(scale.size());
        geak_device_buffer<float> dbase(base.size());
        geak_device_buffer<float> dout(got.size());
        geak_copy_to_device(dmixes, mixes);
        geak_copy_to_device(dscale, scale);
        geak_copy_to_device(dbase, base);
        GEAK_HIP_CHECK(hipMemset(dout.ptr, 0, sizeof(float) * got.size()));

        auto run_once = [&]() {
            geak_dsv4_hc_split_sinkhorn_launch(
                dmixes.ptr, dscale.ptr, dbase.ptr, dout.ptr,
                s.rows, s.n_hc, s.iters, eps, nullptr);
        };
        if (opt.correctness || opt.full_benchmark) {
            run_once();
            GEAK_HIP_CHECK(hipDeviceSynchronize());
            geak_copy_to_host(got, dout);
            ok_all = geak_check_close(got, ref, 2.0e-5f, 2.0e-4f, "dsv4_hc_split_sinkhorn") && ok_all;
        }
        if (opt.benchmark || opt.profile || opt.full_benchmark) {
            const float ms = geak_measure_ms(run_once, opt.iterations);
            std::printf("dsv4_hc_split_sinkhorn shape=%d rows=%d n_hc=%d iters=%d median_ms=%g\n",
                        si, s.rows, s.n_hc, s.iters, ms);
        }
    }

    geak_print_indices(selected);
    std::printf("GEAK_RESULT_GEOMEAN_SPEEDUP=1.0\n");
    return ok_all ? 0 : 1;
}
