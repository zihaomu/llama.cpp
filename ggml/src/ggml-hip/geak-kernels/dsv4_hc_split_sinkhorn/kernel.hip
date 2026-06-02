#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

__launch_bounds__(64, 1)
static __global__ void geak_dsv4_hc_split_sinkhorn_kernel(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float * __restrict__ dst,
        int rows,
        int n_hc,
        int sinkhorn_iters,
        float eps) {
    constexpr int HC_MAX = 16;

    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
        const float * mix = mixes + r * ((2 + n_hc) * n_hc);
        float * out = dst + r * ((2 + n_hc) * n_hc);

        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        for (int i = 0; i < n_hc; ++i) {
            const float z = mix[i] * pre_scale + base[i];
            out[i] = 1.0f / (1.0f + expf(-z)) + eps;
        }

        for (int i = 0; i < n_hc; ++i) {
            const int off = n_hc + i;
            const float z = mix[off] * post_scale + base[off];
            out[off] = 2.0f / (1.0f + expf(-z));
        }

        float c[HC_MAX * HC_MAX];
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_max = -INFINITY;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                const int off = 2 * n_hc + idx;
                const float v = mix[off] * comb_scale + base[off];
                c[idx] = v;
                row_max = fmaxf(row_max, v);
            }

            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }

            const float inv_sum = 1.0f / row_sum;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                float sum = 0.0f;
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                const float inv_denom = 1.0f / (sum + eps);
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    c[src_hc + dst_hc * n_hc] *= inv_denom;
                }
            }

            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                float sum = 0.0f;
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                const float inv_denom = 1.0f / (sum + eps);
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    c[src_hc + dst_hc * n_hc] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < n_hc * n_hc; ++i) {
            out[2 * n_hc + i] = c[i];
        }
    }
}

extern "C" void geak_dsv4_hc_split_sinkhorn_launch(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int rows,
        int n_hc,
        int sinkhorn_iters,
        float eps,
        hipStream_t stream) {
    const int blocks = (rows + 63) / 64;
    geak_dsv4_hc_split_sinkhorn_kernel<<<dim3(blocks), dim3(64), 0, stream>>>(
        mixes, scale, base, dst, rows, n_hc, sinkhorn_iters, eps);
}
