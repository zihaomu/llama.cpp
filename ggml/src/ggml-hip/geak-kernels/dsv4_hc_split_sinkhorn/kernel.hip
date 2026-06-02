#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

static __device__ __forceinline__ float geak_hc_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

template <int N, int ITERS>
__launch_bounds__(256, 1)
static __global__ void geak_dsv4_hc_split_sinkhorn_block_kernel(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float * __restrict__ dst,
        int rows,
        float eps) {
    constexpr int NN = N * N;
    constexpr int WIDTH = (2 + N) * N;

    const int r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const float * mix = mixes + r * WIDTH;
    float * out = dst + r * WIDTH;

    __shared__ float c[NN];
    __shared__ float row_tmp[N];
    __shared__ float col_tmp[N];

    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (tid < N) {
        const float pre_z = mix[tid] * pre_scale + base[tid];
        out[tid] = geak_hc_sigmoid(pre_z) + eps;

        const int off = N + tid;
        const float post_z = mix[off] * post_scale + base[off];
        out[off] = 2.0f * geak_hc_sigmoid(post_z);
    }

    if (tid < NN) {
        const int off = 2 * N + tid;
        c[tid] = mix[off] * comb_scale + base[off];
    }
    __syncthreads();

    if (tid < N) {
        const int row = tid;
        const int row_base = row * N;
        float row_max = c[row_base];
#pragma unroll
        for (int src = 1; src < N; ++src) {
            row_max = fmaxf(row_max, c[row_base + src]);
        }
        row_tmp[row] = row_max;
    }
    __syncthreads();

    if (tid < NN) {
        const int row = tid / N;
        c[tid] = expf(c[tid] - row_tmp[row]);
    }
    __syncthreads();

    if (tid < N) {
        const int row = tid;
        const int row_base = row * N;
        float row_sum = 0.0f;
#pragma unroll
        for (int src = 0; src < N; ++src) {
            row_sum += c[row_base + src];
        }
        row_tmp[row] = row_sum;
    }
    __syncthreads();

    if (tid < NN) {
        const int row = tid / N;
        c[tid] = c[tid] * (1.0f / row_tmp[row]) + eps;
    }
    __syncthreads();

    if (tid < N) {
        const int src = tid;
        float col_sum = 0.0f;
#pragma unroll
        for (int row = 0; row < N; ++row) {
            col_sum += c[src + row * N];
        }
        col_tmp[src] = col_sum;
    }
    __syncthreads();

    if (tid < NN) {
        const int src = tid - (tid / N) * N;
        c[tid] *= 1.0f / (col_tmp[src] + eps);
    }
    __syncthreads();

#pragma unroll
    for (int iter = 1; iter < ITERS; ++iter) {
        if (tid < N) {
            const int row = tid;
            const int row_base = row * N;
            float row_sum = 0.0f;
#pragma unroll
            for (int src = 0; src < N; ++src) {
                row_sum += c[row_base + src];
            }
            row_tmp[row] = row_sum;
        }
        __syncthreads();

        if (tid < NN) {
            const int row = tid / N;
            c[tid] *= 1.0f / (row_tmp[row] + eps);
        }
        __syncthreads();

        if (tid < N) {
            const int src = tid;
            float col_sum = 0.0f;
#pragma unroll
            for (int row = 0; row < N; ++row) {
                col_sum += c[src + row * N];
            }
            col_tmp[src] = col_sum;
        }
        __syncthreads();

        if (tid < NN) {
            const int src = tid - (tid / N) * N;
            c[tid] *= 1.0f / (col_tmp[src] + eps);
        }
        __syncthreads();
    }

    if (tid < NN) {
        out[2 * N + tid] = c[tid];
    }
}

__launch_bounds__(64, 1)
static __global__ void geak_dsv4_hc_split_sinkhorn_scalar_kernel(
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
            out[i] = geak_hc_sigmoid(z) + eps;
        }

        for (int i = 0; i < n_hc; ++i) {
            const int off = n_hc + i;
            const float z = mix[off] * post_scale + base[off];
            out[off] = 2.0f * geak_hc_sigmoid(z);
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
    if (n_hc == 4 && sinkhorn_iters == 3) {
        geak_dsv4_hc_split_sinkhorn_block_kernel<4, 3><<<dim3(rows), dim3(64), 0, stream>>>(
            mixes, scale, base, dst, rows, eps);
    } else if (n_hc == 8 && sinkhorn_iters == 2) {
        geak_dsv4_hc_split_sinkhorn_block_kernel<8, 2><<<dim3(rows), dim3(64), 0, stream>>>(
            mixes, scale, base, dst, rows, eps);
    } else if (n_hc == 16 && sinkhorn_iters == 2) {
        geak_dsv4_hc_split_sinkhorn_block_kernel<16, 2><<<dim3(rows), dim3(256), 0, stream>>>(
            mixes, scale, base, dst, rows, eps);
    } else {
        const int blocks = (rows + 63) / 64;
        geak_dsv4_hc_split_sinkhorn_scalar_kernel<<<dim3(blocks), dim3(64), 0, stream>>>(
            mixes, scale, base, dst, rows, n_hc, sinkhorn_iters, eps);
    }
}
