#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

__device__ __constant__ float k_rope_freq_32[16] = {
    1.0000000000e+00f, 5.6234132520e-01f, 3.1622776600e-01f, 1.7782794100e-01f,
    1.0000000000e-01f, 5.6234132520e-02f, 3.1622776600e-02f, 1.7782794100e-02f,
    1.0000000000e-02f, 5.6234132520e-03f, 3.1622776600e-03f, 1.7782794100e-03f,
    1.0000000000e-03f, 5.6234132520e-04f, 3.1622776600e-04f, 1.7782794100e-04f,
};

__device__ __constant__ float k_rope_freq_64[32] = {
    1.0000000000e+00f, 7.4989420930e-01f, 5.6234132520e-01f, 4.2169650340e-01f,
    3.1622776600e-01f, 2.3713737060e-01f, 1.7782794100e-01f, 1.3335214320e-01f,
    1.0000000000e-01f, 7.4989420930e-02f, 5.6234132520e-02f, 4.2169650340e-02f,
    3.1622776600e-02f, 2.3713737060e-02f, 1.7782794100e-02f, 1.3335214320e-02f,
    1.0000000000e-02f, 7.4989420930e-03f, 5.6234132520e-03f, 4.2169650340e-03f,
    3.1622776600e-03f, 2.3713737060e-03f, 1.7782794100e-03f, 1.3335214320e-03f,
    1.0000000000e-03f, 7.4989420930e-04f, 5.6234132520e-04f, 4.2169650340e-04f,
    3.1622776600e-04f, 2.3713737060e-04f, 1.7782794100e-04f, 1.3335214320e-04f,
};

static __global__ void geak_dsv4_rope_tail_generic(
        const float * __restrict__ src,
        const int32_t * __restrict__ pos,
        float * __restrict__ dst,
        int rows,
        int ne00,
        int n_dims,
        int mode,
        float theta_scale) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int n_nope = ne00 - n_dims;
    const float p = float(pos[row]);
    const float * src_row = src + row * ne00;
    float * dst_row = dst + row * ne00;

    for (int i = tid; i < n_nope; i += blockDim.x) {
        dst_row[i] = src_row[i];
    }

    if (mode == 1) {
        const int n_half = n_dims / 2;
        for (int ic = tid; ic < n_half; ic += blockDim.x) {
            const int rel = 2 * ic;
            const float theta = p * powf(theta_scale, float(rel) * 0.5f);
            float s;
            float c;
            sincosf(theta, &s, &c);
            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = src_row[j0];
            const float x1 = src_row[j1];
            dst_row[j0] = x0 * c - x1 * s;
            dst_row[j1] = x0 * s + x1 * c;
        }
    } else {
        for (int r = 2 * tid; r < n_dims; r += 2 * blockDim.x) {
            const float theta = p * powf(theta_scale, float(r) * 0.5f);
            float s;
            float c;
            sincosf(theta, &s, &c);
            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = src_row[j0];
            const float x1 = src_row[j1];
            dst_row[j0] = x0 * c - x1 * s;
            dst_row[j1] = x0 * s + x1 * c;
        }
    }
}

static __global__ __launch_bounds__(64) void geak_dsv4_rope_tail_80_32_mode0(
        const float * __restrict__ src,
        const int32_t * __restrict__ pos,
        float * __restrict__ dst) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * src_row = src + row * 80;
    float * dst_row = dst + row * 80;

    if (tid < 12) {
        reinterpret_cast<float4 *>(dst_row)[tid] = reinterpret_cast<const float4 *>(src_row)[tid];
    }

    if (tid < 16) {
        const float theta = float(pos[row]) * k_rope_freq_32[tid];
        float s;
        float c;
        sincosf(theta, &s, &c);
        const float2 x = reinterpret_cast<const float2 *>(src_row + 48)[tid];
        float2 y;
        y.x = x.x * c - x.y * s;
        y.y = x.x * s + x.y * c;
        reinterpret_cast<float2 *>(dst_row + 48)[tid] = y;
    }
}

static __global__ __launch_bounds__(64) void geak_dsv4_rope_tail_128_64_mode1(
        const float * __restrict__ src,
        const int32_t * __restrict__ pos,
        float * __restrict__ dst) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * src_row = src + row * 128;
    float * dst_row = dst + row * 128;

    if (tid < 16) {
        reinterpret_cast<float4 *>(dst_row)[tid] = reinterpret_cast<const float4 *>(src_row)[tid];
    }

    if (tid < 32) {
        const float theta = float(pos[row]) * k_rope_freq_64[tid];
        float s;
        float c;
        sincosf(theta, &s, &c);
        const int j0 = 64 + tid;
        const int j1 = 96 + tid;
        const float x0 = src_row[j0];
        const float x1 = src_row[j1];
        dst_row[j0] = x0 * c - x1 * s;
        dst_row[j1] = x0 * s + x1 * c;
    }
}

static __global__ __launch_bounds__(64) void geak_dsv4_rope_tail_192_64_mode0(
        const float * __restrict__ src,
        const int32_t * __restrict__ pos,
        float * __restrict__ dst) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * src_row = src + row * 192;
    float * dst_row = dst + row * 192;

    if (tid < 32) {
        reinterpret_cast<float4 *>(dst_row)[tid] = reinterpret_cast<const float4 *>(src_row)[tid];
    }

    if (tid < 32) {
        const float theta = float(pos[row]) * k_rope_freq_64[tid];
        float s;
        float c;
        sincosf(theta, &s, &c);
        const float2 x = reinterpret_cast<const float2 *>(src_row + 128)[tid];
        float2 y;
        y.x = x.x * c - x.y * s;
        y.y = x.x * s + x.y * c;
        reinterpret_cast<float2 *>(dst_row + 128)[tid] = y;
    }
}

extern "C" void geak_dsv4_rope_tail_launch(
        const float * src,
        const int32_t * pos,
        float * dst,
        int rows,
        int ne00,
        int n_dims,
        int mode,
        float freq_base,
        hipStream_t stream) {
    if (rows <= 0) {
        return;
    }

    if (freq_base == 10000.0f) {
        if (ne00 == 80 && n_dims == 32 && mode == 0) {
            geak_dsv4_rope_tail_80_32_mode0<<<dim3(rows), dim3(64), 0, stream>>>(src, pos, dst);
            return;
        }
        if (ne00 == 128 && n_dims == 64 && mode == 1) {
            geak_dsv4_rope_tail_128_64_mode1<<<dim3(rows), dim3(64), 0, stream>>>(src, pos, dst);
            return;
        }
        if (ne00 == 192 && n_dims == 64 && mode == 0) {
            geak_dsv4_rope_tail_192_64_mode0<<<dim3(rows), dim3(64), 0, stream>>>(src, pos, dst);
            return;
        }
    }

    const float theta_scale = powf(freq_base, -2.0f / float(n_dims));
    geak_dsv4_rope_tail_generic<<<dim3(rows), dim3(128), 0, stream>>>(
        src, pos, dst, rows, ne00, n_dims, mode, theta_scale);
}
