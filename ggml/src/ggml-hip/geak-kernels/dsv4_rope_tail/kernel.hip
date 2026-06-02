#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

static __global__ void geak_dsv4_rope_tail_kernel(
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
    const float theta_scale = powf(freq_base, -2.0f / float(n_dims));
    geak_dsv4_rope_tail_kernel<<<dim3(rows), dim3(128), 0, stream>>>(
        src, pos, dst, rows, ne00, n_dims, mode, theta_scale);
}
