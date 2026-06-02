#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

static __device__ __forceinline__ float geak_dsv4_e4m3fn_value(int i) {
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * exp2f(float(exp - 7));
}

static __device__ __forceinline__ float geak_dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = geak_dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * geak_dsv4_e4m3fn_value(best);
}

__launch_bounds__(64, 1)
static __global__ void geak_dsv4_fp8_kv_quantize_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int rows,
        int ne00,
        int n_rot) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int n_nope = ne00 - n_rot;
    const float * src_row = src + row * ne00;
    float * dst_row = dst + row * ne00;

    __shared__ float scratch[64];

    for (int off = 0; off < n_nope; off += 64) {
        const float v = src_row[off + tid];
        scratch[tid] = fabsf(v);
        __syncthreads();

        for (int stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        const float clamped = fminf(fmaxf(v / scale, -448.0f), 448.0f);
        dst_row[off + tid] = geak_dsv4_e4m3fn_dequant(clamped) * scale;
        __syncthreads();
    }

    for (int i = n_nope + tid; i < ne00; i += 64) {
        dst_row[i] = src_row[i];
    }
}

extern "C" void geak_dsv4_fp8_kv_quantize_launch(
        const float * src,
        float * dst,
        int rows,
        int ne00,
        int n_rot,
        hipStream_t stream) {
    geak_dsv4_fp8_kv_quantize_kernel<<<dim3(rows), dim3(64), 0, stream>>>(
        src, dst, rows, ne00, n_rot);
}
