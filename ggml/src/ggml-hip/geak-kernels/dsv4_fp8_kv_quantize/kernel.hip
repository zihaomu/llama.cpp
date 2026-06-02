#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

static __device__ __forceinline__ uint32_t geak_dsv4_float_as_u32(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    return v.u;
}

static __device__ __forceinline__ float geak_dsv4_u32_as_float(uint32_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = x;
    return v.f;
}

static __device__ __forceinline__ int geak_dsv4_round_even_positive(float x) {
    const float base_f = floorf(x);
    int base = int(base_f);
    const float frac = x - base_f;
    if (frac > 0.5f || (frac == 0.5f && (base & 1) != 0)) {
        ++base;
    }
    return base;
}

static __device__ __forceinline__ float geak_dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    if (ax < 0.033203125f) {
        int q = geak_dsv4_round_even_positive(ax * 512.0f);
        if (q > 16) {
            q = 16;
        }
        return sign * (float(q) * 0.001953125f);
    }

    const uint32_t bits = geak_dsv4_float_as_u32(ax);
    int exp = int((bits >> 23) & 0xff) - 127;
    const uint32_t frac_bits = bits & 0x007fffffu;

    uint32_t mant = frac_bits >> 20;
    const uint32_t rem = frac_bits & 0x000fffffu;
    if (rem > 0x00080000u || (rem == 0x00080000u && (mant & 1u) != 0)) {
        ++mant;
        if (mant == 8) {
            mant = 0;
            ++exp;
        }
    }

    if (exp > 8 || (exp == 8 && mant > 6)) {
        exp = 8;
        mant = 6;
    }

    const uint32_t out_bits = (uint32_t(exp + 127) << 23) | (mant << 20);
    return sign * geak_dsv4_u32_as_float(out_bits);
}

static __device__ __forceinline__ float geak_dsv4_wave64_max(float x) {
#pragma unroll
    for (int delta = 32; delta > 0; delta >>= 1) {
        x = fmaxf(x, __shfl_down(x, delta, 64));
    }
    return __shfl(x, 0, 64);
}

static __device__ __forceinline__ void geak_dsv4_quantize_64(
        const float * __restrict__ src_group,
        float * __restrict__ dst_group,
        int tid) {
    const float v = src_group[tid];
    const float max_abs = geak_dsv4_wave64_max(fabsf(v));

    float scale = 1.0f;
    if (tid == 0) {
        const float amax = fmaxf(max_abs, 1.0e-4f);
        scale = exp2f(ceilf(log2f(amax / 448.0f)));
    }
    scale = __shfl(scale, 0, 64);

    const float clamped = fminf(fmaxf(v / scale, -448.0f), 448.0f);
    dst_group[tid] = geak_dsv4_e4m3fn_dequant(clamped) * scale;
}

template <int NE00, int N_NOPE, int TAIL>
__launch_bounds__(64, 1)
static __global__ void geak_dsv4_fp8_kv_quantize_fixed_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float * src_row = src + row * NE00;
    float * dst_row = dst + row * NE00;

    geak_dsv4_quantize_64(src_row, dst_row, tid);
    if (N_NOPE == 128) {
        geak_dsv4_quantize_64(src_row + 64, dst_row + 64, tid);
    }

    if (TAIL == 64 || tid < TAIL) {
        dst_row[N_NOPE + tid] = src_row[N_NOPE + tid];
    }
}

__launch_bounds__(128, 1)
static __global__ void geak_dsv4_fp8_kv_quantize_shape2_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int lane = tid & 63;
    const int group = tid >> 6;
    const float * src_row = src + row * 192;
    float * dst_row = dst + row * 192;

    if (group == 0) {
        geak_dsv4_quantize_64(src_row, dst_row, lane);
    } else {
        geak_dsv4_quantize_64(src_row + 64, dst_row + 64, lane);
    }

    if (tid < 64) {
        dst_row[128 + tid] = src_row[128 + tid];
    }
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

    for (int off = 0; off < n_nope; off += 64) {
        geak_dsv4_quantize_64(src_row + off, dst_row + off, tid);
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
    if (ne00 == 96 && n_rot == 32) {
        geak_dsv4_fp8_kv_quantize_fixed_kernel<96, 64, 32><<<dim3(rows), dim3(64), 0, stream>>>(src, dst, rows);
    } else if (ne00 == 128 && n_rot == 64) {
        geak_dsv4_fp8_kv_quantize_fixed_kernel<128, 64, 64><<<dim3(rows), dim3(64), 0, stream>>>(src, dst, rows);
    } else if (ne00 == 192 && n_rot == 64) {
        geak_dsv4_fp8_kv_quantize_shape2_kernel<<<dim3(rows), dim3(128), 0, stream>>>(src, dst, rows);
    } else {
        geak_dsv4_fp8_kv_quantize_kernel<<<dim3(rows), dim3(64), 0, stream>>>(
            src, dst, rows, ne00, n_rot);
    }
}
