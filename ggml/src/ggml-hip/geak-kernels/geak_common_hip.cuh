#pragma once

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "../../ggml-common.h"

#include <hip/hip_runtime.h>

#define GEAK_Q8_1_D(b)    ((b).data.d)
#define GEAK_Q8_1_S(b)    ((b).data.s)
#define GEAK_Q2_K_D(b)    ((b).data.d)
#define GEAK_Q2_K_DMIN(b) ((b).data.dmin)

static __device__ __forceinline__ float geak_h2f(ggml_half h) {
    return __half2float(h);
}

static __device__ __forceinline__ float geak_q8_0_value(const block_q8_0 & b, int i) {
    return geak_h2f(b.d) * float(b.qs[i]);
}

static __device__ __forceinline__ float geak_q8_1_value(const block_q8_1 * y, int i) {
    const block_q8_1 & b = y[i / QK8_1];
    return geak_h2f(GEAK_Q8_1_D(b)) * float(b.qs[i % QK8_1]);
}

static __device__ __forceinline__ float geak_q2_k_value(const block_q2_K & b, int i) {
    const int super = i / 128;
    const int within = i - 128 * super;
    const int j = within / 32;
    const int half = (within % 32) / 16;
    const int l = within % 16;
    const int scale_idx = super * 8 + j * 2 + half;
    const int q_idx = super * 32 + half * 16 + l;
    const int shift = 2 * j;

    const uint8_t sc = b.scales[scale_idx];
    const float d = geak_h2f(GEAK_Q2_K_D(b));
    const float dmin = geak_h2f(GEAK_Q2_K_DMIN(b));
    const int q = (b.qs[q_idx] >> shift) & 0x3;
    return d * float(sc & 0x0f) * float(q) - dmin * float(sc >> 4);
}

static __device__ __forceinline__ float geak_iq2_xxs_value(const block_iq2_xxs & b, int i) {
    const int ib32 = i / 32;
    const int group = (i % 32) / 8;
    const int j = i % 8;
    const uint16_t * q = b.qs + 4 * ib32;
    const uint32_t aux0 = uint32_t(q[0]) | (uint32_t(q[1]) << 16);
    const uint32_t aux1 = uint32_t(q[2]) | (uint32_t(q[3]) << 16);
    const uint8_t grid_idx = uint8_t((aux0 >> (8 * group)) & 0xffu);
    const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * group)) & 127u];
    const uint8_t grid = uint8_t((iq2xxs_grid[grid_idx] >> (8 * j)) & 0xffu);
    const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
    const float db = geak_h2f(b.d) * (0.5f + float(aux1 >> 28)) * 0.25f;
    return db * float(grid) * sign;
}
