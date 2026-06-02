#include "dsv4-native-mmvq.h"

#if defined(GGML_USE_HIP)

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "../ggml-common.h"

#include <cstdlib>

static __device__ __forceinline__ int ggml_hip_dsv4_dp4a(const int a, const int b, const int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

static __device__ __forceinline__ int ggml_hip_dsv4_get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int ggml_hip_dsv4_get_int_b4(const void * x, const int i32) {
    return ((const int *) x)[i32];
}

static __device__ __forceinline__ float ggml_hip_dsv4_warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, 32);
    }
    return x;
}

static __global__ void ggml_hip_dsv4_mmvq_q8_0_kernel(
        const block_q8_0 * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint32_t nchannels_x,
        const uint32_t nchannels_y, const uint32_t nchannels_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t nsamples_x,
        const uint32_t nsamples_dst, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    const uint32_t row0 = blockIdx.x;
    if (row0 >= nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_ratio = nchannels_dst / nchannels_x;
    const uint32_t sample_ratio  = nsamples_dst  / nsamples_x;

    const uint32_t channel_x = ids ? ids[channel_dst] : channel_dst / channel_ratio;
    const uint32_t channel_y = ids ? channel_dst % nchannels_y : channel_dst;
    const uint32_t sample_x  = sample_dst / sample_ratio;
    const uint32_t sample_y  = sample_dst;

    const block_q8_1 * y = vy + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const uint32_t kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    constexpr int vdr = 2;
    constexpr int qi = QI8_0;
    constexpr int blocks_per_iter = vdr * 32 / qi;

    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (uint32_t kbx = tid / (qi/vdr); kbx < ncols_x / QK8_0; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi/vdr));

        const block_q8_0 * bx = vx + kbx_offset + kbx;
        const block_q8_1 * by = y + kbx;

        int sumi = 0;
#pragma unroll
        for (int i = 0; i < vdr; ++i) {
            const int v = ggml_hip_dsv4_get_int_b2(bx->qs, kqs + i);
            const int u = ggml_hip_dsv4_get_int_b4(by->qs, kqs + i);
            sumi = ggml_hip_dsv4_dp4a(v, u, sumi);
        }

        sum += __half2float(bx->d) * __half2float(__low2half(by->ds)) * (float) sumi;
    }

    sum = ggml_hip_dsv4_warp_reduce_sum(sum);

    if (threadIdx.x == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0] = sum;
    }

    (void) stride_col_y;
    (void) stride_col_dst;
}

bool ggml_hip_dsv4_native_mmvq_q8_0_applicable(
        bool has_fusion, int ncols_x, int ncols_dst,
        int nsamples_x, int nsamples_dst, int warp_size) {
    const char * enabled = std::getenv("LLAMA_HIP_NATIVE_DSV4_Q8_MMVQ");
    if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0') {
        return false;
    }

    return !has_fusion &&
        ncols_dst == 1 &&
        ncols_x % QK8_0 == 0 &&
        nsamples_x > 0 &&
        nsamples_dst % nsamples_x == 0 &&
        warp_size == 32;
}

void ggml_hip_dsv4_native_mmvq_q8_0_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        int ncols_x, int nrows_x,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        int nsamples_x, int nsamples_dst,
        int stride_sample_x, int stride_sample_y, int stride_sample_dst,
        hipStream_t stream) {
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    const dim3 block(32, 1, 1);

    ggml_hip_dsv4_mmvq_q8_0_kernel<<<grid, block, 0, stream>>>(
        (const block_q8_0 *) vx, (const block_q8_1 *) vy, ids, dst,
        ncols_x, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        nsamples_x, nsamples_dst,
        stride_sample_x, stride_sample_y, stride_sample_dst);
}

#endif
