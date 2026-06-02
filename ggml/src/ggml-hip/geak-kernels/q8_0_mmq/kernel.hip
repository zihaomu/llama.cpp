#include "../geak_common_hip.cuh"

static_assert(QK8_0 == QK8_1, "q8 block sizes must match");

static __device__ __forceinline__ int geak_load_i8x4(const int8_t * p) {
    int v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ int geak_q8_0_q8_1_dot_block(
        const block_q8_0 & bx,
        const block_q8_1 & by) {
    int acc = 0;
#pragma unroll
    for (int j = 0; j < QK8_0; j += 4) {
        const int x4 = geak_load_i8x4(bx.qs + j);
        const int y4 = geak_load_i8x4(by.qs + j);
        acc = __builtin_amdgcn_sdot4(x4, y4, acc, false);
    }
    return acc;
}

static __device__ __forceinline__ float geak_wave_sum(float v) {
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 64);
    }
    return v;
}

static __global__ __launch_bounds__(64, 8) void geak_q8_0_mmq_kernel(
        const block_q8_0 * __restrict__ x,
        const block_q8_1 * __restrict__ y,
        float * __restrict__ out,
        int rows,
        int cols,
        int k) {
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }

    const int nb = k / QK8_0;
    const block_q8_0 * xrow = x + row * nb;
    const block_q8_1 * ycol = y + col * nb;

    float sum = 0.0f;
    for (int ib = tid; ib < nb; ib += blockDim.x) {
        const block_q8_0 & bx = xrow[ib];
        const block_q8_1 & by = ycol[ib];
        const int isum = geak_q8_0_q8_1_dot_block(bx, by);
        const float d = geak_h2f(bx.d) * geak_h2f(GEAK_Q8_1_D(by));
        sum = __fmaf_rn(d, float(isum), sum);
    }

    sum = geak_wave_sum(sum);
    if (tid == 0) {
        out[col * rows + row] = sum;
    }
}

extern "C" void geak_q8_0_mmq_launch(
        const block_q8_0 * x,
        const block_q8_1 * y,
        float * out,
        int rows,
        int cols,
        int k,
        hipStream_t stream) {
    const dim3 grid(rows, cols, 1);
    const dim3 block(64, 1, 1);
    geak_q8_0_mmq_kernel<<<grid, block, 0, stream>>>(x, y, out, rows, cols, k);
}
