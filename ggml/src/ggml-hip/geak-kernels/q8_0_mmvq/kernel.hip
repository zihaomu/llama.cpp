#include "../geak_common_hip.cuh"

static_assert(QK8_0 == QK8_1, "q8_0 and q8_1 block sizes must match");

static __device__ __forceinline__ int geak_q8_0_q8_1_dot_i32(
        const block_q8_0 & xb,
        const block_q8_1 & yb) {
    int acc = 0;
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        acc += int(xb.qs[j]) * int(yb.qs[j]);
    }
    return acc;
}

static __global__ __launch_bounds__(256) void geak_q8_0_mmvq_kernel(
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

    const int xb = k / QK8_0;
    const int yb = k / QK8_1;
    const block_q8_0 * __restrict__ xrow = x + row * xb;
    const block_q8_1 * __restrict__ ycol = y + col * yb;

    float sum = 0.0f;
    for (int ib = tid; ib < xb; ib += blockDim.x) {
        const block_q8_0 & xbq = xrow[ib];
        const block_q8_1 & ybq = ycol[ib];
        const int dot = geak_q8_0_q8_1_dot_i32(xbq, ybq);
        sum += (geak_h2f(xbq.d) * geak_h2f(GEAK_Q8_1_D(ybq))) * float(dot);
    }

    __shared__ float scratch[256];
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[col * rows + row] = scratch[0];
    }
}

extern "C" void geak_q8_0_mmvq_launch(
        const block_q8_0 * x,
        const block_q8_1 * y,
        float * out,
        int rows,
        int cols,
        int k,
        hipStream_t stream) {
    const dim3 grid(rows, cols, 1);
    const dim3 block(256, 1, 1);
    geak_q8_0_mmvq_kernel<<<grid, block, 0, stream>>>(x, y, out, rows, cols, k);
}
