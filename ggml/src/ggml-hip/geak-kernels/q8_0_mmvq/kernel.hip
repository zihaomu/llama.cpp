#include "../geak_common_hip.cuh"

static __global__ void geak_q8_0_mmvq_kernel(
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
    const block_q8_0 * xrow = x + row * xb;
    const block_q8_1 * ycol = y + col * yb;

    float sum = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) {
        sum += geak_q8_0_value(xrow[i / QK8_0], i % QK8_0) * geak_q8_1_value(ycol, i);
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
