#include "../geak_common_hip.cuh"

static __global__ void geak_f16_matvec_kernel(
        const ggml_half * __restrict__ x,
        const ggml_half * __restrict__ y,
        float * __restrict__ out,
        int rows,
        int k) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const ggml_half * xrow = x + row * k;
    float sum = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) {
        sum += geak_h2f(xrow[i]) * geak_h2f(y[i]);
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
        out[row] = scratch[0];
    }
}

extern "C" void geak_f16_matvec_launch(
        const ggml_half * x,
        const ggml_half * y,
        float * out,
        int rows,
        int k,
        hipStream_t stream) {
    geak_f16_matvec_kernel<<<dim3(rows), dim3(256), 0, stream>>>(x, y, out, rows, k);
}
