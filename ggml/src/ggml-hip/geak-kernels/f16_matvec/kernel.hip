#include "../geak_common_hip.cuh"

#define GEAK_THREADS_PER_ROW 256
#define GEAK_ROWS_PER_BLOCK 4
#define GEAK_THREADS_PER_BLOCK (GEAK_THREADS_PER_ROW * GEAK_ROWS_PER_BLOCK)

static __global__ __launch_bounds__(GEAK_THREADS_PER_BLOCK, 1) void geak_f16_matvec_kernel(
        const ggml_half * __restrict__ x,
        const ggml_half * __restrict__ y,
        float * __restrict__ out,
        int rows,
        int k) {
    extern __shared__ float y_cache[];
    __shared__ float scratch[GEAK_ROWS_PER_BLOCK][GEAK_THREADS_PER_ROW];

    const int tid = threadIdx.x;
    const int row_in_block = tid / GEAK_THREADS_PER_ROW;
    const int col = tid - row_in_block * GEAK_THREADS_PER_ROW;
    const int row = blockIdx.x * GEAK_ROWS_PER_BLOCK + row_in_block;
    const bool valid_row = row < rows;

    for (int i = tid; i < k; i += GEAK_THREADS_PER_BLOCK) {
        y_cache[i] = geak_h2f(y[i]);
    }
    __syncthreads();

    float sum = 0.0f;
    if (valid_row) {
        const ggml_half * __restrict__ xrow = x + size_t(row) * k;
        for (int i = col; i < k; i += GEAK_THREADS_PER_ROW) {
            sum += geak_h2f(xrow[i]) * y_cache[i];
        }
    }

    scratch[row_in_block][col] = sum;
    __syncthreads();
    for (int stride = GEAK_THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            scratch[row_in_block][col] += scratch[row_in_block][col + stride];
        }
        __syncthreads();
    }
    if (col == 0 && valid_row) {
        out[row] = scratch[row_in_block][0];
    }
}

extern "C" void geak_f16_matvec_launch(
        const ggml_half * x,
        const ggml_half * y,
        float * out,
        int rows,
        int k,
        hipStream_t stream) {
    const int blocks = (rows + GEAK_ROWS_PER_BLOCK - 1) / GEAK_ROWS_PER_BLOCK;
    geak_f16_matvec_kernel<<<dim3(blocks), dim3(GEAK_THREADS_PER_BLOCK), sizeof(float) * size_t(k), stream>>>(x, y, out, rows, k);
}
