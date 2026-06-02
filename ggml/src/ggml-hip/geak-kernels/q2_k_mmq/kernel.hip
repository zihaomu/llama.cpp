#include "../geak_common_hip.cuh"

static __global__ void geak_q2_k_mmq_kernel(
        const block_q2_K * __restrict__ x,
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

    const int xb = k / QK_K;
    const int yb = k / QK8_1;
    const block_q2_K * xrow = x + row * xb;
    const block_q8_1 * ycol = y + col * yb;

    const int super = tid >> 7;
    const int within = tid & 127;
    const int j = within >> 5;
    const int half = (within >> 4) & 1;
    const int l = within & 15;
    const int scale_idx = (super << 3) + (j << 1) + half;
    const int q_idx = (super << 5) + (half << 4) + l;
    const int shift = j << 1;
    const int y_block_local = tid >> 5;
    const int y_q_local = tid & 31;

    float sum = 0.0f;
    for (int ib = 0; ib < xb; ++ib) {
        const block_q2_K & bx = xrow[ib];
        const uint8_t sc = bx.scales[scale_idx];
        const int q = (bx.qs[q_idx] >> shift) & 0x3;
        const float xv = geak_h2f(GEAK_Q2_K_D(bx)) * float(sc & 0x0f) * float(q) -
                         geak_h2f(GEAK_Q2_K_DMIN(bx)) * float(sc >> 4);

        const block_q8_1 & by = ycol[(ib << 3) + y_block_local];
        const float yv = geak_h2f(GEAK_Q8_1_D(by)) * float(by.qs[y_q_local]);
        sum += xv * yv;
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

extern "C" void geak_q2_k_mmq_launch(
        const block_q2_K * x,
        const block_q8_1 * y,
        float * out,
        int rows,
        int cols,
        int k,
        hipStream_t stream) {
    const dim3 grid(rows, cols, 1);
    const dim3 block(256, 1, 1);
    geak_q2_k_mmq_kernel<<<grid, block, 0, stream>>>(x, y, out, rows, cols, k);
}
