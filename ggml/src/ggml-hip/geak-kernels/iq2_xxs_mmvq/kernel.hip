#include "../geak_common_hip.cuh"

static __device__ __forceinline__ float geak_iq2_xxs_group_dot(
        const block_iq2_xxs & xb,
        const block_q8_1 & yb,
        const int group_id) {
    const int ib32 = group_id >> 2;
    const int group = group_id & 3;
    const uint16_t * q = xb.qs + 4 * ib32;
    const uint32_t aux0 = uint32_t(q[0]) | (uint32_t(q[1]) << 16);
    const uint32_t aux1 = uint32_t(q[2]) | (uint32_t(q[3]) << 16);
    const uint8_t grid_idx = uint8_t((aux0 >> (8 * group)) & 0xffu);
    const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * group)) & 127u];
    const auto grid = iq2xxs_grid[grid_idx];
    const int yoff = group << 3;

    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float grid_v = float((grid >> (8 * j)) & 0xffu);
        const float sign_v = (signs & (1u << j)) ? -grid_v : grid_v;
        dot += sign_v * float(yb.qs[yoff + j]);
    }

    const float xd = geak_h2f(xb.d) * (0.5f + float(aux1 >> 28)) * 0.25f;
    const float yd = geak_h2f(GEAK_Q8_1_D(yb));
    return (xd * yd) * dot;
}

static __global__ __launch_bounds__(256, 2) void geak_iq2_xxs_mmvq_kernel(
        const block_iq2_xxs * __restrict__ x,
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
    const block_iq2_xxs * __restrict__ xrow = x + row * xb;
    const block_q8_1 * __restrict__ ycol = y + col * yb;

    constexpr int groups_per_xblock = QK_K / 8;
    constexpr int q8_per_xblock = QK_K / QK8_1;
    const int total_groups = xb * groups_per_xblock;

    float sum = 0.0f;
    for (int task = tid; task < total_groups; task += blockDim.x) {
        const int xb_idx = task / groups_per_xblock;
        const int group_id = task - xb_idx * groups_per_xblock;
        const int y_idx = xb_idx * q8_per_xblock + (group_id >> 2);
        sum += geak_iq2_xxs_group_dot(xrow[xb_idx], ycol[y_idx], group_id);
    }

#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 64);
    }

    __shared__ float wave_sums[4];
    const int lane = tid & 63;
    const int wave = tid >> 6;
    if (lane == 0) {
        wave_sums[wave] = sum;
    }
    __syncthreads();

    if (tid < 64) {
        sum = (tid < 4) ? wave_sums[tid] : 0.0f;
#pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset, 64);
        }
        if (tid == 0) {
            out[col * rows + row] = sum;
        }
    }
}

extern "C" void geak_iq2_xxs_mmvq_launch(
        const block_iq2_xxs * x,
        const block_q8_1 * y,
        float * out,
        int rows,
        int cols,
        int k,
        hipStream_t stream) {
    const dim3 grid(rows, cols, 1);
    const dim3 block(256, 1, 1);
    geak_iq2_xxs_mmvq_kernel<<<grid, block, 0, stream>>>(x, y, out, rows, cols, k);
}
