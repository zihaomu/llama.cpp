#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#if defined(GGML_USE_HIP)

bool ggml_hip_dsv4_native_mmvq_q8_0_applicable(
        bool has_fusion, int ncols_x, int ncols_dst,
        int nsamples_x, int nsamples_dst, int warp_size);

void ggml_hip_dsv4_native_mmvq_q8_0_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        int ncols_x, int nrows_x,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        int nsamples_x, int nsamples_dst,
        int stride_sample_x, int stride_sample_y, int stride_sample_dst,
        hipStream_t stream);

#endif
