#include "dsv4.cuh"

#include <cstdint>

__launch_bounds__(CUDA_DSV4_HC_SPLIT_SINKHORN_BLOCK_SIZE, 1)
static __global__ void dsv4_hc_split_sinkhorn_f32(
        const char * mixes,
        const char * scale,
        const char * base,
        char *       dst,
        const int64_t n_rows,
        const uint64_t nb_mix1,
        const uint64_t nb_dst1,
        const int n_hc,
        const int sinkhorn_iters,
        const float eps) {
    constexpr int HC_MAX = 16;

    const float * scale_data = reinterpret_cast<const float *>(scale);
    const float * base_data  = reinterpret_cast<const float *>(base);

    for (int64_t r = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; r < n_rows; r += (int64_t) blockDim.x*gridDim.x) {
        const float * mix = reinterpret_cast<const float *>(mixes + r*nb_mix1);
        float *       out = reinterpret_cast<float *>(dst + r*nb_dst1);

        const float pre_scale  = scale_data[0];
        const float post_scale = scale_data[1];
        const float comb_scale = scale_data[2];

        for (int i = 0; i < n_hc; ++i) {
            const float z = mix[i]*pre_scale + base_data[i];
            out[i] = 1.0f/(1.0f + expf(-z)) + eps;
        }

        for (int i = 0; i < n_hc; ++i) {
            const int off = n_hc + i;
            const float z = mix[off]*post_scale + base_data[off];
            out[off] = 2.0f/(1.0f + expf(-z));
        }

        float c[HC_MAX*HC_MAX];

        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_max = -INFINITY;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                const int off = 2*n_hc + idx;
                const float v = mix[off]*comb_scale + base_data[off];
                c[idx] = v;
                row_max = fmaxf(row_max, v);
            }

            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }

            const float inv_sum = 1.0f/row_sum;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                c[idx] = c[idx]*inv_sum + eps;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc*n_hc];
            }

            const float inv_denom = 1.0f/(sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc*n_hc] *= inv_denom;
            }
        }

        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                float sum = 0.0f;
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    sum += c[src_hc + dst_hc*n_hc];
                }

                const float inv_denom = 1.0f/(sum + eps);
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    c[src_hc + dst_hc*n_hc] *= inv_denom;
                }
            }

            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                float sum = 0.0f;
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    sum += c[src_hc + dst_hc*n_hc];
                }

                const float inv_denom = 1.0f/(sum + eps);
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    c[src_hc + dst_hc*n_hc] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < n_hc*n_hc; ++i) {
            out[2*n_hc + i] = c[i];
        }
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(mixes->nb[0] == sizeof(float));
    GGML_ASSERT(scale->nb[0] == sizeof(float));
    GGML_ASSERT(base->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[0]   == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous_rows(mixes));
    GGML_ASSERT(ggml_is_contiguous(scale));
    GGML_ASSERT(ggml_is_contiguous(base));
    GGML_ASSERT(ggml_are_same_shape(mixes, dst));

    const int n_hc = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps = ggml_get_op_params_f32(dst, 2);
    const int64_t mix_hc = mixes->ne[0];
    const int64_t n_rows = ggml_nrows(mixes);

    GGML_ASSERT(n_hc > 0 && n_hc <= 16);
    GGML_ASSERT(sinkhorn_iters > 0);
    GGML_ASSERT(mix_hc == (2 + n_hc)*n_hc);
    GGML_ASSERT(mixes->ne[2] == 1);
    GGML_ASSERT(mixes->ne[3] == 1);
    GGML_ASSERT(ggml_nelements(scale) >= 3);
    GGML_ASSERT(ggml_nelements(base) >= mix_hc);

    const int64_t n_blocks = (n_rows + CUDA_DSV4_HC_SPLIT_SINKHORN_BLOCK_SIZE - 1) / CUDA_DSV4_HC_SPLIT_SINKHORN_BLOCK_SIZE;
    const dim3 block_nums(n_blocks, 1, 1);
    const dim3 block_dims(CUDA_DSV4_HC_SPLIT_SINKHORN_BLOCK_SIZE, 1, 1);

    dsv4_hc_split_sinkhorn_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const char *) mixes->data,
        (const char *) scale->data,
        (const char *) base->data,
        (char *) dst->data,
        n_rows,
        mixes->nb[1],
        dst->nb[1],
        n_hc,
        sinkhorn_iters,
        eps);
}

__launch_bounds__(CUDA_DSV4_HC_EXPAND_BLOCK_SIZE, 1)
static __global__ void dsv4_hc_expand_f32(
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        char *       dst,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens,
        const uint64_t nb_block0,
        const uint64_t nb_block1,
        const uint64_t nb_res0,
        const uint64_t nb_res1,
        const uint64_t nb_res2,
        const uint64_t nb_post0,
        const uint64_t nb_post1,
        const uint64_t nb_comb0,
        const uint64_t nb_comb1,
        const uint64_t nb_comb2,
        const uint64_t nb0,
        const uint64_t nb1,
        const uint64_t nb2) {
    const int64_t n_elem = n_embd*n_hc*n_tokens;

    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n_elem; i += (int64_t) blockDim.x*gridDim.x) {
        const int64_t d      = i % n_embd;
        const int64_t tmp    = i / n_embd;
        const int64_t dst_hc = tmp % n_hc;
        const int64_t t      = tmp / n_hc;

        const float block_v = *reinterpret_cast<const float *>(block_out + d*nb_block0 + t*nb_block1);
        const float post_v  = *reinterpret_cast<const float *>(post      + dst_hc*nb_post0 + t*nb_post1);

        float acc = block_v*post_v;
        for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
            const float comb_v = *reinterpret_cast<const float *>(comb     + dst_hc*nb_comb0 + src_hc*nb_comb1 + t*nb_comb2);
            const float res_v  = *reinterpret_cast<const float *>(residual + d*nb_res0 + src_hc*nb_res1 + t*nb_res2);
            acc += comb_v*res_v;
        }

        *reinterpret_cast<float *>(dst + d*nb0 + dst_hc*nb1 + t*nb2) = acc;
    }
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);
    GGML_ASSERT(block_out->ne[0] == dst->ne[0]);
    GGML_ASSERT(block_out->ne[1] == dst->ne[2]);
    GGML_ASSERT(block_out->ne[2] == 1);
    GGML_ASSERT(block_out->ne[3] == 1);
    GGML_ASSERT(residual->ne[0]  == dst->ne[0]);
    GGML_ASSERT(residual->ne[1]  == dst->ne[1]);
    GGML_ASSERT(residual->ne[2]  == dst->ne[2]);
    GGML_ASSERT(residual->ne[3]  == 1);
    GGML_ASSERT(post->ne[0]      == dst->ne[1]);
    GGML_ASSERT(post->ne[1]      == dst->ne[2]);
    GGML_ASSERT(post->ne[2]      == 1);
    GGML_ASSERT(post->ne[3]      == 1);
    GGML_ASSERT(comb->ne[0]      == dst->ne[1]);
    GGML_ASSERT(comb->ne[1]      == dst->ne[1]);
    GGML_ASSERT(comb->ne[2]      == dst->ne[2]);
    GGML_ASSERT(comb->ne[3]      == 1);
    GGML_ASSERT(dst->ne[3]       == 1);

    const int64_t n_elem = dst->ne[0]*dst->ne[1]*dst->ne[2];
    const int64_t n_blocks = (n_elem + CUDA_DSV4_HC_EXPAND_BLOCK_SIZE - 1) / CUDA_DSV4_HC_EXPAND_BLOCK_SIZE;
    const dim3 block_nums(n_blocks, 1, 1);
    const dim3 block_dims(CUDA_DSV4_HC_EXPAND_BLOCK_SIZE, 1, 1);

    dsv4_hc_expand_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const char *) block_out->data,
        (const char *) residual->data,
        (const char *) post->data,
        (const char *) comb->data,
        (char *) dst->data,
        dst->ne[0], dst->ne[1], dst->ne[2],
        block_out->nb[0], block_out->nb[1],
        residual->nb[0], residual->nb[1], residual->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}
