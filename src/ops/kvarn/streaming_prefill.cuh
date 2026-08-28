#pragma once

#include "ops/kvarn/materialized_prefill.cuh"

#include <math_constants.h>

namespace ninfer::ops::kvarn::detail {

template <typename Geometry>
__device__ __forceinline__ std::int64_t prefill_stat_index(int q_head, int token) {
    return q_head + static_cast<std::int64_t>(Geometry::QHeads) * token;
}

// Computes one materialized context slab and merges its unnormalized online-softmax state into
// the FP32 state shared by successive slab launches.
template <typename Geometry, typename Metadata>
__launch_bounds__(kCausalPromptThreads, 1) __global__ void attention_prefill_slab_kernel(
    const __nv_bfloat16* __restrict__ q, MaterializedPrefillInput cache, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, float* __restrict__ running_acc,
    float* __restrict__ running_m, float* __restrict__ running_l, std::int32_t width,
    std::int32_t key_begin, std::int32_t key_end) {
    constexpr int Br = kCausalPromptBr;
    constexpr int Bc = kCausalPromptBc;
    constexpr int Threads = kCausalPromptThreads;
    constexpr int QKNt = Bc / 8;
    constexpr int QKKs = D / 16;
    constexpr int PVNt = D / 8;
    constexpr int PVKs = Bc / 16;
    constexpr float Log2E = 1.4426950408889634074F;
    constexpr unsigned FullMask = 0xffffffffU;

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* q_s = shared;
    __nv_bfloat16* k_s = q_s + Br * D;
    __nv_bfloat16* v_s = k_s + Bc * D;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= tokens) { return; }

    const int base_pos = positions[0];
    const int tile_rows = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int block_begin = key_begin / Bc;
    const int block_end = min((max_query_abs / Bc) + 1, (key_end + Bc - 1) / Bc);
    if (block_begin >= block_end) { return; }
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r = static_cast<unsigned>(b_rin << 4);

    {
        constexpr int VecPerRow = D / 8;
        constexpr int QRowStride = D * Geometry::QHeads;
        const __nv_bfloat16* q_block_ptr = q + causal_prompt_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row = chunk >> 5;
                const int d = (chunk & 31) << 3;
                cp_async<16, Cache::cg>(q_s + row * D + causal_prompt_swz(row, d),
                                        q_block_ptr + row * QRowStride + d);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row = chunk >> 5;
                const int d = (chunk & 31) << 3;
                __nv_bfloat16* destination = q_s + row * D + causal_prompt_swz(row, d);
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(destination, q_block_ptr + row * QRowStride + d);
                } else {
                    store_vec(destination, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int item = 0; item < 4; ++item) acc[n][item] = 0.0F;
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0F;
    float l1 = 0.0F;
    const float scale_l2 = scale * Log2E;
    int physical_page = block_table[block_begin];

    ninfer::ops::cp_commit();
    cache.template stage<true>(k_s, kv_head, block_begin * Bc, max_query_abs, physical_page, tid);
    ninfer::ops::cp_commit();

    for (int kb = block_begin; kb < block_end; ++kb) {
        const int k0 = kb * Bc;
        const int next_physical_page = kb + 1 < block_end ? block_table[kb + 1] : physical_page;
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        cache.template stage<false>(v_s, kv_head, k0, max_query_abs, physical_page, tid);
        ninfer::ops::cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_prompt_swz_addr(q_lane_base, 0U, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0],
                        bf[0][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                               0U, k_as, k_r));
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int current = k & 1;
            const int next = current ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[next][0], af[next][1], af[next][2], af[next][3],
                             causal_prompt_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[next][nt2][0], bf[next][nt2][1], bf[next][nt2 + 1][0],
                        bf[next][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                               k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[current][0],
                         af[current][1], af[current][2], af[current][3], bf[current][nt][0],
                         bf[current][nt][1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const int qabs0 = qrow0 < tokens ? base_pos + qrow0 : -1;
        const int qabs1 = qrow1 < tokens ? base_pos + qrow1 : -1;
        const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
        float block_m0 = -CUDART_INF_F;
        float block_m1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int key0 = k0 + nt * 8 + 2 * lid;
            const int key1 = key0 + 1;
            if (!full_score_tile) {
                score[nt][0] = qrow0 < tokens && key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1] = qrow0 < tokens && key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2] = qrow1 < tokens && key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3] = qrow1 < tokens && key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            block_m0 = fmaxf(block_m0, fmaxf(score[nt][0], score[nt][1]));
            block_m1 = fmaxf(block_m1, fmaxf(score[nt][2], score[nt][3]));
        }
        block_m0 = warp_max<4>(block_m0, FullMask);
        block_m1 = warp_max<4>(block_m1, FullMask);
        const float next_m0 = fmaxf(m0, block_m0);
        const float next_m1 = fmaxf(m1, block_m1);
        const float next_m0_scaled = next_m0 * scale_l2;
        const float next_m1_scaled = next_m1 * scale_l2;
        const float alpha0 = m0 > -CUDART_INF_F
                                 ? exp2_approx(__fmaf_rn(m0, scale_l2, -next_m0_scaled))
                                 : 0.0F;
        const float alpha1 = m1 > -CUDART_INF_F
                                 ? exp2_approx(__fmaf_rn(m1, scale_l2, -next_m1_scaled))
                                 : 0.0F;
        float block_l0 = 0.0F;
        float block_l1 = 0.0F;
        unsigned probability[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = score[nt][0] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -next_m0_scaled))
                                  : 0.0F;
            const float p01 = score[nt][1] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -next_m0_scaled))
                                  : 0.0F;
            const float p10 = score[nt][2] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -next_m1_scaled))
                                  : 0.0F;
            const float p11 = score[nt][3] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -next_m1_scaled))
                                  : 0.0F;
            block_l0 += p00 + p01;
            block_l1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                probability[pk][0] = pack_bf16x2(p00, p01);
                probability[pk][1] = pack_bf16x2(p10, p11);
            } else {
                probability[pk][2] = pack_bf16x2(p00, p01);
                probability[pk][3] = pack_bf16x2(p10, p11);
            }
        }
        l0 = __fmaf_rn(l0, alpha0, block_l0);
        l1 = __fmaf_rn(l1, alpha1, block_l1);
        m0 = next_m0;
        m1 = next_m1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        ninfer::ops::cp_wait<0>();
        __syncthreads();
        if (kb + 1 < block_end) {
            physical_page = next_physical_page;
            cache.template stage<true>(k_s, kv_head, (kb + 1) * Bc, max_query_abs, physical_page,
                                       tid);
            ninfer::ops::cp_commit();
        }

        constexpr int PVHalf = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                       causal_prompt_swz_addr(v_lane_base, 0U, v_as, v_r));
#pragma unroll
        for (int item = 0; item < PVLoads; ++item) {
            const int k = item / PVHalf;
            const int n2 = (item % PVHalf) * 2;
            const int current = item & 1;
            const int next = current ^ 1;
            if (item + 1 < PVLoads) {
                const int next_k = (item + 1) / PVHalf;
                const int next_n2 = ((item + 1) % PVHalf) * 2;
                ldmatrix_x4_t(
                    vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                    causal_prompt_swz_addr(v_lane_base + static_cast<unsigned>(next_k * 8192),
                                           static_cast<unsigned>(next_n2 << 4), v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], probability[k][0],
                     probability[k][1], probability[k][2], probability[k][3], vf[current][0],
                     vf[current][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                     probability[k][0], probability[k][1], probability[k][2], probability[k][3],
                     vf[current][2], vf[current][3]);
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);
    const int row0 = q0 + warp_row0 + gid;
    const int row1 = row0 + 8;
    const std::int64_t stat0 = prefill_stat_index<Geometry>(q_head, row0);
    const std::int64_t stat1 = prefill_stat_index<Geometry>(q_head, row1);
    const float previous_l0 = row0 < tokens ? running_l[stat0] : 0.0F;
    const float previous_l1 = row1 < tokens ? running_l[stat1] : 0.0F;
    const float previous_m0 = previous_l0 > 0.0F ? running_m[stat0] : -CUDART_INF_F;
    const float previous_m1 = previous_l1 > 0.0F ? running_m[stat1] : -CUDART_INF_F;
    const float combined_m0 = fmaxf(previous_m0, m0);
    const float combined_m1 = fmaxf(previous_m1, m1);
    const float previous_weight0 = previous_l0 > 0.0F
                                       ? exp2_approx((previous_m0 - combined_m0) * scale_l2)
                                       : 0.0F;
    const float previous_weight1 = previous_l1 > 0.0F
                                       ? exp2_approx((previous_m1 - combined_m1) * scale_l2)
                                       : 0.0F;
    const float local_weight0 = l0 > 0.0F ? exp2_approx((m0 - combined_m0) * scale_l2) : 0.0F;
    const float local_weight1 = l1 > 0.0F ? exp2_approx((m1 - combined_m1) * scale_l2) : 0.0F;

    // The stream orders slab launches, and each CTA exclusively owns its query rows and head.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (row0 < tokens) {
            const std::int64_t index = causal_prompt_q_index<Geometry>(q_head, d0, row0);
            const float2 previous = previous_l0 > 0.0F
                                        ? *reinterpret_cast<const float2*>(running_acc + index)
                                        : make_float2(0.0F, 0.0F);
            *reinterpret_cast<float2*>(running_acc + index) =
                make_float2(__fmaf_rn(previous.x, previous_weight0,
                                      acc[n][0] * local_weight0),
                            __fmaf_rn(previous.y, previous_weight0,
                                      acc[n][1] * local_weight0));
        }
        if (row1 < tokens) {
            const std::int64_t index = causal_prompt_q_index<Geometry>(q_head, d0, row1);
            const float2 previous = previous_l1 > 0.0F
                                        ? *reinterpret_cast<const float2*>(running_acc + index)
                                        : make_float2(0.0F, 0.0F);
            *reinterpret_cast<float2*>(running_acc + index) =
                make_float2(__fmaf_rn(previous.x, previous_weight1,
                                      acc[n][2] * local_weight1),
                            __fmaf_rn(previous.y, previous_weight1,
                                      acc[n][3] * local_weight1));
        }
    }
    if (lid == 0) {
        if (row0 < tokens) {
            running_m[stat0] = combined_m0;
            running_l[stat0] = __fmaf_rn(previous_l0, previous_weight0, l0 * local_weight0);
        }
        if (row1 < tokens) {
            running_m[stat1] = combined_m1;
            running_l[stat1] = __fmaf_rn(previous_l1, previous_weight1, l1 * local_weight1);
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(D) __global__ void finalize_prefill_slab_kernel(
    const float* __restrict__ running_acc, const float* __restrict__ running_l, Metadata metadata,
    std::int32_t width, __nv_bfloat16* __restrict__ output) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int d = static_cast<int>(threadIdx.x);
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || token >= width) { return; }
    float value = 0.0F;
    if (token < tokens) {
        const float denominator = running_l[prefill_stat_index<Geometry>(q_head, token)];
        if (denominator > 0.0F) {
            value = running_acc[causal_prompt_q_index<Geometry>(q_head, d, token)] /
                    denominator;
        }
    }
    output[causal_prompt_q_index<Geometry>(q_head, d, token)] = __float2bfloat16_rn(value);
}

} // namespace ninfer::ops::kvarn::detail
