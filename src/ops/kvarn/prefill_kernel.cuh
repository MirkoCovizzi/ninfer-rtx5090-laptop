#pragma once

#include "ninfer/ops/kvarn.h"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kvarn/config.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::kvarn::detail {

inline constexpr int kPrefillWarps = 16;
inline constexpr int kPrefillThreads = kPrefillWarps * 32;
inline constexpr int kPrefillBr = 64;
inline constexpr int kPrefillBc = Group;
inline constexpr int kPrefillProducerWarps = kPrefillBr / 16;
inline constexpr int kPrefillConsumerWarps = kPrefillWarps - kPrefillProducerWarps;
inline constexpr int kPrefillConsumersPerTile =
    kPrefillConsumerWarps / kPrefillProducerWarps;
inline constexpr int kPrefillSmemBytes =
    (kPrefillBr + 2 * kPrefillBc) * D * static_cast<int>(sizeof(__nv_bfloat16));

static_assert(kPrefillProducerWarps == 4);
static_assert(kPrefillConsumerWarps == 12);
static_assert(kPrefillConsumersPerTile == 3);
static_assert(kPrefillSmemBytes == 98304);

struct PrefillCache {
    const std::uint8_t* records;
    const __nv_bfloat16* tail_k;
    const __nv_bfloat16* tail_v;
    const std::int32_t* markers;
    const std::int32_t* table_rows;
    int heads;
};

__device__ __forceinline__ int prefill_tail_slot(const PrefillCache& cache, int logical_page) {
    const int table_row = cache.table_rows[0];
    int tail_slot = -1;
#pragma unroll
    for (int slot = 0; slot < kKvarnTailSlots; ++slot) {
        if (cache.markers[slot + kKvarnTailSlots * table_row] == logical_page) {
            tail_slot = slot;
        }
    }
    return tail_slot;
}

template <bool Key>
__device__ __forceinline__ void stage_prefill_tail(__nv_bfloat16* destination,
                                                   const PrefillCache& cache, int head, int k0,
                                                   int max_query_abs, int tail_slot, int tid,
                                                   int threads) {
    constexpr int VecPerRow = D / 8;
    const int table_row = cache.table_rows[0];
    const auto* tail = Key ? cache.tail_k : cache.tail_v;
    for (int chunk = tid; chunk < Group * VecPerRow; chunk += threads) {
        const int token = chunk / VecPerRow;
        const int d = (chunk % VecPerRow) * 8;
        __nv_bfloat16* output = destination + token * D + gqa_prefill_swz(token, d);
        if (k0 + token > max_query_abs) {
            store_vec(output, make_int4(0, 0, 0, 0));
            continue;
        }
        const std::int64_t source =
            static_cast<std::int64_t>(d) + static_cast<std::int64_t>(D) *
                                               (token + Group *
                                                            (head + cache.heads *
                                                                        (tail_slot +
                                                                         kKvarnTailSlots *
                                                                             table_row)));
        store_vec(output, load_vec<int4>(tail + source));
    }
}

__device__ __forceinline__ void stage_prefill_key(__nv_bfloat16* destination,
                                                  const PrefillCache& cache, int head, int k0,
                                                  int max_query_abs, int physical_page, int tid) {
    const int tail_slot = prefill_tail_slot(cache, k0 / Group);
    if (tail_slot >= 0) {
        stage_prefill_tail<true>(destination, cache, head, k0, max_query_abs, tail_slot, tid,
                                 kPrefillThreads);
        return;
    }

    const int dim = tid & (D - 1);
    const int half = tid / D;
    const std::uint8_t* record =
        cache.records +
        (static_cast<std::int64_t>(physical_page) * cache.heads + head) * kKvarnRecordBytes;
    const auto* scale = reinterpret_cast<const __half*>(record + kKvarnKScaleOffset);
    const auto* zero = reinterpret_cast<const __half*>(record + kKvarnKZeroOffset);
    const auto* token_scale =
        reinterpret_cast<const __half*>(record + kKvarnKTokenScaleOffset);
    const float column_scale = __half2float(scale[dim]);
    const float column_zero = __half2float(zero[dim]);
    const int4 packed = load_vec<int4>(record + kKvarnKPackedOffset + dim * (Group / 2) +
                                       half * 16);
#pragma unroll
    for (int pair = 0; pair < 16; ++pair) {
        const int word_index = pair >> 2;
        const unsigned word =
            word_index == 0   ? static_cast<unsigned>(packed.x)
            : word_index == 1 ? static_cast<unsigned>(packed.y)
            : word_index == 2 ? static_cast<unsigned>(packed.z)
                              : static_cast<unsigned>(packed.w);
        const unsigned codes = (word >> (8 * (pair & 3))) & 0xffU;
        const int token = half * 32 + pair * 2;
        const auto token_scales =
            __half22float2(*reinterpret_cast<const __half2*>(token_scale + token));
        const float decoded0 =
            fmaf(static_cast<float>(codes & 15U), column_scale, column_zero) * token_scales.x;
        const float decoded1 =
            fmaf(static_cast<float>(codes >> 4), column_scale, column_zero) * token_scales.y;
        destination[token * D + gqa_prefill_swz(token, dim)] =
            k0 + token <= max_query_abs ? __float2bfloat16_rn(decoded0)
                                        : __float2bfloat16_rn(0.0F);
        destination[(token + 1) * D + gqa_prefill_swz(token + 1, dim)] =
            k0 + token + 1 <= max_query_abs ? __float2bfloat16_rn(decoded1)
                                            : __float2bfloat16_rn(0.0F);
    }
}

__device__ __forceinline__ void stage_prefill_value(__nv_bfloat16* destination,
                                                    const PrefillCache& cache, int head, int k0,
                                                    int max_query_abs, int physical_page, int tid,
                                                    int threads) {
    const int tail_slot = prefill_tail_slot(cache, k0 / Group);
    if (tail_slot >= 0) {
        stage_prefill_tail<false>(destination, cache, head, k0, max_query_abs, tail_slot, tid,
                                  threads);
        return;
    }

    const std::uint8_t* record =
        cache.records +
        (static_cast<std::int64_t>(physical_page) * cache.heads + head) * kKvarnRecordBytes;
    const auto* channel =
        reinterpret_cast<const __half*>(record + kKvarnVChannelScaleOffset);
    const auto* scale = reinterpret_cast<const __half*>(record + kKvarnVTokenScaleOffset);
    const auto* zero = reinterpret_cast<const __half*>(record + kKvarnVTokenZeroOffset);
    for (int packed_item = tid; packed_item < Group * (D / 4); packed_item += threads) {
        const int token = packed_item / (D / 4);
        const int packed_dim = packed_item - token * (D / 4);
        const int dim = 4 * packed_dim;
        __nv_bfloat16* output = destination + token * D + gqa_prefill_swz(token, dim);
        if (k0 + token > max_query_abs) {
            *reinterpret_cast<uint2*>(output) = make_uint2(0, 0);
            continue;
        }
        const std::uint8_t packed =
            record[kKvarnVPackedOffset + token * (D / 4) + packed_dim];
        const float row_scale = __half2float(scale[token]);
        const float row_zero = __half2float(zero[token]);
        const auto channel01 =
            __half22float2(*reinterpret_cast<const __half2*>(channel + dim));
        const auto channel23 =
            __half22float2(*reinterpret_cast<const __half2*>(channel + dim + 2));
        const float channel_scales[4] = {channel01.x, channel01.y, channel23.x, channel23.y};
        unsigned values[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int code = (packed >> (2 * item)) & 3;
            values[item] = __bfloat16_as_ushort(__float2bfloat16_rn(
                fmaf(static_cast<float>(code), row_scale, row_zero) * channel_scales[item]));
        }
        *reinterpret_cast<uint2*>(output) =
            make_uint2(values[0] | (values[1] << 16), values[2] | (values[3] << 16));
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(kPrefillThreads, 1) __global__ void attention_prefill_kernel(
    const __nv_bfloat16* __restrict__ q, PrefillCache cache, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int QKNt = kPrefillBc / 8;
    constexpr int QKKs = D / 16;
    constexpr int PVNt = D / 8;
    constexpr int PVNtPerWarp = (PVNt + kPrefillConsumersPerTile - 1) /
                                kPrefillConsumersPerTile;
    constexpr int PVKs = kPrefillBc / 16;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(PVNtPerWarp == 11);

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* q_s = shared;
    __nv_bfloat16* k_s = q_s + kPrefillBr * D;
    __nv_bfloat16* v_s = k_s + kPrefillBc * D;
    __nv_bfloat16* p_s = k_s;
    float* alpha_s = reinterpret_cast<float*>(p_s + kPrefillBr * kPrefillBc);
    float* final_l_s = alpha_s + kPrefillBr;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = q_block * kPrefillBr;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + kPrefillBr, width), tid,
                                               kPrefillThreads);
        return;
    }

    const int base_pos = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows = min(kPrefillBr, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks = max_query_abs / kPrefillBc + 1;

    constexpr int VecPerRow = D / 8;
    constexpr int QRowStride = D * Geometry::QHeads;
    const __nv_bfloat16* q_block_ptr = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
    for (int chunk = tid; chunk < kPrefillBr * VecPerRow; chunk += kPrefillThreads) {
        const int row = chunk / VecPerRow;
        const int d = (chunk % VecPerRow) * 8;
        __nv_bfloat16* destination = q_s + row * D + gqa_prefill_swz(row, d);
        if (row < tile_rows) {
            cp_async<16, Cache::cg>(destination, q_block_ptr + row * QRowStride + d);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }
    ninfer::ops::cp_commit();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const bool producer = warp < kPrefillProducerWarps;

    union {
        unsigned probability[QKNt][2];
        float accumulator[PVNtPerWarp][4];
    } warp_state;
    if (!producer) {
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                warp_state.accumulator[n][item] = 0.0F;
            }
        }
    }

    float running_m0 = -CUDART_INF_F;
    float running_m1 = -CUDART_INF_F;
    float running_l0 = 0.0F;
    float running_l1 = 0.0F;
    const float scale_l2 = scale * Log2E;
    for (int block = 0; block < key_blocks; ++block) {
        const int k0 = block * kPrefillBc;
        const int physical_page = block_table[block];
        stage_prefill_key(k_s, cache, kv_head, k0, max_query_abs, physical_page, tid);
        if (block == 0) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();

        float alpha0 = 0.0F;
        float alpha1 = 0.0F;
        if (producer) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                score[tile][0] = score[tile][1] = score[tile][2] = score[tile][3] = 0.0F;
            }
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned query_fragment[4];
                const int query_row = row_base + a_rowoff;
                const int query_col = k * 16 + a_coloff;
                ldmatrix_x4(query_fragment[0], query_fragment[1], query_fragment[2],
                            query_fragment[3],
                            smem_addr(q_s + query_row * D +
                                      gqa_prefill_swz(query_row, query_col)));
#pragma unroll
                for (int tile = 0; tile < QKNt; ++tile) {
                    unsigned key_fragment[2];
                    const int key_row = tile * 8 + b_rin;
                    const int key_col = k * 16 + b_koff;
                    ldmatrix_x2(key_fragment[0], key_fragment[1],
                                smem_addr(k_s + key_row * D +
                                          gqa_prefill_swz(key_row, key_col)));
                    mma_bf16(score[tile][0], score[tile][1], score[tile][2], score[tile][3],
                             query_fragment[0], query_fragment[1], query_fragment[2],
                             query_fragment[3], key_fragment[0], key_fragment[1]);
                }
            }

            const int row0 = row_base + gid;
            const int row1 = row0 + 8;
            const int qabs0 = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1 = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile =
                q0 + kPrefillBr <= tokens && k0 + kPrefillBc - 1 <= base_pos + q0;
            float block_m0 = -CUDART_INF_F;
            float block_m1 = -CUDART_INF_F;
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                const int key0 = k0 + tile * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[tile][0] = key0 <= qabs0 ? score[tile][0] : -CUDART_INF_F;
                    score[tile][1] = key1 <= qabs0 ? score[tile][1] : -CUDART_INF_F;
                    score[tile][2] = key0 <= qabs1 ? score[tile][2] : -CUDART_INF_F;
                    score[tile][3] = key1 <= qabs1 ? score[tile][3] : -CUDART_INF_F;
                }
                block_m0 = fmaxf(block_m0, fmaxf(score[tile][0], score[tile][1]));
                block_m1 = fmaxf(block_m1, fmaxf(score[tile][2], score[tile][3]));
            }
            block_m0 = warp_max<4>(block_m0, FullMask);
            block_m1 = warp_max<4>(block_m1, FullMask);
            const float next_m0 = fmaxf(running_m0, block_m0);
            const float next_m1 = fmaxf(running_m1, block_m1);
            const float next_m0_scaled = next_m0 * scale_l2;
            const float next_m1_scaled = next_m1 * scale_l2;
            alpha0 = running_m0 == -CUDART_INF_F
                         ? 0.0F
                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -next_m0_scaled));
            alpha1 = running_m1 == -CUDART_INF_F
                         ? 0.0F
                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -next_m1_scaled));
            float block_l0 = 0.0F;
            float block_l1 = 0.0F;
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                const float p00 = score[tile][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[tile][0], scale_l2,
                                                             -next_m0_scaled))
                                      : 0.0F;
                const float p01 = score[tile][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[tile][1], scale_l2,
                                                             -next_m0_scaled))
                                      : 0.0F;
                const float p10 = score[tile][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[tile][2], scale_l2,
                                                             -next_m1_scaled))
                                      : 0.0F;
                const float p11 = score[tile][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[tile][3], scale_l2,
                                                             -next_m1_scaled))
                                      : 0.0F;
                block_l0 += p00 + p01;
                block_l1 += p10 + p11;
                warp_state.probability[tile][0] = pack_bf16x2(p00, p01);
                warp_state.probability[tile][1] = pack_bf16x2(p10, p11);
            }
            running_l0 = __fmaf_rn(running_l0, alpha0, block_l0);
            running_l1 = __fmaf_rn(running_l1, alpha1, block_l1);
            running_m0 = next_m0;
            running_m1 = next_m1;
        } else {
            const int worker_tid = tid - kPrefillProducerWarps * 32;
            stage_prefill_value(v_s, cache, kv_head, k0, max_query_abs, physical_page, worker_tid,
                                kPrefillConsumerWarps * 32);
        }
        __syncthreads();

        if (producer) {
            const int row_base = warp * 16;
            const int row0 = row_base + gid;
            const int row1 = row0 + 8;
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                const int col0 = tile * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const unsigned p0 = warp_state.probability[tile][0];
                const unsigned p1 = warp_state.probability[tile][1];
                p_s[row0 * kPrefillBc + gqa_prefill_swz(row0, col0)] =
                    __ushort_as_bfloat16(static_cast<std::uint16_t>(p0));
                p_s[row0 * kPrefillBc + gqa_prefill_swz(row0, col1)] =
                    __ushort_as_bfloat16(static_cast<std::uint16_t>(p0 >> 16));
                p_s[row1 * kPrefillBc + gqa_prefill_swz(row1, col0)] =
                    __ushort_as_bfloat16(static_cast<std::uint16_t>(p1));
                p_s[row1 * kPrefillBc + gqa_prefill_swz(row1, col1)] =
                    __ushort_as_bfloat16(static_cast<std::uint16_t>(p1 >> 16));
            }
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        }
        __syncthreads();

        if (!producer) {
            const int consumer_warp = warp - kPrefillProducerWarps;
            const int row_tile = consumer_warp % kPrefillProducerWarps;
            const int dimension_slice = consumer_warp / kPrefillProducerWarps;
            const int row_base = row_tile * 16;
            const float row_alpha0 = alpha_s[row_base + gid];
            const float row_alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                warp_state.accumulator[n][0] *= row_alpha0;
                warp_state.accumulator[n][1] *= row_alpha0;
                warp_state.accumulator[n][2] *= row_alpha1;
                warp_state.accumulator[n][3] *= row_alpha1;
            }
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned probability_fragment[4];
                const int probability_col = k * 16 + a_coloff;
                ldmatrix_x4(probability_fragment[0], probability_fragment[1],
                            probability_fragment[2], probability_fragment[3],
                            smem_addr(p_s + (row_base + a_rowoff) * kPrefillBc +
                                      gqa_prefill_swz(row_base + a_rowoff, probability_col)));
#pragma unroll
                for (int n = 0; n < PVNtPerWarp; ++n) {
                    const int global_n = dimension_slice * PVNtPerWarp + n;
                    if (global_n >= PVNt) { continue; }
                    unsigned value_fragment[2];
                    const int value_row = k * 16 + b_koff + b_rin;
                    const int value_col = global_n * 8;
                    ldmatrix_x2_t(value_fragment[0], value_fragment[1],
                                  smem_addr(v_s + value_row * D +
                                            gqa_prefill_swz(value_row, value_col)));
                    mma_bf16(warp_state.accumulator[n][0], warp_state.accumulator[n][1],
                             warp_state.accumulator[n][2], warp_state.accumulator[n][3],
                             probability_fragment[0], probability_fragment[1],
                             probability_fragment[2], probability_fragment[3], value_fragment[0],
                             value_fragment[1]);
                }
            }
        }
        __syncthreads();
    }

    if (producer) {
        const float final_l0 = warp_sum<4>(running_l0, FullMask);
        const float final_l1 = warp_sum<4>(running_l1, FullMask);
        if (lid == 0) {
            const int row0 = warp * 16 + gid;
            const int row1 = row0 + 8;
            final_l_s[row0] = final_l0;
            final_l_s[row1] = final_l1;
        }
    }
    __syncthreads();

    if (!producer) {
        const int consumer_warp = warp - kPrefillProducerWarps;
        const int row_tile = consumer_warp % kPrefillProducerWarps;
        const int dimension_slice = consumer_warp / kPrefillProducerWarps;
        const int row_base = row_tile * 16;
        const int row0 = row_base + gid;
        const int row1 = row0 + 8;
        const float inv_l0 = final_l_s[row0] > 0.0F ? __frcp_rn(final_l_s[row0]) : 0.0F;
        const float inv_l1 = final_l_s[row1] > 0.0F ? __frcp_rn(final_l_s[row1]) : 0.0F;
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = dimension_slice * PVNtPerWarp + n;
            if (global_n >= PVNt) { continue; }
            const int d0 = global_n * 8 + 2 * lid;
            if (row0 < tile_rows) {
                *reinterpret_cast<unsigned*>(
                    out + gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)) =
                    pack_bf16x2(warp_state.accumulator[n][0] * inv_l0,
                                warp_state.accumulator[n][1] * inv_l0);
            }
            if (row1 < tile_rows) {
                *reinterpret_cast<unsigned*>(
                    out + gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)) =
                    pack_bf16x2(warp_state.accumulator[n][2] * inv_l1,
                                warp_state.accumulator[n][3] * inv_l1);
            }
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + kPrefillBr, width), tid,
                                           kPrefillThreads);
}

} // namespace ninfer::ops::kvarn::detail
