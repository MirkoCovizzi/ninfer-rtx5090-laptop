#pragma once

#include "ninfer/ops/kvarn.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/kvarn/config.cuh"
#include "ops/kvarn/hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::kvarn {
namespace detail {

inline constexpr int kDecodeBc = 32;
inline constexpr int kDecodeWarps = 8;
inline constexpr int kPackedKWordsPerHalf = 4;
inline constexpr int kPackedKWords = 2 * kPackedKWordsPerHalf * D;
inline constexpr int kPackedVBytes = Group * (D / 4);
inline constexpr int kVCodeValues = 4;
static_assert(VBits == 2);

struct alignas(16) DecodeRecordMetadata {
    __half k_scale[D];
    __half k_zero[D];
    float k_token_scale[Group];
    // Plane by position in the packed V byte so consumer warps read consecutive banks.
    float v_channel_scale[kVCodeValues][D / kVCodeValues];
    float v_base[Group][kVCodeValues];
};

template <typename Geometry>
__device__ __forceinline__ int kvarn_decode_active_splits(int window, int launch_capacity,
                                                          int tokens) {
    if constexpr (Geometry::QHeads == 24) {
        if (tokens == 1 && window > 8198) {
            int splits = div_up(window, 192);
            const int cap = window <= DecodeMidWindow ? DecodeMidSplits : DecodeLongSplits;
            splits = min(splits, cap);
            return min(splits, launch_capacity);
        }
    }
    return causal_small_t_active_splits<Geometry, false>(window, launch_capacity, tokens);
}

__device__ __forceinline__ int decode_tail_slot(const std::int32_t* markers, int table_row,
                                                int logical_page) {
    int tail_slot = -1;
#pragma unroll
    for (int slot = 0; slot < kKvarnTailSlots; ++slot) {
        if (markers[slot + kKvarnTailSlots * table_row] == logical_page) { tail_slot = slot; }
    }
    return tail_slot;
}

__device__ __forceinline__ void stage_decode_record(
    unsigned* packed_k, std::uint8_t* packed_v, DecodeRecordMetadata* metadata,
    const std::uint8_t* record, int tid, int threads) {
    constexpr int KChunks = kKvarnKScaleOffset / 16;
    for (int chunk = tid; chunk < KChunks; chunk += threads) {
        const int4 packed = load_vec<int4>(record + kKvarnKPackedOffset + chunk * 16);
        const int dim = chunk >> 1;
        const int half = chunk & 1;
        packed_k[(half * kPackedKWordsPerHalf + 0) * D + dim] =
            static_cast<unsigned>(packed.x);
        packed_k[(half * kPackedKWordsPerHalf + 1) * D + dim] =
            static_cast<unsigned>(packed.y);
        packed_k[(half * kPackedKWordsPerHalf + 2) * D + dim] =
            static_cast<unsigned>(packed.z);
        packed_k[(half * kPackedKWordsPerHalf + 3) * D + dim] =
            static_cast<unsigned>(packed.w);
    }
    constexpr int VChunks = kPackedVBytes / 16;
    for (int chunk = tid; chunk < VChunks; chunk += threads) {
        store_vec(packed_v + chunk * 16,
                  load_vec<int4>(record + kKvarnVPackedOffset + chunk * 16));
    }

    const auto* k_scale = reinterpret_cast<const __half*>(record + kKvarnKScaleOffset);
    const auto* k_zero = reinterpret_cast<const __half*>(record + kKvarnKZeroOffset);
    const auto* k_token = reinterpret_cast<const __half*>(record + kKvarnKTokenScaleOffset);
    const auto* v_channel = reinterpret_cast<const __half*>(record + kKvarnVChannelScaleOffset);
    const auto* v_scale = reinterpret_cast<const __half*>(record + kKvarnVTokenScaleOffset);
    const auto* v_zero = reinterpret_cast<const __half*>(record + kKvarnVTokenZeroOffset);
    for (int dim = tid; dim < D; dim += threads) {
        metadata->k_scale[dim] = k_scale[dim];
        metadata->k_zero[dim] = k_zero[dim];
        metadata->v_channel_scale[dim & 3][dim >> 2] = __half2float(v_channel[dim]);
    }
    for (int token = tid; token < Group; token += threads) {
        metadata->k_token_scale[token] = __half2float(k_token[token]);
        const float scale = __half2float(v_scale[token]);
        const float zero = __half2float(v_zero[token]);
#pragma unroll
        for (int code = 0; code < kVCodeValues; ++code) {
            metadata->v_base[token][code] = fmaf(static_cast<float>(code), scale, zero);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void stage_decode_key(
    __nv_bfloat16* destination, const unsigned* packed_k, const DecodeRecordMetadata* metadata,
    const __nv_bfloat16* tail_k, int table_row, int tail_slot, int heads, int head,
    int logical_begin, int valid_begin, int valid_end, int tid, int threads) {
    constexpr int Bc = kDecodeBc;
    const int token_base = logical_begin & (Group - 1);
    if (tail_slot >= 0) {
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += threads) {
            const int token = chunk / (D / 8);
            const int d = (chunk % (D / 8)) * 8;
            __nv_bfloat16* output = destination + token * D + causal_small_t_tc_swz(token, d);
            const int position = logical_begin + token;
            if (position < valid_begin || position >= valid_end) {
                store_vec(output, make_int4(0, 0, 0, 0));
                continue;
            }
            const std::int64_t source =
                static_cast<std::int64_t>(d) + static_cast<std::int64_t>(D) *
                                                   (token_base + token + Group *
                                                                             (head + heads *
                                                                                         (tail_slot +
                                                                                           kKvarnTailSlots *
                                                                                               table_row)));
            store_vec(output, load_vec<int4>(tail_k + source));
        }
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int record_half = token_base / Bc;
    for (int dim = warp * 32 + lane; dim < D; dim += (threads / 32) * 32) {
        const int word_base = record_half * kPackedKWordsPerHalf;
        const int4 packed = make_int4(
            static_cast<int>(packed_k[(word_base + 0) * D + dim]),
            static_cast<int>(packed_k[(word_base + 1) * D + dim]),
            static_cast<int>(packed_k[(word_base + 2) * D + dim]),
            static_cast<int>(packed_k[(word_base + 3) * D + dim]));
        const float scale = __half2float(metadata->k_scale[dim]);
        const float zero = __half2float(metadata->k_zero[dim]);
#pragma unroll
        for (int pair = 0; pair < Bc / 2; ++pair) {
            const int token = 2 * pair;
            const int word_index = pair >> 2;
            const unsigned word =
                word_index == 0   ? static_cast<unsigned>(packed.x)
                : word_index == 1 ? static_cast<unsigned>(packed.y)
                : word_index == 2 ? static_cast<unsigned>(packed.z)
                                  : static_cast<unsigned>(packed.w);
            const unsigned codes = (word >> (8 * (pair & 3))) & 0xffU;
            const auto token_scales = *reinterpret_cast<const float2*>(
                metadata->k_token_scale + token_base + token);
            const float decoded0 = fmaf(static_cast<float>(codes & 15U), scale, zero) *
                                   token_scales.x;
            const float decoded1 = fmaf(static_cast<float>(codes >> 4), scale, zero) *
                                   token_scales.y;
            const int position0 = logical_begin + token;
            const int position1 = position0 + 1;
            destination[token * D + causal_small_t_tc_swz(token, dim)] =
                position0 >= valid_begin && position0 < valid_end ? __float2bfloat16_rn(decoded0)
                                                                   : __float2bfloat16_rn(0.0F);
            destination[(token + 1) * D + causal_small_t_tc_swz(token + 1, dim)] =
                position1 >= valid_begin && position1 < valid_end ? __float2bfloat16_rn(decoded1)
                                                                   : __float2bfloat16_rn(0.0F);
        }
    }
}

__device__ __forceinline__ void stage_decode_value(
    __nv_bfloat16* destination, const std::uint8_t* packed_v,
    const DecodeRecordMetadata* metadata,
    const __nv_bfloat16* tail_v, int table_row, int tail_slot, int heads, int head,
    int logical_begin, int valid_begin, int valid_end, int tid, int threads) {
    constexpr int Bc = kDecodeBc;
    const int token_base = logical_begin & (Group - 1);
    if (tail_slot >= 0) {
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += threads) {
            const int token = chunk / (D / 8);
            const int d = (chunk % (D / 8)) * 8;
            __nv_bfloat16* output = destination + token * D + causal_small_t_tc_swz(token, d);
            const int position = logical_begin + token;
            if (position < valid_begin || position >= valid_end) {
                store_vec(output, make_int4(0, 0, 0, 0));
                continue;
            }
            const std::int64_t source =
                static_cast<std::int64_t>(d) + static_cast<std::int64_t>(D) *
                                                   (token_base + token + Group *
                                                                             (head + heads *
                                                                                         (tail_slot +
                                                                                          kKvarnTailSlots *
                                                                                              table_row)));
            store_vec(output, load_vec<int4>(tail_v + source));
        }
        return;
    }

    for (int item = tid; item < Bc * (D / 4); item += threads) {
        const int token = item / (D / 4);
        const int packed_dim = item - token * (D / 4);
        const int record_token = token_base + token;
        const int position = logical_begin + token;
        const std::uint8_t packed = packed_v[record_token * (D / 4) + packed_dim];
        unsigned values[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int code = (packed >> (2 * item)) & 3;
            const float decoded = metadata->v_base[record_token][code] *
                                  metadata->v_channel_scale[item][packed_dim];
            const __nv_bfloat16 value =
                position >= valid_begin && position < valid_end ? __float2bfloat16_rn(decoded)
                                                                : __float2bfloat16_rn(0.0F);
            values[item] = __bfloat16_as_ushort(value);
        }
        const uint2 packed_values =
            make_uint2(values[0] | (values[1] << 16), values[2] | (values[3] << 16));
        auto* output = reinterpret_cast<uint2*>(
            destination + token * D + causal_small_t_tc_swz(token, 4 * packed_dim));
        *output = packed_values;
    }
}

template <typename Geometry, bool MultiBatch, bool Masked>
__launch_bounds__(kDecodeWarps * 32, 2) __global__ void attention_decode_kernel(
    const __nv_bfloat16* q, const std::uint8_t* records, const __nv_bfloat16* tail_k,
    const __nv_bfloat16* tail_v, const std::int32_t* markers, const std::int32_t* positions,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    std::int32_t heads, float scale, __nv_bfloat16* partial_acc, float* partial_m,
    float* partial_l) {
    constexpr int Wc = kDecodeWarps;
    constexpr int ProducerWarps = 1;
    constexpr int Br = 16;
    constexpr int Bc = kDecodeBc;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt = Bc / 8;
    constexpr int QKKs = D / 16;
    constexpr int PVNt = D / 8;
    constexpr int ConsumerWarps = Wc - ProducerWarps;
    constexpr int PVNtPerWarp = div_up(PVNt, ConsumerWarps);
    constexpr int PVKs = Bc / 16;
    constexpr int PageIds = 64;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(Group == 2 * Bc);
    static_assert(Geometry::GroupSize <= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[2 * Bc * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) unsigned packed_k_s[kPackedKWords];
    __shared__ __align__(16) std::uint8_t packed_v_s[kPackedVBytes];
    __shared__ DecodeRecordMetadata metadata_s;
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int flat_column = static_cast<int>(blockIdx.z);
    const int batch = MultiBatch ? flat_column / tokens : 0;
    const int column = MultiBatch ? flat_column - batch * tokens : flat_column;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_count = Geometry::GroupSize;
    const int current_column = column_begin + column;
    const std::int64_t column_base =
        current_column + (MultiBatch ? static_cast<std::int64_t>(batch) * full_width : 0);
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    positions += column_base;
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * D * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            const int q_head = kv_head * Geometry::GroupSize + row;
            partial_m[causal_partial_stat_index<Geometry>(q_head, column, split, tokens)] =
                -CUDART_INF_F;
            partial_l[causal_partial_stat_index<Geometry>(q_head, column, split, tokens)] = 0.0f;
        }
        for (int index = tid; index < row_count * D; index += Threads) {
            const int row = index / D;
            const int d = index % D;
            const int q_head = kv_head * Geometry::GroupSize + row;
            partial_acc[causal_partial_acc_index<Geometry>(q_head, d, column, split, tokens)] =
                __float2bfloat16(0.0f);
        }
    };

    const bool valid = !Masked || current_column < valid_columns[batch];
    if (!valid || kv_head >= Geometry::KVHeads || column >= tokens || split_count <= 0) {
        if (kv_head < Geometry::KVHeads && column < tokens) { write_neutral(); }
        return;
    }
    const int query_position = positions[0];
    if (query_position < 0 || query_position >= logical_capacity) {
        write_neutral();
        return;
    }
    const int window = query_position + 1;
    const int active_splits = kvarn_decode_active_splits<Geometry>(window, split_count, tokens);
    if (split >= active_splits) { return; }
    const int logical_tiles = div_up(window, Bc);
    const bool tile_split = logical_tiles >= active_splits;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_splits) : div_up(window, active_splits);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_end = min(split_start + units_per_split * (tile_split ? Bc : 1), window);
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }
    for (int index = tid; index < Br * D; index += Threads) {
        const int row = index / D;
        const int d = index % D;
        const int q_head = kv_head * Geometry::GroupSize + row;
        qkv_s[row * D + causal_small_t_tc_swz(row, d)] =
            row < row_count ? q[causal_q_index<Geometry>(q_head, d)] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    // Both admitted GQA groups fit one MMA row tile. The producer retains Q fragments while the
    // other seven warps partition the valid PV tile, keeping two CTAs resident without spills.
    union {
        unsigned query_fragment[QKKs][4];
        float accumulator[PVNtPerWarp][4];
    } warp_state;
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int query_col = k * 16 + a_coloff;
            ldmatrix_x4(warp_state.query_fragment[k][0], warp_state.query_fragment[k][1],
                        warp_state.query_fragment[k][2], warp_state.query_fragment[k][3],
                        smem_addr(&qkv_s[a_rowoff * D +
                                          causal_small_t_tc_swz(a_rowoff, query_col)]));
        }
    }
    __syncthreads();

    int physical_page = physical_pages_s[0];
    if (warp >= ProducerWarps) {
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
            for (int item = 0; item < 4; ++item) { warp_state.accumulator[n][item] = 0.0f; }
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    for (int block = 0; block < key_blocks; ++block) {
        const int k0 = first_tile + block * Bc;
        if (block != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        const int tail_slot = decode_tail_slot(markers, table_row, k0 / Group);
        const std::uint8_t* record =
            records + (static_cast<std::int64_t>(physical_page) * heads + kv_head) *
                          kKvarnRecordBytes;
        if (tail_slot < 0 && (block == 0 || (k0 & (Group - 1)) == 0)) {
            stage_decode_record(packed_k_s, packed_v_s, &metadata_s, record, tid, Threads);
        }
        stage_decode_key(k_s, packed_k_s, &metadata_s, tail_k, table_row, tail_slot, heads, kv_head,
                         k0, max(k0, split_start), min(k0 + Bc, split_end), tid, Threads);
        __syncthreads();

        if (warp == 0) {
            float score[QKNt][4];
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                score[tile][0] = score[tile][1] = score[tile][2] = score[tile][3] = 0.0f;
            }
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
#pragma unroll
                for (int tile = 0; tile < QKNt; ++tile) {
                    unsigned key_fragment[2];
                    const int row = tile * 8 + b_rin;
                    const int col = k * 16 + b_koff;
                    ldmatrix_x2(key_fragment[0], key_fragment[1],
                                 smem_addr(&k_s[row * D + causal_small_t_tc_swz(row, col)]));
                    mma_bf16(score[tile][0], score[tile][1], score[tile][2], score[tile][3],
                             warp_state.query_fragment[k][0], warp_state.query_fragment[k][1],
                             warp_state.query_fragment[k][2], warp_state.query_fragment[k][3],
                             key_fragment[0], key_fragment[1]);
                }
            }
            const int row0 = gid;
            const int row1 = row0 + 8;
            float block_m0 = -CUDART_INF_F, block_m1 = -CUDART_INF_F;
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                const int col0 = tile * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[tile][0] = row0 < row_count && key0 >= split_start && key0 < split_end &&
                                         key0 <= query_position
                                     ? score[tile][0] * scale
                                     : -CUDART_INF_F;
                score[tile][1] = row0 < row_count && key1 >= split_start && key1 < split_end &&
                                         key1 <= query_position
                                     ? score[tile][1] * scale
                                     : -CUDART_INF_F;
                score[tile][2] = row1 < row_count && key0 >= split_start && key0 < split_end &&
                                         key0 <= query_position
                                     ? score[tile][2] * scale
                                     : -CUDART_INF_F;
                score[tile][3] = row1 < row_count && key1 >= split_start && key1 < split_end &&
                                         key1 <= query_position
                                     ? score[tile][3] * scale
                                     : -CUDART_INF_F;
                block_m0 = fmaxf(block_m0, fmaxf(score[tile][0], score[tile][1]));
                block_m1 = fmaxf(block_m1, fmaxf(score[tile][2], score[tile][3]));
            }
            block_m0 = warp_max<4>(block_m0, FullMask);
            block_m1 = warp_max<4>(block_m1, FullMask);
            const float next_m0 = fmaxf(m0, block_m0);
            const float next_m1 = fmaxf(m1, block_m1);
            const float alpha0 =
                m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - next_m0) * Log2E);
            const float alpha1 =
                m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - next_m1) * Log2E);
            float block_l0 = 0.0f, block_l1 = 0.0f;
#pragma unroll
            for (int tile = 0; tile < QKNt; ++tile) {
                const int col0 = tile * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const float p00 = next_m0 > -CUDART_INF_F && score[tile][0] > -CUDART_INF_F
                                      ? exp2_approx((score[tile][0] - next_m0) * Log2E)
                                      : 0.0f;
                const float p01 = next_m0 > -CUDART_INF_F && score[tile][1] > -CUDART_INF_F
                                      ? exp2_approx((score[tile][1] - next_m0) * Log2E)
                                      : 0.0f;
                const float p10 = next_m1 > -CUDART_INF_F && score[tile][2] > -CUDART_INF_F
                                      ? exp2_approx((score[tile][2] - next_m1) * Log2E)
                                      : 0.0f;
                const float p11 = next_m1 > -CUDART_INF_F && score[tile][3] > -CUDART_INF_F
                                      ? exp2_approx((score[tile][3] - next_m1) * Log2E)
                                      : 0.0f;
                block_l0 += p00 + p01;
                block_l1 += p10 + p11;
                p_s[gid * Bc + causal_small_t_tc_swz32(gid, col0)] = __float2bfloat16(p00);
                p_s[gid * Bc + causal_small_t_tc_swz32(gid, col1)] = __float2bfloat16(p01);
                p_s[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)] =
                    __float2bfloat16(p10);
                p_s[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col1)] =
                    __float2bfloat16(p11);
            }
            block_l0 = warp_sum<4>(block_l0, FullMask);
            block_l1 = warp_sum<4>(block_l1, FullMask);
            l0 = l0 * alpha0 + block_l0;
            l1 = l1 * alpha1 + block_l1;
            m0 = next_m0;
            m1 = next_m1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int worker_tid = tid - ProducerWarps * 32;
            stage_decode_value(v_s, packed_v_s, &metadata_s, tail_v, table_row, tail_slot, heads,
                               kv_head, k0, max(k0, split_start), min(k0 + Bc, split_end),
                               worker_tid, ConsumerWarps * 32);
        }
        __syncthreads();

        if (warp >= ProducerWarps) {
            const int consumer_warp = warp - ProducerWarps;
            const int output_tile = consumer_warp * PVNtPerWarp;
            const float alpha0 = alpha_s[gid];
            const float alpha1 = alpha_s[gid + 8];
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                warp_state.accumulator[n][0] *= alpha0;
                warp_state.accumulator[n][1] *= alpha0;
                warp_state.accumulator[n][2] *= alpha1;
                warp_state.accumulator[n][3] *= alpha1;
            }
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = output_tile + n;
                if (global_n >= PVNt) { continue; }
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    unsigned probability_fragment[4];
                    const int probability_col = k * 16 + a_coloff;
                    ldmatrix_x4(
                        probability_fragment[0], probability_fragment[1],
                        probability_fragment[2], probability_fragment[3],
                        smem_addr(&p_s[a_rowoff * Bc +
                                         causal_small_t_tc_swz32(a_rowoff, probability_col)]));
                    unsigned value_fragment[2];
                    const int value_row = k * 16 + b_koff + b_rin;
                    const int value_col = global_n * 8;
                    ldmatrix_x2_t(value_fragment[0], value_fragment[1],
                                  smem_addr(&v_s[value_row * D +
                                                   causal_small_t_tc_swz(value_row, value_col)]));
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

    if (warp == 0 && lid == 0) {
        const int row0 = gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            const int q_head = kv_head * Geometry::GroupSize + row0;
            partial_m[causal_partial_stat_index<Geometry>(q_head, column, split, tokens)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, column, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            const int q_head = kv_head * Geometry::GroupSize + row1;
            partial_m[causal_partial_stat_index<Geometry>(q_head, column, split, tokens)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, column, split, tokens)] = l1;
        }
    }
    if (warp >= ProducerWarps) {
        const int consumer_warp = warp - ProducerWarps;
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_warp * PVNtPerWarp + n;
            if (global_n >= PVNt) { continue; }
            const int d0 = global_n * 8 + 2 * lid;
            const int d1 = d0 + 1;
            const int row0 = gid;
            const int row1 = row0 + 8;
            if (row0 < row_count) {
                qkv_s[row0 * D + d0] = __float2bfloat16(warp_state.accumulator[n][0]);
                qkv_s[row0 * D + d1] = __float2bfloat16(warp_state.accumulator[n][1]);
            }
            if (row1 < row_count) {
                qkv_s[row1 * D + d0] = __float2bfloat16(warp_state.accumulator[n][2]);
                qkv_s[row1 * D + d1] = __float2bfloat16(warp_state.accumulator[n][3]);
            }
        }
    }
    __syncthreads();
    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d = (chunk % (D / 8)) * 8;
        const int q_head = kv_head * Geometry::GroupSize + row;
        const std::int64_t destination =
            causal_partial_acc_index<Geometry>(q_head, d, column, split, tokens);
        store_vec(partial_acc + destination, load_vec<int4>(qkv_s + row * D + d));
    }
}

template <typename Geometry, bool MultiBatch, bool Masked, bool PerTokenSplits>
__launch_bounds__(D) __global__ void reduce_output_hadamard_kernel(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* output) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch = 0;
    int token = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = static_cast<int>(threadIdx.x);
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    positions += column_begin;
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int query_position = positions[PerTokenSplits ? token : tokens - 1];
    int output_column = column_begin + token;
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    std::int64_t partial_acc_offset = 0;
    if constexpr (MultiBatch) {
        partial_acc_offset = static_cast<std::int64_t>(batch) * D * Geometry::QHeads * tokens *
                             split_count;
        const std::int64_t stat_offset =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_m += stat_offset;
        partial_l += stat_offset;
    }
    const int active_splits =
        kvarn_decode_active_splits<Geometry>(query_position + 1, split_count, tokens);

    __shared__ float stage[2][D];
    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_splits; split += D) {
        local_m = fmaxf(
            local_m, partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    stage[0][tid] = local_m;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { stage[0][tid] = fmaxf(stage[0][tid], stage[0][tid + stride]); }
        __syncthreads();
    }
    const float head_m = stage[0][0];

    float local_l = 0.0F;
    if (head_m > -CUDART_INF_F) {
        for (int split = tid; split < active_splits; split += D) {
            const float tile_l =
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l > 0.0F) {
                local_l += tile_l *
                           expf(partial_m[causal_partial_stat_index<Geometry>(q_head, token, split,
                                                                             tokens)] -
                                head_m);
            }
        }
    }
    stage[0][tid] = local_l;
    __syncthreads();
    for (int stride = D / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { stage[0][tid] += stage[0][tid + stride]; }
        __syncthreads();
    }
    const float head_l = stage[0][0];

    float numerator = 0.0F;
    if (head_l > 0.0F) {
        for (int split = 0; split < active_splits; ++split) {
            const float tile_l =
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0F) { continue; }
            const float weight = expf(
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                head_m);
            const std::int64_t index =
                partial_acc_offset +
                causal_partial_acc_index<Geometry>(q_head, tid, token, split, tokens);
            numerator += __bfloat162float(partial_acc[index]) * weight;
        }
    }
    bool valid = true;
    if constexpr (Masked) { valid = column_begin + token < valid_columns[batch]; }
    const float combined = valid && head_l > 0.0F ? numerator / head_l : 0.0F;
    float value = __bfloat162float(__float2bfloat16_rn(combined));

#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffU, value, span);
        value = (tid & span) == 0 ? value + other : other - value;
    }
    stage[0][tid] = value;
    __syncthreads();
    int current = 0;
#pragma unroll
    for (int span = 32; span < D; span <<= 1) {
        const int group = tid / (2 * span);
        const int lane = tid & (span - 1);
        const int left = group * 2 * span + lane;
        const int right = left + span;
        const float left_value = stage[current][left];
        const float right_value = stage[current][right];
        value = (tid & span) == 0 ? left_value + right_value : left_value - right_value;
        stage[current ^ 1][tid] = value;
        current ^= 1;
        __syncthreads();
    }
    output[causal_q_index<Geometry>(q_head, tid, output_column)] =
        __float2bfloat16_rn(value * 0.0625F);
}

} // namespace detail
} // namespace ninfer::ops::kvarn
