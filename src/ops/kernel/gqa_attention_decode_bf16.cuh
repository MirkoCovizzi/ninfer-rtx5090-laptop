#pragma once

// ninfer::ops - split-KV GQA small-T attention, BF16 KV-cache partial kernel.
// Standalone from the int8 kernel (gqa_attention_decode_i8.cuh): shared scaffolding
// lives in gqa_attention_decode.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/common/mma.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

struct GqaBf16PagedStorage {
    static constexpr bool kvarn = false;
    __nv_bfloat16* k;
    __nv_bfloat16* v;
};

struct GqaKvarnPagedStorage {
    static constexpr bool kvarn = true;
    const std::uint8_t* k_codes;
    const std::uint8_t* k_block_scales;
    const __half* k_channel_scales;
    const std::uint8_t* v_codes;
    const __half* v_channel_scales;
    const __half* v_token_scales;
    const __half* v_token_zeros;
    const __nv_bfloat16* tail_k;
    const __nv_bfloat16* tail_v;
    const std::int32_t* tail_logical_pages;
    std::int32_t heads;
};

__device__ __forceinline__ void gqa_kvarn_decode_sixteen(
    const GqaKvarnPagedStorage& storage, std::int64_t record, int token, int d,
    __nv_bfloat16* k_destination0, __nv_bfloat16* v_destination0,
    __nv_bfloat16* k_destination1, __nv_bfloat16* v_destination1) {
    const std::int64_t token_record = record * kPagedKVPageSize + token;
    const float k_block = detail::decode_nvfp4_e4m3(
        storage.k_block_scales[token_record * (kGqaHeadDim / 16) + d / 16]);
    const float v_scale = __half2float(storage.v_token_scales[token_record]);
    const float v_zero  = __half2float(storage.v_token_zeros[token_record]);
#pragma unroll
    for (int segment = 0; segment < 2; ++segment) {
        const int segment_d = d + 8 * segment;
        const std::uint8_t v_packed0 =
            storage.v_codes[token_record * (kGqaHeadDim / 4) + segment_d / 4];
        const std::uint8_t v_packed1 =
            storage.v_codes[token_record * (kGqaHeadDim / 4) + segment_d / 4 + 1];
        __nv_bfloat16 k_values[8];
        __nv_bfloat16 v_values[8];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int pair_d = segment_d + 2 * pair;
            const std::uint8_t k_packed =
                storage.k_codes[token_record * (kGqaHeadDim / 2) + pair_d / 2];
            const float2 k_codes = detail::decode_nvfp4_e2m1x2(k_packed);
            const float2 k_channel = __half22float2(*reinterpret_cast<const __half2*>(
                storage.k_channel_scales + record * kGqaHeadDim + pair_d));
            k_values[2 * pair] = __float2bfloat16(k_codes.x * k_block * k_channel.x);
            k_values[2 * pair + 1] = __float2bfloat16(k_codes.y * k_block * k_channel.y);

            const std::uint8_t v_packed = pair < 2 ? v_packed0 : v_packed1;
            const int shift             = 4 * (pair & 1);
            const float v_code0         = static_cast<float>((v_packed >> shift) & 3);
            const float v_code1         = static_cast<float>((v_packed >> (shift + 2)) & 3);
            const float2 v_channel = __half22float2(*reinterpret_cast<const __half2*>(
                storage.v_channel_scales + record * kGqaHeadDim + pair_d));
            v_values[2 * pair] =
                __float2bfloat16((v_code0 * v_scale + v_zero) * v_channel.x);
            v_values[2 * pair + 1] =
                __float2bfloat16((v_code1 * v_scale + v_zero) * v_channel.y);
        }
        store_vec(segment == 0 ? k_destination0 : k_destination1, load_vec<int4>(k_values));
        store_vec(segment == 0 ? v_destination0 : v_destination1, load_vec<int4>(v_values));
    }
}

__device__ __forceinline__ void gqa_kvarn_decode_key_sixteen(
    const GqaKvarnPagedStorage& storage, std::int64_t record, int token, int d,
    __nv_bfloat16* destination0, __nv_bfloat16* destination1) {
    const std::int64_t token_record = record * kPagedKVPageSize + token;
    const float block = detail::decode_nvfp4_e4m3(
        storage.k_block_scales[token_record * (kGqaHeadDim / 16) + d / 16]);
#pragma unroll
    for (int segment = 0; segment < 2; ++segment) {
        const int segment_d = d + 8 * segment;
        __nv_bfloat16 values[8];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int pair_d = segment_d + 2 * pair;
            const std::uint8_t packed =
                storage.k_codes[token_record * (kGqaHeadDim / 2) + pair_d / 2];
            const float2 codes = detail::decode_nvfp4_e2m1x2(packed);
            const float2 channel = __half22float2(*reinterpret_cast<const __half2*>(
                storage.k_channel_scales + record * kGqaHeadDim + pair_d));
            values[2 * pair]     = __float2bfloat16(codes.x * block * channel.x);
            values[2 * pair + 1] = __float2bfloat16(codes.y * block * channel.y);
        }
        store_vec(segment == 0 ? destination0 : destination1, load_vec<int4>(values));
    }
}

__device__ __forceinline__ void gqa_kvarn_decode_value_sixteen(
    const GqaKvarnPagedStorage& storage, std::int64_t record, int token, int d,
    __nv_bfloat16* destination0, __nv_bfloat16* destination1) {
    const std::int64_t token_record = record * kPagedKVPageSize + token;
    const float scale               = __half2float(storage.v_token_scales[token_record]);
    const float zero                = __half2float(storage.v_token_zeros[token_record]);
#pragma unroll
    for (int segment = 0; segment < 2; ++segment) {
        const int segment_d = d + 8 * segment;
        const std::uint8_t packed0 =
            storage.v_codes[token_record * (kGqaHeadDim / 4) + segment_d / 4];
        const std::uint8_t packed1 =
            storage.v_codes[token_record * (kGqaHeadDim / 4) + segment_d / 4 + 1];
        __nv_bfloat16 values[8];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int pair_d             = segment_d + 2 * pair;
            const std::uint8_t packed    = pair < 2 ? packed0 : packed1;
            const int shift              = 4 * (pair & 1);
            const float code0            = static_cast<float>((packed >> shift) & 3);
            const float code1            = static_cast<float>((packed >> (shift + 2)) & 3);
            const float2 channel = __half22float2(*reinterpret_cast<const __half2*>(
                storage.v_channel_scales + record * kGqaHeadDim + pair_d));
            values[2 * pair]     = __float2bfloat16((code0 * scale + zero) * channel.x);
            values[2 * pair + 1] = __float2bfloat16((code1 * scale + zero) * channel.y);
        }
        store_vec(segment == 0 ? destination0 : destination1, load_vec<int4>(values));
    }
}

// Qwen3.8-27B T=1 KVarN attention. One warp produces the six QK/softmax rows while the
// remaining warps decode V; all eight warps then partition the 256-wide PV output. This avoids
// carrying a full 256-wide accumulator in every lane and eliminates the old empty row-tile warp.
template <typename Geometry, bool MultiBatch, bool Masked>
__launch_bounds__(256, 2) __global__ void gqa_attention_decode_kvarn_t1_kernel(
    const __nv_bfloat16* q, const std::int32_t* pos, GqaKvarnPagedStorage storage,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t tokens, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(Geometry::QHeads == 24 && Geometry::KVHeads == 4 && Geometry::GroupSize == 6);

    constexpr int Wc            = 8;
    constexpr int Threads       = Wc * 32;
    constexpr int RowCount      = Geometry::GroupSize;
    constexpr int Br            = 16;
    constexpr int Bc            = 32;
    constexpr int D             = kGqaHeadDim;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNtPerWarp   = D / (Wc * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int PageIds       = 64;
    constexpr int VLoaderWarps  = Wc - 1;
    constexpr int VLoaderThreads = VLoaderWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) __nv_bfloat16 q_s[Br * D];
    __shared__ __align__(16) __nv_bfloat16 k_s[Bc * D];
    __shared__ __align__(16) __nv_bfloat16 v_s[Bc * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ std::int32_t tail_pages_s[kKvarnTailSlots];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int flat_column = static_cast<int>(blockIdx.z);
    const int batch       = MultiBatch ? flat_column / tokens : 0;
    const int column      = flat_column - batch * tokens;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;

    const int valid_tokens =
        Masked ? (valid_columns[batch] > column_begin + column ? 1 : 0) : 1;
    std::int64_t column_base = column_begin + column;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
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
        for (int row = tid; row < RowCount; row += Threads) {
            const int q_head = kv_head * Geometry::GroupSize + row;
            partial_m[gqa_partial_stat_index<Geometry>(q_head, column, split, tokens)] =
                -CUDART_INF_F;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, column, split, tokens)] = 0.0f;
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            const int q_head = kv_head * Geometry::GroupSize + row;
            partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, column, split, tokens)] =
                __float2bfloat16(0.0f);
        }
    };

    if (kv_head >= Geometry::KVHeads || valid_tokens == 0) {
        write_neutral();
        return;
    }
    const int query_pos = pos[0];
    if (query_pos < 0 || query_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = query_pos + 1;
    int active_split_count = gqa_small_t_active_splits<Geometry, false>(window, split_count, 1);
    if (window > 8198) {
        int kvarn_splits       = div_up(window, 192 / Geometry::DecodeSplitScale);
        const int cap = window <= kGqa27KvarnT1MidWindow ? kGqa27KvarnT1MidSplits
                                                          : kGqa27KvarnT1LongSplits;
        kvarn_splits = kvarn_splits < cap ? kvarn_splits : cap;
        active_split_count = kvarn_splits < split_count ? kvarn_splits : split_count;
    }
    if (split >= active_split_count) { return; }

    const int logical_tiles  = div_up(window, Bc);
    const bool tile_split    = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = split_limit < window ? split_limit : window;
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
    for (int candidate = tid; candidate < kKvarnTailSlots; candidate += Threads) {
        const std::int64_t marker =
            (static_cast<std::int64_t>(table_row) * kKvarnTailSlots + candidate) * storage.heads +
            kv_head;
        tail_pages_s[candidate] = storage.tail_logical_pages[marker];
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        const __nv_bfloat16 value =
            row < RowCount ? q[gqa_q_index<Geometry>(kv_head * Geometry::GroupSize + row, d, 0)]
                           : __float2bfloat16(0.0f);
        q_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    int physical_page = physical_pages_s[0];

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        int tail_slot             = -1;
        const int logical_page    = k0 >> kPagedKVPageShift;
#pragma unroll
        for (int candidate = 0; candidate < kKvarnTailSlots; ++candidate) {
            if (tail_pages_s[candidate] == logical_page &&
                logical_page == (query_pos >> kPagedKVPageShift)) {
                tail_slot = candidate;
            }
        }

        const std::int64_t record =
            static_cast<std::int64_t>(physical_page) * storage.heads + kv_head;
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int d     = (chunk - key_l * (D / 16)) * 16;
            const int key   = k0 + key_l;
            __nv_bfloat16* dst0 = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* dst1 = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d + 8)];
            if (key < split_start || key >= split_end) {
                store_vec(dst0, make_int4(0, 0, 0, 0));
                store_vec(dst1, make_int4(0, 0, 0, 0));
            } else if (tail_slot >= 0) {
                const std::int64_t tail =
                    (((static_cast<std::int64_t>(table_row) * kKvarnTailSlots + tail_slot) *
                          storage.heads +
                      kv_head) *
                         kPagedKVPageSize +
                     (key & kPagedKVPageMask)) *
                        D +
                    d;
                store_vec(dst0, load_vec<int4>(storage.tail_k + tail));
                store_vec(dst1, load_vec<int4>(storage.tail_k + tail + 8));
            } else {
                gqa_kvarn_decode_key_sixteen(storage, record, key & kPagedKVPageMask, d, dst0,
                                             dst1);
            }
        }
        __syncthreads();

        if (warp == 0) {
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned af[4];
                const int acol = k * 16 + a_coloff;
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(&q_s[a_rowoff * D + gqa_small_t_tc_swz(a_rowoff, acol)]));
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    unsigned bf[2];
                    const int brow = nt * 8 + b_rin;
                    const int bcol = k * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                    mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                             af[1], af[2], af[3], bf[0], bf[1]);
                }
            }

            const int row0 = gid;
            const int row1 = gid + 8;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] = (row0 < RowCount && key0 >= split_start && key0 < split_end &&
                                key0 <= query_pos)
                                   ? score[nt][0] * scale
                                   : -CUDART_INF_F;
                score[nt][1] = (row0 < RowCount && key1 >= split_start && key1 < split_end &&
                                key1 <= query_pos)
                                   ? score[nt][1] * scale
                                   : -CUDART_INF_F;
                score[nt][2] = (row1 < RowCount && key0 >= split_start && key0 < split_end &&
                                key0 <= query_pos)
                                   ? score[nt][2] * scale
                                   : -CUDART_INF_F;
                score[nt][3] = (row1 < RowCount && key1 >= split_start && key1 < split_end &&
                                key1 <= query_pos)
                                   ? score[nt][3] * scale
                                   : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * Log2E);
            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                      : 0.0f;
                const float p01 = nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                      : 0.0f;
                const float p10 = nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                      : 0.0f;
                const float p11 = nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           =
                    __float2bfloat16(p00);
                p_s[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           =
                    __float2bfloat16(p01);
                p_s[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] =
                    __float2bfloat16(p10);
                p_s[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] =
                    __float2bfloat16(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            l0  = l0 * alpha0 + bl0;
            l1  = l1 * alpha1 + bl1;
            m0  = nm0;
            m1  = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int loader_tid = tid - 32;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 16); chunk += VLoaderThreads) {
                const int key_l = chunk / (D / 16);
                const int d     = (chunk - key_l * (D / 16)) * 16;
                const int key   = k0 + key_l;
                __nv_bfloat16* dst0 = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                __nv_bfloat16* dst1 = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d + 8)];
                if (key < split_start || key >= split_end) {
                    store_vec(dst0, make_int4(0, 0, 0, 0));
                    store_vec(dst1, make_int4(0, 0, 0, 0));
                } else if (tail_slot >= 0) {
                    const std::int64_t tail =
                        (((static_cast<std::int64_t>(table_row) * kKvarnTailSlots + tail_slot) *
                              storage.heads +
                          kv_head) *
                             kPagedKVPageSize +
                         (key & kPagedKVPageMask)) *
                            D +
                        d;
                    store_vec(dst0, load_vec<int4>(storage.tail_v + tail));
                    store_vec(dst1, load_vec<int4>(storage.tail_v + tail + 8));
                } else {
                    gqa_kvarn_decode_value_sixteen(storage, record, key & kPagedKVPageMask, d,
                                                   dst0, dst1);
                }
            }
        }
        __syncthreads();

        const int consumer_n0 = warp * PVNtPerWarp;
        const float alpha0    = alpha_s[gid];
        const float alpha1    = alpha_s[gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_s[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = (consumer_n0 + n) * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
    }

    if (warp == 0 && lid == 0) {
        const int row0 = gid;
        const int row1 = gid + 8;
        if (row0 < RowCount) {
            const int q_head = kv_head * Geometry::GroupSize + row0;
            partial_m[gqa_partial_stat_index<Geometry>(q_head, column, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, column, split, tokens)] = l0;
        }
        if (row1 < RowCount) {
            const int q_head = kv_head * Geometry::GroupSize + row1;
            partial_m[gqa_partial_stat_index<Geometry>(q_head, column, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, column, split, tokens)] = l1;
        }
    }
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0   = (warp * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0 = gid;
        const int row1 = gid + 8;
        if (row0 < RowCount) {
            const int q_head = kv_head * Geometry::GroupSize + row0;
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, column, split, tokens);
            partial_acc[dst]     = __float2bfloat16(acc[n][0]);
            partial_acc[dst + 1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < RowCount) {
            const int q_head = kv_head * Geometry::GroupSize + row1;
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, column, split, tokens);
            partial_acc[dst]     = __float2bfloat16(acc[n][2]);
            partial_acc[dst + 1] = __float2bfloat16(acc[n][3]);
        }
    }
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          bool CanonicalColumns, typename CacheInput, typename CacheStorage>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_bf16_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, CacheStorage storage,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = kGqaHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The GQA Op's 262144-key maximum envelope spans at most 49 pages in one 27B split.
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int flat_column = static_cast<int>(blockIdx.z);
    const int batch       = MultiBatch ? (CanonicalColumns ? flat_column / tokens : flat_column) : 0;
    const int column      = CanonicalColumns ? flat_column - batch * tokens : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int active_tokens = CanonicalColumns ? 1 : tokens;
    int valid_tokens        = active_tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin - column;
        valid_tokens = remaining <= 0 ? 0 : (remaining < active_tokens ? remaining : active_tokens);
    }
    const int row_count = active_tokens * Geometry::GroupSize;

    std::int64_t current_base = column_begin;
    if constexpr (MultiBatch) { current_base += static_cast<std::int64_t>(batch) * full_width; }
    const std::int32_t* current_pos = pos + current_base;
    const int current_remaining = Masked ? valid_columns[batch] - column_begin : tokens;
    const int current_tokens = current_remaining <= 0
                                   ? 0
                                   : (current_remaining < tokens ? current_remaining : tokens);
    const std::int32_t current_first_pos = current_tokens > 0 ? current_pos[0] : -1;
    const std::int64_t column_base       = current_base + column;
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        static_assert(!CacheStorage::kvarn);
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * current_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * current_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, active_tokens, kv_head, q_head, token);
            const int output_token = column + token;
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, output_token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, output_token, split, tokens)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, active_tokens, kv_head, q_head, token);
            const int output_token = column + token;
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[
                    gqa_partial_acc_index<Geometry>(q_head, d, output_token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || active_tokens < 1 ||
        active_tokens > TokenTile || column < 0 || column >= tokens ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[active_tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, active_tokens);
    if constexpr (CacheStorage::kvarn) {
        if (window > 8198) {
            const int kvarn_splits = div_up(window, 192 / Geometry::DecodeSplitScale);
            active_split_count = kvarn_splits < split_count ? kvarn_splits : split_count;
        }
    }
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
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

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const int input_token = CanonicalColumns ? column + token : token;
                const std::int64_t new_off =
                    gqa_kv_new_index<Geometry>(kv_head, d, input_token);
                const int lane             = tid & 31;
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page     = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&storage.k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&storage.v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, active_tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        int kvarn_tail_slot = -1;
        if constexpr (CacheStorage::kvarn) {
            const int logical_page = k0 >> kPagedKVPageShift;
#pragma unroll
            for (int candidate = 0; candidate < kKvarnTailSlots; ++candidate) {
                const std::int64_t marker =
                    (static_cast<std::int64_t>(table_row) * kKvarnTailSlots + candidate) *
                        storage.heads +
                    kv_head;
                if (storage.tail_logical_pages[marker] == logical_page &&
                    logical_page == (last_pos >> kPagedKVPageShift)) {
                    kvarn_tail_slot = candidate;
                }
            }

        }
        // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
        // Current-step tokens come from k_new/v_new; tail slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l      = chunk / (D / 16);
            const int d          = (chunk - key_l * (D / 16)) * 16;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst0 = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst0 = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* k_dst1 = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d + 8)];
            __nv_bfloat16* v_dst1 = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d + 8)];
            if (key >= split_start && key < split_end) {
                if constexpr (CacheStorage::kvarn) {
                    const int offset    = key & kPagedKVPageMask;
                    const std::int64_t record =
                        static_cast<std::int64_t>(physical_page) * storage.heads + kv_head;
                    if (kvarn_tail_slot >= 0) {
                        const std::int64_t tail =
                            (((static_cast<std::int64_t>(table_row) * kKvarnTailSlots +
                               kvarn_tail_slot) *
                                  storage.heads +
                              kv_head) * kPagedKVPageSize +
                             offset) *
                                D +
                            d;
                        store_vec(k_dst0, load_vec<int4>(storage.tail_k + tail));
                        store_vec(v_dst0, load_vec<int4>(storage.tail_v + tail));
                        store_vec(k_dst1, load_vec<int4>(storage.tail_k + tail + 8));
                        store_vec(v_dst1, load_vec<int4>(storage.tail_v + tail + 8));
                    } else {
                        gqa_kvarn_decode_sixteen(storage, record, offset, d, k_dst0, v_dst0,
                                                 k_dst1, v_dst1);
                    }
                } else if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - current_first_pos;
                    const bool from_new = new_token >= 0 && new_token < current_tokens;
                    if (from_new) {
                        const std::int64_t off = gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                        ninfer::ops::cp_async<16>(k_dst0, &input.k[off]);
                        ninfer::ops::cp_async<16>(v_dst0, &input.v[off]);
                        ninfer::ops::cp_async<16>(k_dst1, &input.k[off + 8]);
                        ninfer::ops::cp_async<16>(v_dst1, &input.v[off + 8]);
                    } else {
                        const std::int64_t off = gqa_cache_index<Geometry>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        ninfer::ops::cp_async<16>(k_dst0, &storage.k[off]);
                        ninfer::ops::cp_async<16>(v_dst0, &storage.v[off]);
                        ninfer::ops::cp_async<16>(k_dst1, &storage.k[off + 8]);
                        ninfer::ops::cp_async<16>(v_dst1, &storage.v[off + 8]);
                    }
                } else {
                    const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                                       key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(k_dst0, &storage.k[off]);
                    ninfer::ops::cp_async<16>(v_dst0, &storage.v[off]);
                    ninfer::ops::cp_async<16>(k_dst1, &storage.k[off + 8]);
                    ninfer::ops::cp_async<16>(v_dst1, &storage.v[off + 8]);
                }
            } else {
                store_vec(k_dst0, make_int4(0, 0, 0, 0));
                store_vec(v_dst0, make_int4(0, 0, 0, 0));
                store_vec(k_dst1, make_int4(0, 0, 0, 0));
                store_vec(v_dst1, make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (!CacheStorage::kvarn) {
            ninfer::ops::cp_commit();
            ninfer::ops::cp_wait<0>();
        }
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, active_tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, active_tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                    ? score[nt][3] * scale
                    : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, active_tokens, kv_head, q_head, token);
            const int output_token = column + token;
            partial_m[gqa_partial_stat_index<Geometry>(q_head, output_token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, output_token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, active_tokens, kv_head, q_head, token);
            const int output_token = column + token;
            partial_m[gqa_partial_stat_index<Geometry>(q_head, output_token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, output_token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. Stage the final split-local
    // accumulator through shared memory so partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, active_tokens, kv_head, q_head, token);
        const int output_token = column + token;
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, output_token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ninfer::ops
