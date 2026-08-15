#include "hip/hip_runtime.h"
#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kBidirectionalGqaHeadDim  = 128;
inline constexpr int kBidirectionalGqaQHeads   = 32;
inline constexpr int kBidirectionalGqaKVHeads  = 8;
inline constexpr int kBidirectionalGqaGroup    = 4;
inline constexpr int kBidirectionalGqaMaxSplit = 85;
inline constexpr int kSwaWindow                = 4096;

__device__ __forceinline__ int bidirectional_gqa_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned bidirectional_gqa_swz_addr(unsigned lane_base, unsigned ck,
                                                               unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

__device__ __forceinline__ std::int64_t bidirectional_gqa_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kBidirectionalGqaQHeads) * token);
}

__device__ __forceinline__ std::int64_t bidirectional_gqa_query_kv_index(int kv_head, int d,
                                                                         int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kBidirectionalGqaKVHeads) * token);
}

__device__ __forceinline__ std::int64_t
bidirectional_gqa_cyclic_context_index(int kv_head, int d, int position, int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

template <int Tokens>
__device__ __forceinline__ std::int64_t bidirectional_gqa_partial_index(int q_head, int d,
                                                                        int token, int split) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kBidirectionalGqaQHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(Tokens) * split));
}

template <int Tokens>
__device__ __forceinline__ std::int64_t bidirectional_gqa_stat_index(int q_head, int token,
                                                                     int split) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(kBidirectionalGqaQHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(Tokens) * split);
}

__device__ __forceinline__ void noncausal_gqa_row_to_qt(int row, int kv_head, int& q_head,
                                                        int& token) {
    token             = row / kBidirectionalGqaGroup;
    const int q_local = row - token * kBidirectionalGqaGroup;
    q_head            = kv_head * kBidirectionalGqaGroup + q_local;
}

template <bool CyclicSwa, int KeyBlock, int Threads>
__device__ __forceinline__ void
bidirectional_gqa_stage_tile(__hip_bfloat16* dst, const __hip_bfloat16* context,
                             const __hip_bfloat16* query, int key0, int valid_keys, bool query_tile,
                             int kv_head, int context_stride, int physical_page, int tid) {
    constexpr int VecsPerRow = kBidirectionalGqaHeadDim / 8;
    constexpr int Page       = 64;
    const std::int64_t paged_base =
        static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
        ((key0 & (Page - 1)) + Page * (physical_page + context_stride * kv_head));
    for (int chunk = tid; chunk < KeyBlock * VecsPerRow; chunk += Threads) {
        const int row      = chunk / VecsPerRow;
        const int d        = (chunk - row * VecsPerRow) * 8;
        const bool live    = row < valid_keys;
        const int safe_row = live ? row : 0;
        std::int64_t src_index;
        if constexpr (CyclicSwa) {
            const int context_position = (live ? key0 + row : 0) & (kSwaWindow - 1);
            src_index = query_tile ? bidirectional_gqa_query_kv_index(kv_head, d, safe_row)
                                   : bidirectional_gqa_cyclic_context_index(
                                         kv_head, d, context_position, context_stride);
        } else {
            src_index = query_tile
                            ? bidirectional_gqa_query_kv_index(kv_head, d, safe_row)
                            : paged_base + d +
                                  static_cast<std::int64_t>(kBidirectionalGqaHeadDim) * safe_row;
        }
        const __hip_bfloat16* src = query_tile ? query + src_index : context + src_index;
        __hip_bfloat16* smem = &dst[row * kBidirectionalGqaHeadDim + bidirectional_gqa_swz(row, d)];
        cp_async_zfill<16, Cache::cg>(smem, src, live ? 16 : 0);
    }
}

// Transposed V staging (dual of bidirectional_gqa_stage_tile) for the WMMA PV
// B-operand. The B-fragment for O += P*V holds column n = d and elements k = row,
// reading v_t[d * KeyBlock + swz(d, row)] — contiguous per 8-key group. The
// untransposed [row][d] layout would scatter the fragment over one element per
// smem row, so V is stored transposed into the same v_s region.
template <bool CyclicSwa, int KeyBlock, int Threads>
__device__ __forceinline__ void
bidirectional_gqa_stage_tile_t(__hip_bfloat16* dst, const __hip_bfloat16* context,
                               const __hip_bfloat16* query, int key0, int valid_keys,
                               bool query_tile, int kv_head, int context_stride,
                               int physical_page, int tid) {
    constexpr int VecsPerRow = kBidirectionalGqaHeadDim / 8;
    constexpr int Page       = 64;
    const std::int64_t paged_base =
        static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
        ((key0 & (Page - 1)) + Page * (physical_page + context_stride * kv_head));
    for (int chunk = tid; chunk < KeyBlock * VecsPerRow; chunk += Threads) {
        const int key      = chunk / VecsPerRow;
        const int d8       = chunk - key * VecsPerRow;
        const bool live    = key < valid_keys;
        const int safe_row = live ? key : 0;
        std::int64_t src_index;
        if constexpr (CyclicSwa) {
            const int context_position = (live ? key0 + key : 0) & (kSwaWindow - 1);
            src_index = query_tile ? bidirectional_gqa_query_kv_index(kv_head, d8 * 8, safe_row)
                                   : bidirectional_gqa_cyclic_context_index(
                                         kv_head, d8 * 8, context_position, context_stride);
        } else {
            src_index = query_tile
                            ? bidirectional_gqa_query_kv_index(kv_head, d8 * 8, safe_row)
                            : paged_base + d8 * 8 +
                                  static_cast<std::int64_t>(kBidirectionalGqaHeadDim) * safe_row;
        }
        const __hip_bfloat16* src = query_tile ? query + src_index : context + src_index;
        if (live) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = d8 * 8 + i;
                dst[d * KeyBlock + bidirectional_gqa_swz(d, key)] = src[i];
            }
        }
    }
}

template <bool CyclicSwa, int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__device__ __forceinline__ void noncausal_gqa_split_partial_body(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ query_k,
    const __hip_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ context_state,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ selectors,
    const __hip_bfloat16* __restrict__ context_k, const __hip_bfloat16* __restrict__ context_v,
    const std::int32_t* __restrict__ block_tables, int context_stride, int logical_pages,
    int max_context, int split_capacity, float scale, __hip_bfloat16* __restrict__ partial_acc,
    float* __restrict__ partial_m, float* __restrict__ partial_l, __hip_bfloat16* __restrict__ out) {
    static_assert(Tokens >= 1 && Tokens <= 16);
    static_assert(WarpsPerCta == (Tokens + 3) / 4);
    static_assert(KeyBlock == 32 || KeyBlock == 64);

    constexpr int D             = kBidirectionalGqaHeadDim;
    constexpr int Wc            = WarpsPerCta;
    constexpr int Threads       = Wc * 32;
    constexpr int Br            = Wc * 16;
    constexpr int RowCount      = Tokens * kBidirectionalGqaGroup;
    // gfx1151 WMMA: atoms are 16x16, so the QK score tile has KeyBlock/16
    // n-atoms and the PV accumulator has D/16 n-atoms.
    constexpr int WQKNt         = KeyBlock / 16;
    constexpr int QKKs          = D / 16;
    constexpr int WPVNt         = D / 16;
    constexpr int PVKs          = KeyBlock / 16;
    constexpr int RowBytes      = D * static_cast<int>(sizeof(__hip_bfloat16));
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned long long FullMask = 0xffffffffull;

    static_assert(RowCount <= Br);
    static_assert(Br <= 2 * KeyBlock);
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;

    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(D) * kBidirectionalGqaQHeads * Tokens;
    constexpr std::int64_t QueryKvElements =
        static_cast<std::int64_t>(D) * kBidirectionalGqaKVHeads * Tokens;
    constexpr std::int64_t PartialElements = QueryElements;
    constexpr std::int64_t StatElements =
        static_cast<std::int64_t>(kBidirectionalGqaQHeads) * Tokens;
    q += QueryElements * batch;
    query_k += QueryKvElements * batch;
    query_v += QueryKvElements * batch;
    out += QueryElements * batch;
    partial_acc += PartialElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    const int valid = valid_columns[batch];
    if constexpr (CyclicSwa) {
        context_state += static_cast<std::int64_t>(Tokens) * batch;
        const std::int64_t lane_elements =
            static_cast<std::int64_t>(D) * context_stride * kBidirectionalGqaKVHeads;
        context_k += lane_elements * selectors[batch];
        context_v += lane_elements * selectors[batch];
    } else {
        context_state += batch;
        block_tables += static_cast<std::int64_t>(logical_pages) * selectors[batch];
    }
    const int length = context_state[0];
    if (kv_head >= kBidirectionalGqaKVHeads || split >= split_capacity || length < 0 ||
        length > max_context || valid < 1 || valid > Tokens) {
        return;
    }

    const int context_count = CyclicSwa ? min(length, kSwaWindow - 1) : length;
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }

    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const bool owns_query        = split == active_splits - 1;
    const int context_tile_count = tile_end - tile_begin;
    const int iterations         = context_tile_count + (owns_query ? 1 : 0);

    int table_group       = -1;
    int table_lane_page   = 0;
    const auto paged_page = [&](int key0) {
        const int logical_page = key0 >> 6;
        if constexpr (Tokens == 4) {
            const int group = logical_page & ~3;
            if (group != table_group) {
                const int table_index = group + lane;
                if (lane < 4 && table_index < logical_pages) {
                    table_lane_page = __ldg(block_tables + table_index);
                }
                table_group = group;
            }
            return __shfl_sync(FullMask, table_lane_page, logical_page - group);
        } else {
            const int physical_page = lane == 0 ? __ldg(block_tables + logical_page) : 0;
            return __shfl_sync(FullMask, physical_page, 0);
        }
    };
    if constexpr (!CyclicSwa && Tokens == 4) {
        if (context_tile_count > 0) {
            const int first_logical_page =
                (context_start + static_cast<int>(tile_begin) * KeyBlock) >> 6;
            const int first_group = first_logical_page & ~3;
            const int table_index = first_group + lane;
            if (lane < 4 && table_index < logical_pages) {
                table_lane_page = __ldg(block_tables + table_index);
            }
            table_group = first_group;
        }
    }

    extern __shared__ __align__(16) __hip_bfloat16 shared[];
    __hip_bfloat16* k_s = shared;
    __hip_bfloat16* v_s = shared + KeyBlock * D;

    // The two K/V buffers together hold at least Br rows. Use them once as Q staging, then retain
    // all Q MMA fragments in registers for the complete split.
    for (int chunk = tid; chunk < Br * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head = 0, token = 0;
        noncausal_gqa_row_to_qt(row, kv_head, q_head, token);
        const bool live = row < RowCount && token < valid;
        const __hip_bfloat16* src =
            q + bidirectional_gqa_q_index(live ? q_head : 0, d, live ? token : 0);
        __hip_bfloat16* dst = &shared[row * D + bidirectional_gqa_swz(row, d)];
        cp_async_zfill<16, Cache::cg>(dst, src, live ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int warp_row0 = warp * 16;
    const bool lane_hi  = (lane >> 4) != 0;
    // Retain all Q WMMA A-fragments (D/16 steps over head_dim) in registers for
    // the whole split; the shared buffer is reused for K/V each iteration, so Q
    // cannot stay resident in smem.
    unsigned af_q[QKKs][8];
#pragma unroll
    for (int ks = 0; ks < QKKs; ++ks) {
        const int arow = warp_row0 + (lane >> 1);
        if (wmma_a_lane_active(lane)) {
            wmma_load_a_bf16(af_q[ks], shared, arow, ks * 16, D, bidirectional_gqa_swz);
        }
    }
    __syncthreads();

    float acc[WPVNt][8];
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
        for (int item = 0; item < 8; ++item) { acc[n][item] = 0.0f; }
    }
    // Running row max/sum. Lanes 0-15 track rows 0..7 (register r = row r); lanes
    // 16-31 track rows 8..15 (register r = row r + 8).
    float m0[8], m1[8], l0[8], l1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        m0[r] = -HIP_INF_F;
        m1[r] = -HIP_INF_F;
        l0[r] = 0.0f;
        l1[r] = 0.0f;
    }

    auto tile_metadata = [&](int iteration, bool& is_query, int& key0, int& valid_keys) {
        is_query = iteration >= context_tile_count;
        if (is_query) {
            key0       = 0;
            valid_keys = valid;
        } else {
            key0       = context_start + (tile_begin + iteration) * KeyBlock;
            valid_keys = min(KeyBlock, length - key0);
        }
    };
    const auto tile_page = [&](bool is_query, int key0) {
        if constexpr (CyclicSwa) {
            return 0;
        } else {
            return is_query ? 0 : paged_page(key0);
        }
    };

    bool current_is_query = false;
    int current_key0      = 0;
    int current_valid     = 0;
    tile_metadata(0, current_is_query, current_key0, current_valid);
    int current_page = tile_page(current_is_query, current_key0);
    bidirectional_gqa_stage_tile<CyclicSwa, KeyBlock, Threads>(
        k_s, context_k, query_k, current_key0, current_valid, current_is_query, kv_head,
        context_stride, current_page, tid);
    cp_commit();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        cp_wait<0>();
        __syncthreads();

        bidirectional_gqa_stage_tile_t<CyclicSwa, KeyBlock, Threads>(
            v_s, context_v, query_v, current_key0, current_valid, current_is_query, kv_head,
            context_stride, current_page, tid);
        cp_commit();

        float score[WQKNt][8];
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int item = 0; item < 8; ++item) { score[nt][item] = 0.0f; }
#pragma unroll
            for (int ks = 0; ks < QKKs; ++ks) {
                unsigned bfrag[8];
                const int bcol = nt * 16 + (lane & 15);
                wmma_load_b_bf16(bfrag, k_s, bcol, ks * 16, D, bidirectional_gqa_swz);
                WmmaC8& c = *reinterpret_cast<WmmaC8*>(score[nt]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(af_q[ks]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bfrag);
                c = wmma_bf16(a, b, c);
            }
        }

        cp_wait<0>();
        __syncthreads();

        bool next_is_query = false;
        int next_key0      = 0;
        int next_valid     = 0;
        int next_page      = 0;
        if (iteration + 1 < iterations) {
            tile_metadata(iteration + 1, next_is_query, next_key0, next_valid);
            if constexpr (CyclicSwa) {
                next_page = tile_page(next_is_query, next_key0);
            } else {
                next_page =
                    !next_is_query && !current_is_query && (next_key0 >> 6) == (current_key0 >> 6)
                        ? current_page
                        : tile_page(next_is_query, next_key0);
            }
            bidirectional_gqa_stage_tile<CyclicSwa, KeyBlock, Threads>(
                k_s, context_k, query_k, next_key0, next_valid, next_is_query, kv_head,
                context_stride, next_page, tid);
            cp_commit();
        }

        // Block row-max on scaled scores; per-row max reduces across the 16 lanes
        // of the row's half (register r, WMMA C layout).
        float bm0[8], bm1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bm0[r] = -HIP_INF_F; bm1[r] = -HIP_INF_F; }
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int brow      = r + (lane_hi ? 8 : 0);
                const int bkey      = nt * 16 + (lane & 15);
                const int token     = brow < RowCount ? brow / kBidirectionalGqaGroup : 0;
                const bool row_live = brow < RowCount && token < valid;
                const int q_pos     = CyclicSwa ? context_state[token] : 0;
                const bool allow    = row_live && bkey < current_valid &&
                                      (!CyclicSwa || current_is_query ||
                                       current_key0 + bkey >= q_pos - 4095);
                const float s = allow ? score[nt][r] * scale : -HIP_INF_F;
                score[nt][r]  = s;
                bm0[r]        = fmaxf(bm0[r], s);
                bm1[r]        = fmaxf(bm1[r], s);
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            bm0[r] = warp_max<16>(bm0[r], FullMask);
            bm1[r] = warp_max<16>(bm1[r], FullMask);
        }

        float nm0[8], nm1[8], alpha0[8], alpha1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            nm0[r]    = fmaxf(m0[r], bm0[r]);
            nm1[r]    = fmaxf(m1[r], bm1[r]);
            alpha0[r] = m0[r] == -HIP_INF_F ? 0.0f : exp2_approx((m0[r] - nm0[r]) * Log2E);
            alpha1[r] = m1[r] == -HIP_INF_F ? 0.0f : exp2_approx((m1[r] - nm1[r]) * Log2E);
        }

        unsigned pv_a[PVKs][8];
        float bl0[8], bl1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bl0[r] = 0.0f; bl1[r] = 0.0f; }
        const int arow = lane >> 1;
        const int a_rr = arow & 7;
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
            float p[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float s  = score[nt][r];
                const float nm = lane_hi ? nm1[r] : nm0[r];
                p[r] = (s > -HIP_INF_F) ? exp2_approx((s - nm) * Log2E) : 0.0f;
                bl0[r] += p[r];
                bl1[r] += p[r];
            }
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                // A-fragment element (row q, key i) lives in lane (i & 15) +
                // 16 * (q >= 8) at register (q & 7) of the score C fragment.
                const int src0 = 2 * jj + (lane_hi ? 16 : 0);
                const int src1 = 2 * jj + 1 + (lane_hi ? 16 : 0);
                const float e0 = __shfl_sync(FullMask, p[a_rr], src0);
                const float e1 = __shfl_sync(FullMask, p[a_rr], src1);
                pv_a[nt][jj]   = pack_bf16x2(e0, e1);
            }
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            bl0[r] = warp_sum<16>(bl0[r], FullMask);
            bl1[r] = warp_sum<16>(bl1[r], FullMask);
            l0[r]  = l0[r] * alpha0[r] + bl0[r];
            l1[r]  = l1[r] * alpha1[r] + bl1[r];
            m0[r]  = nm0[r];
            m1[r]  = nm1[r];
        }
#pragma unroll
        for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[n][r] *= (lane_hi ? alpha1[r] : alpha0[r]);
            }
        }

        // O += P * V, contracting over the KeyBlock rows (16 per step).
#pragma unroll
        for (int kk = 0; kk < PVKs; ++kk) {
            WmmaA16I a = *reinterpret_cast<WmmaA16I*>(pv_a[kk]);
#pragma unroll
            for (int n = 0; n < WPVNt; ++n) {
                unsigned bfrag[8];
                const int dcol = n * 16 + (lane & 15);
                wmma_load_b_bf16(bfrag, v_s, dcol, kk * 16, KeyBlock, bidirectional_gqa_swz);
                WmmaC8& c = *reinterpret_cast<WmmaC8*>(acc[n]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bfrag);
                c         = wmma_bf16(a, b, c);
            }
        }

        current_is_query = next_is_query;
        current_key0     = next_key0;
        current_valid    = next_valid;
        current_page     = next_page;
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        l0[r] = warp_sum<16>(l0[r], FullMask);
        l1[r] = warp_sum<16>(l1[r], FullMask);
    }

    if constexpr (!DirectOutput) {
        // Each row writes its own running max/sum to the partial stats. Register
        // r of half0/half1 carries row r / row 8+r; lane (r + 16*half) stores it.
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if ((lane & 15) == r) {
                const int brow = r + (lane_hi ? 8 : 0);
                if (brow < RowCount) {
                    int q_head = 0, token = 0;
                    noncausal_gqa_row_to_qt(brow, kv_head, q_head, token);
                    const float mm = lane_hi ? m1[r] : m0[r];
                    const float ll = lane_hi ? l1[r] : l0[r];
                    partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)] = mm;
                    partial_l[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)] = ll;
                }
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            l0[r] = l0[r] > 0.0f ? 1.0f / l0[r] : 0.0f;
            l1[r] = l1[r] > 0.0f ? 1.0f / l1[r] : 0.0f;
        }
    }

#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
        const int d = n * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int brow = r + (lane_hi ? 8 : 0);
            if (brow < RowCount) {
                int q_head = 0, token = 0;
                noncausal_gqa_row_to_qt(brow, kv_head, q_head, token);
                if constexpr (DirectOutput) {
                    const auto dst = bidirectional_gqa_q_index(q_head, d, token);
                    out[dst] = __float2bfloat16_rn(
                        acc[n][r] * (lane_hi ? l1[r] : l0[r]));
                } else {
                    const auto dst =
                        bidirectional_gqa_partial_index<Tokens>(q_head, d, token, split);
                    partial_acc[dst] = __float2bfloat16_rn(acc[n][r]);
                }
            }
        }
    }
}
template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void bidirectional_gqa_split_partial_kernel(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ query_k,
    const __hip_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ context_length,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const __hip_bfloat16* __restrict__ context_k, const __hip_bfloat16* __restrict__ context_v,
    const std::int32_t* __restrict__ block_tables, int physical_pages, int logical_pages,
    int max_context, int split_capacity, float scale, __hip_bfloat16* __restrict__ partial_acc,
    float* __restrict__ partial_m, float* __restrict__ partial_l, __hip_bfloat16* __restrict__ out) {
    noncausal_gqa_split_partial_body<false, Tokens, WarpsPerCta, KeyBlock, DirectOutput>(
        q, query_k, query_v, context_length, valid_columns, table_rows, context_k, context_v,
        block_tables, physical_pages, logical_pages, max_context, split_capacity, scale,
        partial_acc, partial_m, partial_l, out);
}

template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void swa_split_partial_kernel(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ query_k,
    const __hip_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __hip_bfloat16* __restrict__ context_k, const __hip_bfloat16* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    __hip_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, __hip_bfloat16* __restrict__ out) {
    noncausal_gqa_split_partial_body<true, Tokens, WarpsPerCta, KeyBlock, DirectOutput>(
        q, query_k, query_v, positions, valid_columns, lanes, context_k, context_v, nullptr,
        padded_context, 0, max_context, split_capacity, scale, partial_acc, partial_m, partial_l,
        out);
}

template <bool CyclicSwa, int Tokens, int KeyBlock>
__device__ __forceinline__ void
noncausal_gqa_reduce_body(const __hip_bfloat16* __restrict__ partial_acc,
                          const float* __restrict__ partial_m, const float* __restrict__ partial_l,
                          const std::int32_t* __restrict__ context_state,
                          const std::int32_t* __restrict__ valid_columns, int max_context,
                          int split_capacity, __hip_bfloat16* __restrict__ out) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int tid    = static_cast<int>(threadIdx.x);
    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(kBidirectionalGqaHeadDim) * kBidirectionalGqaQHeads * Tokens;
    constexpr std::int64_t StatElements =
        static_cast<std::int64_t>(kBidirectionalGqaQHeads) * Tokens;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    out += QueryElements * batch;
    if constexpr (CyclicSwa) {
        context_state += static_cast<std::int64_t>(Tokens) * batch;
    } else {
        context_state += batch;
    }
    const int length = context_state[0];
    if (q_head >= kBidirectionalGqaQHeads || token >= Tokens) { return; }
    if (length < 0 || length > max_context || token >= valid_columns[batch]) {
        if (tid < kBidirectionalGqaHeadDim) {
            out[bidirectional_gqa_q_index(q_head, tid, token)] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int context_count = CyclicSwa ? min(length, kSwaWindow - 1) : length;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    __shared__ float reduce[128];

    float local_m = -HIP_INF_F;
    for (int split = tid; split < active_splits; split += blockDim.x) {
        local_m =
            fmaxf(local_m, partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)]);
    }
    reduce[tid] = local_m;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float global_m = reduce[0];
    __syncthreads();

    float local_l = 0.0f;
    for (int split = tid; split < active_splits; split += blockDim.x) {
        const auto idx = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
        local_l += partial_l[idx] * expf(partial_m[idx] - global_m);
    }
    reduce[tid] = local_l;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float global_l = reduce[0];

    if (tid < kBidirectionalGqaHeadDim) {
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            const auto stat    = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
            const float weight = expf(partial_m[stat] - global_m);
            numerator += __bfloat162float(partial_acc[bidirectional_gqa_partial_index<Tokens>(
                             q_head, tid, token, split)]) *
                         weight;
        }
        const float value = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[bidirectional_gqa_q_index(q_head, tid, token)] = __float2bfloat16(value);
    }
}

template <int Tokens, int KeyBlock>
__launch_bounds__(128, 2) __global__
    void bidirectional_gqa_reduce_kernel(const __hip_bfloat16* __restrict__ partial_acc,
                                         const float* __restrict__ partial_m,
                                         const float* __restrict__ partial_l,
                                         const std::int32_t* __restrict__ context_length,
                                         const std::int32_t* __restrict__ valid_columns,
                                         int max_context, int split_capacity,
                                         __hip_bfloat16* __restrict__ out) {
    noncausal_gqa_reduce_body<false, Tokens, KeyBlock>(partial_acc, partial_m, partial_l,
                                                       context_length, valid_columns, max_context,
                                                       split_capacity, out);
}

template <int Tokens, int KeyBlock, int WarpsPerBlock>
__launch_bounds__(WarpsPerBlock * 32, 2) __global__
    void swa_reduce_kernel(const __hip_bfloat16* __restrict__ partial_acc,
                           const float* __restrict__ partial_m, const float* __restrict__ partial_l,
                           const std::int32_t* __restrict__ positions,
                           const std::int32_t* __restrict__ valid_columns, int max_context,
                           int split_capacity, __hip_bfloat16* __restrict__ out) {
    static_assert(WarpsPerBlock >= 1 && WarpsPerBlock <= 8);
    constexpr int MaxSplits = 128;
    constexpr unsigned long long Mask = 0xffffffffull;
    __shared__ float weights[WarpsPerBlock][MaxSplits];

    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int batch      = static_cast<int>(blockIdx.z);
    const int output_row = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const int token      = output_row / kBidirectionalGqaQHeads;
    const int q_head     = output_row - token * kBidirectionalGqaQHeads;
    if (warp >= WarpsPerBlock || token >= Tokens) return;

    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(kBidirectionalGqaHeadDim) * kBidirectionalGqaQHeads * Tokens;
    constexpr std::int64_t StatElements =
        static_cast<std::int64_t>(kBidirectionalGqaQHeads) * Tokens;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    out += QueryElements * batch;

    const int length = positions[0];
    if (length < 0 || length > max_context || token >= valid_columns[batch]) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d                                      = lane + item * 32;
            out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int context_count = min(length, kSwaWindow - 1);
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;

    float local_m = -HIP_INF_F;
    for (int split = lane; split < active_splits; split += 32) {
        local_m =
            fmaxf(local_m, partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)]);
    }
    const float global_m = warp_max<32>(local_m, Mask);

    float local_l = 0.0f;
    for (int split = lane; split < active_splits; split += 32) {
        const auto stat      = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
        const float weight   = expf(partial_m[stat] - global_m);
        weights[warp][split] = weight;
        local_l += partial_l[stat] * weight;
    }
    const float global_l = warp_sum<32>(local_l, Mask);
    __syncwarp(Mask);

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d     = lane + item * 32;
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            numerator +=
                __bfloat162float(
                    partial_acc[bidirectional_gqa_partial_index<Tokens>(q_head, d, token, split)]) *
                weights[warp][split];
        }
        const float value = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(value);
    }
}

} // namespace ninfer::ops
