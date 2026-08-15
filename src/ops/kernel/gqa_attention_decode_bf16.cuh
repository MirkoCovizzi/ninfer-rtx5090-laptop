#include "hip/hip_runtime.h"
#pragma once

// ninfer::ops - split-KV GQA small-T attention, BF16 KV-cache partial kernel.
// Standalone from the int8 kernel (gqa_attention_decode_i8.cuh): shared scaffolding
// lives in gqa_attention_decode.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_bf16_kernel(
    const __hip_bfloat16* q, CacheInput input, const std::int32_t* pos, __hip_bfloat16* cache_k,
    __hip_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __hip_bfloat16* partial_acc, float* partial_m, float* partial_l) {
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
    constexpr unsigned long long FullMask = 0xffffffffull;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __hip_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __hip_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __hip_bfloat16* k_s = qkv_s;
    __hip_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
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
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -HIP_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
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
                const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                const int lane             = tid & 31;
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                physical_page     = __shfl_sync(FullMask, physical_page, 0);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __hip_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    // gfx1151 WMMA: atoms are 16x16 (not 16x8). QK covers the warp's 16 rows x 32
    // keys as Bc/16 = 2 column atoms; PV covers 16 rows x 256 dims as D/16 = 16
    // column atoms; each atom carries 8 accumulator registers.
    constexpr int WQKNt = QKNt / 2; // Bc / 16
    constexpr int WPVNt = PVNt / 2; // D / 16
    static_assert(WQKNt * 16 == Bc);
    static_assert(WPVNt * 16 == D);

    const bool lane_hi    = (lane >> 4) != 0;
    const int warp_row0   = warp * 16;
    __hip_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    // Q A-fragments are captured here because the key loop overwrites the Q rows
    // of qkv_s with the K tile (the original kernel preloaded af_q the same way);
    // the QK MMAs below reference these registers.
    unsigned a_frag[QKKs][8];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + (lane >> 1);
        if (wmma_a_lane_active(lane)) {
            wmma_load_a_bf16(a_frag[k], qkv_s, arow, k * 16, D, gqa_small_t_tc_swz);
        }
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[WPVNt][8];
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[n][r] = 0.0f; }
    }
    // Running per-row softmax state. Lanes 0-15 track rows warp_row0+0..7 (register
    // r = row r), lanes 16-31 track rows warp_row0+8..15 (register r = row r + 8).
    float m[8], l[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { m[r] = -HIP_INF_F; l[r] = 0.0f; }

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Stage the bf16 K tile with a 16B cp.async wave (k_s stays [key_l][D]
        // swizzled); V is staged transposed as v_s[d][key] so the WMMA PV
        // B-fragments read contiguous 8-key groups. Current-step tokens come from
        // k_new/v_new; tail slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            const int key   = k0 + key_l;
            __hip_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    const std::int64_t off =
                        from_new ? gqa_kv_new_index<Geometry>(kv_head, d, new_token)
                                 : gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                             key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(k_dst, from_new ? &input.k[off] : &cache_k[off]);
                } else {
                    const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                                       key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
            }
        }
        // V transposed scatter: v_s[d * Bc + swz(d, key)]. Each 8-wide d source
        // lands on 8 distinct swizzled slots of the d-row, so it is fanned out
        // one bf16 at a time (mirrors the prefill BF16 kernel's stage_kv_t).
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            const int key   = k0 + key_l;
            const __hip_bfloat16* src = nullptr;
            if (key >= split_start && key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    const std::int64_t off =
                        from_new ? gqa_kv_new_index<Geometry>(kv_head, d, new_token)
                                 : gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                             key & kPagedKVPageMask);
                    src = from_new ? &input.v[off] : &cache_v[off];
                } else {
                    const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                                       key & kPagedKVPageMask);
                    src = &cache_v[off];
                }
            }
            const bool valid = src != nullptr;
            // Bc is only 32 here, so the swizzle must keep its 8-key groups inside
            // the 32-wide row (gqa_small_t_tc_swz32); the 64-range swizzle would
            // spill past the row end.
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int dd         = d + i;
                __hip_bfloat16* v_dst = &v_s[dd * Bc + gqa_small_t_tc_swz32(dd, key_l)];
                *v_dst                = valid ? src[i] : __hip_bfloat16(0);
            }
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        // S = Q K^T for this warp's 16 rows over all Bc keys. With 16x16 WMMA the
        // tile splits into WQKNt column atoms; each atom's C fragment holds one
        // column per lane across 8 rows (lanes 0-15 / 16-31 = rows 0-7 / 8-15).
        float score[WQKNt][8];
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { score[nt][r] = 0.0f; }
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned b_frag[8];
                wmma_load_b_bf16(b_frag, k_s, nt * 16 + (lane & 15), k * 16, D,
                                 gqa_small_t_tc_swz);
                WmmaC8& c  = *reinterpret_cast<WmmaC8*>(score[nt]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag[k]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag);
                c          = wmma_bf16(a, b, c);
            }
        }

        // Per-query validity/absolute position, uniform across the 16 lanes of an
        // 8-row half (each of those lanes holds the same rows at different cols).
        int row_qabs[8];
        bool row_valid[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row_abs = warp_row0 + (lane_hi ? 8 : 0) + r;
            row_valid[r]      = row_abs < row_count;
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row_abs, tokens, kv_head, q_head, token);
            row_qabs[r] = row_valid[r] ? pos[token] : -1;
        }

        float bm[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bm[r] = -HIP_INF_F; }
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int key  = k0 + nt * 16 + (lane & 15);
                const bool keep = row_valid[r] && key >= split_start && key < split_end &&
                                  key <= row_qabs[r];
                score[nt][r] = keep ? score[nt][r] * scale : -HIP_INF_F;
                bm[r]        = fmaxf(bm[r], score[nt][r]);
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) { bm[r] = warp_max<16>(bm[r], FullMask); }

        float nm[8], alpha[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            nm[r]    = fmaxf(m[r], bm[r]);
            alpha[r] = (m[r] == -HIP_INF_F) ? 0.0f : exp2_approx((m[r] - nm[r]) * Log2E);
        }

        float bl[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bl[r] = 0.0f; }
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row_l = r + (lane_hi ? 8 : 0);
                const int col   = nt * 16 + (lane & 15);
                const float p   = (nm[r] > -HIP_INF_F && score[nt][r] > -HIP_INF_F)
                                      ? exp2_approx((score[nt][r] - nm[r]) * Log2E)
                                      : 0.0f;
                bl[r] += p;
                p_sw[row_l * Bc + gqa_small_t_tc_swz32(row_l, col)] = __float2bfloat16(p);
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            bl[r] = warp_sum<16>(bl[r], FullMask);
            l[r]  = l[r] * alpha[r] + bl[r];
            m[r]  = nm[r];
        }
#pragma unroll
        for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[n][r] *= alpha[r]; }
        }
        __syncwarp();

        // O += P V, contracting over the Bc keys. P (A-operand) is read from the
        // swizzled p_sw tile; V is read from the transposed v_s[d][key] tile.
        unsigned pv_a[PVKs][8];
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            if (wmma_a_lane_active(lane)) {
                wmma_load_a_bf16(pv_a[k], p_sw, lane >> 1, k * 16, Bc, gqa_small_t_tc_swz32);
            }
        }
#pragma unroll
        for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned vf[8];
                wmma_load_b_bf16(vf, v_s, n * 16 + (lane & 15), k * 16, Bc,
                                 gqa_small_t_tc_swz32);
                WmmaC8& c  = *reinterpret_cast<WmmaC8*>(acc[n]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(pv_a[k]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(vf);
                c          = wmma_bf16(a, b, c);
            }
        }
        __syncthreads();
    }
    // Publish the running per-row softmax stats. Each 8-row half's rows are uniform
    // across its 16 lanes after the per-iteration warp reductions, so one lane of
    // each half writes all eight, and the row-to-(q_head, token) map diverges per row.
    if (lane == 0 || lane == 16) {
        const int half_off = (lane >> 4) << 3;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row_abs = warp_row0 + half_off + r;
            if (row_abs < row_count) {
                int q_head = 0;
                int token  = 0;
                gqa_small_t_tc_row_to_qt<Geometry>(row_abs, tokens, kv_head, q_head, token);
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m[r];
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l[r];
            }
        }
    }

    // WMMA C-fragments hold one output column per lane across 8 rows. Stage the
    // final split-local accumulator through shared memory so partial_acc is
    // written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
        const int d = n * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row_abs = warp_row0 + r + (lane_hi ? 8 : 0);
            if (row_abs < row_count) {
                qkv_s[row_abs * D + d] = __float2bfloat16(acc[n][r]);
            }
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ninfer::ops
