#include "hip/hip_runtime.h"
#pragma once

// ninfer::ops - split-KV GQA small-T attention, int8 KV-cache partial kernel.
// Historical design: docs/archive/optimization-era/2026-07-08-gqa-decode-int8-kernel-redesign.md.
//
//   * QK runs on native m16n8k32.s8 tensor cores. Q is quantized on-chip to int8
//     per (row, 64-group); K stays int8 in the cache and is read straight into
//     smem (no dequant). The int32 MMA output is rescaled per 64-group by
//     qs[row,g]*ks[key,g]. This halves the QK MMA count vs bf16 and removes the
//     entire K dequant.
//   * PV stays bf16 (V is quantized per key, so its scale cannot be factored out
//     of a key-contracted int8 accumulation): V int8 is staged, dequanted once to
//     a bf16 tile, then the existing bf16 PV MMA runs. V is still read from DRAM
//     as int8, so the bandwidth win is kept.
//   * All keys (history AND the current/diagonal tokens) are read from the
//     quantized cache; the fused append writes the new tokens first and a
//     __syncthreads orders the in-block readback. No from_new special-casing.
//
// Standalone from the bf16 kernel; shared scaffolding (layout constants, ldmatrix
// helpers, the s8/bf16 MMA helpers, the reducer) lives in gqa_attention_decode.cuh.

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops {

// Store one int8 code into a d-contiguous-as-b16 swizzled tile so the same
// gqa_small_t_tc_swz / ldmatrix path that serves bf16 tiles serves the int8 tile.
// A b16 lane holds two packed int8 (d even = low byte, d odd = high byte); this
// matches the byte layout a 16 B cp.async of d-contiguous cache bytes produces
// (see the design doc / kernel comments), so Q (byte stores) and K (cp.async)
// agree.
__device__ __forceinline__ void gqa_small_t_i8_store_swz(std::int8_t* tile, int row, int d,
                                                         int d_b16_stride, std::int8_t code) {
    const int c   = d >> 1;
    const int lo  = d & 1;
    const int off = (row * d_b16_stride + gqa_small_t_tc_swz(row, c)) * 2 + lo;
    tile[off]     = code;
}

// gfx1151 i8 WMMA (wmma_i32_16x16x16_iu8) fragment loads. The atom holds 16 raw
// int8 per lane (one v4i32 = WmmaA8I). In the [row][D-byte] b16-lane XOR-swizzled
// tile a 16-byte k-window (k0 multiple of 16) is contiguous at
// tile[row*D + swz(row, k0/2)*2], so each fragment is ONE 16-byte ds read. The i8
// A-fragment: participating lanes ((l&1)==((l>>4)&1)) hold row l>>1; the i8
// B-fragment: every lane holds column n = l&15. Both use k = byte index.
__device__ __forceinline__ void gqa_wmma_load_a_i8(unsigned (&frag)[4], const std::int8_t* base,
                                                   int row, int k0, int stride_bytes) {
    const int cn          = k0 >> 1;
    const unsigned addr   = smem_addr(&base[row * stride_bytes + gqa_small_t_tc_swz(row, cn) * 2]);
    wmma_ds_load_b128(frag, addr, 0);
}
__device__ __forceinline__ void gqa_wmma_load_b_i8(unsigned (&frag)[4], const std::int8_t* base,
                                                   int col, int k0, int stride_bytes) {
    const int cn          = k0 >> 1;
    const unsigned addr   = smem_addr(&base[col * stride_bytes + gqa_small_t_tc_swz(col, cn) * 2]);
    wmma_ds_load_b128(frag, addr, 0);
}

// Decode-specialized producer/consumer kernel for T=1..6. One producer warp per
// m16 row tile computes QK + online softmax, while all CTA warps partition the
// tile's 256-wide PV output. This keeps each thread's PV accumulator at 16, 32,
// or 64 floats instead of 128 and uses otherwise-idle warps for useful output
// work.
//
// Q has a dedicated shared tile so producers can reload one 64-dimension group
// at a time. K/V codes and scales are staged asynchronously; non-producer warps
// dequantize V while producers execute QK. After both consume the code tile, the
// next K/V tile is prefetched into the same arena while the current PV runs.
template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_decode_i8_tiled_kernel(
        const __hip_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float scale,
        __hip_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kGqaHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kGqaKvQuantGroups;
    constexpr int GroupKc              = kGqaKvQuantGroup / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    // The GQA Op's 262144-key maximum envelope spans at most 49 pages in one 27B split.
    constexpr int PageIds         = 64;
    constexpr int ProducerThreads = RowTiles * 32;
    constexpr int VLoaderThreads  = Threads - ProducerThreads;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned long long FullMask = 0xffffffffull;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V bf16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s      = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8     = q_s;
    float* q_scale_tmp    = reinterpret_cast<float*>(r_s);
    std::int8_t* k_i8     = r_s;
    std::int8_t* v_i8     = r_s + Bc * D;
    __hip_bfloat16* v_bf16 = reinterpret_cast<__hip_bfloat16*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) __hip_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
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
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -HIP_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
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
        // The owning split quantizes each current row before its cache tile is consumed.
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            int physical_page       = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kGqaKvQuantGroup + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            const float kv0         = __bfloat162float(input.k[src0]);
            const float kv1         = __bfloat162float(input.k[src1]);
            const float vv0         = __bfloat162float(input.v[src0]);
            const float vv1         = __bfloat162float(input.v[src1]);
            float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax                   = warp_max(kamax, FullMask);
            vamax                   = warp_max(vamax, FullMask);
            const __half ksh        = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
            const __half vsh        = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
            const float ks          = __half2float(ksh);
            const float vs          = __half2float(vsh);
            const float k_inv       = ks > 0.0f ? 1.0f / ks : 0.0f;
            const float v_inv       = vs > 0.0f ? 1.0f / vs : 0.0f;
            physical_page           = __shfl_sync(FullMask, physical_page, 0);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(kv0, k_inv);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(kv1, k_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(vv0, v_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(vv1, v_inv);
            if (lane == 0) {
                const std::int64_t so =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = ksh;
                cache_v_scale[so] = vsh;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int unit = warp; unit < RowCount * Groups; unit += Wc) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroup + lane;
        const int d1  = d0 + 32;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        const float x0  = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d0, token)]);
        const float x1  = __bfloat162float(q[gqa_q_index<Geometry>(q_head, d1, token)]);
        float amax      = fmaxf(fabsf(x0), fabsf(x1));
        amax            = warp_max(amax, FullMask);
        const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_small_t_i8_store_swz(q_i8, row, d0, DB16, gqa_kv_quant_code(x0, inv));
        gqa_small_t_i8_store_swz(q_i8, row, d1, DB16, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
    }
    __syncthreads();

    // gfx1151 WMMA atoms are 16x16 (not 16x8). QK covers Bc keys as Bc/16 column
    // atoms; PV covers D dims as D/16 column atoms; each atom carries 8 regs.
    constexpr int WQKNt         = QKNt / 2;      // Bc / 16
    constexpr int WPVNt         = D / 16;
    constexpr int WPVNtPerWarp  = PVNtPerWarp / 2;
    constexpr int I8KSteps      = D / 16;        // 16-byte QK k-steps
    constexpr int I8KPerGroup   = kGqaKvQuantGroup / 16; // 64 / 16 = 4, == I8KSteps / Groups
    static_assert(WQKNt * 16 == Bc);
    static_assert(WPVNtPerWarp * 2 == PVNtPerWarp);
    static_assert(I8KSteps == Groups * I8KPerGroup);

    const bool lane_hi = (lane >> 4) != 0;

    // Per-row query quant scale for this lane's 8 rows (rows r + 8*(l>=16)).
    float q_scale_r[8][Groups];
    if (warp < RowTiles) {
        const int producer_base = warp * 16 + (lane_hi ? 8 : 0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row_abs = producer_base + r;
            const bool valid  = row_abs < RowCount;
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                q_scale_r[r][g] = valid ? q_scale_tmp[row_abs * Groups + g] : 0.0f;
            }
        }
    }
    __syncthreads();

    float acc[WPVNtPerWarp][8];
#pragma unroll
    for (int n = 0; n < WPVNtPerWarp; ++n) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[n][r] = 0.0f; }
    }

    // Running per-row softmax state (lanes 0-15 -> rows 0..7, 16-31 -> rows 8..15).
    float m[8], l[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { m[r] = -HIP_INF_F; l[r] = 0.0f; }

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = gqa_kv_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int2(0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = gqa_kv_quant_code_index<Geometry>(
                    physical_page, kv_head, d, key & kPagedKVPageMask);
                std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                ninfer::ops::cp_async<16>(dst, &cache_k_i8[off]);
                ninfer::ops::cp_async<16>(&v_i8[key_l * D + d], &cache_v_i8[off]);
            } else {
                std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                store_vec(dst, make_int4(0, 0, 0, 0));
                store_vec(&v_i8[key_l * D + d], make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = physical_pages_s[0];
    issue_kv_tile(first_tile, physical_page);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            __hip_bfloat16* p_sw         = &p_s[producer_row_base * Bc];
            float score[WQKNt][8];
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) { score[nt][r] = 0.0f; }
            }

            // Int8 QK: one per-64-group int32 accumulation (I8KPerGroup 16-byte
            // k-steps each), rescaled by qs[row,g]*ks[key,g] into the fp32 score.
            const int arow = producer_row_base + (lane >> 1);
            unsigned a_frag[4];
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                int acc_i[WQKNt][8];
#pragma unroll
                for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) { acc_i[nt][r] = 0; }
                }
#pragma unroll
                for (int kk = 0; kk < I8KPerGroup; ++kk) {
                    const int k = g * I8KPerGroup + kk;
                    if (wmma_a_lane_active(lane)) {
                        gqa_wmma_load_a_i8(a_frag, q_i8, arow, k * 16, D);
                    }
                    WmmaA8I a = *reinterpret_cast<WmmaA8I*>(a_frag);
#pragma unroll
                    for (int nt = 0; nt < WQKNt; ++nt) {
                        unsigned b_frag[4];
                        gqa_wmma_load_b_i8(b_frag, k_i8, nt * 16 + (lane & 15), k * 16, D);
                        WmmaC8I& c = *reinterpret_cast<WmmaC8I*>(acc_i[nt]);
                        WmmaA8I b  = *reinterpret_cast<WmmaA8I*>(b_frag);
                        c          = wmma_i8(a, b, c);
                    }
                }
#pragma unroll
                for (int nt = 0; nt < WQKNt; ++nt) {
                    const float ks = __half2float(k_scale_s[(nt * 16 + (lane & 15)) * Groups + g]);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        score[nt][r] += q_scale_r[r][g] * ks * static_cast<float>(acc_i[nt][r]);
                    }
                }
            }

            // Per-query validity/absolute position, uniform across each 8-row half.
            int row_qabs[8];
            bool row_valid[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row_abs = producer_row_base + (lane_hi ? 8 : 0) + r;
                row_valid[r]      = row_abs < RowCount;
                int q_head = 0, token = 0;
                gqa_small_t_tc_row_to_qt<Geometry>(row_abs, TokenTile, kv_head, q_head, token);
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
                const int row_abs = producer_row_base + r + (lane_hi ? 8 : 0);
                alpha_s[row_abs]  = alpha[r];
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                int4 deq        = make_int4(0, 0, 0, 0);
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    float vs      = 0.0f;
                    if ((lane & 7) == 0) { vs = __half2float(v_scale_s[key_l * Groups + grp]); }
                    vs  = __shfl_sync(FullMask, vs, grp * 8);
                    deq = gqa_kv_dequant_i8x8_from(&v_i8[key_l * D + d], vs);
                }
                const __hip_bfloat16* src8 = reinterpret_cast<const __hip_bfloat16*>(&deq);
                // Bc is only 32, so keep 8-key groups inside the 32-wide row.
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int dd = d + i;
                    v_bf16[dd * Bc + gqa_small_t_tc_swz32(dd, key_l)] = src8[i];
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            }
            issue_kv_tile(next_k0, physical_page);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __hip_bfloat16* p_consumer  = &p_s[consumer_row_base * Bc];
        float alpha[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            alpha[r] = alpha_s[consumer_row_base + r + (lane_hi ? 8 : 0)];
        }
#pragma unroll
        for (int n = 0; n < WPVNtPerWarp; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[n][r] *= alpha[r]; }
        }

        unsigned pv_a[PVKs][8];
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            if (wmma_a_lane_active(lane)) {
                wmma_load_a_bf16(pv_a[k], p_consumer, lane >> 1, k * 16, Bc,
                                 gqa_small_t_tc_swz32);
            }
        }
#pragma unroll
        for (int n = 0; n < WPVNtPerWarp; ++n) {
            const int global_nw = consumer_slice * WPVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned vf[8];
                wmma_load_b_bf16(vf, v_bf16, global_nw * 16 + (lane & 15), k * 16, Bc,
                                 gqa_small_t_tc_swz32);
                WmmaC8& c  = *reinterpret_cast<WmmaC8*>(acc[n]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(pv_a[k]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(vf);
                c          = wmma_bf16(a, b, c);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    // Publish the running per-row softmax stats (one lane per 8-row half).
    if (warp < RowTiles && (lane == 0 || lane == 16)) {
        const int half_off = (lane >> 4) << 3;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row_abs = warp * 16 + half_off + r;
            if (row_abs < RowCount) {
                int q_head = 0;
                int token  = 0;
                gqa_small_t_tc_row_to_qt<Geometry>(row_abs, TokenTile, kv_head, q_head, token);
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m[r];
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l[r];
            }
        }
    }

#pragma unroll
    for (int n = 0; n < WPVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d                 = (consumer_slice * WPVNtPerWarp + n) * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row_abs = consumer_row_base + r + (lane_hi ? 8 : 0);
            if (row_abs < RowCount) {
                int q_head = 0;
                int token  = 0;
                gqa_small_t_tc_row_to_qt<Geometry>(row_abs, TokenTile, kv_head, q_head, token);
                const std::int64_t dst =
                    gqa_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile);
                partial_acc[dst] = __float2bfloat16(acc[n][r]);
            }
        }
    }

}

} // namespace ninfer::ops
