#include "hip/hip_runtime.h"
#pragma once

// INT8-native GQA prompt kernel for the registered Qwen3.6 head geometries. QK stays INT8 through
// m16n8k32.s8 Tensor Cores; V alone is dequantized with packed FP16 arithmetic while
// producer warps execute QK. Sixteen warps split each 16-row FP16 PV output across
// four 64-dimension slices.

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillI8Warps      = 16;
inline constexpr int kGqaPrefillI8Threads    = kGqaPrefillI8Warps * 32;
inline constexpr int kGqaPrefillI8Br         = 64;
inline constexpr int kGqaPrefillI8Bc         = 64;
inline constexpr int kGqaPrefillI8Groups     = kGqaPrefillHeadDim / kGqaKvQuantGroup;
inline constexpr int kGqaPrefillI8DB16       = kGqaPrefillHeadDim / 2;
inline constexpr int kGqaPrefillI8RowTiles   = kGqaPrefillI8Br / 16;
inline constexpr int kGqaPrefillI8DConsumers = kGqaPrefillI8Warps / kGqaPrefillI8RowTiles;

inline constexpr int kGqaPrefillI8QBytes = kGqaPrefillI8Br * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI8QScaleBytes =
    kGqaPrefillI8Br * kGqaPrefillI8Groups * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillI8KBytes = kGqaPrefillI8Bc * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI8VBytes = kGqaPrefillI8Bc * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI8VStageBytes =
    kGqaPrefillI8Bc * kGqaPrefillHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI8PBytes =
    kGqaPrefillI8Br * kGqaPrefillI8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI8ScaleBytes =
    2 * kGqaPrefillI8Bc * kGqaPrefillI8Groups * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI8StatsBytes =
    2 * kGqaPrefillI8Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillI8SmemBytes = kGqaPrefillI8QBytes + kGqaPrefillI8QScaleBytes +
                                              kGqaPrefillI8KBytes + kGqaPrefillI8VBytes +
                                              kGqaPrefillI8VStageBytes + kGqaPrefillI8PBytes +
                                              kGqaPrefillI8ScaleBytes + kGqaPrefillI8StatsBytes;

static_assert(kGqaPrefillI8Groups == 4);
static_assert(kGqaPrefillI8DConsumers == 4);
static_assert(kGqaPrefillI8SmemBytes == 92672);

__device__ __forceinline__ void gqa_prefill_i8_store_swz(std::int8_t* tile, int row, int d,
                                                         std::int8_t code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * kGqaPrefillI8DB16 + gqa_prefill_swz(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

__device__ __forceinline__ int gqa_prefill_i8_p_swz(int row, int col) {
    if constexpr (kGqaPrefillI8Bc == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return gqa_prefill_swz(row, col);
}

__device__ __forceinline__ int4 gqa_prefill_i8_dequant_f16x8(const std::int8_t* codes8,
                                                             __half scale) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    const __half2 s2     = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Eight independent quantization units per CTA; one warp owns one
// (token, kv_head, 64-d group), with two dimensions per lane.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_i8_kernel(const __hip_bfloat16* __restrict__ k,
                                              const __hip_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, std::int8_t* __restrict__ cache_k,
                                              std::int8_t* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned long long FullMask = 0xffffffffull;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaPrefillI8Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaPrefillI8Groups;
    const int tmp                   = unit / kGqaPrefillI8Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    const int d0                    = group * kGqaKvQuantGroup + lane;
    const int d1                    = d0 + 32;

    const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
    const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, d1, token);
    const float k0          = __bfloat162float(k[src0]);
    const float k1          = __bfloat162float(k[src1]);
    const float v0          = __bfloat162float(v[src0]);
    const float v1          = __bfloat162float(v[src1]);

    float k_abs = fmaxf(fabsf(k0), fabsf(k1));
    float v_abs = fmaxf(fabsf(v0), fabsf(v1));
    k_abs       = warp_max(k_abs, FullMask);
    v_abs       = warp_max(v_abs, FullMask);

    const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 127.0f : 0.0f);
    const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / 127.0f : 0.0f);
    const float ks   = __half2float(ksh);
    const float vs   = __half2float(vsh);
    const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
    page             = __shfl_sync(FullMask, page, 0);

    const std::int64_t code_base =
        gqa_kv_quant_code_index<Geometry>(page, kv_head, group * kGqaKvQuantGroup, page_off);
    cache_k[code_base + lane]      = gqa_kv_quant_code(k0, kinv);
    cache_k[code_base + lane + 32] = gqa_kv_quant_code(k1, kinv);
    cache_v[code_base + lane]      = gqa_kv_quant_code(v0, vinv);
    cache_v[code_base + lane + 32] = gqa_kv_quant_code(v1, vinv);
    if (lane == 0) {
        const std::int64_t scale_off =
            gqa_kv_quant_scale_index<Geometry>(page, kv_head, group, page_off);
        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

// Large appends are scheduled in absolute eight-token tiles. Eight divides P=64, so each CTA is
// page-local while an unknown base offset costs at most one empty tail CTA in the launch envelope.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_i8_page_kernel(
    const __hip_bfloat16* __restrict__ k, const __hip_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::int8_t* __restrict__ cache_k, std::int8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned long long FullMask = 0xffffffffull;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int group             = static_cast<int>(blockIdx.z);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) { return; }

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;

    const int token  = token_begin + warp;
    const bool valid = token < token_end;
    const int d0     = group * kGqaKvQuantGroup + lane;
    const int d1     = d0 + 32;
    float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, d1, token);
        k0                      = __bfloat162float(k[src0]);
        k1                      = __bfloat162float(k[src1]);
        v0                      = __bfloat162float(v[src0]);
        v1                      = __bfloat162float(v[src1]);
    }
    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    const __half ksh  = __float2half_rn(k_abs > 0.0f ? k_abs / 127.0f : 0.0f);
    const __half vsh  = __float2half_rn(v_abs > 0.0f ? v_abs / 127.0f : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv  = vs > 0.0f ? 1.0f / vs : 0.0f;
    physical_page     = __shfl_sync(FullMask, physical_page, 0);
    if (!valid) { return; }

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;
    const std::int64_t code_base =
        paged_kv_page_head_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head) +
        static_cast<std::int64_t>(page_off) * kGqaKvQuantHeadDim + group * kGqaKvQuantGroup;
    cache_k[code_base + lane]      = gqa_kv_quant_code(k0, kinv);
    cache_k[code_base + lane + 32] = gqa_kv_quant_code(k1, kinv);
    cache_v[code_base + lane]      = gqa_kv_quant_code(v0, vinv);
    cache_v[code_base + lane + 32] = gqa_kv_quant_code(v1, vinv);
    if (lane == 0) {
        const std::int64_t scale_offset =
            paged_kv_page_head_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page,
                                                                            kv_head) +
            static_cast<std::int64_t>(page_off) * kGqaKvQuantGroups + group;
        scale_k[scale_offset] = ksh;
        scale_v[scale_offset] = vsh;
    }
}

template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_i8_kernel(
    const __hip_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __hip_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillI8Br;
    constexpr int Bc            = kGqaPrefillI8Bc;
    constexpr int DB16          = kGqaPrefillI8DB16;
    constexpr int Groups        = kGqaPrefillI8Groups;
    constexpr int GroupKc       = kGqaKvQuantGroup / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kGqaPrefillI8DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kGqaPrefillI8RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillI8Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned long long FullMask = 0xffffffffull;

    static_assert(GroupKc == 2);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kGqaPrefillI8QBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       kGqaPrefillI8QScaleBytes);
    std::int8_t* v_i8 = k_i8 + kGqaPrefillI8KBytes;
    __half* v_f16     = reinterpret_cast<__half*>(v_i8 + kGqaPrefillI8VBytes);
    __half* p_s       = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                  kGqaPrefillI8VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kGqaPrefillI8PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    __hip_bfloat16* q_b16 = reinterpret_cast<__hip_bfloat16*>(q_i8);
    __hip_bfloat16* k_b16 = reinterpret_cast<__hip_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kGqaPrefillI8Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One warp owns one (row, 64-d group) at a time.
    for (int unit = warp; unit < Br * Groups; unit += kGqaPrefillI8Warps) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroup + lane;
        const int d1  = d0 + 32;
        float x0      = 0.0f;
        float x1      = 0.0f;
        if (row < tile_rows) {
            x0 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row)]);
            x1 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d1, q0 + row)]);
        }
        float absmax    = fmaxf(fabsf(x0), fabsf(x1));
        absmax          = warp_max(absmax, FullMask);
        const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_prefill_i8_store_swz(q_i8, row, d0, gqa_kv_quant_code(x0, inv));
        gqa_prefill_i8_store_swz(q_i8, row, d1, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale[row * Groups + grp] = qs; }
    }
    __syncthreads();

    auto issue_kv_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kGqaPrefillI8Threads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * Groups];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
                store_vec(vd, make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += kGqaPrefillI8Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            std::int8_t* kd = &k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, dc * 8)) * 2];
            std::int8_t* vd = &v_i8[key_l * D + d];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d, key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                cp_async<16, Cache::cg>(vd, &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    issue_kv_tile(0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int warp_row0 = warp * 16;
    const bool lane_hi  = (lane >> 4) != 0;
    // gfx1151 WMMA: QK uses i8 16x16x16 atoms (D/16 contraction steps grouped in
    // 64-wide quant groups); PV uses f16 atoms. Per-warp tiles: WQKNt = Bc/16 key
    // atoms; WPVNtPerWarp = (D/DConsumers)/16 output atoms.
    constexpr int WQKNt       = Bc / 16;
    constexpr int WGroupKKs   = kGqaKvQuantGroup / 16; // 4 i8 steps per quant group
    constexpr int WPVNtPerWarp = (D / kGqaPrefillI8DConsumers) / 16;

    float acc[WPVNtPerWarp][8];
#pragma unroll
    for (int n = 0; n < WPVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
    }
    // Running row max/sum. Lanes 0-15 track rows 0..7 (register r = row r); lanes
    // 16-31 track rows 8..15 (register r = row r + 8).
    float running_m0[8], running_m1[8], running_l0[8], running_l1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        running_m0[r] = -HIP_INF_F;
        running_m1[r] = -HIP_INF_F;
        running_l0[r] = 0.0f;
        running_l1[r] = 0.0f;
    }
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float scoref[WQKNt][8];
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) { scoref[nt][i] = 0.0f; }
            }

            // INT8 QK over D=256 (16 i8 steps), accumulated per 64-wide quant group
            // then scaled by q_scale * k_scale for that group.
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                WmmaC8I rawi32[WQKNt];
#pragma unroll
                for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { rawi32[nt][i] = 0; }
                }
#pragma unroll
                for (int kk = 0; kk < WGroupKKs; ++kk) {
                    const int d0 = grp * kGqaKvQuantGroup + kk * 16;
                    unsigned af[4] = {0u, 0u, 0u, 0u};
                    if (wmma_a_lane_active(lane)) {
                        const int arow = row_base + (lane >> 1);
                        const unsigned aa =
                            smem_addr(&q_i8[arow * D + 2 * gqa_prefill_swz(arow, d0 >> 1)]);
                        wmma_ds_load_b128(af, aa, 0);
                    }
#pragma unroll
                    for (int nt = 0; nt < WQKNt; ++nt) {
                        unsigned bf[4];
                        const int key = nt * 16 + (lane & 15);
                        const unsigned bb =
                            smem_addr(&k_i8[key * D + 2 * gqa_prefill_swz(key, d0 >> 1)]);
                        wmma_ds_load_b128(bf, bb, 0);
                        WmmaA8I a = *reinterpret_cast<WmmaA8I*>(af);
                        WmmaA8I b = *reinterpret_cast<WmmaA8I*>(bf);
                        rawi32[nt] = wmma_i8(a, b, rawi32[nt]);
                    }
                }
#pragma unroll
                for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int brow = r + (lane_hi ? 8 : 0);
                        const float qs = q_scale[(row_base + brow) * Groups + grp];
                        const float ks = __half2float(k_scale_s[(nt * 16 + (lane & 15)) * Groups +
                                                                grp]);
                        scoref[nt][r] = __fmaf_rn(qs * ks, static_cast<float>(rawi32[nt][r]),
                                                  scoref[nt][r]);
                    }
                }
            }

            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0[8], bm1[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { bm0[r] = -HIP_INF_F; bm1[r] = -HIP_INF_F; }
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int brow = r + (lane_hi ? 8 : 0);
                    if (!full_score_tile) {
                        const int qabs = brow < tile_rows ? base_pos + q0 + brow : -1;
                        const int key_abs = k0 + nt * 16 + (lane & 15);
                        scoref[nt][r] = key_abs <= qabs ? scoref[nt][r] : -HIP_INF_F;
                    }
                    bm0[r] = fmaxf(bm0[r], scoref[nt][r]);
                    bm1[r] = fmaxf(bm1[r], scoref[nt][r]);
                }
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                bm0[r] = warp_max<16>(bm0[r], FullMask);
                bm1[r] = warp_max<16>(bm1[r], FullMask);
            }

            float nm0[8], nm1[8], alpha0[8], alpha1[8], nm0s[8], nm1s[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                nm0[r]    = fmaxf(running_m0[r], bm0[r]);
                nm1[r]    = fmaxf(running_m1[r], bm1[r]);
                nm0s[r]   = nm0[r] * scale_l2;
                nm1s[r]   = nm1[r] * scale_l2;
                alpha0[r] = running_m0[r] == -HIP_INF_F
                                ? 0.0f
                                : exp2_approx(__fmaf_rn(running_m0[r], scale_l2, -nm0s[r]));
                alpha1[r] = running_m1[r] == -HIP_INF_F
                                ? 0.0f
                                : exp2_approx(__fmaf_rn(running_m1[r], scale_l2, -nm1s[r]));
            }
            float bl0[8], bl1[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { bl0[r] = 0.0f; bl1[r] = 0.0f; }
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float s  = scoref[nt][r];
                    const float nm = lane_hi ? nm1s[r] : nm0s[r];
                    const float p  = s > -HIP_INF_F ? exp2_approx(__fmaf_rn(s, scale_l2, -nm))
                                                    : 0.0f;
                    bl0[r] += p;
                    bl1[r] += p;
                    const int brow = r + (lane_hi ? 8 : 0);
                    const int key  = nt * 16 + (lane & 15);
                    p_s[(row_base + brow) * Bc + gqa_prefill_i8_p_swz(row_base + brow, key)] =
                        __float2half_rn(p);
                }
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                bl0[r]        = warp_sum<16>(bl0[r], FullMask);
                bl1[r]        = warp_sum<16>(bl1[r], FullMask);
                running_l0[r] = __fmaf_rn(running_l0[r], alpha0[r], bl0[r]);
                running_l1[r] = __fmaf_rn(running_l1[r], alpha1[r], bl1[r]);
                running_m0[r] = nm0[r];
                running_m1[r] = nm1[r];
                if (lane == r) { alpha_s[row_base + r] = alpha0[r]; }
                if (lane == 16 + r) { alpha_s[row_base + 8 + r] = alpha1[r]; }
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            // Dequantize V transposed so the WMMA PV B-fragment (column n = d,
            // elements k = key) reads v_f16[d * Bc + swz(d, key)] contiguously.
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                if (key <= max_query_abs) {
                    const int grp = d >> 6;
                    __half vs     = __float2half_rn(0.0f);
                    if ((lane & 7) == 0) { vs = v_scale_s[key_l * Groups + grp]; }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    const int4 packed = gqa_prefill_i8_dequant_f16x8(&v_i8[key_l * D + d], vs);
                    const __half* f8  = reinterpret_cast<const __half*>(&packed);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int df = d + i;
                        v_f16[df * Bc + gqa_prefill_swz(df, key_l)] = f8[i];
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int df = d + i;
                        v_f16[df * Bc + gqa_prefill_swz(df, key_l)] = __half(0.0f);
                    }
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_tile((kb + 1) * Bc); }

        const int row_tile = warp % kGqaPrefillI8RowTiles;
        const int d_slice  = warp / kGqaPrefillI8RowTiles;
        const int row_base = row_tile * 16;
#pragma unroll
        for (int n = 0; n < WPVNtPerWarp; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[n][r] *= alpha_s[row_base + r + (lane_hi ? 8 : 0)];
            }
        }

        // O += P * V, contracting over the Bc keys (16 per step). Operands are f16.
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned af[8];
            wmma_load_a_bf16(af, reinterpret_cast<const __hip_bfloat16*>(p_s),
                             row_base + (lane >> 1), k * 16, Bc, gqa_prefill_swz);
            WmmaA16 a = *reinterpret_cast<WmmaA16*>(af);
#pragma unroll
            for (int n = 0; n < WPVNtPerWarp; ++n) {
                const int dcol = d_slice * (D / kGqaPrefillI8DConsumers) + n * 16 + (lane & 15);
                unsigned bfrag[8];
                wmma_load_b_bf16(bfrag, reinterpret_cast<const __hip_bfloat16*>(v_f16), dcol,
                                 k * 16, Bc, gqa_prefill_swz);
                WmmaC8& c = *reinterpret_cast<WmmaC8*>(acc[n]);
                WmmaA16 b  = *reinterpret_cast<WmmaA16*>(bfrag);
                c         = wmma_f16(a, b, c);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    // Producers publish final per-row normalization.
    if (warp < ProducerWarps) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            running_l0[r] = warp_sum<16>(running_l0[r], FullMask);
            running_l1[r] = warp_sum<16>(running_l1[r], FullMask);
            if (lane == r) { final_l_s[warp * 16 + r] = running_l0[r]; }
            if (lane == 16 + r) { final_l_s[warp * 16 + 8 + r] = running_l1[r]; }
        }
    }
    __syncthreads();

    const int row_tile = warp % kGqaPrefillI8RowTiles;
    const int d_slice  = warp / kGqaPrefillI8RowTiles;
    const int row_base = row_tile * 16;
    float inv_l[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const float lv = final_l_s[row_base + r];
        inv_l[r]       = lv > 0.0f ? __frcp_rn(lv) : 0.0f;
    }
#pragma unroll
    for (int n = 0; n < WPVNtPerWarp; ++n) {
        const int d = d_slice * (D / kGqaPrefillI8DConsumers) + n * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int brow = r + (lane_hi ? 8 : 0);
            if (brow < tile_rows) {
                out[gqa_prefill_q_index<Geometry>(q_head, d, q0 + row_base + brow)] =
                    __float2bfloat16_rn(acc[n][r] * inv_l[brow]);
            }
        }
    }

        gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           kGqaPrefillI8Threads);
}

} // namespace ninfer::ops
