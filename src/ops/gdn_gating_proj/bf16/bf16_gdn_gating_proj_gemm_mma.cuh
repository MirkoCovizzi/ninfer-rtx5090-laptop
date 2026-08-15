#include "hip/hip_runtime.h"
#pragma once

// SM120 BF16 GDN gating projection for the two exact registered geometries:
//
//   Qwen3.6-27B:     a/b = W[48,5120] @ x[5120,T]
//   Qwen3.6-35B-A3B: a/b = W[32,2048] @ x[2048,T]
//
// A CTA computes the same 16 output rows from both weights over 128 (27B) or
// 64 (35B) tokens.
// Split-K routes use a tuned eight- or sixteen-warp specialization and an
// in-kernel cooperative grid reduction; the unsplit long-context route uses
// eight warps for more independent MMA accumulators. Both preserve a single
// kernel launch.

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/warp.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_cooperative_groups.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kBf16GdnBlockM     = 16;
inline constexpr int kBf16GdnBlockK     = 64;
inline constexpr int kBf16GdnWarps      = 16;
// Single-stage synchronous pipeline. HIP's cp_async helpers are synchronous loads, so
// a double buffer buys no overlap; keeping one stage also halves smem (20 KiB), which
// is required for the cooperative SplitK launch on gfx1151: with 2 stages (40 KiB) the
// grid (e.g. 24 blocks at SplitK=8) exceeds the 20-block residency ceiling (1 block per
// 20-CU WGP), and a plain kStages=1 flip breaks the `it & 1` buffer indexing on
// multi-tile runs. stage is therefore always 0 below.
inline constexpr int kBf16GdnStages     = 1;
inline constexpr int kBf16GdnMFragments = kBf16GdnBlockM / 16;

template <int BlockN>
inline constexpr int kBf16GdnSmemElements =
    kBf16GdnStages * (BlockN * kBf16GdnBlockK + 2 * kBf16GdnBlockM * kBf16GdnBlockK);

template <int BlockN>
inline constexpr int kBf16GdnSmemBytes = kBf16GdnSmemElements<BlockN> * sizeof(__hip_bfloat16);

struct Bf16Gdn27Geometry {
    static constexpr int kHeads  = 48;
    static constexpr int kHidden = 5120;
    static constexpr int kBlockN = 128;
};

struct Bf16Gdn35Geometry {
    static constexpr int kHeads  = 32;
    static constexpr int kHidden = 2048;
    static constexpr int kBlockN = 64;
};

static_assert(Bf16Gdn27Geometry::kHidden % kBf16GdnBlockK == 0);
static_assert(Bf16Gdn35Geometry::kHidden % kBf16GdnBlockK == 0);

__device__ __forceinline__ int bf16_gdn_swizzle(int row, int col) {
    return (col & ~63) + gemm_swz64(row, col & 63);
}

template <class Geometry, int SplitK, bool FullTokens, int Warps, bool NormalizeInput = false,
          int NormTokenCapacity = 0>
__global__ __launch_bounds__(Warps * 32, 1) void bf16_gdn_gating_proj_gemm_mma_kernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ norm_weight,
    __hip_bfloat16* __restrict__ normalized_x, float norm_eps,
    const __hip_bfloat16* __restrict__ a_weight, const __hip_bfloat16* __restrict__ b_weight,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias, float* __restrict__ partial,
    float* __restrict__ g, float* __restrict__ beta, std::int32_t t) {
    constexpr int kBf16GdnHeads       = Geometry::kHeads;
    constexpr int kBf16GdnHidden      = Geometry::kHidden;
    constexpr int kBf16GdnBlockN      = Geometry::kBlockN;
    constexpr int kBf16GdnLogicalRows = 2 * kBf16GdnHeads;
    constexpr int kBf16GdnKTiles      = kBf16GdnHidden / kBf16GdnBlockK;
    static_assert(SplitK >= 1 && kBf16GdnKTiles % SplitK == 0,
                  "BF16 GDN split-K must divide the exact geometry's K tiles");
    static_assert(Warps == 8 || Warps == 16);
    static_assert(kBf16GdnBlockN % Warps == 0);
    constexpr int kTilesPerSplit = kBf16GdnKTiles / SplitK;
    constexpr int kKSubtiles     = kBf16GdnBlockK / 16;
    constexpr int kThreads       = Warps * 32;
    constexpr int kWarpN         = kBf16GdnBlockN / Warps;
    static_assert(kWarpN % 8 == 0);
    // gfx1151 WMMA atoms are 16x16 (not the legacy m16n8 pair).
    // Each warp covers WMT x WNT 16x16 atoms; when kWarpN is not a multiple
    // of 16 the final atom overlaps the neighbouring warp's columns
    // (duplicate compute, correct results) and the store below emits only
    // the warp's own columns.
    constexpr int WMT = kBf16GdnMFragments;
    constexpr int WNT = (kWarpN + 15) / 16;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* xs  = reinterpret_cast<__hip_bfloat16*>(smem_raw);
    auto* aws = xs + kBf16GdnStages * kBf16GdnBlockN * kBf16GdnBlockK;
    auto* bws = aws + kBf16GdnStages * kBf16GdnBlockM * kBf16GdnBlockK;

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int head0  = static_cast<int>(blockIdx.y) * kBf16GdnBlockM;
    const int token0   = static_cast<int>(blockIdx.x) * kBf16GdnBlockN;
    const int split    = static_cast<int>(blockIdx.z);
    const int kt_begin = split * kTilesPerSplit;

    if constexpr (NormalizeInput) {
        static_assert(SplitK == 32, "fused input normalization is tuned for split-32");
        static_assert(NormTokenCapacity > 0 && NormTokenCapacity <= 16);
        constexpr int kLocalPairs     = kBf16GdnBlockK / 2;
        const auto* x2                = reinterpret_cast<const __hip_bfloat162*>(x);
        const int token_count         = min(kBf16GdnBlockN, t - token0);
        float sums[NormTokenCapacity] = {};
        // One warp from row tile zero contributes a 64-element norm slice. The existing post-MMA
        // cooperative handoff reduces the 32 slices, so normalization adds no grid-wide barrier.
        if (blockIdx.y == 0 && warp == 0) {
            for (int pair = lane; pair < kLocalPairs; pair += kWarpSize) {
#pragma unroll
                for (int token_local = 0; token_local < NormTokenCapacity; ++token_local) {
                    if (token_local >= token_count) { continue; }
                    const std::int64_t row_base =
                        static_cast<std::int64_t>(token0 + token_local) * (kBf16GdnHidden / 2);
                    const int global_pair = kt_begin * (kBf16GdnBlockK / 2) + pair;
                    const float2 value    = __bfloat1622float2(x2[row_base + global_pair]);
                    sums[token_local] += value.x * value.x + value.y * value.y;
                }
            }
#pragma unroll
            for (int token_local = 0; token_local < NormTokenCapacity; ++token_local) {
                sums[token_local] = warp_reduce_sum(sums[token_local]);
                if (lane == 0 && token_local < token_count) {
                    float* norm_partial =
                        partial + static_cast<std::int64_t>(SplitK) * t * kBf16GdnLogicalRows;
                    norm_partial[static_cast<std::int64_t>(split) * t + token0 + token_local] =
                        sums[token_local];
                }
            }
        }
    }

    float a_acc[WMT][WNT][8] = {};
    float b_acc[WMT][WNT][8] = {};

    auto stage_load = [&](int stage, int kt) {
        const int k0 = kt * kBf16GdnBlockK;

        // x is shared by all 16 head rows and by both projections.
        for (int vec = tid; vec < kBf16GdnBlockN * (kBf16GdnBlockK / 8); vec += kThreads) {
            const int token_local = vec / (kBf16GdnBlockK / 8);
            const int k_vec       = vec - token_local * (kBf16GdnBlockK / 8);
            const int kk          = k_vec * 8;
            const int token       = token0 + token_local;
            __hip_bfloat16* dst =
                &xs[stage * kBf16GdnBlockN * kBf16GdnBlockK + token_local * kBf16GdnBlockK +
                    bf16_gdn_swizzle(token_local, kk)];
            if constexpr (NormalizeInput) {
                const bool valid = FullTokens || token < t;
                if (valid) {
                    const auto* source = reinterpret_cast<const __hip_bfloat162*>(
                        &x[static_cast<std::int64_t>(token) * kBf16GdnHidden + k0 + kk]);
                    const auto* gain =
                        reinterpret_cast<const __hip_bfloat162*>(&norm_weight[k0 + kk]);
                    auto* target = reinterpret_cast<__hip_bfloat162*>(dst);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 value              = __bfloat1622float2(source[pair]);
                        const float2 weight             = __bfloat1622float2(gain[pair]);
                        const __hip_bfloat162 normalized = __floats2bfloat162_rn(
                            value.x * (1.0f + weight.x), value.y * (1.0f + weight.y));
                        target[pair] = normalized;
                    }
                } else {
                    *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
                }
            } else if constexpr (FullTokens) {
                cp_async<16, Cache::cg>(
                    dst, &x[static_cast<std::int64_t>(token) * kBf16GdnHidden + k0 + kk]);
            } else {
                const bool valid     = token < t;
                const int safe_token = valid ? token : 0;
                cp_async_zfill<16, Cache::cg>(
                    dst, &x[static_cast<std::int64_t>(safe_token) * kBf16GdnHidden + k0 + kk],
                    valid ? 16 : 0);
            }
        }

        // Each thread moves one 16-byte vector: 128 vectors from Wa and 128
        // from Wb. Both contiguous weights remain BF16 all the way into ldmatrix.
        const int weight_vecs = kBf16GdnBlockM * (kBf16GdnBlockK / 8);
        for (int all_vec = tid; all_vec < 2 * weight_vecs; all_vec += kThreads) {
            const bool is_b    = all_vec >= weight_vecs;
            const int vec      = is_b ? all_vec - weight_vecs : all_vec;
            const int row      = vec / (kBf16GdnBlockK / 8);
            const int k_vec    = vec - row * (kBf16GdnBlockK / 8);
            const int kk       = k_vec * 8;
            __hip_bfloat16* dst = (is_b ? bws : aws) + stage * kBf16GdnBlockM * kBf16GdnBlockK +
                                 row * kBf16GdnBlockK + bf16_gdn_swizzle(row, kk);
            const __hip_bfloat16* weight = is_b ? b_weight : a_weight;
            cp_async<16, Cache::cg>(
                dst, &weight[static_cast<std::int64_t>(head0 + row) * kBf16GdnHidden + k0 + kk]);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kBf16GdnStages; ++stage) {
        if (stage < kTilesPerSplit) { stage_load(stage, kt_begin + stage); }
        ninfer::ops::cp_commit();
    }

    // WMMA fragment loads. A is the [16, BlockK] weight strip: head rows
    // mi*16 + (lane>>1), K window ki*16, stride kBf16GdnBlockK. The token-major
    // x tile is the column-major B representation: column warp*kWarpN + ni*16
    // + (lane&15), K window ki*16.
    auto load_fragments = [&](int stage, int kcol, unsigned (&xf)[WNT][8],
                            unsigned (&af)[WMT][8], unsigned (&bf)[WMT][8]) {
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
            const int col = warp * kWarpN + ni * 16 + (lane & 15);
            // The 16-wide WMMA atom overlaps the neighbouring warp's columns
            // when kWarpN < 16; for the trailing warp it reaches past the
            // block's token tile, which xs never stages. Zero those columns so
            // the duplicated compute contributes nothing (the store already
            // filters local_col >= kWarpN).
            if (col < kBf16GdnBlockN) {
                wmma_load_b_bf16(xf[ni], xs + stage * kBf16GdnBlockN * kBf16GdnBlockK, col, kcol,
                                 kBf16GdnBlockK, bf16_gdn_swizzle);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) { xf[ni][e] = 0u; }
            }
        }
#pragma unroll
        for (int mi = 0; mi < WMT; ++mi) {
            if (!wmma_a_lane_active(lane)) { continue; }
            const int arow = mi * 16 + (lane >> 1);
            const __hip_bfloat16* base  = aws + stage * kBf16GdnBlockM * kBf16GdnBlockK;
            const __hip_bfloat16* bbase = bws + stage * kBf16GdnBlockM * kBf16GdnBlockK;
            wmma_load_a_bf16(af[mi], base, arow, kcol, kBf16GdnBlockK, bf16_gdn_swizzle);
            wmma_load_a_bf16(bf[mi], bbase, arow, kcol, kBf16GdnBlockK, bf16_gdn_swizzle);
        }
    };

#pragma unroll 1
    for (int it = 0; it < kTilesPerSplit; ++it) {
        // Single-stage: the buffer is always stage 0; the trailing barrier + reload
        // of the next tile keeps the pipeline synchronous and correct for any
        // kTilesPerSplit (the kStages=1 flip with `it & 1` read/wrote OOB smem on
        // odd iterations).
        constexpr int stage = 0;
        ninfer::ops::cp_wait<kBf16GdnStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kKSubtiles; ++ki) {
            unsigned xf[WNT][8];
            unsigned af[WMT][8];
            unsigned bf[WMT][8];
            load_fragments(stage, ki * 16, xf, af, bf);
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < WNT; ++ni) {
                    WmmaC8& ca = *reinterpret_cast<WmmaC8*>(a_acc[mi][ni]);
                    WmmaC8& cb = *reinterpret_cast<WmmaC8*>(b_acc[mi][ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(af[mi]);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bf[mi]);
                    WmmaA16I x = *reinterpret_cast<WmmaA16I*>(xf[ni]);
                    ca = wmma_bf16(a, x, ca);
                    cb = wmma_bf16(b, x, cb);
                }
            }
        }

        __syncthreads();
        const int next = it + kBf16GdnStages;
        if (next < kTilesPerSplit) { stage_load(stage, kt_begin + next); }
        ninfer::ops::cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
        const int row_base = head0 + mi * 16;
        const int row_lo   = row_base + (lane >= 16 ? 8 : 0);
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
            const int col_base  = token0 + warp * kWarpN + ni * 16;
            const int local_col = ni * 16 + (lane & 15);
            const int col       = col_base + (lane & 15);
            // Skip the overlap columns when kWarpN is not a multiple of 16.
            if (local_col >= kWarpN) { continue; }
            auto store = [&](int token, int row, float av, float bv) {
                if constexpr (SplitK == 1) {
                    const std::int64_t out_i =
                        static_cast<std::int64_t>(token) * kBf16GdnHeads + row;
                    g[out_i]    = -expf(A_log[row]) * softplus(av + dt_bias[row]);
                    beta[out_i] = sigmoid(bv);
                } else {
                    const std::int64_t base =
                        (static_cast<std::int64_t>(split) * t + token) * kBf16GdnLogicalRows;
                    partial[base + row]                 = av;
                    partial[base + kBf16GdnHeads + row] = bv;
                }
            };
            if constexpr (FullTokens) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    store(col, row_lo + r, a_acc[mi][ni][r], b_acc[mi][ni][r]);
                }
            } else {
                if (col < t) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        store(col, row_lo + r, a_acc[mi][ni][r], b_acc[mi][ni][r]);
                    }
                }
            }
        }
    }

    if constexpr (SplitK > 1) {
        cooperative_groups::this_grid().sync();
        const int block_linear = (static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.y) +
                                  static_cast<int>(blockIdx.y)) *
                                     static_cast<int>(gridDim.x) +
                                 static_cast<int>(blockIdx.x);
        const int grid_threads = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y) *
                                 static_cast<int>(gridDim.z) * kThreads;
        const int elems = kBf16GdnHeads * t;
        if constexpr (NormalizeInput) {
            const float* norm_partial =
                partial + static_cast<std::int64_t>(SplitK) * t * kBf16GdnLogicalRows;
            const int hidden_elems = kBf16GdnHidden * t;
            for (int i = block_linear * kThreads + tid; i < hidden_elems; i += grid_threads) {
                const int k     = i % kBf16GdnHidden;
                const int token = i / kBf16GdnHidden;
                float sum       = 0.0F;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += norm_partial[static_cast<std::int64_t>(s) * t + token];
                }
                const float inv    = rsqrtf(sum / static_cast<float>(kBf16GdnHidden) + norm_eps);
                const float value  = __bfloat162float(x[i]);
                const float weight = __bfloat162float(norm_weight[k]);
                normalized_x[i]    = __float2bfloat16_rn(value * inv * (1.0F + weight));
            }
        }
        for (int i = block_linear * kThreads + tid; i < elems; i += grid_threads) {
            const int row   = i % kBf16GdnHeads;
            const int token = i / kBf16GdnHeads;
            float av        = 0.0f;
            float bv        = 0.0f;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(s) * t + token) * kBf16GdnLogicalRows;
                av += partial[base + row];
                bv += partial[base + kBf16GdnHeads + row];
            }
            if constexpr (NormalizeInput) {
                const float* norm_partial =
                    partial + static_cast<std::int64_t>(SplitK) * t * kBf16GdnLogicalRows;
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += norm_partial[static_cast<std::int64_t>(s) * t + token];
                }
                const float inv = rsqrtf(sum / static_cast<float>(kBf16GdnHidden) + norm_eps);
                av *= inv;
                bv *= inv;
            }
            g[i]    = -expf(A_log[row]) * softplus(av + dt_bias[row]);
            beta[i] = sigmoid(bv);
        }
    }
}

} // namespace ninfer::ops::detail
