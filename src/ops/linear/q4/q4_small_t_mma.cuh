#include "hip/hip_runtime.h"
#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q4SmallTMmaStoreEpilogue {};

struct Q4SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int InputRows>
struct Q4DraftHeadGeometry {
    static constexpr int kOutputRows   = 131072;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q4DraftSmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

__device__ __forceinline__ int q4_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q4SmallTBf16PairBits {
    __hip_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q4_small_t_bf16_pair(std::uint8_t packed) {
    const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
    const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
    Q4SmallTBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4SmallTMmaStoreEpilogue,
          class RowPolicy = Q4SmallTMmaIdentityRows>
__launch_bounds__(256, 6) __global__
    void q4_small_t_mma_kernel(const __hip_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __hip_bfloat16* __restrict__ out, Epilogue epilogue = {},
                               RowPolicy row_policy = {}) {
    using Schedule              = Q4DraftSmallTSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    // gfx1151 WMMA atoms are 16x16; use div_up(kTileCols,16) 16-column atoms, the final one
    // overlapping the tile edge when kTileCols is not a multiple of 16 (filtered at store).
    constexpr int kWmmaNt       = (kTileCols + 15) / 16;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 2 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK / 2];
            __hip_bfloat16 activations[kWarps][kTileCols * kTileK];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;

        float partial[kWarps * kWmmaNt * 32 * 8];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int row0    = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    const auto stage_x = [&](int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[warp][col * kTileK + q4_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16>(
                dst,
                &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK + k8 * 8]);
        }
    };

    const auto stage_weight = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(row0, row);
            for (int chunk = lane; chunk < kGroupK / 32; chunk += 32) {
                cp_async<16, Schedule::kCodeCache>(
                    &code_shared[row][chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                        chunk * 16);
            }
        }
        for (int row = tid; row < kRowsPerCta; row += kWarps * 32) {
            const int weight_row = row_policy.weight_row(row0, row);
            cp_async<16>(&scale_shared[row][0],
                         scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                                   group_k0 / 64) *
                                      2);
        }
    };

    const int warp_koff = k_split * kTileK;
    float acc[kWmmaNt][8] = {};
#pragma unroll
    for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[ni][r] = 0.0f; }
    }

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0      = group_index * kGroupK;
        float group_acc[kWmmaNt][8] = {};
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { group_acc[ni][r] = 0.0f; }
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // gfx1151 WMMA A fragment: active lane l holds row m = l>>1 and the full 16-K
            // window. Each code_shared byte decodes to 2 bf16 weights for two adjacent K.
            const int m                = lane >> 1;
            const int base_byte        = warp_koff / 2 + ks * 8;
            const unsigned char* base  = &code_shared[m][base_byte];
            const bool a_active        = wmma_a_lane_active(lane);
            unsigned a_frag[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned pair = q4_small_t_bf16_pair(base[j]);
                a_frag[j]           = a_active ? pair : 0u;
            }
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
                unsigned b_frag[8];
                const int col = ni * 16 + (lane & 15);
                wmma_load_b_bf16(b_frag, x_shared[k_split], col, ks * 16, kTileK,
                                 q4_small_t_swizzle_64);
                WmmaC8& c  = *reinterpret_cast<WmmaC8*>(group_acc[ni]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag);
                c          = wmma_bf16(a, b, c);
            }
        }

        // WMMA C layout: lane l holds rows r + 8*(l>=16) at column l&15. Each of those rows
        // has its OWN per-64-K scale, so scale per register (per row), not per lane.
        const int rb = 8 * (lane >= 16 ? 1 : 0);
        float row_scale[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            row_scale[r] = __half2float(__ushort_as_half(scale_shared[rb + r][k_split]));
        }
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[ni][r] = fmaf(group_acc[ni][r], row_scale[r], acc[ni][r]);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_weight(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                partial[((k_split * kWmmaNt + ni) * 32 + lane) * 8 + r] = acc[ni][r];
            }
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[ni][r] += partial[(((k_split + 1) * kWmmaNt + ni) * 32 + lane) * 8 + r];
            }
            if (k_split != 0) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    partial[((k_split * kWmmaNt + ni) * 32 + lane) * 8 + r] = acc[ni][r];
                }
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
            float sum[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { sum[r] = acc[ni][r]; }
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    sum[r] += partial[((split * kWmmaNt + ni) * 32 + lane) * 8 + r];
                }
            }
            // WMMA C layout: lane l holds (row r + 8*(l>=16), col = lane&15).
            const int row_lo     = row0 + (lane < 16 ? 0 : 8);
            const int local_col  = ni * 16 + (lane & 15);
            const int col        = local_col;
            if constexpr (std::is_same_v<Epilogue, Q4SmallTMmaStoreEpilogue>) {
                if (local_col >= kTileCols) { continue; }
                if (col < ActiveCols) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        out[static_cast<std::int64_t>(col) * Geometry::kOutputRows + row_lo + r] =
                            __float2bfloat16_rn(sum[r]);
                    }
                }
            } else {
                // A fused (e.g. SwiGLU) epilogue combines a gate/up row-half pair. Exchange
                // the partner across the warp converged (all 32 lanes) BEFORE the per-column
                // guard, so a narrow tile width or partial column range never traps.
                float partner[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) { partner[r] = __shfl_sync(0xffffffffull, sum[r], lane ^ 16); }
                if (local_col >= kTileCols) { continue; }
                epilogue.template store_wmma<ActiveCols>(row_lo, col, sum, partner);
            }
        }
    }
}

} // namespace ninfer::ops::detail
