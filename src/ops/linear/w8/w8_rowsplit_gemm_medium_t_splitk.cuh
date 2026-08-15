#include "hip/hip_runtime.h"
#pragma once

// W8G32 RowSplit medium-T split-K MMA core. K-split warps share one 16-row
// weight tile while N-groups cover disjoint column ranges. Output owns the
// physical direct-write policy.

#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks, class Output,
          bool AddResidual = false>
__global__
__launch_bounds__(KSplits* NGroups * 32, MinBlocks) void w8_rowsplit_medium_t_splitk_kernel(
    const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, int active_cols) {
    constexpr int kTileK       = 64;
    constexpr int kMmaRows     = 16;
    constexpr int kRowsPerCta  = 16;
    constexpr int kKernelWarps = KSplits * NGroups;
    constexpr int kGroupK      = KSplits * kTileK;
    constexpr int kGroups      = Hidden / kGroupK;
    constexpr int kWarpCols    = TileCols / NGroups;
    constexpr int kNt          = kWarpCols / 8;
    // gfx1151 WMMA atoms are 16x16; the warp column tile uses div_up(kWarpCols,16) atoms
    // and the final atom may overlap the neighboring warp/global columns (filtered at store).
    constexpr int kWmmaNt = (kWarpCols + 15) / 16;
    constexpr unsigned long long kMask = 0xffffffffull;
    static_assert(KSplits == 2 || KSplits == 4 || KSplits == 8);
    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(Hidden % kGroupK == 0 && kKernelWarps <= 32);

    __shared__ __align__(16) std::uint8_t code_shared[kMmaRows][kGroupK];
    __shared__ __align__(16) __hip_bfloat16 b_shared[kKernelWarps][kWarpCols * kTileK];

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int n_group    = warp / KSplits;
    const int k_split    = warp - n_group * KSplits;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    // blockIdx.y selects a contiguous TileCols-wide column range so a wide problem can be
    // split across more CTAs (grid.y > 1) to keep the per-CTA static shared tile within the
    // gfx1151 64 KiB LDS cap; all single-column-block callers use blockIdx.y == 0.
    const int col_base   = static_cast<int>(blockIdx.y) * TileCols;
    const int n_base     = col_base + n_group * kWarpCols;
    const int remaining  = active_cols - n_base;
    const int local_cols = remaining <= 0 ? 0 : (remaining < kWarpCols ? remaining : kWarpCols);
    const int cta_row0   = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int k0) {
        for (int item = lane; item < local_cols * (kTileK / 8); item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16, Cache::cg>(
                dst, &x[static_cast<std::int64_t>(n_base + col) * Hidden + k0 + k8 * 8]);
        }
        cp_commit();
    };

    const auto stage_codes = [&](int group_k0) {
        constexpr int kChunks = kGroupK / 16;
        for (int item = tid; item < kMmaRows * kChunks; item += kKernelWarps * 32) {
            const int row            = item / kChunks;
            const int chunk          = item - row * kChunks;
            const int swizzled_chunk = chunk ^ (row & 7);
            cp_async<16, Cache::cg>(&code_shared[row][swizzled_chunk * 16],
                                    codes + static_cast<std::int64_t>(cta_row0 + row) * Hidden +
                                        group_k0 + chunk * 16);
        }
        cp_commit();
    };

    const int warp_koff = k_split * kTileK;
    float acc[kWmmaNt][8];
#pragma unroll
    for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[ni][r] = 0.0f; }
    }

    stage_codes(0);
    stage_x(warp_koff);
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        const int k0       = group_k0 + warp_koff;

        // WMMA C layout: lane l holds rows r + 8*(l>=16); each row has its own 32-K scale.
        const int rb = (lane & 16) ? 8 : 0;
        unsigned row_scale_pair[8];
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) {
            const int scale_row = cta_row0 + rb + rr;
            row_scale_pair[rr]  = *reinterpret_cast<const unsigned*>(
                scales + (static_cast<std::int64_t>(scale_row) * (Hidden / 32) + k0 / 32) * 2);
        }

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            float group_acc[kWmmaNt][8];
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
                for (int r = 0; r < 8; ++r) { group_acc[ni][r] = 0.0f; }
            }
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                const int ks = group * 2 + ki;
                // WMMA A fragment: active lane l holds row m = l>>1 and the full 16-K window.
                const int m  = lane >> 1;
                const int chunk = (warp_koff + ks * 16) >> 4;
                const unsigned short* base = reinterpret_cast<const unsigned short*>(
                    &code_shared[m][(chunk ^ (m & 7)) * 16]);
                unsigned a_frag[8];
                const bool a_active = wmma_a_lane_active(lane);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned pair = w8_small_t_bf16_pair_from_s8(static_cast<unsigned>(base[j]));
                    a_frag[j] = a_active ? pair : 0u;
                }
#pragma unroll
                for (int ni = 0; ni < kWmmaNt; ++ni) {
                    unsigned b_frag[8];
                    const int col = ni * 16 + (lane & 15);
                    wmma_load_b_bf16(b_frag, b_shared[warp], col, ks * 16, kTileK,
                                     w8_small_t_swizzle_64);
                    WmmaC8& c  = *reinterpret_cast<WmmaC8*>(group_acc[ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag);
                    c          = wmma_bf16(a, b, c);
                }
            }
            float row_scale[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const unsigned bits = group == 0 ? row_scale_pair[r] & 0xffffu
                                                 : row_scale_pair[r] >> 16;
                row_scale[r]        = __half2float(__ushort_as_half(bits));
            }
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc[ni][r] = fmaf(group_acc[ni][r], row_scale[r], acc[ni][r]);
                }
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_codes(group_k0 + kGroupK);
            stage_x(k0 + kGroupK);
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = reinterpret_cast<float*>(b_shared);
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                partial[((warp * kWmmaNt + ni) * 32 + lane) * 8 + r] = acc[ni][r];
            }
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[ni][r] += partial[(((warp + 1) * kWmmaNt + ni) * 32 + lane) * 8 + r];
            }
            if (k_split != 0) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    partial[((warp * kWmmaNt + ni) * 32 + lane) * 8 + r] = acc[ni][r];
                }
            }
        }
    }

    if constexpr (KSplits > 2) {
        __syncthreads();
        if (k_split == 0) {
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
                for (int split = 2; split < KSplits; split += 2) {
                    const int partner_warp = n_group * KSplits + split;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        acc[ni][r] +=
                            partial[((partner_warp * kWmmaNt + ni) * 32 + lane) * 8 + r];
                    }
                }
            }
        }
    }

    if (k_split == 0) {
        const W8OutputTile output_tile = output.tile(cta_row0);
        const auto store               = [&](int row, int col, float value) {
            __hip_bfloat16* destination = output_tile.at(row, col);
            if constexpr (AddResidual) { value += __bfloat162float(*destination); }
            *destination = __float2bfloat16_rn(value);
        };
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
            // WMMA C layout: lane l holds (row r + 8*(l>=16), col = n_base + ni*16 + (l&15)).
            const int col       = n_base + ni * 16 + (lane & 15);
            const int local_col = ni * 16 + (lane & 15);
            const int row_lo    = cta_row0 + (lane < 16 ? 0 : 8);
            if (local_col >= kWarpCols) { continue; }
            if (col < active_cols) {
#pragma unroll
                for (int r = 0; r < 8; ++r) { store(row_lo + r, col, acc[ni][r]); }
            }
        }
    }
}

} // namespace ninfer::ops::detail
