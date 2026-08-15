#include "hip/hip_runtime.h"
#pragma once

// Reusable W8G32 RowSplit exact-small-T MMA core.
//
// Geometry, active tokens, and scheduling are independent compile-time values. K-split warps
// cooperatively own one 16-row output tile; each warp evaluates a disjoint 64-wide K slice, then
// the CTA reduces FP32 partials in shared memory. Output owns physical row/token addressing; an
// optional caller epilogue may instead consume the FP32 tile.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct W8SmallTMmaStoreEpilogue {};

struct W8SmallTMmaResidualEpilogue {};

struct W8SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

__device__ __forceinline__ int w8_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union W8SmallTBf16PairBits {
    __hip_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned w8_small_t_bf16_pair_from_s8(unsigned values) {
    W8SmallTBf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    W8SmallTBf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    W8SmallTBf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

template <class Geometry, int ActiveCols, class Schedule, class Output,
          class Epilogue = W8SmallTMmaStoreEpilogue, class RowPolicy = W8SmallTMmaIdentityRows,
          bool DirectPairEpilogue = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void w8_small_t_mma_kernel(
    const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue = {},
    RowPolicy row_policy = {}) {
    constexpr int kHidden     = Geometry::kInputRows;
    constexpr int kTileK      = Schedule::kTileKPerWarp;
    constexpr int kWarps      = Schedule::kKWarps;
    constexpr int kMmaRows    = Schedule::kRowsPerCta;
    constexpr int kRowsPerCta = Schedule::kRowsPerCta;
    constexpr int kGroupK     = Schedule::kGroupK;
    constexpr int kGroups     = kHidden / kGroupK;
    constexpr int kTileCols   = Schedule::kTileTokens;
    static_assert((kHidden % kGroupK) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);
    constexpr int kNt        = kTileCols / 8;
    // gfx1151 WMMA atoms are 16x16 (not 16x8); a warp tile of kTileCols columns uses
    // kWmmaNt 16-column atoms (the final atom may overlap the tile edge when kTileCols
    // is not a multiple of 16; the store filters those columns out).
    constexpr int kWmmaNt = (kTileCols + 15) / 16;
    constexpr unsigned long long kMask = 0xffffffffull;
    // gfx1151 WMMA atoms are 16 columns wide: the B-fragment load reads the full atom
    // width (col = lane&15), so activations are staged in one 16-column buffer per warp
    // and re-staged per atom (kWmmaNt atoms per K group). Staging fewer columns than
    // the atom width made the fragment reads walk past the shared union into stale LDS,
    // corrupting random tiles with garbage; stage_x zero-fills columns past ActiveCols,
    // and the fixed per-atom buffer keeps every schedule under the 64 KiB LDS cap.
    constexpr int kAtomCols = 16;

    union SharedStorage {
        struct {
            std::uint8_t codes[kMmaRows][kGroupK];
            __hip_bfloat16 activations[kWarps][kAtomCols * kTileK];
            std::uint8_t scales[kMmaRows][Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared
                                              ? Schedule::kScaleBytesPerRow
                                              : 1];
        } staging;

        float partial[kWarps * kWmmaNt * 32 * 8];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& b_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;

    const int cta_row0 = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    // Stage atom ni (16 columns [ni*16, ni*16+16)) into the per-warp buffer. Columns
    // past ActiveCols are zero-filled; the zfill path with src_bytes == 0 never
    // dereferences the source pointer, so no OOB global read happens.
    const auto stage_x = [&](int group_k0, int col_base) {
        constexpr int kItemsPerSplit = kAtomCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)];
            if (col_base + col < ActiveCols) {
                cp_async<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(col_base + col) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8]);
            } else {
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(0) * kHidden + group_k0 + warp * kTileK + k8 * 8],
                    0);
            }
        }
    };

    const auto stage_codes = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(cta_row0, row);
            for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[row][swizzled_chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kHidden + group_k0 +
                        chunk * 16);
            }
        }
        if constexpr (Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared) {
            constexpr int kScaleChunksPerRow = Schedule::kScaleBytesPerRow / 16;
            for (int item = tid; item < kMmaRows * kScaleChunksPerRow; item += kWarps * 32) {
                const int row        = item / kScaleChunksPerRow;
                const int chunk      = item - row * kScaleChunksPerRow;
                const int weight_row = row_policy.weight_row(cta_row0, row);
                cp_async<16, Schedule::kWeightCache>(
                    &scale_shared[row][chunk * 16],
                    scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                              group_k0 / 32 + chunk * 8) *
                                 2);
            }
        }
    };

    const int warp_koff = k_split * kTileK;
    float acc[kWmmaNt][8];
#pragma unroll
    for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[ni][r] = 0.0f; }
    }

    stage_codes(0);

    constexpr int kGroupUnroll = kHidden <= 6144 ? kGroups : 12;
#pragma unroll kGroupUnroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        const int rb       = (lane & 16) ? 8 : 0;
        unsigned row_scale_pair[8];

#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
            // Each 16-column atom stages its own B tile into the shared per-warp buffer.
            stage_x(group_k0, ni * 16);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            // WMMA C layout: lane l holds rows r + 8*(l>=16) at the column l&15; each of
            // those rows carries its OWN 32-K scale, so collect one fp16-pair per row of
            // this lane once per group (code_shared/scale_shared are staged by the first
            // atom's barrier).
            if (ni == 0) {
                if constexpr (Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared) {
#pragma unroll
                    for (int rr = 0; rr < 8; ++rr) {
                        row_scale_pair[rr] = *reinterpret_cast<const unsigned*>(
                            &scale_shared[rb + rr][warp_koff / 16]);
                    }
                } else {
#pragma unroll
                    for (int rr = 0; rr < 8; ++rr) {
                        const int scale_row = row_policy.weight_row(cta_row0, rb + rr);
                        row_scale_pair[rr]  = *reinterpret_cast<const unsigned*>(
                            scales + (static_cast<std::int64_t>(scale_row) *
                                      Geometry::kGroupsPerRow +
                                      group_k0 / 32 + warp_koff / 32) *
                                         2);
                    }
                }
            }
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                float group_acc[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) { group_acc[r] = 0.0f; }
#pragma unroll
                for (int ki = 0; ki < 2; ++ki) {
                    const int ks = group * 2 + ki;
                    // gfx1151 WMMA A fragment: active lane l holds row m = l>>1 and all 16
                    // K elements of the window. The 16 x w8 weights for (row m, k in
                    // [warp_koff + ks*16, +16)) lie in one swizzled 16-byte code_shared
                    // chunk.
                    const int m  = lane >> 1;
                    const int chunk = (warp_koff + ks * 16) >> 4;
                    unsigned a_raw[4];
                    // Same asm-load path as the B fragment (leading lgkmcnt wait +
                    // memory clobber) so the A side cannot see stale LDS either.
                    wmma_ds_load_b128(a_raw, smem_addr(&code_shared[m][(chunk ^ (m & 7)) * 16]), 0);
                    unsigned a_frag[8];
                    const bool a_active = wmma_a_lane_active(lane);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const unsigned short code_pair =
                            static_cast<unsigned short>(a_raw[j >> 1] >> ((j & 1) * 16));
                        const unsigned pair = w8_small_t_bf16_pair_from_s8(code_pair);
                        a_frag[j] = a_active ? pair : 0u;
                    }
                    unsigned b_frag[8];
                    wmma_load_b_bf16(b_frag, b_shared[k_split], lane & 15, ks * 16, kTileK,
                                     w8_small_t_swizzle_64);
                    WmmaC8& c  = *reinterpret_cast<WmmaC8*>(group_acc);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag);
                    c = wmma_bf16(a, b, c);
                }
                float row_scale[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const unsigned bits = group == 0 ? row_scale_pair[r] & 0xffffu
                                                     : row_scale_pair[r] >> 16;
                    row_scale[r]        = __half2float(__ushort_as_half(bits));
                }
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc[ni][r] = fmaf(group_acc[r], row_scale[r], acc[ni][r]);
                }
            }
            // All warps finished reading b_shared before the next atom re-stages it.
            __syncthreads();
        }

        if (group_index + 1 < kGroups) {
            stage_codes(group_k0 + kGroupK);
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
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
    __syncthreads();

    if (k_split == 0) {
        const W8OutputTile output_tile = output.tile(cta_row0);
        float* projected               = partial;
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
            const int row_lo = cta_row0 + (lane < 16 ? 0 : 8);
            const int local_col = ni * 16 + (lane & 15);
            const int col = local_col;
            // DirectPairEpilogue pulls the partner row-half with a warp-uniform shuffle
            // (all 32 lanes), so run it before the lane-dependent local_col>=kTileCols guard
            // that would otherwise exclude some lanes from the __shfl_sync.
            float partner[8];
            if constexpr (DirectPairEpilogue) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    partner[r] = __shfl_sync(kMask, sum[r], lane ^ 16);
                }
            }
            if (local_col >= kTileCols) { continue; }
            if constexpr (std::is_same_v<Epilogue, W8SmallTMmaStoreEpilogue> ||
                          std::is_same_v<Epilogue, W8SmallTMmaResidualEpilogue>) {
                if (col < ActiveCols) {
                    const auto store = [&](int row, float value) {
                        __hip_bfloat16* destination = output_tile.at(row, col);
                        if constexpr (std::is_same_v<Epilogue, W8SmallTMmaResidualEpilogue>) {
                            value += __bfloat162float(*destination);
                        }
                        *destination = __float2bfloat16_rn(value);
                    };
#pragma unroll
                    for (int r = 0; r < 8; ++r) { store(row_lo + r, sum[r]); }
                }
            } else if constexpr (DirectPairEpilogue) {
                if (col < ActiveCols && lane < 16) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int gate_row = row_policy.weight_row(cta_row0, r);
                        epilogue.template store_wmma<ActiveCols>(gate_row, col, sum[r], partner[r]);
                    }
                }
            } else {
                // FP32 projection layout is [row x kTileCols]; each lane writes its 8 rows.
                const float* values = sum;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (col < ActiveCols) { projected[(row_lo - cta_row0 + r) * kTileCols + col] = sum[r]; }
                }
            }
        }
        if constexpr (!std::is_same_v<Epilogue, W8SmallTMmaStoreEpilogue> &&
                      !std::is_same_v<Epilogue, W8SmallTMmaResidualEpilogue> &&
                      !DirectPairEpilogue) {
            __syncwarp();
            if (lane < kRowsPerCta) {
                float row_values[ActiveCols];
#pragma unroll
                for (int token = 0; token < ActiveCols; ++token) {
                    row_values[token] = projected[lane * kTileCols + token];
                }
                epilogue.store(cta_row0 + lane, row_values);
            }
        }
    }
}

} // namespace ninfer::ops::detail
