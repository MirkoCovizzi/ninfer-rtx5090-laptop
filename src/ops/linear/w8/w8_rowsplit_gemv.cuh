#pragma once

// W8 row-split small-T GEMV (warp per row, streamed weight tiles).
// Mirrors the q5 GEMV-T structure for the W8G32 head / MTP-module shapes where the
// WMMA small-T path underutilizes the WMMA atoms (T < 16) and runs far below DRAM
// bandwidth on gfx1151. Weights are streamed once per kTt-column slice; x is either
// staged once per block (kStageX, kK*kTt must fit shared) or re-read from L2.

#include "hip/hip_runtime.h"
#include "ops/common/memory.cuh"
#include "ops/common/hip_compat.cuh"
#include "ops/common/warp.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
constexpr int kW8GroupK = 32;
constexpr int kW8GroupsPerTile = 16;
constexpr int kW8CodeBytesPerGroup = 32;
constexpr int kW8ScaleBytesPerGroup = 2;
} // namespace

__device__ __forceinline__ void
w8_gemv_issue_tile(uint4* __restrict__ s_code, uint4* __restrict__ s_sc,
                   const std::uint8_t* __restrict__ code_row,
                   const std::uint8_t* __restrict__ scale_row, int tile, int lane) {
    const int g0 = tile * kW8GroupsPerTile;
    s_code[lane] = *reinterpret_cast<const uint4*>(code_row + g0 * kW8CodeBytesPerGroup +
                                                   lane * 16);
    if (lane < 2) {
        s_sc[lane] = *reinterpret_cast<const uint4*>(scale_row + g0 * kW8ScaleBytesPerGroup +
                                                     lane * 16);
    }
}

// kTt token columns; x2 column base x2_col with stride kK2 (bf16x2 words).
template <int kTt, int kK2>
__device__ __forceinline__ void w8_gemv_consume_tile(const __hip_bfloat162* __restrict__ x2_col,
                                                     const uint4* __restrict__ s_code,
                                                     const uint4* __restrict__ s_sc, int tile,
                                                     int lane, float (&acc)[kTt]) {
    const int g0   = tile * kW8GroupsPerTile;
    const int e0   = lane * 16;
    const float scale =
        __half2float(__ushort_as_half(reinterpret_cast<const std::uint16_t*>(s_sc)[e0 >> 5]));
    const auto* codes = reinterpret_cast<const std::int8_t*>(s_code) + e0;
    const int k0      = tile * (kW8GroupsPerTile * kW8GroupK) + e0;
    const int xoff    = k0 >> 1;
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        const float w0 = static_cast<float>(codes[j]) * scale;
        const float w1 = static_cast<float>(codes[j + 1]) * scale;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const float2 xv =
                __bfloat1622float2(x2_col[static_cast<std::int64_t>(tt) * kK2 + xoff + (j >> 1)]);
            acc[tt] = fmaf(w0, xv.x, acc[tt]);
            acc[tt] = fmaf(w1, xv.y, acc[tt]);
        }
    }
}

template <int kN, int kK, int kRowsPerBlock, int kStages, bool kStageX, int kTt,
          int kActiveCols = kTt>
__global__ void __launch_bounds__(kRowsPerBlock * 32)
    w8_rowsplit_gemv_t_kernel(const __hip_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ scales,
                              __hip_bfloat16* __restrict__ out) {
    constexpr int kGroups   = kK / kW8GroupK;
    constexpr int kTiles    = kGroups / kW8GroupsPerTile;
    constexpr int kPrefetch = kStages - 1;
    static_assert(kGroups % kW8GroupsPerTile == 0, "K must be a multiple of 512");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of kRowsPerBlock");
    constexpr int kK2 = kK / 2;
    // x staging plus the per-warp tile buffers must fit 64 KiB dynamic LDS.
    static_assert(!kStageX ||
                      kK * kTt * 2 + kRowsPerBlock * kStages * (32 + 2) * 16 <= 64 * 1024,
                  "staged x must fit shared");

    __shared__ __align__(16) __hip_bfloat16 x_sh[kStageX ? kK * kTt : 1];
    __shared__ uint4 s_code[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    if constexpr (kStageX) {
        auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        for (int i = static_cast<int>(threadIdx.x); i < kK * kTt / 8;
             i += static_cast<int>(blockDim.x)) {
            x_sh_v[i] = x_g[i];
        }
        __syncthreads();
    }

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kK;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * (kK / kW8GroupK) * kW8ScaleBytesPerGroup;
    const auto* x2 = kStageX ? reinterpret_cast<const __hip_bfloat162*>(x_sh)
                             : reinterpret_cast<const __hip_bfloat162*>(x);

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            w8_gemv_issue_tile(s_code[warp][p], s_sc[warp][p], code_row, scale_row, p, lane);
        }
    }

    float acc[kActiveCols];
#pragma unroll
    for (int tt = 0; tt < kActiveCols; ++tt) { acc[tt] = 0.0f; }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            w8_gemv_issue_tile(s_code[warp][buf], s_sc[warp][buf], code_row, scale_row, fetch, lane);
        }
        const int buf = tile % kStages;
        w8_gemv_consume_tile<kActiveCols, kK2>(x2, s_code[warp][buf], s_sc[warp][buf], tile, lane,
                                               acc);
        __syncwarp();
    }

#pragma unroll
    for (int tt = 0; tt < kActiveCols; ++tt) {
        float v = warp_reduce_sum(acc[tt]);
        if (lane == 0) { out[static_cast<std::int64_t>(tt) * kN + row] = __float2bfloat16_rn(v); }
    }
}

// kTt-column slices of a T-column forward (partial remainders keep compile-time
// ActiveCols so the weight matrix is never re-read for leftover columns).
namespace {
// Dispatch a runtime column count in [1..8] to the matching compile-time
// ActiveCols and call launch(full_tile_size, active_cols, t0).
template <class Launch>
void dispatch_gemv_t_cols(int cols, Launch&& launch) {
    switch (cols) {
    case 1: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}); return;
    case 2: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}); return;
    case 3: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 3>{}); return;
    case 4: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{}); return;
    case 5: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 5>{}); return;
    case 6: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 6>{}); return;
    case 7: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 7>{}); return;
    case 8: launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{}); return;
    default: throw std::invalid_argument("gemv_t: unsupported column dispatch");
    }
}
} // namespace


template <int kN, int kK, int kTt, int kRowsPerBlock = 16, int kStages = 2>
inline void w8_rowsplit_gemv_t_launch_kernel(const __hip_bfloat16* x, const std::uint8_t* codes,
                                             const std::uint8_t* scales, __hip_bfloat16* out,
                                             int cols, hipStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    auto launch = [&](auto full, auto active, int t0) {
        constexpr int kFull   = decltype(full)::value;
        constexpr int kActive = decltype(active)::value;
        constexpr bool kStageX =
            kK * kFull * 2 + kRowsPerBlock * kStages * (32 + 2) * 16 <= 64 * 1024;
        w8_rowsplit_gemv_t_kernel<kN, kK, kRowsPerBlock, kStages, kStageX, kFull, kActive>
            <<<grid, kBlockThreads, 0, stream>>>(x + static_cast<std::int64_t>(t0) * kK, codes,
                                                 scales, out + static_cast<std::int64_t>(t0) * kN);
    };
    if (cols <= 8) {
        dispatch_gemv_t_cols(cols, [&](auto full, auto active) { launch(full, active, 0); });
        return;
    }
    int t0 = 0;
    for (; t0 + kTt <= cols; t0 += kTt) {
        launch(std::integral_constant<int, kTt>{}, std::integral_constant<int, kTt>{}, t0);
    }
    if (t0 < cols) {
        const int rem = cols - t0;
        dispatch_gemv_t_cols(rem, [&](auto full, auto active) { launch(full, active, t0); });
    }
}

} // namespace ninfer::ops::detail
