#pragma once

// Q4 row-split small-T GEMV (warp per row, streamed weight tiles) for the T=2..8
// batched forward, mirroring the q5 GEMV-T: the SIMT r8_c4 path that previously
// served these shapes runs far below DRAM bandwidth on gfx1151, while the q4
// T=1 GEMV already streams at the wall. Weights are read once per kTt-column
// slice; x column slices come from global memory (L2-resident).

#include "hip/hip_runtime.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
constexpr int kQ4TGroupK = 64;
constexpr int kQ4TGroupsPerTile = 16;
constexpr int kQ4TCodeBytesPerGroup = 32;
constexpr int kQ4TScaleBytesPerGroup = 2;
} // namespace

__device__ __forceinline__ void
q4_gemv_t_issue_tile(uint4* __restrict__ s_code, uint4* __restrict__ s_sc,
                     const std::uint8_t* __restrict__ code_row,
                     const std::uint8_t* __restrict__ scale_row, int tile, int lane) {
    const int g0 = tile * kQ4TGroupsPerTile;
    s_code[lane] = *reinterpret_cast<const uint4*>(code_row + g0 * kQ4TCodeBytesPerGroup +
                                                   lane * 16);
    if (lane < 2) {
        s_sc[lane] = *reinterpret_cast<const uint4*>(scale_row + g0 * kQ4TScaleBytesPerGroup +
                                                     lane * 16);
    }
}

template <int kTt, int kK2>
__device__ __forceinline__ void q4_gemv_t_consume_tile(const __hip_bfloat162* __restrict__ x2_col,
                                                       const uint4* __restrict__ s_code,
                                                       const uint4* __restrict__ s_sc, int tile,
                                                       int lane, float (&acc)[kTt]) {
    const int g0 = tile * kQ4TGroupsPerTile;
    const auto* codes = reinterpret_cast<const std::uint8_t*>(s_code);
    const auto* tsc   = reinterpret_cast<const std::uint16_t*>(s_sc);
#pragma unroll
    for (int tg = 0; tg < kQ4TGroupsPerTile; ++tg) {
        const float scale = __half2float(__ushort_as_half(tsc[tg]));
        const int packed  = static_cast<int>(codes[tg * kQ4TCodeBytesPerGroup + lane]);
        const int q0      = sign_extend<4>(packed & 0x0f);
        const int q1      = sign_extend<4>(packed >> 4);
        const int k0      = (g0 + tg) * kQ4TGroupK + lane * 2;
        const int xoff    = k0 >> 1;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const float2 xv = __bfloat1622float2(x2_col[static_cast<std::int64_t>(tt) * kK2 + xoff]);
            acc[tt]         = fmaf(static_cast<float>(q0) * scale, xv.x, acc[tt]);
            acc[tt]         = fmaf(static_cast<float>(q1) * scale, xv.y, acc[tt]);
        }
    }
}

template <int kN, int kK, int kRowsPerBlock, int kStages, int kTt, int kActiveCols = kTt>
__global__ void __launch_bounds__(kRowsPerBlock * 32)
    q4_rowsplit_gemv_t_kernel(const __hip_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ scales,
                              __hip_bfloat16* __restrict__ out) {
    constexpr int kGroups   = kK / kQ4TGroupK;
    constexpr int kTiles    = kGroups / kQ4TGroupsPerTile;
    constexpr int kPrefetch = kStages - 1;
    static_assert(kGroups % kQ4TGroupsPerTile == 0, "K must be a multiple of 1024");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of kRowsPerBlock");
    constexpr int kK2 = kK / 2;

    __shared__ uint4 s_code[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * (kK / 2);
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * kQ4TScaleBytesPerGroup;
    const auto* x2 = reinterpret_cast<const __hip_bfloat162*>(x);

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_gemv_t_issue_tile(s_code[warp][p], s_sc[warp][p], code_row, scale_row, p, lane);
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
            q4_gemv_t_issue_tile(s_code[warp][buf], s_sc[warp][buf], code_row, scale_row, fetch,
                                 lane);
        }
        const int buf = tile % kStages;
        q4_gemv_t_consume_tile<kActiveCols, kK2>(x2, s_code[warp][buf], s_sc[warp][buf], tile, lane,
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
inline void q4_rowsplit_gemv_t_launch_kernel(const __hip_bfloat16* x, const std::uint8_t* codes,
                                             const std::uint8_t* scales, __hip_bfloat16* out,
                                             int cols, hipStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    auto launch = [&](auto full, auto active, int t0) {
        constexpr int kFull = decltype(full)::value;
        constexpr int kActive = decltype(active)::value;
        q4_rowsplit_gemv_t_kernel<kN, kK, kRowsPerBlock, kStages, kFull, kActive>
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
