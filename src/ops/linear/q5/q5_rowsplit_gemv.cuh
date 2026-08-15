#include "hip/hip_runtime.h"
#pragma once

// Shared Q5 row-split decode (T=1) GEMV core.
//
// Memory-bound design for the narrow Qwen3.6 decode shapes (N ~= 5-7K):
//   - A block owns kRowsPerBlock output rows; one warp computes one full row.
//   - Each warp streams its row's 16-group weight tiles through shared memory with
//     a cp.async pipeline kStages buffers deep: tiles j+1 .. j+kStages-1 are in
//     flight while tile j is unpacked/accumulated, so DRAM latency is hidden
//     without relying on very high occupancy (these matrices cannot fill the GPU
//     with one warp per row otherwise).
//   - The activation x (full K vector, reused by every row) is optionally staged
//     into shared once per block (kStageX); disable it when K is so large that x
//     would not fit alongside the weight buffers (e.g. mlp_down, K=17408).
//   - Each warp does a warp-shuffle reduction and writes one bf16 output; there is
//     no block barrier on the hot path.
// A tile is 16 groups: 512 nibble bytes (32 uint4, one per lane), 128 high bytes
// (8 uint4), 32 scale bytes (2 uint4). All weight reads are fully coalesced 128-bit
// loads, so the kernel runs DRAM-bound instead of L1/LSU- or latency-bound.

#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Issue the cp.async copies for one 16-group tile into a shared buffer slot, then
// commit them as one pipeline group. Every lane commits (even lanes that issue
// fewer copies) so a uniform pipe_wait works warp-wide.
__device__ __forceinline__ void
q5_gemv_issue_tile(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi, uint4* __restrict__ s_sc,
                   const std::uint8_t* __restrict__ code_row,
                   const std::uint8_t* __restrict__ high_row,
                   const std::uint8_t* __restrict__ scale_row, int tile, int lane) {
    constexpr int kGroupsPerTile = 16;
    const int g0                 = tile * kGroupsPerTile;
    pipe_copy<16>(&s_nib[lane], reinterpret_cast<const uint4*>(code_row + g0 * 32) + lane);
    if (lane < 8) {
        pipe_copy<16>(&s_hi[lane], reinterpret_cast<const uint4*>(high_row + g0 * 8) + lane);
    }
    if (lane < 2) {
        pipe_copy<16>(&s_sc[lane], reinterpret_cast<const uint4*>(scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

// Unpack + accumulate one staged 16-group tile. x is read from x2 (shared or global).
// kTt > 1 processes kTt token columns per tile; each column's x slice starts at
// x2_col[tt * kK2] (kK2 = kK/2 bf16x2 words). Weight tiles are shared across columns.
template <int kTt, int kK2>
__device__ __forceinline__ void q5_gemv_consume_tile(const __hip_bfloat162* __restrict__ x2_col,
                                                     const uint4* __restrict__ s_nib,
                                                     const uint4* __restrict__ s_hi,
                                                     const uint4* __restrict__ s_sc, int tile,
                                                     int lane, float (&acc)[kTt]) {
    constexpr int kGroupK              = 64;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    const auto* tn                     = reinterpret_cast<const std::uint8_t*>(s_nib);
    const auto* th                     = reinterpret_cast<const std::uint8_t*>(s_hi);
    const auto* tsc                    = reinterpret_cast<const std::uint16_t*>(s_sc);
    const int g0                       = tile * kGroupsPerTile;
#pragma unroll
    for (int tg = 0; tg < kGroupsPerTile; ++tg) {
        const float scale       = __half2float(__ushort_as_half(tsc[tg]));
        const std::uint8_t low  = tn[tg * kNibbleBytesPerGroup + lane];
        const std::uint8_t high = th[tg * kHighBytesPerGroup + (lane >> 2)] >> ((lane & 3) * 2);
        const int q0 = sign_extend<5>(static_cast<int>((low & 0x0fu) | ((high & 0x01u) << 4)));
        const int q1 = sign_extend<5>(static_cast<int>((low >> 4) | ((high & 0x02u) << 3)));

        const int k0 = (g0 + tg) * kGroupK + lane * 2;
        const int xoff = k0 >> 1;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const float2 xv = __bfloat1622float2(x2_col[static_cast<std::int64_t>(tt) * kK2 + xoff]);
            acc[tt]         = fmaf(static_cast<float>(q0) * scale, xv.x, acc[tt]);
            acc[tt]         = fmaf(static_cast<float>(q1) * scale, xv.y, acc[tt]);
        }
    }
}

// kN  : output rows, kK : reduction dim (multiple of 1024).
// kRowsPerBlock : rows (= warps) per block.
// kStages : cp.async pipeline depth (shared buffers; >=2 to overlap).
// kStageX : stage the activation vector into shared (false when x is too large).
struct Q5GemvStoreEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__hip_bfloat16* out, __hip_bfloat16* out_tail, int row,
                                               float value) const {
        if constexpr (SplitOutput) {
            if (row < SplitRow) {
                out[row] = __float2bfloat16_rn(value);
            } else {
                out_tail[row - SplitRow] = __float2bfloat16_rn(value);
            }
        } else {
            out[row] = __float2bfloat16_rn(value);
        }
    }
};

// kTt = number of token columns computed per launch (kTt == 1 keeps the T=1 path
// bit-identical; kTt > 1 reads x from global memory column slices, weights once).
template <int kN, int kK, int kRowsPerBlock, int kStages, bool kStageX, bool kResidual,
          bool kSplitOutput = false, int kSplitRow = 0, class Epilogue = Q5GemvStoreEpilogue,
          bool TriggerPdl = false, bool JoinPdl = false, int kTt = 1,
          int kActiveCols = kTt>
__global__ void
q5_rowsplit_gemv_kernel(const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                        const std::uint8_t* __restrict__ high_bits,
                        const std::uint8_t* __restrict__ scales, __hip_bfloat16* __restrict__ out,
                        __hip_bfloat16* __restrict__ out_tail, Epilogue epilogue = {}) {
    constexpr int kGroupK              = 64;
    constexpr int kGroups              = kK / kGroupK;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kTiles               = kGroups / kGroupsPerTile;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    constexpr int kXVecs               = kK / 8; // x as uint4 (8 bf16 each)
    constexpr int kPrefetch            = kStages - 1;
    static_assert(kGroups % kGroupsPerTile == 0, "K must be a multiple of 16 groups (1024)");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of kRowsPerBlock");
    static_assert(kStages >= 2, "need at least double buffering");
    static_assert(!kSplitOutput || (kSplitRow > 0 && kSplitRow < kN),
                  "split-output Q5 GEMV requires an interior compile-time seam");
    static_assert(!kResidual || !kSplitOutput, "the Q5 residual GEMV epilogue is contiguous-only");
    static_assert(!kStageX || kTt == 1, "kTt > 1 reads x column slices from global memory");
    static_assert(!kSplitOutput || kTt == 1, "split-output Q5 GEMV is T=1 only");
    static_assert(kK % (kGroupK * kGroupsPerTile) == 0, "K must be a multiple of 1024");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    // __align__(16) so the uint4 staging below is well-defined by construction.
    __shared__ __align__(16) __hip_bfloat16 x_sh[kStageX ? kK : 1];
    __shared__ uint4 s_nib[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_hi[kRowsPerBlock][kStages][8];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    if constexpr (kStageX) {
        // x must be 16-byte aligned for the uint4 staging (guaranteed: activations
        // come from the 256-byte-aligned workspace arena).
        auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
            x_sh_v[i] = x_g[i];
        }
        __syncthreads();
    }

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kNibbleBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2                = kStageX ? reinterpret_cast<const __hip_bfloat162*>(x_sh)
                                            : reinterpret_cast<const __hip_bfloat162*>(x);
    constexpr int kK2             = kK / 2;

    // Prime up to kPrefetch tiles; empty commits keep the group count uniform so the
    // constant pipe_wait<kPrefetch>() in the loop is always valid.
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q5_gemv_issue_tile(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row, high_row,
                               scale_row, p, lane);
        } else {
            pipe_commit();
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
            q5_gemv_issue_tile(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row,
                               high_row, scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = tile % kStages;
        q5_gemv_consume_tile<kActiveCols, kK2>(x2, s_nib[warp][buf], s_hi[warp][buf],
                                               s_sc[warp][buf], tile, lane, acc);
        __syncwarp();
    }

#pragma unroll
    for (int tt = 0; tt < kActiveCols; ++tt) {
        float v = warp_reduce_sum(acc[tt]);
        if (lane == 0) {
            if constexpr (kResidual) { v += __bfloat162float(out[static_cast<std::int64_t>(tt) * kN + row]); }
            if constexpr (kActiveCols == 1) {
                epilogue.template operator()<kSplitOutput, kSplitRow>(out, out_tail, row, v);
            } else {
                out[static_cast<std::int64_t>(tt) * kN + row] = __float2bfloat16_rn(v);
            }
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

// One block per kRowsPerBlock rows; kRowsPerBlock warps per block.
template <int kN, int kK, int kRowsPerBlock, int kStages = 2, bool kStageX = true>
inline void q5_rowsplit_gemv_launch_kernel(const __hip_bfloat16* x, const std::uint8_t* codes,
                                           const std::uint8_t* high_bits,
                                           const std::uint8_t* scales, __hip_bfloat16* out,
                                           hipStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kN, kK, kRowsPerBlock, kStages, kStageX, false>
        <<<grid, kBlockThreads, 0, stream>>>(x, codes, high_bits, scales, out, nullptr);
}

template <int kN, int kK, int kRowsPerBlock, int kStages = 2, bool kStageX = true>
inline void
q5_rowsplit_gemv_residual_launch_kernel(const __hip_bfloat16* x, const std::uint8_t* codes,
                                        const std::uint8_t* high_bits, const std::uint8_t* scales,
                                        __hip_bfloat16* residual_out, hipStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kN, kK, kRowsPerBlock, kStages, kStageX, true>
        <<<grid, kBlockThreads, 0, stream>>>(x, codes, high_bits, scales, residual_out, nullptr);
}

// kTt-column slices of a T-column forward. Each slice reads the weight matrix once and
// streams the kTt x column slices (global, L2-resident); keeps the decode/memory-bound
// GEMV structure for small-T batches where the split GEMM paths underutilize the GPU.

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

// One weight stream per launch: cols <= 4 uses a kTt=4 tile with kActiveCols compile-time
// columns; cols in 5..8 uses a kTt=8 tile; larger T slices at kTt=4 with an exact partial
// remainder (never a per-column re-read of the weight matrix).
template <int kN, int kK, int kTt, int kRowsPerBlock = 16, int kStages = 2>
inline void q5_rowsplit_gemv_t_launch_kernel(const __hip_bfloat16* x, const std::uint8_t* codes,
                                             const std::uint8_t* high_bits,
                                             const std::uint8_t* scales, __hip_bfloat16* out,
                                             int cols, hipStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    auto launch = [&](auto full, auto active, int t0) {
        constexpr int kFull = decltype(full)::value;
        constexpr int kActive = decltype(active)::value;
        q5_rowsplit_gemv_kernel<kN, kK, kRowsPerBlock, kStages, false, false, false, 0,
                                Q5GemvStoreEpilogue, false, false, kFull, kActive>
            <<<grid, kBlockThreads, 0, stream>>>(x + static_cast<std::int64_t>(t0) * kK, codes,
                                                 high_bits, scales,
                                                 out + static_cast<std::int64_t>(t0) * kN, nullptr);
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

template <int kN, int kK, int kTt, int kRowsPerBlock = 16, int kStages = 2>
inline void
q5_rowsplit_gemv_t_residual_launch_kernel(const __hip_bfloat16* x, const std::uint8_t* codes,
                                          const std::uint8_t* high_bits, const std::uint8_t* scales,
                                          __hip_bfloat16* residual_out, int cols,
                                          hipStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    auto launch = [&](auto full, auto active, int t0) {
        constexpr int kFull = decltype(full)::value;
        constexpr int kActive = decltype(active)::value;
        q5_rowsplit_gemv_kernel<kN, kK, kRowsPerBlock, kStages, false, true, false, 0,
                                Q5GemvStoreEpilogue, false, false, kFull, kActive>
            <<<grid, kBlockThreads, 0, stream>>>(x + static_cast<std::int64_t>(t0) * kK, codes,
                                                 high_bits, scales,
                                                 residual_out + static_cast<std::int64_t>(t0) * kN,
                                                 nullptr);
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
