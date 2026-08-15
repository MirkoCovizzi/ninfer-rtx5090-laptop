#include "hip/hip_runtime.h"
#pragma once

// Q4G64 RowSplit x BF16 Tensor Core GEMM.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// Each CTA owns a BlockRows x BlockCols output tile and advances K in one
// 64-value quant group at a time. Raw Q4 codes and FP16 scales are staged
// independently from the BF16 activation tile. Q4 values are decoded to BF16
// in shared memory, then consumed by m16n8k16 BF16 MMA with FP32 accumulation.
//
// The template describes only physical kernel structure. Rows, Cols, and K
// remain runtime dimensions; registered shapes select a closed schedule and a
// statically compiled boundary variant outside the kernel.

#include "ops/common/mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4FragmentPipeline {
    Serial,
    PingPong,
};

enum class Q4ScaleLoad {
    Scalar16,
    Pair32,
};

template <int BlockRows_, int BlockCols_, int BlockK_, int WarpRows_, int WarpCols_,
          int PipelineStages_, int LaunchBoundsMinBlocks_, Q4FragmentPipeline FragmentPipeline_,
          Cache QuantCache_, Cache ActivationCache_, Q4ScaleLoad ScaleLoadMode_>
struct Q4RowSplitMmaGemmSchedule {
    static constexpr int kBlockRows = BlockRows_;
    static constexpr int kBlockCols = BlockCols_;
    static constexpr int kBlockK    = BlockK_;
    static constexpr int kWarpRows  = WarpRows_;
    static constexpr int kWarpCols  = WarpCols_;

    static constexpr int kPipelineStages                  = PipelineStages_;
    static constexpr int kLaunchBoundsMinBlocks           = LaunchBoundsMinBlocks_;
    static constexpr Q4FragmentPipeline kFragmentPipeline = FragmentPipeline_;
    static constexpr Cache kQuantCache                    = QuantCache_;
    static constexpr Cache kActivationCache               = ActivationCache_;
    static constexpr Q4ScaleLoad kScaleLoadMode           = ScaleLoadMode_;

    static constexpr int kWarpGridRows = kBlockRows / kWarpRows;
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMmaRows      = kWarpRows / 16;
    static constexpr int kMmaCols      = kWarpCols / 8;
    static constexpr int kMmaKSteps    = kBlockK / 16;
    static constexpr int kGroupsPerK   = kBlockK / Q4RowSplitStorage::kGroupK;
    static constexpr int kScaleBytes =
        kScaleLoadMode == Q4ScaleLoad::Pair32 ? 4 : Q4RowSplitStorage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__hip_bfloat16)) +
        kPipelineStages * kBlockCols * kBlockK * static_cast<int>(sizeof(__hip_bfloat16)) +
        kPipelineStages * kBlockRows * kGroupsPerK * Q4RowSplitStorage::kCodeBytesPerGroup +
        kPipelineStages * kBlockRows * kGroupsPerK * kScaleBytes;

    static_assert(kBlockRows > 0 && kBlockCols > 0);
    static_assert(kBlockK == Q4RowSplitStorage::kGroupK,
                  "Q4 MMA currently stages exactly one quant group per K tile");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0,
                  "Q4 MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0,
                  "Q4 MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "Q4 MMA cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q4 MMA staged shared memory exceeds the static 48 KiB budget");
};

// XOR swizzle for a [rows][64] BF16 tile. The eight 16-byte column groups are
// permuted by the low row bits so ldmatrix reads do not repeatedly hit the same
// shared-memory bank group.
__device__ __forceinline__ int q4_mma_swizzle_k64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// clang-format off
template <class Schedule_, bool Full>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kLaunchBoundsMinBlocks)
void q4_rowsplit_gemm_mma_kernel(
    const __hip_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k) {
    // clang-format on
    using Schedule       = Schedule_;
    constexpr bool kFull = Full;
    constexpr int BM     = Schedule::kBlockRows;
    constexpr int BN     = Schedule::kBlockCols;
    constexpr int BK     = Schedule::kBlockK;
    constexpr int WM     = Schedule::kWarpRows;
    constexpr int WN     = Schedule::kWarpCols;
    constexpr int MT     = Schedule::kMmaRows;
    constexpr int NT     = Schedule::kMmaCols;
    constexpr int KSUB   = Schedule::kMmaKSteps;
    constexpr int S      = Schedule::kPipelineStages;
    constexpr int GPB    = Schedule::kGroupsPerK;
    constexpr int SB     = Schedule::kScaleBytes;

    __shared__ __align__(16) __hip_bfloat16 As[BM * BK];
    __shared__ __align__(16) __hip_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * GPB * Q4RowSplitStorage::kCodeBytesPerGroup];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * GPB * SB];

    const int groups_per_row = padded_k / Q4RowSplitStorage::kGroupK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridCols;
    const int warp_col       = warp % Schedule::kWarpGridCols;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    // gfx1151 WMMA atoms are 16x16 (not 16x8). Each warp covers WMT x WNT 16x16
    // atoms; when kWarpCols is not a multiple of 16 the final atom overlaps the
    // neighbouring warp's columns (duplicate compute, correct results) and the
    // store below emits only the warp's own columns.
    constexpr int WMT = MT;
    constexpr int WNT = (Schedule::kWarpCols + 15) / 16;
    float accum[WMT][WNT][8];
#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { accum[mi][ni][r] = 0.0f; }
        }
    }

    const int k_tiles = padded_k / BK;


    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst = &Bs[stage][local_col * BK + q4_mma_swizzle_k64(local_col, k8 * 8)];
            if constexpr (kFull) {
                if (col < cols) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            } else {
                if (col < cols && kk + 8 <= k) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group0 = (k_tile * BK) / Q4RowSplitStorage::kGroupK;
#pragma unroll 1
        for (int item = tid; item < BM * GPB * 2; item += Schedule::kThreads) {
            const int row_group = item >> 1;
            const int half      = item & 1;
            const int local_row = row_group / GPB;
            const int group     = row_group - local_row * GPB;
            const int row       = row0 + local_row;
            auto* dst = &Cr[stage][row_group * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                cp_async<16, Schedule::kQuantCache>(
                    dst, &codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16]);
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                    cp_async<16, Schedule::kQuantCache>(
                        dst,
                        &codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }

#pragma unroll 1
        for (int row_group = tid; row_group < BM * GPB; row_group += Schedule::kThreads) {
            const int local_row   = row_group / GPB;
            const int group       = row_group - local_row * GPB;
            const int row         = row0 + local_row;
            const int scale_group = group0 + group;
            auto* dst             = &Sr[stage][row_group * SB];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                if constexpr (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) {
                    const int aligned_group = scale_group & ~1;
                    const std::int64_t aligned_index =
                        static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                    if (aligned_group + 1 < groups_per_row) {
                        cp_async<4>(
                            dst, &scales[aligned_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(
                            &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                }
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                    if constexpr (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) {
                        const int aligned_group = scale_group & ~1;
                        const std::int64_t aligned_index =
                            static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                        if (aligned_group + 1 < groups_per_row) {
                            cp_async<4>(
                                dst,
                                &scales[aligned_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                        } else {
                            *reinterpret_cast<std::uint16_t*>(dst) =
                                *reinterpret_cast<const std::uint16_t*>(
                                    &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                            *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                        }
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                    }
                } else {
                    dst[0] = 0;
                    dst[1] = 0;
                    if constexpr (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) {
                        dst[2] = 0;
                        dst[3] = 0;
                    }
                }
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    auto decode_weight = [&](int stage, int k_tile) {
        const int scale_group = (k_tile * BK) / Q4RowSplitStorage::kGroupK;
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            auto* dst = &As[local_row * BK];
#pragma unroll
            for (int group = 0; group < GPB; ++group) {
                const int staged_group = local_row * GPB + group;
                const std::uint8_t* scale_ptr =
                    &Sr[stage][staged_group * SB + (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32
                                                        ? ((scale_group + group) & 1) *
                                                              Q4RowSplitStorage::kScaleBytesPerGroup
                                                        : 0)];
                const __hip_bfloat162 weights =
                    Q4MmaDecodeAtom::decode_pair(Cr[stage], scale_ptr, staged_group, lane);
                const int shared_col =
                    q4_mma_swizzle_k64(local_row, group * Q4RowSplitStorage::kGroupK + 2 * lane);
                store_vec(&dst[shared_col], weights);
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        decode_weight(stage, k_tile);
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_frag)[WMT][8], unsigned(&b_frag)[WNT][8]) {
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
                const int row = warp_row * WM + mi * 16 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    wmma_load_a_bf16(a_frag[mi], As, row, k_step, BK, q4_mma_swizzle_k64);
                }
            }
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int col = warp_col * WN + ni * 16 + (lane & 15);
                wmma_load_b_bf16(b_frag[ni], Bs[stage], col, k_step, BK, q4_mma_swizzle_k64);
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Q4FragmentPipeline::PingPong) {
            unsigned a_frag[2][WMT][8];
            unsigned b_frag[2][WNT][8];
            load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int current = ki & 1;
                const int next    = (ki + 1) & 1;
                if (ki + 1 < KSUB) { load_fragments((ki + 1) * 16, a_frag[next], b_frag[next]); }
#pragma unroll
                for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < WNT; ++ni) {
                        WmmaC8& c = *reinterpret_cast<WmmaC8*>(accum[mi][ni]);
                        WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag[current][mi]);
                        WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag[current][ni]);
                        c = wmma_bf16(a, b, c);
                    }
                }
            }
        } else {
            unsigned a_frag[WMT][8];
            unsigned b_frag[WNT][8];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                load_fragments(ki * 16, a_frag, b_frag);
#pragma unroll
                for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < WNT; ++ni) {
                        WmmaC8& c = *reinterpret_cast<WmmaC8*>(accum[mi][ni]);
                        WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag[mi]);
                        WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag[ni]);
                        c = wmma_bf16(a, b, c);
                    }
                }
            }
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
        const int row_base = row0 + warp_row * WM + mi * 16;
        const int row_lo   = row_base + (lane >= 16 ? 8 : 0);
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
            const int col_base   = col0 + warp_col * WN + ni * 16;
            const int local_col  = ni * 16 + (lane & 15);
            const int output_col = col_base + (lane & 15);
            // Skip the overlap columns when kWarpCols is not a multiple of 16.
            if (local_col >= WN) { continue; }
            if constexpr (kFull) {
                if (output_col < cols) {
                    const float* values = accum[mi][ni];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        out[static_cast<std::int64_t>(output_col) * rows + row_lo + r] =
                            __float2bfloat16_rn(values[r]);
                    }
                }
            } else {
                const float* values = accum[mi][ni];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int row = row_lo + r;
                    if (row < rows && output_col < cols) {
                        out[static_cast<std::int64_t>(output_col) * rows + row] =
                            __float2bfloat16_rn(values[r]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail