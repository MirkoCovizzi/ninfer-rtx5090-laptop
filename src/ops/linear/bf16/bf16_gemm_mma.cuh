#include "hip/hip_runtime.h"
#pragma once

// Dense BF16 x BF16 Tensor Core GEMM shared by pure Linear and semantic Ops.
//
//   C[M,N] = A[M,K] * B[K,N]
//
// A is row-major with contiguous K. Public NInfer activations store one contiguous
// K vector per token, which is the column-major B representation consumed by the
// MMA atom. M and K come from a compiled geometry; N remains runtime.
//
// The tensor core on gfx1151 is the wave32 WMMA instruction (see mma.cuh). Each
// warp computes a WarpRows x WarpCols tile from MmaRows x MmaCols 16x16x16 atoms;
// the k dimension of one atom is 16, matching the schedule's 16-element K steps.

#include "ops/common/mma.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16MmaFragmentPipeline : std::uint8_t {
    Serial,
    PingPong,
};

enum class Bf16MmaRaster : std::uint8_t {
    TokenFast,
    RowFast,
    Grouped,
};

enum class Bf16MmaSwizzle : std::uint8_t {
    Plain,
    Xor64,
};

template <int BlockRows, int BlockCols, int BlockK, int WarpRows, int WarpCols, int PipelineStages,
          int MinBlocks, Cache WeightCache, Cache ActivationCache,
          Bf16MmaFragmentPipeline FragmentPipeline, Bf16MmaRaster Raster,
          Bf16MmaSwizzle Swizzle = Bf16MmaSwizzle::Xor64, int RasterGroupRows = 1>
struct Bf16MmaSchedule {
    static constexpr int kBlockRows                            = BlockRows;
    static constexpr int kBlockCols                            = BlockCols;
    static constexpr int kBlockK                               = BlockK;
    static constexpr int kWarpRows                             = WarpRows;
    static constexpr int kWarpCols                             = WarpCols;
    static constexpr int kPipelineStages                       = PipelineStages;
    static constexpr int kMinBlocks                            = MinBlocks;
    static constexpr Cache kWeightCache                        = WeightCache;
    static constexpr Cache kActivationCache                    = ActivationCache;
    static constexpr Bf16MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Bf16MmaRaster kRaster                     = Raster;
    static constexpr Bf16MmaSwizzle kSwizzle                   = Swizzle;
    static constexpr int kRasterGroupRows                      = RasterGroupRows;

    static constexpr int kWarpsM         = kBlockRows / kWarpRows;
    static constexpr int kWarpsN         = kBlockCols / kWarpCols;
    static constexpr int kWarps          = kWarpsM * kWarpsN;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaCols        = kWarpCols / 16;
    static constexpr int kMmaK           = kBlockK / 16;
    static constexpr int kSharedElements = kPipelineStages * (kBlockRows + kBlockCols) * kBlockK;
    static constexpr int kSharedBytes = kSharedElements * static_cast<int>(sizeof(__hip_bfloat16));

    static_assert(kBlockRows > 0 && kBlockCols > 0 && kBlockK > 0);
    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0);
    // gfx1151 WMMA atoms are 16x16 (not 16x8). kWarpRows must tile 16-row atoms;
    // kWarpCols spins div_up(kWarpCols,16) atoms and the last overlaps the neighbour warp's
    // columns (filtered at store), so only an 8-element alignment is required for the B tile.
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0);
    static_assert(kBlockK % 64 == 0);
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8);
    static_assert(kMinBlocks >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
    static_assert(kRaster != Bf16MmaRaster::Grouped || kRasterGroupRows > 0);
};

struct Bf16MmaOutputTile {
    __hip_bfloat16* data;
    std::int32_t leading_dim;
    std::int32_t parent_row_begin;

    __device__ __forceinline__ __hip_bfloat16* at(std::int32_t parent_row,
                                                  std::int32_t token) const {
        return data + static_cast<std::int64_t>(token) * leading_dim + parent_row -
               parent_row_begin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *at(parent_row, token) = __float2bfloat16_rn(value);
    }
};

struct Bf16MmaContiguousOutput {
    __hip_bfloat16* data;
    std::int32_t leading_dim;

    __device__ __forceinline__ Bf16MmaOutputTile tile(std::int32_t) const {
        return {data, leading_dim, 0};
    }
};

template <class Schedule>
__device__ __forceinline__ int bf16_mma_shared_col(int row, int col) {
    if constexpr (Schedule::kSwizzle == Bf16MmaSwizzle::Plain) {
        return col;
    } else {
        return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
    }
}

template <class Schedule>
__device__ __forceinline__ void
bf16_mma_tile_coordinates(std::int32_t linear, std::int32_t tiles_m, std::int32_t tiles_n,
                          std::int32_t& tile_m, std::int32_t& tile_n) {
    if constexpr (Schedule::kRaster == Bf16MmaRaster::TokenFast) {
        tile_m = linear / tiles_n;
        tile_n = linear - tile_m * tiles_n;
    } else if constexpr (Schedule::kRaster == Bf16MmaRaster::RowFast) {
        tile_n = linear / tiles_m;
        tile_m = linear - tile_n * tiles_m;
    } else {
        constexpr int group_rows      = Schedule::kRasterGroupRows;
        const std::int32_t group_span = group_rows * tiles_n;
        const std::int32_t group      = linear / group_span;
        const std::int32_t first_m    = group * group_rows;
        const std::int32_t active_m   = min(group_rows, tiles_m - first_m);
        const std::int32_t within     = linear - group * group_span;
        tile_m                        = first_m + within % active_m;
        tile_n                        = within / active_m;
    }
}

// Loads one lane's 16 bf16 fragment elements from shared memory. The Xor64 swizzle
// keeps each 8-element K group contiguous, so two 16-byte loads cover k 0..7 and
// k 8..15. The fragment register order matches the WMMA k = element index layout.
__device__ __forceinline__ void ds_load_b128(unsigned* out, unsigned addr, int offset_dwords) {
    uint4 v;
    // Leading s_waitcnt lgkmcnt(0) drains this wave's LDS pipe after the staging
    // barrier (gfx1151 does not always make other waves' ds_stores visible to a
    // subsequent ds_read under multi-wave occupancy). The "memory" clobber is
    // required: without it the read is not a memory operation at IR level and LLVM
    // hoists it across __syncthreads(), racing the staging stores.
    asm volatile("s_waitcnt lgkmcnt(0);\n ds_read_b128 %0, %1 offset:%2;\n s_waitcnt lgkmcnt(0);\n"
                 : "=v"(v)
                 : "v"(addr), "n"(offset_dwords)
                 : "memory");
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}

template <class Schedule>
__device__ __forceinline__ void bf16_wmma_load_a(unsigned (&frag)[8], const __hip_bfloat16* base,
                                                 int row, int k0, int stride_k) {
    const unsigned chunk1 =
        smem_addr(&base[row * stride_k + bf16_mma_shared_col<Schedule>(row, k0)]);
    const unsigned chunk2 =
        smem_addr(&base[row * stride_k + bf16_mma_shared_col<Schedule>(row, k0 + 8)]);
    ds_load_b128(frag, chunk1, 0);
    ds_load_b128(frag + 4, chunk2, 0);
}

// Same layout for the B tile: each lane covers one token column of 16 K values.
template <class Schedule>
__device__ __forceinline__ void bf16_wmma_load_b(unsigned (&frag)[8], const __hip_bfloat16* base,
                                                 int col, int k0, int stride_k) {
    const unsigned chunk1 =
        smem_addr(&base[col * stride_k + bf16_mma_shared_col<Schedule>(col, k0)]);
    const unsigned chunk2 =
        smem_addr(&base[col * stride_k + bf16_mma_shared_col<Schedule>(col, k0 + 8)]);
    ds_load_b128(frag, chunk1, 0);
    ds_load_b128(frag + 4, chunk2, 0);
}

template <class Geometry, class Schedule, bool FullTokens, class Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) void bf16_gemm_mma_kernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ weight, Output output,
    std::int32_t tokens) {
    constexpr int M       = Geometry::kOutputRows;
    constexpr int K       = Geometry::kInputRows;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockCols;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int WM      = Schedule::kWarpRows;
    constexpr int WN      = Schedule::kWarpCols;
    constexpr int MT      = Schedule::kMmaRows;
    // gfx1151 WMMA atoms are 16x16 (not 16x8). Each warp covers WMT x WNT 16x16 atoms;
    // when kWarpCols is not a multiple of 16 the final atom overlaps the neighbour warp's
    // columns (duplicate compute, correct results) and the store below emits only this warp's
    // own columns.
    constexpr int WNT     = (Schedule::kWarpCols + 15) / 16;
    constexpr int KSUB    = Schedule::kMmaK;
    constexpr int S       = Schedule::kPipelineStages;
    constexpr int WARPS_N = Schedule::kWarpsN;
    constexpr int THREADS = Schedule::kThreads;
    static_assert(M % BM == 0);
    static_assert(K % BK == 0);
    static_assert(K / BK >= S);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__hip_bfloat16*>(shared_raw);
    auto* Bs = As + S * BM * BK;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / WARPS_N;
    const int wn   = warp - wm * WARPS_N;

    constexpr int tiles_m = M / BM;
    const int tiles_n     = tokens / BN + static_cast<int>(tokens % BN != 0);
    int tile_m            = 0;
    int tile_n            = 0;
    bf16_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), tiles_m, tiles_n, tile_m,
                                        tile_n);
    const int m0           = tile_m * BM;
    const int n0           = tile_n * BN;
    const auto output_tile = output.tile(m0);

    float accum[MT][WNT][8] = {};

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k0  = k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;

#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8);
            const int k8  = item - row * (BK / 8);
            const int kk  = k8 * 8;
            cp_async<16, Schedule::kWeightCache>(
                &a_stage[row * BK + bf16_mma_shared_col<Schedule>(row, kk)],
                &weight[static_cast<std::int64_t>(m0 + row) * K + k0 + kk]);
        }

#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int col   = item / (BK / 8);
            const int k8    = item - col * (BK / 8);
            const int kk    = k8 * 8;
            auto* dst       = &b_stage[col * BK + bf16_mma_shared_col<Schedule>(col, kk)];
            const int token = n0 + col;
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(token) * K + k0 + kk]);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(valid ? token : 0) * K + k0 + kk],
                    valid ? 16 : 0);
            }
        }
    };

    constexpr int kTiles = K / BK;
#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= kTiles) {
            cp_wait<S - 1>();
        } else {
            // Once the producer stops refilling the ring, fewer than S groups remain. Drain them
            // instead of applying the steady-state wait count to the final consumer stage.
            cp_wait<0>();
        }
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_frag)[MT][8], unsigned(&b_frag)[WNT][8]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    bf16_wmma_load_a<Schedule>(a_frag[mi], As + stage * BM * BK, row,
                                               k_step * 16, BK);
                }
            }
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int col = wn * WN + ni * 16 + (lane & 15);
                bf16_wmma_load_b<Schedule>(b_frag[ni], Bs + stage * BN * BK, col, k_step * 16,
                                           BK);
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Bf16MmaFragmentPipeline::PingPong) {
            unsigned a_frag[2][MT][8];
            unsigned b_frag[2][WNT][8];
            load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
            for (int k_step = 0; k_step < KSUB; ++k_step) {
                const int slot = k_step & 1;
                if (k_step + 1 < KSUB) {
                    load_fragments(k_step + 1, a_frag[slot ^ 1], b_frag[slot ^ 1]);
                }
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < WNT; ++ni) {
                        WmmaC8& c = *reinterpret_cast<WmmaC8*>(accum[mi][ni]);
                        WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag[slot][mi]);
                        WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag[slot][ni]);
                        c = wmma_bf16(a, b, c);
                    }
                }
            }
        } else {
            unsigned a_frag[MT][8];
            unsigned b_frag[WNT][8];
#pragma unroll
            for (int k_step = 0; k_step < KSUB; ++k_step) {
                load_fragments(k_step, a_frag, b_frag);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
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
        const int next = k_tile + S;
        if (next < kTiles) {
            stage_inputs(stage, next);
            cp_commit();
        }
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row_base = m0 + wm * WM + mi * 16;
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
            const int n_base    = n0 + wn * WN + ni * 16 + (lane & 15);
            const float* value  = accum[mi][ni];
            const int row_lo    = row_base + (lane >= 16 ? 8 : 0);
            // Skip the overlap columns when kWarpCols is not a multiple of 16.
            if ((ni * 16 + (lane & 15)) >= WN) { continue; }
            if constexpr (FullTokens) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    output_tile.store(row_lo + r, n_base, value[r]);
                }
            } else {
                if (n_base < tokens) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        output_tile.store(row_lo + r, n_base, value[r]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
