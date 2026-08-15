#include "hip/hip_runtime.h"
#pragma once

// Folded gate/up GEMM.  A logical BM=64 weight tile is laid out as 32 gate
// rows followed by their 32 matching up rows.  One normal BM64 tensor-core
// contraction therefore produces both projections in one accumulator array;
// the warp's first and second row halves pair directly in the SiLU epilogue.

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

template <class Cfg, bool FullTiles>
__global__
__launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void q4_linear_swiglu_mma_split_half_pair_kernel(
    const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __hip_bfloat16* __restrict__ out,
    std::int32_t intermediate, std::int32_t k, std::int32_t t, std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WM   = Cfg::WM;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int KSUB = BK / 16;
    constexpr int S    = Cfg::STAGES;
    constexpr int SB   = Cfg::SCALE_BYTES;
    constexpr int PM   = BM / 2;
    static_assert(BK == 64, "folded gate/up Q4 kernel requires one group per K tile");
    static_assert(BM == 64 && WM == 64 && MT == 4,
                  "folded gate/up mapping requires one 64-row warp tile");
    // gfx1151 WMMA atoms are 16x16 (not 16x8). Each warp covers WMT x WNT 16x16
    // atoms; when kWarpCols is not a multiple of 16 the final atom overlaps the
    // neighbouring warp's columns (duplicate compute, correct results) and the
    // store below emits only the warp's own columns.
    constexpr int WMT = MT;
    constexpr int WNT = (Cfg::WN + 15) / 16;

    __shared__ __align__(16) __hip_bfloat16 As[BM * BK];
    __shared__ __align__(16) __hip_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * 32];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * SB];

    const int kg   = padded_k >> 6;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wn   = warp;
    const int m0   = static_cast<int>(blockIdx.x) * PM;
    const int t0   = static_cast<int>(blockIdx.y) * BN;

    float acc[WMT][WNT][8];
#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[mi][ni][r] = 0.0f; }
        }
    }

    const int NKT      = padded_k / BK;

    auto stage_load_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int c = tid; c < BN * (BK / 8); c += Cfg::THREADS) {
            const int tl       = c / (BK / 8);
            const int kg8      = c - tl * (BK / 8);
            const int kl       = kg8 * 8;
            const int col      = t0 + tl;
            const int kk       = k0 + kl;
            __hip_bfloat16* dst = &Bs[stage][tl * BK + gemm_swz64(tl, kl)];
            if constexpr (FullTiles) {
                gemm_cp_async<16, Cfg>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else if (col < t && kk + 8 <= k) {
                gemm_cp_async<16, Cfg>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto global_row = [&](int row) { return m0 + (row & (PM - 1)) + (row / PM) * intermediate; };

    auto stage_load_quant = [&](int stage, int kt) {
        const int g = (kt * BK) >> 6;
#pragma unroll 1
        for (int c = tid; c < BM * 2; c += Cfg::THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            const int grow = global_row(row);
            auto* dst      = &Cr[stage][row * 32 + half * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &codes[gi * 32 + half * 16]);
            } else if (m0 + (row & (PM - 1)) < intermediate) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &codes[gi * 32 + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int row = tid; row < BM; row += Cfg::THREADS) {
            const int grow = global_row(row);
            auto* dst      = &Sr[stage][row * SB];
            if constexpr (FullTiles) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if (aligned_g + 1 < kg) {
                    gemm_cp_async<4, Cfg>(dst, &scales[aligned_gi * 2]);
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&scales[gi * 2]);
                    *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                }
            } else if (m0 + (row & (PM - 1)) < intermediate) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if (aligned_g + 1 < kg) {
                    gemm_cp_async<4, Cfg>(dst, &scales[aligned_gi * 2]);
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&scales[gi * 2]);
                    *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                }
            } else {
                *reinterpret_cast<std::uint32_t*>(dst) = 0;
            }
        }
    };

    auto stage_load = [&](int stage, int kt) {
        stage_load_x(stage, kt);
        stage_load_quant(stage, kt);
    };

    auto dequant_to_As = [&](int stage, int kt) {
        const int scale_off = ((kt * BK >> 6) & 1) * 2;
        for (int row = warp; row < BM; row += Cfg::WARPS) {
            const __hip_bfloat162 w = Q4MmaDecodeAtom::decode_pair(
                Cr[stage], &Sr[stage][row * SB + scale_off], row, lane);
            const int sc = gemm_swz64(row, 2 * lane);
            store_vec(&As[row * BK + sc], w);
        }
    };

#pragma unroll
    for (int s = 0; s < S; ++s) {
        if (s < NKT) { stage_load(s, s); }
        ninfer::ops::cp_commit();
    }

    for (int it = 0; it < NKT; ++it) {
        const int stage = it % S;
        ninfer::ops::cp_wait<S - 1>();
        __syncthreads();
        dequant_to_As(stage, it);
        __syncthreads();

        unsigned af[WMT][8];
        unsigned bf[WNT][8];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ks = ki * 16;
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
                const int arow = mi * 16 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    wmma_load_a_bf16(af[mi], As, arow, ks, BK, gemm_swz64);
                }
            }
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int brow = wn * WN + ni * 16 + (lane & 15);
                wmma_load_b_bf16(bf[ni], Bs[stage], brow, ks, BK, gemm_swz64);
            }
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < WNT; ++ni) {
                    WmmaC8& c = *reinterpret_cast<WmmaC8*>(acc[mi][ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(af[mi]);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bf[ni]);
                    c = wmma_bf16(a, b, c);
                }
            }
        }

        __syncthreads();
        const int next = it + S;
        if (next < NKT) { stage_load(stage, next); }
        ninfer::ops::cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < WMT / 2; ++mi) {
        const int row_lo = m0 + mi * 16 + 8 * (lane >= 16);
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
            const int local_col = ni * 16 + (lane & 15);
            const int col       = t0 + wn * WN + ni * 16 + (lane & 15);
            // Skip the overlap columns when kWarpCols is not a multiple of 16.
            if (local_col >= Cfg::WN) { continue; }
            auto store = [&](int col_idx, int row, float gv, float uv) {
                out[static_cast<std::int64_t>(col_idx) * intermediate + row] =
                    __float2bfloat16_rn(silu(gv) * uv);
            };
            if constexpr (FullTiles) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    store(col, row_lo + r, acc[mi][ni][r], acc[mi + WMT / 2][ni][r]);
                }
            } else {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int row = row_lo + r;
                    if (row < intermediate && col < t) {
                        store(col, row, acc[mi][ni][r], acc[mi + WMT / 2][ni][r]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
