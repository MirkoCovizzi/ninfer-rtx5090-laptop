#include "hip/hip_runtime.h"
#pragma once

// Closed Q4/Q5 RowSplit grouped-MMA mechanism. Semantic Ops own the exact job
// set, route plan, workspace, and fixed instantiations.

#include "ops/common/math.h"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "core/tensor.h"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

struct RowSplitGroupedMmaJob {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint8_t* scales  = nullptr;
    __hip_bfloat16* out          = nullptr;
    std::int32_t n              = 0;
    std::int32_t out_ld         = 0;
    std::int32_t out_row_offset = 0;
    bool q5                     = false;
};

enum class RowSplitGroupedMmaCodec : std::uint8_t {
    Mixed,
    Q4,
    Q5,
};

template <class Cfg, bool FullTiles, RowSplitGroupedMmaCodec Codec = RowSplitGroupedMmaCodec::Mixed,
          int Jobs = 4>
__global__ __launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void rowsplit_grouped_mma_kernel(
    const __hip_bfloat16* __restrict__ x, RowSplitGroupedMmaJob job0, RowSplitGroupedMmaJob job1,
    RowSplitGroupedMmaJob job2, RowSplitGroupedMmaJob job3, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WM   = Cfg::WM;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int GPB  = Cfg::GROUPS_PER_BK;
    constexpr int KSUB = BK / 16;
    constexpr int S    = Cfg::STAGES;
    constexpr int SB   = Cfg::SCALE_BYTES;
    constexpr int HB   = Codec == RowSplitGroupedMmaCodec::Q4 ? 1 : 8;
    static_assert(GPB == 1, "grouped input GEMM requires BK=group_size=64");
    static_assert(Jobs == 2 || Jobs == 4, "grouped input GEMM supports two or four jobs");

    __shared__ __align__(16) __hip_bfloat16 As[BM * BK];
    __shared__ __align__(16) __hip_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * 32];
    __shared__ __align__(16) std::uint8_t Hr[S][BM * HB];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * SB];

    const int tiles0 = div_up(job0.n, BM);
    int tile         = static_cast<int>(blockIdx.x);
    RowSplitGroupedMmaJob job;
    if constexpr (Jobs == 2) {
        if (tile < tiles0) {
            job = job0;
        } else {
            tile -= tiles0;
            job = job1;
        }
    } else {
        const int tiles1 = div_up(job1.n, BM);
        const int tiles2 = div_up(job2.n, BM);
        if (tile < tiles0) {
            job = job0;
        } else if ((tile -= tiles0) < tiles1) {
            job = job1;
        } else if ((tile -= tiles1) < tiles2) {
            job = job2;
        } else {
            tile -= tiles2;
            job = job3;
        }
    }

    const int kg   = padded_k >> 6;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::WARPS_N;
    const int wn   = warp % Cfg::WARPS_N;
    const int m0   = tile * BM;
    const int t0   = static_cast<int>(blockIdx.y) * BN;

    // gfx1151 WMMA atoms are 16x16 (not 16x8). Each warp covers WMT x WNT
    // 16x16 atoms; kWarpCols (Cfg::WN) is a multiple of the block tile width
    // in these schedules, so WNT divides evenly.
    constexpr int WMT = MT;
    constexpr int WNT = (Cfg::WN + 15) / 16;
    float acc[WMT][WNT][8];
#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[mi][ni][r] = 0.0f; }
        }
    }

    const int NKT = padded_k / BK;

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

    auto stage_load_quant = [&](int stage, int kt) {
        const int g = (kt * BK) >> 6;
#pragma unroll 1
        for (int c = tid; c < BM * 2; c += Cfg::THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            const int grow = m0 + row;
            auto* dst      = &Cr[stage][row * 32 + half * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &job.codes[gi * 32 + half * 16]);
            } else if (grow < job.n) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &job.codes[gi * 32 + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (Codec == RowSplitGroupedMmaCodec::Q5) {
#pragma unroll 1
            for (int row = tid; row < BM; row += Cfg::THREADS) {
                const int grow = m0 + row;
                auto* dst      = &Hr[stage][row * 8];
                if constexpr (FullTiles) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                    gemm_cp_async<8, Cfg>(dst, &job.high[gi * 8]);
                } else if (grow < job.n) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                    gemm_cp_async<8, Cfg>(dst, &job.high[gi * 8]);
                } else {
                    *reinterpret_cast<std::uint64_t*>(dst) = 0;
                }
            }
        } else if constexpr (Codec == RowSplitGroupedMmaCodec::Mixed) {
            if (job.q5) {
#pragma unroll 1
                for (int row = tid; row < BM; row += Cfg::THREADS) {
                    const int grow = m0 + row;
                    auto* dst      = &Hr[stage][row * 8];
                    if constexpr (FullTiles) {
                        const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                        gemm_cp_async<8, Cfg>(dst, &job.high[gi * 8]);
                    } else if (grow < job.n) {
                        const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                        gemm_cp_async<8, Cfg>(dst, &job.high[gi * 8]);
                    } else {
                        *reinterpret_cast<std::uint64_t*>(dst) = 0;
                    }
                }
            }
        }
#pragma unroll 1
        for (int row = tid; row < BM; row += Cfg::THREADS) {
            const int grow = m0 + row;
            auto* dst      = &Sr[stage][row * SB];
            if constexpr (FullTiles) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    if (aligned_g + 1 < kg) {
                        gemm_cp_async<4, Cfg>(dst, &job.scales[aligned_gi * 2]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                }
            } else if (grow < job.n) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    if (aligned_g + 1 < kg) {
                        gemm_cp_async<4, Cfg>(dst, &job.scales[aligned_gi * 2]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                }
            } else {
                *reinterpret_cast<std::uint16_t*>(dst) = 0;
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                }
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
            __hip_bfloat162 w;
            if constexpr (Codec == RowSplitGroupedMmaCodec::Q5) {
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    w = Q5MmaDecodeAtom::decode_pair(Cr[stage], Hr[stage],
                                                     &Sr[stage][row * SB + scale_off], row, lane);
                } else {
                    w = Q5MmaDecodeAtom::decode_pair(Cr[stage], Hr[stage], &Sr[stage][row * SB],
                                                     row, lane);
                }
            } else if constexpr (Codec == RowSplitGroupedMmaCodec::Q4) {
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    w = Q4MmaDecodeAtom::decode_pair(Cr[stage], &Sr[stage][row * SB + scale_off],
                                                     row, lane);
                } else {
                    w = Q4MmaDecodeAtom::decode_pair(Cr[stage], &Sr[stage][row * SB], row, lane);
                }
            } else {
                if (job.q5) {
                    if constexpr (Cfg::SCALE_PAIR_LOAD) {
                        w = Q5MmaDecodeAtom::decode_pair(
                            Cr[stage], Hr[stage], &Sr[stage][row * SB + scale_off], row, lane);
                    } else {
                        w = Q5MmaDecodeAtom::decode_pair(Cr[stage], Hr[stage], &Sr[stage][row * SB],
                                                         row, lane);
                    }
                } else if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    w = Q4MmaDecodeAtom::decode_pair(Cr[stage], &Sr[stage][row * SB + scale_off],
                                                     row, lane);
                } else {
                    w = Q4MmaDecodeAtom::decode_pair(Cr[stage], &Sr[stage][row * SB], row, lane);
                }
            }
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

        unsigned a_frag[WMT][8];
        unsigned b_frag[WNT][8];
        auto load_fragments = [&](int ks) {
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
                const int arow = wm * WM + mi * 16 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    wmma_load_a_bf16(a_frag[mi], As, arow, ks, BK, gemm_swz64);
                }
            }
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int bcol = wn * WN + ni * 16 + (lane & 15);
                wmma_load_b_bf16(b_frag[ni], Bs[stage], bcol, ks, BK, gemm_swz64);
            }
        };
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ks = ki * 16;
            load_fragments(ks);
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < WNT; ++ni) {
                    WmmaC8& c = *reinterpret_cast<WmmaC8*>(acc[mi][ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag[mi]);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag[ni]);
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
    for (int mi = 0; mi < WMT; ++mi) {
        const int row_base = m0 + wm * WM + mi * 16;
        const int row_lo   = row_base + (lane >= 16 ? 8 : 0);
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
            const int col_base   = t0 + wn * WN + ni * 16;
            const int local_col  = ni * 16 + (lane & 15);
            const int output_col = col_base + (lane & 15);
            // Skip overlap columns when kWarpCols is not a multiple of 16.
            if (local_col >= Cfg::WN) { continue; }
            const float* values = acc[mi][ni];
            if constexpr (FullTiles) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    job.out[static_cast<std::int64_t>(output_col) * job.out_ld +
                            job.out_row_offset + row_lo + r] = __float2bfloat16_rn(values[r]);
                }
            } else {
                for (int r = 0; r < 8; ++r) {
                    const int row = row_lo + r;
                    if (row < job.n && output_col < t) {
                        job.out[static_cast<std::int64_t>(output_col) * job.out_ld +
                                job.out_row_offset + row] = __float2bfloat16_rn(values[r]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
