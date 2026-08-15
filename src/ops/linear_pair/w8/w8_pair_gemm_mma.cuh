#include "hip/hip_runtime.h"
#pragma once

// Paired MTP K/V W8G32 GEMM. One CTA keeps a single BF16 activation tile in
// shared memory and contracts it against independent K and V weight tiles.

#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {

template <int TileCols>
inline constexpr int kW8PairMmaMinBlocks = TileCols <= 64 ? 3 : 2;

template <int TileCols, bool Full>
__global__
__launch_bounds__((TileCols / 16) * 32, kW8PairMmaMinBlocks<TileCols>) void w8_pair_gemm_mma_kernel(
    const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ k_codes,
    const std::uint8_t* __restrict__ k_scales, const std::uint8_t* __restrict__ v_codes,
    const std::uint8_t* __restrict__ v_scales, __hip_bfloat16* __restrict__ k_out,
    __hip_bfloat16* __restrict__ v_out, std::int32_t m, std::int32_t k, std::int32_t n,
    std::int32_t padded_k) {
    constexpr int BM                = 32;
    constexpr int BN                = TileCols;
    constexpr int BK                = 64;
    constexpr int WN                = 16;
    constexpr int MT                = 2;
    constexpr int NT                = 2;
    constexpr int KSUB              = 4;
    // gfx1151 WMMA atoms are 16x16. The warp tile is BM rows x WN(16) cols; use WMT m16-row
    // atoms and WNT 16-column atoms (WN==16, so exactly one column atom, no overlap).
    constexpr int WMT = BM / 16;
    constexpr int WNT = WN / 16;
    constexpr int WARPS_N           = BN / WN;
    constexpr int WARPS             = WARPS_N;
    constexpr int THREADS           = WARPS * 32;
    constexpr int SCALE_CACHE_BYTES = 16;
    static_assert(TileCols == 64 || TileCols == 80 || TileCols == 96 || TileCols == 112 ||
                  TileCols == 128);

    __shared__ __align__(16) __hip_bfloat16 As[BM * BK];
    __shared__ __align__(16) __hip_bfloat16 Bs[2][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[2][BM * BK];
    __shared__ __align__(16) std::uint8_t Sr[2][BM * SCALE_CACHE_BYTES];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wn   = warp % WARPS_N;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int m0   = static_cast<int>(blockIdx.x) * BM;
    const int n0   = static_cast<int>(blockIdx.y) * BN;
    const int kg   = padded_k / 32;

    float acc_k[WMT][WNT][8];
    float acc_v[WMT][WNT][8];
#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc_k[mi][ni][r] = 0.0f;
                acc_v[mi][ni][r] = 0.0f;
            }
        }
    }

    auto stage_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int nl = item / (BK / 8);
            const int k8 = item - nl * (BK / 8);
            const int kk = k0 + k8 * 8;
            const int nn = n0 + nl;
            auto* dst    = &Bs[stage][nl * BK + w8g32_swz64(nl, k8 * 8)];
            if constexpr (Full) {
                cp_async<16, Cache::cg>(dst, &x[static_cast<std::int64_t>(nn) * k + kk]);
            } else {
                const int valid = (nn < n && kk < k) ? min(8, k - kk) * 2 : 0;
                ninfer::ops::cp_async_zfill<16>(
                    dst, &x[static_cast<std::int64_t>(nn < n ? nn : 0) * k + (kk < k ? kk : 0)],
                    valid);
            }
        }
    };

    auto stage_codes = [&](int p, int kt) {
        const auto* codes = p == 0 ? k_codes : v_codes;
        const int g0      = kt * 2;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += THREADS) {
            const int row   = item / (BK / 16);
            const int chunk = item - row * (BK / 16);
            const int grow  = m0 + row;
            auto* dst       = &Cr[p][row * BK + chunk * 16];
            if constexpr (Full) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cache::cg>(dst, &codes[gi * 32 + chunk * 16]);
            } else {
                const std::int64_t gi = static_cast<std::int64_t>(grow < m ? grow : 0) * kg + g0;
                ninfer::ops::cp_async_zfill<16>(dst, &codes[gi * 32 + chunk * 16],
                                                grow < m ? 16 : 0);
            }
        }
    };

    auto stage_scales = [&](int p, int kt) {
        if ((kt & 3) == 0) {
            const auto* scales = p == 0 ? k_scales : v_scales;
            const int g0       = kt * 2;
            for (int row = tid; row < BM; row += THREADS) {
                const int grow = m0 + row;
                auto* dst      = &Sr[p][row * SCALE_CACHE_BYTES];
                if constexpr (Full) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                    if (g0 + 8 <= kg) {
                        cp_async<16, Cache::cg>(dst, &scales[gi * 2]);
                    } else {
                        ninfer::ops::cp_async_zfill<16>(dst, &scales[gi * 2], max(0, kg - g0) * 2);
                    }
                } else {
                    const bool valid_row   = grow < m;
                    const int valid_scales = valid_row && g0 < kg ? min(8, kg - g0) : 0;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(valid_row ? grow : 0) * kg + min(g0, kg - 1);
                    ninfer::ops::cp_async_zfill<16>(dst, &scales[gi * 2], valid_scales * 2);
                }
            }
        }
    };

    auto dequant = [&](int p, int kt) {
        const int scale_pair_offset = (kt & 3) * 4;
        const int half              = lane >> 4;
        const int half_lane         = lane & 15;
        for (int row_pair = warp * 2; row_pair < BM; row_pair += WARPS * 2) {
            const int row       = row_pair + half;
            unsigned scale_pair = half_lane == 0
                                      ? *reinterpret_cast<const std::uint32_t*>(
                                            &Sr[p][row * SCALE_CACHE_BYTES + scale_pair_offset])
                                      : 0;
            scale_pair          = __shfl_sync(0xffffffffffffffffull, scale_pair, half * 16);
#pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                const float scale =
                    __half2float(__ushort_as_half((scale_pair >> (gg * 16)) & 0xffffu));
                const int col = gg * 32 + half_lane * 2;
                const std::uint16_t packed =
                    *reinterpret_cast<const std::uint16_t*>(&Cr[p][row * BK + col]);
                const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                const __hip_bfloat162 values = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                                    static_cast<float>(q1) * scale);
                store_vec(&As[row * BK + w8g32_swz64(row, col)], values);
            }
        }
    };

    auto mma_pair = [&](int p, int stage) {
        unsigned af[2][WMT][8];
        unsigned bf[2][WNT][8];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
                const int row = mi * 16 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    wmma_load_a_bf16(af[slot][mi], As, row, ks * 16, BK, w8g32_swz64);
                }
            }
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int col = wn * WN + ni * 16 + (lane & 15);
                wmma_load_b_bf16(bf[slot][ni], Bs[stage], col, ks * 16, BK, w8g32_swz64);
            }
        };
        load_fragments(0, 0);
#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
            const int slot = ks & 1;
            if (ks + 1 < KSUB) { load_fragments(slot ^ 1, ks + 1); }
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < WNT; ++ni) {
                    WmmaC8& c  = *reinterpret_cast<WmmaC8*>(p == 0 ? acc_k[mi][ni] : acc_v[mi][ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(af[slot][mi]);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bf[slot][ni]);
                    c          = wmma_bf16(a, b, c);
                }
            }
        }
    };

    const int nkt = padded_k / BK;
    stage_x(0, 0);
    stage_codes(0, 0);
    stage_codes(1, 0);
    stage_scales(0, 0);
    stage_scales(1, 0);
    ninfer::ops::cp_commit();

#pragma unroll 2
    for (int kt = 0; kt < nkt; ++kt) {
        const int stage = kt & 1;
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        dequant(0, kt);
        __syncthreads();
        mma_pair(0, stage);
        __syncthreads();

        dequant(1, kt);
        __syncthreads();

        const int next = kt + 1;
        if (next < nkt) {
            stage_x(next & 1, next);
            stage_codes(0, next);
            stage_codes(1, next);
            stage_scales(0, next);
            stage_scales(1, next);
            ninfer::ops::cp_commit();
        }
        mma_pair(1, stage);
    }

#pragma unroll
    for (int p = 0; p < 2; ++p) {
        auto* out = p == 0 ? k_out : v_out;
#pragma unroll
        for (int mi = 0; mi < WMT; ++mi) {
            const int row_lo = m0 + mi * 16 + (lane < 16 ? 0 : 8);
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int col       = n0 + wn * WN + ni * 16 + (lane & 15);
                const int local_col = ni * 16 + (lane & 15);
                if (local_col >= WN) { continue; }
                const float* values = p == 0 ? acc_k[mi][ni] : acc_v[mi][ni];
                if constexpr (Full) {
                    if (col < n) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            out[static_cast<std::int64_t>(col) * m + row_lo + r] =
                                __float2bfloat16_rn(values[r]);
                        }
                    }
                } else {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int row = row_lo + r;
                        if (row < m && col < n) {
                            out[static_cast<std::int64_t>(col) * m + row] =
                                __float2bfloat16_rn(values[r]);
                        }
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
