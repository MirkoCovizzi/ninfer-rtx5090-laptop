#include "hip/hip_runtime.h"
#pragma once

// Dedicated Large-T W8G32 x BF16 Tensor Core GEMM.
//
// out[M,N] = W[M,K] * x[K,N], where W stores one signed int8 code per element
// and one FP16 scale per 32 K elements. Raw codes and eight quantization groups' scales are
// staged with cp.async before dequantization into a swizzled BF16
// shared tile; x uses a two-stage cp.async pipeline. Tensor Cores execute
// m16n8k16 BF16 MMA with FP32 accumulation.

#include "ops/common/mma.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

union alignas(16) W8Bf16x8Bits {
    uint4 raw;
    __hip_bfloat162 pair[4];
};

static_assert(sizeof(W8Bf16x8Bits) == 16);

template <int BM_, int BN_, int WM_, int WN_, int MIN_BLOCKS_, int STAGES_ = 2, int BK_ = 64,
          int ACTIVATION_STAGES_ = STAGES_>
struct W8RowSplitMmaGemmSchedule {
    static constexpr int BM                = BM_;
    static constexpr int BN                = BN_;
    static constexpr int BK                = BK_;
    static constexpr int WM                = WM_;
    static constexpr int WN                = WN_;
    static constexpr int MIN_BLOCKS        = MIN_BLOCKS_;
    static constexpr int WARPS_M           = BM / WM;
    static constexpr int WARPS_N           = BN / WN;
    static constexpr int WARPS             = WARPS_M * WARPS_N;
    static constexpr int THREADS           = WARPS * 32;
    static constexpr int MT                = WM / 16;
    static constexpr int NT                = WN / 8;
    static constexpr int KSUB              = BK / 16;
    static constexpr int STAGES            = STAGES_;
    static constexpr int ACTIVATION_STAGES = ACTIVATION_STAGES_;
    static constexpr int SCALE_CACHE_BYTES = 16;
    static constexpr int SMEM_BYTES =
        BM * BK * 2 + ACTIVATION_STAGES * BN * BK * 2 + BM * BK + BM * SCALE_CACHE_BYTES;

    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(WM % 16 == 0 && WN % 8 == 0);
    static_assert(THREADS <= 1024);
    static_assert(BK == 64 || BK == 128);
    static_assert(STAGES == 2, "W8G32 MMA uses a two-stage cp.async pipeline");
    static_assert(ACTIVATION_STAGES == 1 || ACTIVATION_STAGES == STAGES,
                  "W8G32 MMA activation staging is single-buffered or follows the pipeline");
    static_assert(SMEM_BYTES <= 48 * 1024);
};

__device__ __forceinline__ int w8g32_swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Cfg, bool Full, W8Epilogue Epilogue = W8Epilogue::Store,
          class Output = W8ContiguousOutput>
__global__ __launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void w8_rowsplit_gemm_mma_kernel(
    const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, std::int32_t m, std::int32_t k,
    std::int32_t n, std::int32_t padded_k) {
    constexpr int BM                = Cfg::BM;
    constexpr int BN                = Cfg::BN;
    constexpr int BK                = Cfg::BK;
    constexpr int WM                = Cfg::WM;
    constexpr int WN                = Cfg::WN;
    constexpr int MT                = Cfg::MT;
    constexpr int NT                = Cfg::NT;
    constexpr int KSUB              = Cfg::KSUB;
    constexpr bool kSwiGlu          = Epilogue == W8Epilogue::SwiGluSplitHalf;
    constexpr int kOutputRowsPerCta = kSwiGlu ? BM / 2 : BM;
    static_assert(!kSwiGlu || (BM % 32) == 0);
    static_assert(!kSwiGlu || Cfg::WARPS_M == 1 || Cfg::WARPS_M == 2,
                  "SwiGLU supports warp-local or shared-memory row pairing");
    // gfx1151 WMMA atoms are 16x16 (not 16x8). Each warp covers WMT x WNT 16x16
    // atoms; when kWarpCols is not a multiple of 16 the final atom overlaps the
    // neighbouring warp's columns (duplicate compute, correct results) and the
    // store below emits only the warp's own columns.
    constexpr int WMT = MT;
    constexpr int WNT = (Cfg::WN + 15) / 16;

    __shared__ __align__(16) __hip_bfloat16 As[BM * BK];
    __shared__ __align__(16) __hip_bfloat16 Bs[Cfg::ACTIVATION_STAGES][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[BM * BK];
    __shared__ __align__(16) std::uint8_t Sr[BM * Cfg::SCALE_CACHE_BYTES];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::WARPS_N;
    const int wn   = warp % Cfg::WARPS_N;

    const int m0           = output.row_begin(static_cast<int>(blockIdx.x), kOutputRowsPerCta);
    const int n0           = static_cast<int>(blockIdx.y) * BN;
    const int kg           = padded_k / 32;
    const auto output_tile = output.tile(m0);

    float acc[WMT][WNT][8];
#pragma unroll
    for (int mi = 0; mi < WMT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNT; ++ni) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { acc[mi][ni][r] = 0.0f; }
        }
    }

    auto stage_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Cfg::THREADS) {
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

    auto stage_w = [&](int kt) {
        constexpr int GROUPS            = BK / 32;
        constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
        const int g0                    = kt * GROUPS;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += Cfg::THREADS) {
            const int row   = item / (BK / 16);
            const int chunk = item - row * (BK / 16);
            const int grow =
                kSwiGlu ? m0 + (row % (BM / 2)) + (row >= BM / 2 ? m / 2 : 0) : m0 + row;
            auto* dst = &Cr[row * BK + chunk * 16];
            if constexpr (Full) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cache::cg>(dst, &codes[gi * 32 + chunk * 16]);
            } else {
                const bool valid_row  = output_tile.valid(grow, m);
                const std::int64_t gi = static_cast<std::int64_t>(valid_row ? grow : 0) * kg + g0;
                ninfer::ops::cp_async_zfill<16>(dst, &codes[gi * 32 + chunk * 16],
                                                valid_row ? 16 : 0);
            }
        }
        if ((kt % SCALE_CACHE_TILES) == 0) {
            for (int row = tid; row < BM; row += Cfg::THREADS) {
                const int grow =
                    kSwiGlu ? m0 + (row % (BM / 2)) + (row >= BM / 2 ? m / 2 : 0) : m0 + row;
                auto* dst = &Sr[row * Cfg::SCALE_CACHE_BYTES];
                if constexpr (Full) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                    cp_async<16, Cache::cg>(dst, &scales[gi * 2]);
                } else {
                    const bool valid_row   = output_tile.valid(grow, m);
                    const int valid_scales = valid_row && g0 < kg ? min(8, kg - g0) : 0;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(valid_row ? grow : 0) * kg + min(g0, kg - 1);
                    ninfer::ops::cp_async_zfill<16>(dst, &scales[gi * 2], valid_scales * 2);
                }
            }
        }
    };

    auto dequant_w = [&](int kt) {
        constexpr int GROUPS            = BK / 32;
        constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
        const int scale_pair_offset     = (kt % SCALE_CACHE_TILES) * GROUPS * 2;
        const int half                  = lane >> 4;
        const int half_lane             = lane & 15;
        for (int row_pair = warp * 2; row_pair < BM; row_pair += Cfg::WARPS * 2) {
            const int row = row_pair + half;
            unsigned scale_pair0;
            unsigned scale_pair1 = 0;
            if constexpr (GROUPS == 2) {
                scale_pair0 = half_lane == 0
                                  ? *reinterpret_cast<const std::uint32_t*>(
                                        &Sr[row * Cfg::SCALE_CACHE_BYTES + scale_pair_offset])
                                  : 0;
                scale_pair0 = __shfl_sync(0xffffffffffffffffull, scale_pair0, half * 16);
            } else {
                static_assert(GROUPS == 4);
                const unsigned lane_scale_pair =
                    half_lane < 2
                        ? *reinterpret_cast<const std::uint32_t*>(
                              &Sr[row * Cfg::SCALE_CACHE_BYTES + scale_pair_offset + half_lane * 4])
                        : 0;
                scale_pair0 = __shfl_sync(0xffffffffffffffffull, lane_scale_pair, half * 16);
                scale_pair1 = __shfl_sync(0xffffffffffffffffull, lane_scale_pair, half * 16 + 1);
            }
#pragma unroll
            for (int gg = 0; gg < GROUPS; ++gg) {
                const unsigned scale_pair = gg < 2 ? scale_pair0 : scale_pair1;
                const unsigned scale_bits = (scale_pair >> ((gg & 1) * 16)) & 0xffffu;
                const float scale         = __half2float(__ushort_as_half(scale_bits));
                const int col             = gg * 32 + half_lane * 2;
                const std::uint16_t packed =
                    *reinterpret_cast<const std::uint16_t*>(&Cr[row * BK + col]);
                const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                const __hip_bfloat162 values = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                                    static_cast<float>(q1) * scale);
                store_vec(&As[row * BK + w8g32_swz64(row, col)], values);
            }
        }
    };

    const int nkt = padded_k / BK;
    stage_x(0, 0);
    stage_w(0);
    ninfer::ops::cp_commit();

#pragma unroll 4
    for (int kt = 0; kt < nkt; ++kt) {
        const int stage = kt % Cfg::STAGES;
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        dequant_w(kt);
        __syncthreads();

        const int next = kt + 1;
        if (next < nkt) {
            if constexpr (Cfg::ACTIVATION_STAGES == Cfg::STAGES) {
                stage_x(next % Cfg::STAGES, next);
            }
            stage_w(next);
            ninfer::ops::cp_commit();
        }

        unsigned af[2][WMT][8];
        unsigned bf[2][WNT][8];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < WMT; ++mi) {
                const int ar = wm * WM + mi * 16 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    wmma_load_a_bf16(af[slot][mi], As, ar, ks * 16, BK, w8g32_swz64);
                }
            }
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int br = wn * WN + ni * 16 + (lane & 15);
                wmma_load_b_bf16(bf[slot][ni], Bs[Cfg::ACTIVATION_STAGES == 1 ? 0 : stage], br,
                                 ks * 16, BK, w8g32_swz64);
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
                    WmmaC8& c = *reinterpret_cast<WmmaC8*>(acc[mi][ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(af[slot][mi]);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bf[slot][ni]);
                    c = wmma_bf16(a, b, c);
                }
            }
        }

        if constexpr (Cfg::ACTIVATION_STAGES == 1) {
            if (next < nkt) {
                __syncthreads();
                stage_x(0, next);
                ninfer::ops::cp_commit();
            }
        }
    }

    if constexpr (kSwiGlu) {
        if constexpr (Cfg::WARPS_M == 1) {
            static_assert((MT % 2) == 0);
            constexpr int kGateMt = MT / 2;
#pragma unroll
            for (int mi = 0; mi < kGateMt; ++mi) {
                const int r_lo = m0 + mi * 16 + 8 * (lane >= 16);
#pragma unroll
                for (int ni = 0; ni < WNT; ++ni) {
                    const int local_col = ni * 16 + (lane & 15);
                    const int c0        = n0 + wn * WN + ni * 16 + (lane & 15);
                    // Skip the overlap columns when kWarpCols is not a multiple of 16.
                    if (local_col >= WN) { continue; }
                    const float* gate_acc = acc[mi][ni];
                    const float* up_acc   = acc[mi + kGateMt][ni];
                    if constexpr (Full) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            *output_tile.at(r_lo + r, c0) = __float2bfloat16_rn(
                                silu(gate_acc[r]) * up_acc[r]);
                        }
                    } else {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            if (r_lo + r < m / 2 && c0 < n) {
                                *output_tile.at(r_lo + r, c0) = __float2bfloat16_rn(
                                    silu(gate_acc[r]) * up_acc[r]);
                            }
                        }
                    }
                }
            }
        } else {
            static_assert(Cfg::WARPS_M == 2);
            auto* up_shared = reinterpret_cast<float*>(Bs);
            __syncthreads();
            if (wm == 1) {
#pragma unroll
                for (int mi = 0; mi < WMT; ++mi) {
                    const int local_r_lo = mi * 16 + 8 * (lane >= 16);
#pragma unroll
                    for (int ni = 0; ni < WNT; ++ni) {
                        const int block_col  = wn * WN + ni * 16 + (lane & 15);
                        const int warp_local = ni * 16 + (lane & 15);
                        // Skip the overlap columns when kWarpCols is not a multiple of 16.
                        if (warp_local >= WN) { continue; }
                        const float* up_acc = acc[mi][ni];
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            up_shared[(local_r_lo + r) * BN + block_col] = up_acc[r];
                        }
                    }
                }
            }
            __syncthreads();
            if (wm == 0) {
#pragma unroll
                for (int mi = 0; mi < WMT; ++mi) {
                    const int local_r_lo = mi * 16 + 8 * (lane >= 16);
                    const int r_lo       = m0 + local_r_lo;
#pragma unroll
                    for (int ni = 0; ni < WNT; ++ni) {
                        const int block_col  = wn * WN + ni * 16 + (lane & 15);
                        const int warp_local = ni * 16 + (lane & 15);
                        const int c0         = n0 + block_col;
                        // Skip the overlap columns when kWarpCols is not a multiple of 16.
                        if (warp_local >= WN) { continue; }
                        const float* gate_acc = acc[mi][ni];
                        if constexpr (Full) {
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float upv = up_shared[(local_r_lo + r) * BN + block_col];
                                *output_tile.at(r_lo + r, c0) =
                                    __float2bfloat16_rn(silu(gate_acc[r]) * upv);
                            }
                        } else {
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                if (r_lo + r < m / 2 && c0 < n) {
                                    const float upv =
                                        up_shared[(local_r_lo + r) * BN + block_col];
                                    *output_tile.at(r_lo + r, c0) =
                                        __float2bfloat16_rn(silu(gate_acc[r]) * upv);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if constexpr (Epilogue == W8Epilogue::Residual) {
        static_assert(BM <= Cfg::STAGES * BK && (BM % 8) == 0,
                      "W8 residual epilogue reuses the x pipeline as a BF16 output tile");
        __syncthreads();
        __hip_bfloat16* projected_shared = Bs[0];
#pragma unroll
        for (int mi = 0; mi < WMT; ++mi) {
            const int local_r_lo = wm * WM + mi * 16 + 8 * (lane >= 16);
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int block_col  = wn * WN + ni * 16 + (lane & 15);
                const int warp_local = ni * 16 + (lane & 15);
                // Skip the overlap columns when kWarpCols is not a multiple of 16.
                if (warp_local >= WN) { continue; }
                const float* a = acc[mi][ni];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    projected_shared[block_col * BM + local_r_lo + r] =
                        __float2bfloat16_rn(a[r]);
                }
            }
        }
        __syncthreads();

        constexpr int kRowsPerPack = 8;
        constexpr int kPacksPerCol = BM / kRowsPerPack;
        constexpr int kPacks       = BN * kPacksPerCol;
        for (int pack = tid; pack < kPacks; pack += Cfg::THREADS) {
            const int local_col = pack / kPacksPerCol;
            const int row_pack  = pack - local_col * kPacksPerCol;
            const int local_row = row_pack * kRowsPerPack;
            const int col       = n0 + local_col;
            const int row       = m0 + local_row;
            if constexpr (Full) {
                W8Bf16x8Bits projected;
                projected.raw = load_vec<uint4>(&projected_shared[local_col * BM + local_row]);
                W8Bf16x8Bits residual;
                residual.raw = load_vec<uint4>(output_tile.at(row, col));
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    residual.pair[pair] = __floats2bfloat162_rn(
                        __low2float(residual.pair[pair]) + __low2float(projected.pair[pair]),
                        __high2float(residual.pair[pair]) + __high2float(projected.pair[pair]));
                }
                store_vec(output_tile.at(row, col), residual.raw);
            } else if (col < n && row < m) {
                if (row + kRowsPerPack <= m) {
                    W8Bf16x8Bits projected;
                    projected.raw = load_vec<uint4>(&projected_shared[local_col * BM + local_row]);
                    W8Bf16x8Bits residual;
                    residual.raw = load_vec<uint4>(output_tile.at(row, col));
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        residual.pair[pair] = __floats2bfloat162_rn(
                            __low2float(residual.pair[pair]) + __low2float(projected.pair[pair]),
                            __high2float(residual.pair[pair]) + __high2float(projected.pair[pair]));
                    }
                    store_vec(output_tile.at(row, col), residual.raw);
                } else {
#pragma unroll
                    for (int i = 0; i < kRowsPerPack; ++i) {
                        if (row + i < m) {
                            __hip_bfloat16* destination = output_tile.at(row + i, col);
                            *destination               = __float2bfloat16_rn(
                                __bfloat162float(*destination) +
                                __bfloat162float(projected_shared[local_col * BM + local_row + i]));
                        }
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < WMT; ++mi) {
            const int r_lo = m0 + wm * WM + mi * 16 + 8 * (lane >= 16);
#pragma unroll
            for (int ni = 0; ni < WNT; ++ni) {
                const int local_col = ni * 16 + (lane & 15);
                const int c0        = n0 + wn * WN + ni * 16 + (lane & 15);
                // Skip the overlap columns when kWarpCols is not a multiple of 16.
                if (local_col >= WN) { continue; }
                const float* a = acc[mi][ni];
                if constexpr (Full) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        *output_tile.at(r_lo + r, c0) = __float2bfloat16_rn(a[r]);
                    }
                } else {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int row = r_lo + r;
                        if (output_tile.valid(row, m) && c0 < n) {
                            *output_tile.at(row, c0) = __float2bfloat16_rn(a[r]);
                        }
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
