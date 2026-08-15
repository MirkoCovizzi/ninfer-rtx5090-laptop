#include "hip/hip_runtime.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_fp16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows                    = 12288;
constexpr int kHidden                  = 2048;
constexpr int kTileK                   = 64;
constexpr int kMmaRows                 = 16;
constexpr int kRowsPerCta              = 16;
constexpr int kFirstExactCols          = 2;
constexpr int kLastProjectionExactCols = 32;
constexpr int kLastSnapshotExactCols   = 16;
using Output                           = W8SplitOutput2<8192, 4096>;

template <class Publish>
struct W8GdnSplitKConvEpilogue {
    GdnConvEpilogue<Publish> conv;
    __hip_bfloat16* z;

    template <int ActiveCols>
    __device__ __forceinline__ void store(std::int32_t row,
                                          const float (&projected)[ActiveCols]) const {
        if (row < 8192) {
            conv.store(row, projected);
        } else {
#pragma unroll
            for (int token = 0; token < ActiveCols; ++token) {
                z[static_cast<std::int64_t>(token) * 4096 + row - 8192] =
                    __float2bfloat16_rn(projected[token]);
            }
        }
    }
};

__device__ __forceinline__ int swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Bf16PairBits {
    __hip_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned bf16_pair_from_s8(unsigned values) {
    Bf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    Bf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    Bf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
__global__
__launch_bounds__(KSplits* NGroups * 32, MinBlocks) void w8_gdn_input_medium_t_splitk_kernel(
    const __hip_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, int active_cols) {
    constexpr int kKernelWarps = KSplits * NGroups;
    constexpr int kGroupK      = KSplits * kTileK;
    constexpr int kGroups      = kHidden / kGroupK;
    constexpr int kWarpCols    = TileCols / NGroups;
    // gfx1151 WMMA atoms are 16x16 (not 16x8). Each warp covers kWmmaNt 16-column atoms;
    // when kWarpCols is not a multiple of 16 the final atom overlaps the neighbouring warp's
    // columns (duplicate compute, correct results) and the store emits only this warp's own
    // columns.
    constexpr int kWmmaNt = (kWarpCols + 15) / 16;
    static_assert(KSplits == 2 || KSplits == 4);
    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(kHidden % kGroupK == 0 && kKernelWarps <= 32);

    __shared__ __align__(16) std::uint8_t code_shared[kMmaRows][kGroupK];
    __shared__ __align__(16) __hip_bfloat16 b_shared[kKernelWarps][kWarpCols * kTileK];

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int n_group    = warp / KSplits;
    const int k_split    = warp - n_group * KSplits;
    const int n_base     = n_group * kWarpCols;
    const int remaining  = active_cols - n_base;
    const int local_cols = remaining <= 0 ? 0 : (remaining < kWarpCols ? remaining : kWarpCols);
    const int cta_row0   = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int k0) {
        for (int item = lane; item < local_cols * (kTileK / 8); item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + swizzle_64(col, k8 * 8)];
            cp_async<16, Cache::cg>(
                dst, &x[static_cast<std::int64_t>(n_base + col) * kHidden + k0 + k8 * 8]);
        }
        cp_commit();
    };

    const auto stage_codes = [&](int group_k0) {
        constexpr int kChunks = kGroupK / 16;
        for (int item = tid; item < kMmaRows * kChunks; item += kKernelWarps * 32) {
            const int row            = item / kChunks;
            const int chunk          = item - row * kChunks;
            const int swizzled_chunk = chunk ^ (row & 7);
            cp_async<16, Cache::cg>(&code_shared[row][swizzled_chunk * 16],
                                    codes + static_cast<std::int64_t>(cta_row0 + row) * kHidden +
                                        group_k0 + chunk * 16);
        }
        cp_commit();
    };

    const int warp_koff = k_split * kTileK;
    float acc[kWmmaNt][8];
#pragma unroll
    for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc[ni][r] = 0.0f; }
    }

    stage_codes(0);
    stage_x(warp_koff);
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        const int k0       = group_k0 + warp_koff;

        // WMMA C layout: lane l holds rows r + 8*(l>=16) at column l&15; each of those rows
        // carries its OWN 32-K scale.
        const int rb = (lane & 16) ? 8 : 0;
        unsigned row_scale_pair[8];
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) {
            const int scale_row = cta_row0 + rb + rr;
            row_scale_pair[rr]  = *reinterpret_cast<const unsigned*>(
                scales + (static_cast<std::int64_t>(scale_row) * (kHidden / 32) + k0 / 32) * 2);
        }

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            float group_acc[kWmmaNt][8];
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
                for (int r = 0; r < 8; ++r) { group_acc[ni][r] = 0.0f; }
            }
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                const int ks = group * 2 + ki;
                // WMMA A fragment: active lane l holds row m = l>>1 and the full 16-K window
                // [warp_koff + ks*16, +16). The 16 w8 codes lie in one swizzled 16-byte chunk.
                const int m                 = lane >> 1;
                const int chunk             = (warp_koff + ks * 16) >> 4;
                const unsigned short* base  = reinterpret_cast<const unsigned short*>(
                    &code_shared[m][(chunk ^ (m & 7)) * 16]);
                unsigned a_frag[8];
                const bool a_active = wmma_a_lane_active(lane);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned pair = bf16_pair_from_s8(static_cast<unsigned>(base[j]));
                    a_frag[j]           = a_active ? pair : 0u;
                }
#pragma unroll
                for (int ni = 0; ni < kWmmaNt; ++ni) {
                    unsigned b_frag[8];
                    const int col = ni * 16 + (lane & 15);
                    wmma_load_b_bf16(b_frag, b_shared[warp], col, ks * 16, kTileK, swizzle_64);
                    WmmaC8& c  = *reinterpret_cast<WmmaC8*>(group_acc[ni]);
                    WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag);
                    WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag);
                    c          = wmma_bf16(a, b, c);
                }
            }
            float row_scale[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const unsigned bits = group == 0 ? row_scale_pair[r] & 0xffffu
                                                 : row_scale_pair[r] >> 16;
                row_scale[r]        = __half2float(__ushort_as_half(bits));
            }
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc[ni][r] = fmaf(group_acc[ni][r], row_scale[r], acc[ni][r]);
                }
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_codes(group_k0 + kGroupK);
            stage_x(k0 + kGroupK);
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = reinterpret_cast<float*>(b_shared);
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
                acc[ni][r] +=
                    partial[(((warp + 1) * kWmmaNt + ni) * 32 + lane) * 8 + r];
            }
            if constexpr (KSplits == 4) {
                if (k_split == 2) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        partial[((warp * kWmmaNt + ni) * 32 + lane) * 8 + r] = acc[ni][r];
                    }
                }
            }
        }
    }

    if constexpr (KSplits == 4) {
        __syncthreads();
        if (k_split == 0) {
#pragma unroll
            for (int ni = 0; ni < kWmmaNt; ++ni) {
                const int partner_warp = n_group * KSplits + 2;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc[ni][r] +=
                        partial[((partner_warp * kWmmaNt + ni) * 32 + lane) * 8 + r];
                }
            }
        }
    }

    if (k_split == 0) {
        const W8OutputTile output_tile = output.tile(cta_row0);
#pragma unroll
        for (int ni = 0; ni < kWmmaNt; ++ni) {
            // WMMA C layout: lane l holds (row r + 8*(l>=16), col = n_base + ni*16 + (l&15)).
            const int col       = n_base + ni * 16 + (lane & 15);
            const int local_col = ni * 16 + (lane & 15);
            const int row_lo    = cta_row0 + (lane < 16 ? 0 : 8);
            // Skip the overlap columns when kWarpCols is not a multiple of 16.
            if (local_col >= kWarpCols) { continue; }
            if (col < active_cols) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    *output_tile.at(row_lo + r, col) = __float2bfloat16_rn(acc[ni][r]);
                }
            }
        }
    }
}

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        hipStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Geometry = W8LinearGeometry<kRows, kHidden>;
    using Schedule = W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__hip_bfloat16*>(qkv.data), static_cast<__hip_bfloat16*>(z.data)};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols, class Publish>
void launch_active_cols_conv(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                             Tensor& z, Publish publish, hipStream_t stream) {
    static_assert(ActiveCols >= 2 && ActiveCols <= 16);
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = W8LinearGeometry<kRows, kHidden>;
    using Schedule         = W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>;
    const Output ignored_output{static_cast<__hip_bfloat16*>(query.data),
                                static_cast<__hip_bfloat16*>(z.data)};
    const W8GdnSplitKConvEpilogue<Publish> epilogue{
        {
            static_cast<const __hip_bfloat16*>(conv_weight.data),
            static_cast<const __hip_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__hip_bfloat16*>(query.data),
            static_cast<__hip_bfloat16*>(key.data),
            static_cast<__hip_bfloat16*>(value.data),
            8192,
            2048,
            2048,
            4096,
            0,
            ActiveCols,
            0,
            publish,
        },
        static_cast<__hip_bfloat16*>(z.data),
    };
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, Output, W8GdnSplitKConvEpilogue<Publish>>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
}

template <int ActiveCols>
void launch_active_cols_conv_snapshot(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, hipStream_t stream) {
    launch_active_cols_conv<ActiveCols>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__hip_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data), 8192},
        stream);
}

template <int ActiveCols>
void launch_active_cols_conv_record(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight, const Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_slot,
                                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                    Tensor& z, hipStream_t stream) {
    launch_active_cols_conv<ActiveCols>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        RecordColumnPublish{static_cast<__hip_bfloat16*>(conv_record.data), 8192, ActiveCols},
        stream);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium_cols(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        hipStream_t stream) {
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__hip_bfloat16*>(qkv.data), static_cast<__hip_bfloat16*>(z.data)};
    w8_gdn_input_medium_t_splitk_kernel<TileCols, KSplits, NGroups, MinBlocks>
        <<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

using ProjectionLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, hipStream_t);
using SnapshotLauncher   = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                  const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                  Tensor&, Tensor&, hipStream_t);
using RecordLauncher     = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                Tensor&, hipStream_t);

template <std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_snapshot_launchers(std::index_sequence<Offsets...>) {
    return std::array<SnapshotLauncher, sizeof...(Offsets)>{
        &launch_active_cols_conv_snapshot<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLauncher, sizeof...(Offsets)>{
        &launch_active_cols_conv_record<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kProjectionLaunchers = make_projection_launchers(
    std::make_index_sequence<kLastProjectionExactCols - kFirstExactCols + 1>{});
constexpr auto kSnapshotLaunchers = make_snapshot_launchers(
    std::make_index_sequence<kLastSnapshotExactCols - kFirstExactCols + 1>{});
constexpr auto kRecordLaunchers =
    make_record_launchers(std::make_index_sequence<kLastSnapshotExactCols - kFirstExactCols + 1>{});

} // namespace

void w8_gdn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    hipStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > 96) {
        throw std::invalid_argument("W8 GDN split-K MMA requires T=2..96");
    }
    if (cols <= kLastProjectionExactCols) {
        kProjectionLaunchers[cols - kFirstExactCols](x, weight, qkv, z, stream);
    } else if (cols <= 48) {
        launch_medium_cols<48, 4, 2, 3>(x, weight, qkv, z, stream);
    } else if (cols <= 64) {
        launch_medium_cols<64, 4, 2, 2>(x, weight, qkv, z, stream);
    } else {
        launch_medium_cols<96, 2, 4, 3>(x, weight, qkv, z, stream);
    }
    HIP_CHECK(hipGetLastError());
}

void w8_gdn_input_splitk_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, hipStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("W8 fused GDN input snapshot requires T=2..16");
    }
    kSnapshotLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, valid_columns,
                                               initial_slot, snapshot_base_slot, query, key, value,
                                               z, stream);
    HIP_CHECK(hipGetLastError());
}

void w8_gdn_input_splitk_conv_record_launch(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slot,
                                            Tensor& conv_record, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, hipStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("W8 fused GDN input record requires T=2..16");
    }
    kRecordLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, valid_columns,
                                             initial_slot, conv_record, query, key, value, z,
                                             stream);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
