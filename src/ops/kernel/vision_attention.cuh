#include "hip/hip_runtime.h"
#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"
#include <hip/hip_math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kVisionAttentionHeadDim = 72;
inline constexpr int kVisionAttentionHeads   = 16;
inline constexpr int kVisionAttentionBr      = 64;
inline constexpr int kVisionAttentionBc      = 64;
inline constexpr int kVisionAttentionPaddedD = 128;

struct alignas(16) VisionAttentionTile {
    std::int32_t q0;
    std::int32_t begin;
    std::int32_t end;
    std::int32_t reserved;
};

static_assert(sizeof(VisionAttentionTile) == 16);

__device__ __forceinline__ const __hip_bfloat16*
vision_attention_ptr(const __hip_bfloat16* data, std::int64_t stride_d, std::int64_t stride_h,
                     std::int64_t stride_t, int d, int h, int t) {
    return data + static_cast<std::int64_t>(d) * stride_d +
           static_cast<std::int64_t>(h) * stride_h + static_cast<std::int64_t>(t) * stride_t;
}

__device__ __forceinline__ int vision_attention_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// Identity swizzle for the transposed V tile. The Xor64 swizzle only fits 64-wide
// rows; for Bc = 16/32 the transposed [d][Bc] row is narrower, so feature keys are
// stored contiguously (vs[d * Bc + key]) and loaded with the identity swizzle.
__device__ __forceinline__ int vision_attention_swz_id(int /*row*/, int col) { return col; }

__global__ void vision_attention_prepare_tiles_kernel(const std::int32_t* cu_seqlens,
                                                      std::int32_t segments,
                                                      VisionAttentionTile* tiles,
                                                      std::int32_t max_tiles,
                                                      std::int32_t patches) {
    for (int tile = static_cast<int>(threadIdx.x); tile < max_tiles;
         tile += static_cast<int>(blockDim.x)) {
        tiles[tile] = {-1, 0, 0, 0};
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }

    int next = 0;
    for (int segment = 0; segment < segments; ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        if (begin < 0 || end <= begin || end > patches) { continue; }
        for (int q0 = begin; q0 < end && next < max_tiles; q0 += kVisionAttentionBr) {
            tiles[next++] = {q0, begin, end, 0};
        }
    }
}

template <int Br, int Threads>
__device__ __forceinline__ void
vision_attention_stage_q(__hip_bfloat16* dst, const __hip_bfloat16* q, int q0, int end, int head,
                         int tid, std::int64_t stride_d, std::int64_t stride_h,
                         std::int64_t stride_t) {
    constexpr int VecsPerRow = kVisionAttentionPaddedD / 8;
    for (int chunk = tid; chunk < Br * VecsPerRow; chunk += Threads) {
        const int row       = chunk / VecsPerRow;
        const int d         = (chunk % VecsPerRow) * 8;
        const bool in_range = q0 + row < end && d < kVisionAttentionHeadDim;
        __hip_bfloat16* smem = &dst[row * kVisionAttentionPaddedD + vision_attention_swz(row, d)];
        const __hip_bfloat16* global = vision_attention_ptr(
            q, stride_d, stride_h, stride_t, in_range ? d : 0, head, in_range ? q0 + row : q0);
        cp_async_zfill<16, Cache::cg>(smem, global, in_range ? 16 : 0);
    }
}

template <int Bc, int Threads>
__device__ __forceinline__ void
vision_attention_stage_kv(__hip_bfloat16* dst, const __hip_bfloat16* src, int key0, int end, int head,
                          int tid, std::int64_t stride_d, std::int64_t stride_h,
                          std::int64_t stride_t) {
    constexpr int VecsPerRow = kVisionAttentionPaddedD / 8;
    for (int chunk = tid; chunk < Bc * VecsPerRow; chunk += Threads) {
        const int row       = chunk / VecsPerRow;
        const int d         = (chunk % VecsPerRow) * 8;
        const bool in_range = key0 + row < end && d < kVisionAttentionHeadDim;
        __hip_bfloat16* smem = &dst[row * kVisionAttentionPaddedD + vision_attention_swz(row, d)];
        const __hip_bfloat16* global =
            vision_attention_ptr(src, stride_d, stride_h, stride_t, in_range ? d : 0, head,
                                 in_range ? key0 + row : key0);
        cp_async_zfill<16, Cache::cg>(smem, global, in_range ? 16 : 0);
    }
}

// Transposed V staging for the WMMA PV B-operand. The WMMA B-fragment for
// O += P*V holds column n = d and elements k = key, reading v_t[d * Bc +
// swz(d, key)] — contiguous per 8-key group. The untransposed [key][d] layout
// would scatter the fragment over one element per smem row. Rows d in [D, 80)
// (the unused tail of the last 16-wide atom) are never written and stay zero.
template <int Bc, int Threads>
__device__ __forceinline__ void
vision_attention_stage_kv_t(__hip_bfloat16* dst, const __hip_bfloat16* src, int key0, int end,
                            int head, int tid, std::int64_t stride_d, std::int64_t stride_h,
                            std::int64_t stride_t) {
    constexpr int D      = kVisionAttentionHeadDim;
    constexpr int Chunks = Bc * (D / 8);
    for (int chunk = tid; chunk < Chunks; chunk += Threads) {
        const int key = chunk / (D / 8);
        const int d8  = chunk - key * (D / 8);
        const bool in_range = key0 + key < end;
        const __hip_bfloat16* global =
            vision_attention_ptr(src, stride_d, stride_h, stride_t, d8 * 8, head,
                                 in_range ? key0 + key : key0);
        if (in_range) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = d8 * 8 + i;
                if (d < D) { dst[d * Bc + key] = global[i]; }
            }
        }
    }
}


template <int Br, int Bc>
__launch_bounds__(Br * 2, 128 / Br) __global__ void vision_attention_flash_kernel(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v, const VisionAttentionTile* __restrict__ tiles,
    std::int32_t patches, std::int32_t uniform_segment_length, __hip_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    static_assert(Br == 16 || Br == 32 || Br == 64);
    static_assert(Bc == 16 || Bc == 32 || Bc == 64);
    constexpr int D             = kVisionAttentionHeadDim;
    constexpr int Dp            = kVisionAttentionPaddedD;
    constexpr int Threads       = Br * 2;
    constexpr int QKKs          = 5; // ceil(72 / 16)
    constexpr int WQKNt         = Bc / 16; // QK score n-atoms
    constexpr int PVKs          = Bc / 16; // PV contraction steps over keys
    constexpr int WPVNt         = 5;       // ceil(72 / 16) PV n-atoms
    constexpr float ScaleLog2E  = 0.11785113019775792073f * 1.4426950408889634074f;
    constexpr unsigned long long FullMask = 0xffffffffull;

    VisionAttentionTile tile;
    if (tiles != nullptr) {
        tile = tiles[blockIdx.x];
    } else if (uniform_segment_length > 0) {
        const int tiles_per_segment = (uniform_segment_length + Br - 1) / Br;
        const int segment           = static_cast<int>(blockIdx.x) / tiles_per_segment;
        const int tile_in_segment   = static_cast<int>(blockIdx.x) - segment * tiles_per_segment;
        const int begin             = segment * uniform_segment_length;
        tile = {begin + tile_in_segment * Br, begin, begin + uniform_segment_length, 0};
    } else {
        tile = {static_cast<std::int32_t>(blockIdx.x) * Br, 0, patches, 0};
    }
    if (tile.q0 < 0) { return; }
    const int head = static_cast<int>(blockIdx.y);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const bool lane_hi = (lane >> 4) != 0;
    const int warp_row0 = warp * 16;

    extern __shared__ __align__(16) __hip_bfloat16 shared[];
    __hip_bfloat16* q_s = shared;
    __hip_bfloat16* k_s = q_s + Br * Dp;
    __hip_bfloat16* v_s = k_s + Bc * Dp;

    vision_attention_stage_q<Br, Threads>(q_s, q, tile.q0, tile.end, head, tid, q_stride_d,
                                          q_stride_h, q_stride_t);

    float acc[WPVNt][8];
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
    }
    float m0[8], m1[8], l0[8], l1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        m0[r] = -HIP_INF_F;
        m1[r] = -HIP_INF_F;
        l0[r] = 0.0f;
        l1[r] = 0.0f;
    }

    cp_commit();
    vision_attention_stage_kv<Bc, Threads>(k_s, k, tile.begin, tile.end, head, tid, k_stride_d,
                                           k_stride_h, k_stride_t);
    cp_commit();

    // Pre-zero the transposed V region so the unused d >= 72 tail is finite.
#pragma unroll 1
    for (int i = tid; i < 80 * Bc; i += Threads) v_s[i] = __hip_bfloat16(0);

    const int key_blocks = (tile.end - tile.begin + Bc - 1) / Bc;
    const int qrow_off   = tile.q0 + warp_row0;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int key0 = tile.begin + kb * Bc;
        cp_wait<0>();
        __syncthreads();

        vision_attention_stage_kv_t<Bc, Threads>(v_s, v, key0, tile.end, head, tid, v_stride_d,
                                                 v_stride_h, v_stride_t);
        cp_commit();

        float score[WQKNt][8];
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { score[nt][i] = 0.0f; }
        }
        unsigned a_frag[8];
        unsigned b_frag[WQKNt][8];
#pragma unroll
        for (int ks = 0; ks < QKKs; ++ks) {
            const int k0   = ks * 16;
            const int arow = warp_row0 + (lane >> 1);
            if (wmma_a_lane_active(lane)) {
                wmma_load_a_bf16(a_frag, q_s, arow, k0, Dp, vision_attention_swz);
            }
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
                const int bcol = nt * 16 + (lane & 15);
                wmma_load_b_bf16(b_frag[nt], k_s, bcol, k0, Dp, vision_attention_swz);
            }
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
                WmmaC8& c = *reinterpret_cast<WmmaC8*>(score[nt]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(a_frag);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(b_frag[nt]);
                c         = wmma_bf16(a, b, c);
            }
        }

        const bool full_tile = tile.q0 + Br <= tile.end && key0 + Bc <= tile.end;
        float bm0[8], bm1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bm0[r] = -HIP_INF_F; bm1[r] = -HIP_INF_F; }
        if (full_tile) {
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    bm0[r] = fmaxf(bm0[r], score[nt][r]);
                    bm1[r] = fmaxf(bm1[r], score[nt][r]);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int qrow  = qrow_off + r + (lane_hi ? 8 : 0);
                    const int key   = key0 + nt * 16 + (lane & 15);
                    const bool keep = qrow < tile.end && key < tile.end;
                    score[nt][r]    = keep ? score[nt][r] : -HIP_INF_F;
                    bm0[r]          = fmaxf(bm0[r], score[nt][r]);
                    bm1[r]          = fmaxf(bm1[r], score[nt][r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            bm0[r] = warp_max<16>(bm0[r], FullMask);
            bm1[r] = warp_max<16>(bm1[r], FullMask);
        }

        float nm0[8], nm1[8], alpha0[8], alpha1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            nm0[r]    = fmaxf(m0[r], bm0[r]);
            nm1[r]    = fmaxf(m1[r], bm1[r]);
            alpha0[r] = exp2_approx(__fmaf_rn(m0[r], ScaleLog2E, -nm0[r] * ScaleLog2E));
            alpha1[r] = exp2_approx(__fmaf_rn(m1[r], ScaleLog2E, -nm1[r] * ScaleLog2E));
        }

        float bl0[8], bl1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bl0[r] = 0.0f; bl1[r] = 0.0f; }
        const int arow = lane >> 1;
        const int a_rr = arow & 7;
        unsigned pv_a[PVKs][8];
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
            float p[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float s  = score[nt][r];
                const float nm = lane_hi ? nm1[r] : nm0[r];
                p[r] = (s > -HIP_INF_F) ? exp2_approx(__fmaf_rn(s, ScaleLog2E, -nm * ScaleLog2E))
                                        : 0.0f;
                bl0[r] += p[r];
                bl1[r] += p[r];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int src0 = 2 * j + (lane_hi ? 16 : 0);
                const int src1 = 2 * j + 1 + (lane_hi ? 16 : 0);
                // A-fragment element (row q, key i) lives in lane (i & 15) +
                // 16 * (q >= 8) at register (q & 7) of the score C fragment. The
                // shuffle only sends a uniform register per call, so iterate the 8
                // registers and keep a_rr (src lanes evaluate p[own row], not p[q]).
                float e0 = 0.0f, e1 = 0.0f;
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const float v0 = __shfl_sync(FullMask, p[rr], src0);
                    const float v1 = __shfl_sync(FullMask, p[rr], src1);
                    if (rr == a_rr) { e0 = v0; e1 = v1; }
                }
                pv_a[nt][j]    = pack_bf16x2(e0, e1);
            }
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            l0[r] = __fmaf_rn(l0[r], alpha0[r], bl0[r]);
            l1[r] = __fmaf_rn(l1[r], alpha1[r], bl1[r]);
            m0[r] = nm0[r];
            m1[r] = nm1[r];
        }
#pragma unroll
        for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[n][r] *= (lane_hi ? alpha1[r] : alpha0[r]);
            }
        }

        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) {
            vision_attention_stage_kv<Bc, Threads>(k_s, k, key0 + Bc, tile.end, head, tid,
                                                   k_stride_d, k_stride_h, k_stride_t);
            cp_commit();
        }

#pragma unroll
        for (int kk = 0; kk < PVKs; ++kk) {
            WmmaA16I a = *reinterpret_cast<WmmaA16I*>(pv_a[kk]);
#pragma unroll
            for (int n = 0; n < WPVNt; ++n) {
                unsigned bfrag[8];
                const int dcol = n * 16 + (lane & 15);
                wmma_load_b_bf16(bfrag, v_s, dcol, kk * 16, Bc, vision_attention_swz_id);
                WmmaC8& c   = *reinterpret_cast<WmmaC8*>(acc[n]);
                WmmaA16I b   = *reinterpret_cast<WmmaA16I*>(bfrag);
                c          = wmma_bf16(a, b, c);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        l0[r] = warp_sum<16>(l0[r], FullMask);
        l1[r] = warp_sum<16>(l1[r], FullMask);
    }
    float inv_l0[8], inv_l1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        inv_l0[r] = l0[r] > 0.0f ? __frcp_rn(l0[r]) : 0.0f;
        inv_l1[r] = l1[r] > 0.0f ? __frcp_rn(l1[r]) : 0.0f;
    }
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
        const int d = n * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int qrow = qrow_off + r + (lane_hi ? 8 : 0);
            if (d < D && qrow < tile.end) {
                const std::int64_t offset =
                    (static_cast<std::int64_t>(qrow) * kVisionAttentionHeads + head) * D + d;
                out[offset] = __float2bfloat16_rn(acc[n][r] * (lane_hi ? inv_l1[r] : inv_l0[r]));
            }
        }
    }
}

} // namespace ninfer::ops
