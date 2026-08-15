#include "hip/hip_runtime.h"
#pragma once

// Shared Qwen3.6 GQA dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillHeadDim = 256;

// gfx1151 limits dynamic LDS to 64 KiB per CTA; the (Br + 2*Bc) * D * 2 staging
// of the bf16 prompt kernel must fit, so the key tile is 32 wide.
inline constexpr int kGqaPrefillBr        = 64;
inline constexpr int kGqaPrefillBc        = 32;
inline constexpr int kGqaPrefillThreads   = 128;
inline constexpr int kGqaPrefillSmemBytes = (kGqaPrefillBr + 2 * kGqaPrefillBc) *
                                            kGqaPrefillHeadDim *
                                            static_cast<int>(sizeof(__hip_bfloat16));

struct GqaPrefillDirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct GqaPrefillBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaPrefillHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_zero_output_rows(__hip_bfloat16* out, int q_head,
                                                             int row_begin, int row_end, int tid,
                                                             int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * kGqaPrefillHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / kGqaPrefillHeadDim;
        const int d   = element - (row - row_begin) * kGqaPrefillHeadDim;
        out[gqa_prefill_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// Identity swizzle for the transposed V tile: keys are stored contiguously per
// dimension row (v_s[d * Bc + key]), which is collision-free and matches the
// b128 fragment load (8 consecutive keys).
__device__ __forceinline__ int gqa_prefill_swz_identity(int /*row*/, int col) {
    return col;
}

__device__ __forceinline__ unsigned gqa_prefill_swz_addr(unsigned lane_base, unsigned ck,
                                                         unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

} // namespace ninfer::ops