#pragma once

// CUDA translation of Huawei KVarN's kvarn_store.py packing and scale-absorption identities.

#include "ops/kvarn/sinkhorn.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::kvarn {

struct StorePointers {
    std::uint8_t* k_codes;
    __half* k_scales;
    __half* k_zeros;
    __half* k_token_scales;
    std::uint8_t* v_codes;
    __half* v_channel_scales;
    __half* v_token_scales;
    __half* v_token_zeros;
};

struct SinkhornWorkspace {
    float* log_column;
    float* log_row;
    float* best_column;
    float* best_row;
    float* row_deviation;
    float* column_deviation;
    float* row_inverse;
    float* column_inverse;
    float* scratch;
};

__device__ __forceinline__ SinkhornWorkspace workspace_after(float* tile) {
    float* cursor = tile + (D + 1) * Group;
    SinkhornWorkspace result;
    result.log_column      = cursor;
    result.log_row         = cursor + D;
    result.best_column     = cursor + 2 * D;
    result.best_row        = cursor + 3 * D;
    result.row_deviation   = cursor + 4 * D;
    result.column_deviation = cursor + 5 * D;
    result.row_inverse     = cursor + 6 * D;
    result.column_inverse  = cursor + 7 * D;
    result.scratch         = cursor + 8 * D;
    return result;
}

__device__ __forceinline__ void store_k_tile(const float* tile, int record, StorePointers out,
                                             SinkhornWorkspace w) {
    variance_normalize<true>(tile, w.log_column, w.log_row, w.best_column, w.best_row,
                             w.row_deviation, w.column_deviation, w.row_inverse,
                             w.column_inverse, w.scratch);
    const int d = static_cast<int>(threadIdx.x);
    if (d < D) {
        float minimum = CUDART_INF_F;
        float maximum = -CUDART_INF_F;
        for (int token = 0; token < Group; ++token) {
            const float balanced = tile[d + (D + 1) * token] *
                                   expf(-w.best_row[d] - w.best_column[token]);
            minimum = fminf(minimum, balanced);
            maximum = fmaxf(maximum, balanced);
        }
        const float rtn_scale = fmaxf((maximum - minimum) / 15.0F, 1.0e-10F);
        const float row_scale = expf(w.best_row[d]);
        out.k_scales[static_cast<std::int64_t>(record) * D + d] =
            __float2half_rn(row_scale * rtn_scale);
        out.k_zeros[static_cast<std::int64_t>(record) * D + d] =
            __float2half_rn(row_scale * minimum);
        const std::int64_t code_base =
            (static_cast<std::int64_t>(record) * D + d) * (Group / 2);
        for (int byte = 0; byte < Group / 2; ++byte) {
            std::uint8_t packed = 0;
#pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int token = 2 * byte + item;
                const float balanced = tile[d + (D + 1) * token] *
                                       expf(-w.best_row[d] - w.best_column[token]);
                const int q = max(0, min(15, __float2int_rn((balanced - minimum) / rtn_scale)));
                packed |= static_cast<std::uint8_t>(q << (4 * item));
            }
            out.k_codes[code_base + byte] = packed;
        }
    }
    if (d < Group) {
        out.k_token_scales[static_cast<std::int64_t>(record) * Group + d] =
            __float2half_rn(expf(w.best_column[d]));
    }
}

__device__ __forceinline__ void store_v_tile(const float* tile, int record, StorePointers out,
                                             SinkhornWorkspace w) {
    variance_normalize<false>(tile, w.log_column, w.log_row, w.best_column, w.best_row,
                              w.row_deviation, w.column_deviation, w.row_inverse,
                              w.column_inverse, w.scratch);
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < D) {
        out.v_channel_scales[static_cast<std::int64_t>(record) * D + tid] =
            __float2half_rn(expf(w.best_column[tid]));
    }
    if (tid < Group) {
        float minimum = CUDART_INF_F;
        float maximum = -CUDART_INF_F;
        for (int d = 0; d < D; ++d) {
            const float balanced = tile[d + (D + 1) * tid] *
                                   expf(-w.best_row[tid] - w.best_column[d]);
            minimum = fminf(minimum, balanced);
            maximum = fmaxf(maximum, balanced);
        }
        const float rtn_scale = fmaxf((maximum - minimum) / 3.0F, 1.0e-10F);
        const float row_scale = expf(w.best_row[tid]);
        out.v_token_scales[static_cast<std::int64_t>(record) * Group + tid] =
            __float2half_rn(row_scale * rtn_scale);
        out.v_token_zeros[static_cast<std::int64_t>(record) * Group + tid] =
            __float2half_rn(row_scale * minimum);
        const std::int64_t code_base =
            (static_cast<std::int64_t>(record) * Group + tid) * (D / 4);
        for (int byte = 0; byte < D / 4; ++byte) {
            std::uint8_t packed = 0;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int d = 4 * byte + item;
                const float balanced = tile[d + (D + 1) * tid] *
                                       expf(-w.best_row[tid] - w.best_column[d]);
                const int q = max(0, min(3, __float2int_rn((balanced - minimum) / rtn_scale)));
                packed |= static_cast<std::uint8_t>(q << (2 * item));
            }
            out.v_codes[code_base + byte] = packed;
        }
    }
}

} // namespace ninfer::ops::kvarn
