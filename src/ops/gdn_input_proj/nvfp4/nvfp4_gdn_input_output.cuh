#include "hip/hip_runtime.h"
#pragma once

#include "ops/common/memory.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4GdnInputOutput {
    static constexpr std::int32_t kQkvRows = 10240;
    static constexpr std::int32_t kZRows   = 6144;

    __hip_bfloat16* qkv;
    __hip_bfloat16* z;

    __device__ __forceinline__ __hip_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((Nvfp4GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Nvfp4GdnInputOutput::kZRows % 128) == 0);

} // namespace ninfer::ops::detail
