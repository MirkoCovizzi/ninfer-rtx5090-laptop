#include "hip/hip_runtime.h"
#pragma once

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4AddResidualEpilogue {
    const __hip_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t token,
                                           float value) const {
        return value + __bfloat162float(residual[static_cast<std::int64_t>(token) * rows + row]);
    }
};

} // namespace ninfer::ops::detail
