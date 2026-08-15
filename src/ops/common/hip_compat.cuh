#pragma once

// HIP compatibility shims for CUDA-named bf16 intrinsics used across the
// kernel tree. HIP's hip_bf16.h provides the corresponding functionality under
// different names; these aliases keep the kernel sources unchanged.

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
    return __hip_bfloat162{__float2bfloat16(lo), __float2bfloat16(hi)};
}
