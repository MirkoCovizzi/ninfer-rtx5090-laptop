#include "hip/hip_runtime.h"
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __hip_bfloat16* output,
                                         std::int32_t tokens, float alpha, hipStream_t stream);

} // namespace ninfer::ops::detail
