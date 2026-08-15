#include "hip/hip_runtime.h"
#pragma once

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __hip_bfloat16* output,
                                  std::int32_t tokens, float alpha, hipStream_t stream);

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __hip_bfloat16* query,
                                     __hip_bfloat16* gate, __hip_bfloat16* key, __hip_bfloat16* value,
                                     std::int32_t tokens, float alpha, hipStream_t stream);

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __hip_bfloat16* qkv, __hip_bfloat16* z, std::int32_t tokens,
                               float alpha, hipStream_t stream);

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __hip_bfloat16* residual,
                                      std::int32_t tokens, float alpha, hipStream_t stream);

} // namespace ninfer::ops::detail
