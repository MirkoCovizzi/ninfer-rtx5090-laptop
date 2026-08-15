#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops::detail {

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, hipStream_t stream);

} // namespace ninfer::ops::detail
