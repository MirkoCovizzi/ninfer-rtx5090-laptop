#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, hipStream_t stream);

} // namespace ninfer::ops::detail
