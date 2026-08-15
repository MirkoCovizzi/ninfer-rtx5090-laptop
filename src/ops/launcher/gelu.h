#pragma once

#include "core/tensor.h"
#include "ninfer/ops/gelu.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, hipStream_t stream);

} // namespace ninfer::ops::detail
