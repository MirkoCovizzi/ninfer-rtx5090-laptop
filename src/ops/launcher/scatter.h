#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, hipStream_t stream);
void scatter_bf16_batch_launch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               hipStream_t stream);
} // namespace ninfer::ops::detail
