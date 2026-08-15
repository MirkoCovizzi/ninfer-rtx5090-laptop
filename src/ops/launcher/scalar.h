#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void set_i32_scalar_launch(Tensor& destination, std::int32_t value, hipStream_t stream);
void assign_i32_scalar_launch(const Tensor& source, Tensor& destination, hipStream_t stream);
void add_i32_scalars_launch(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                            hipStream_t stream);
void increment_i32_scalar_launch(Tensor& scalar, hipStream_t stream);
void increment_i64_scalar_launch(Tensor& scalar, hipStream_t stream);

} // namespace ninfer::ops::detail
