#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void fill_i32_positions_launch(Tensor& positions, std::int32_t start, hipStream_t stream);
void offset_i32_positions_launch(const Tensor& source, const Tensor& delta, Tensor& destination,
                                 hipStream_t stream);
void offset_i32_positions_block_launch(const Tensor& source, const Tensor& delta,
                                       Tensor& destination, int block, hipStream_t stream);

} // namespace ninfer::ops::detail
