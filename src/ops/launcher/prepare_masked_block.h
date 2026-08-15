#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void prepare_masked_block_launch(const Tensor& anchors, const Tensor& lengths,
                                 const Tensor& valid_columns, std::int32_t mask_id, Tensor& ids,
                                 Tensor& positions, std::int32_t block_size, hipStream_t stream);

} // namespace ninfer::ops::detail
