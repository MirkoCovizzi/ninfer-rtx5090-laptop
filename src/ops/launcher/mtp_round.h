#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops::detail {

void mtp_prepare_next_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                   const Tensor& accepted, const Tensor& updated_frontiers,
                                   const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                   const Tensor& rope_deltas, Tensor& alignment_ids,
                                   Tensor& next_extents, Tensor& ar_positions,
                                   Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                   std::int32_t max_context, hipStream_t stream);

} // namespace ninfer::ops::detail
