#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/kvarn.h"

#include <cuda_runtime_api.h>

namespace ninfer::ops::kvarn {

void decode_attention(const Tensor& query, const Tensor& positions,
                      const Tensor& valid_columns, const Tensor& table_rows, float scale,
                      KvarnPagedBatchLayerView cache, GqaExecutionEnvelope envelope,
                      WorkspaceArena& workspace, Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops::kvarn
