#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/kvarn.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

[[nodiscard]] std::size_t kvarn_attention_workspace_capacity_bytes(
    std::int32_t query_heads, CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width);

// Appends and attends in the orthonormal KVarN frame. Every newly complete non-sink page is
// encoded. Provisional calls retain its BF16 tail until commit: queries before the closing position
// read BF16, queries at/after closure read the record. Rejected suffixes remain overwritable in the
// tail. Q is transformed in place; K/V are disposable and may also be transformed in place.
void kvarn_attention(Tensor query, Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                     KvarnPagedBatchLayerView cache, bool provisional,
                     CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                     Tensor& output, cudaStream_t stream);

void kvarn_attention_cached(Tensor query, const Tensor& positions, const Tensor& kv_table_rows,
                            float scale, const KvarnPagedBatchLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& output, cudaStream_t stream);

void kvarn_kv_append(Tensor key, Tensor value, const Tensor& positions, const Tensor& valid_columns,
                     const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache, bool provisional,
                     cudaStream_t stream);

// accepted_columns is an I32 prefix count per batch row. Full non-sink pages in that accepted
// prefix already have encoded records; only their BF16 markers are retired. Sinks remain lossless.
void kvarn_commit_pages(const Tensor& positions, const Tensor& accepted_columns,
                        const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache,
                        cudaStream_t stream);

// Re-establishes the writable represented tail after truncating a retained sequence. A historical
// partial page is decoded from its record; an already-live partial tail is preserved.
void kvarn_restore_tail(std::int32_t frontier, KvarnPagedLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
