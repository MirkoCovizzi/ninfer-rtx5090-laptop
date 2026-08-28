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

// Appends and attends in the orthonormal KVarN frame. Committed calls encode every newly complete
// non-sink page; provisional calls retain all touched pages in BF16 until kvarn_commit_pages.
void kvarn_attention(Tensor query, Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                     KvarnPagedBatchLayerView cache, bool provisional,
                     CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                     Tensor& output,
                     cudaStream_t stream);

void kvarn_attention_cached(Tensor query, const Tensor& positions, const Tensor& kv_table_rows,
                            float scale, const KvarnPagedBatchLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& output, cudaStream_t stream);

void kvarn_kv_append(Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows,
                     KvarnPagedBatchLayerView cache, bool provisional, cudaStream_t stream);

// accepted_columns is an I32 prefix count per batch row. Full non-sink pages in that accepted
// prefix are encoded and their dynamic BF16 markers retired; sink pages remain lossless.
void kvarn_commit_pages(const Tensor& positions, const Tensor& accepted_columns,
                        const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache,
                        cudaStream_t stream);

// Re-establishes the writable represented tail after truncating a retained sequence. A historical
// partial page is decoded from its record; an already-live partial tail is preserved.
void kvarn_restore_tail(std::int32_t frontier, KvarnPagedLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
