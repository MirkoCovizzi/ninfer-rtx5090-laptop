#pragma once

#include "core/paged_kv_cache.h"
#include "ninfer/ops/kvarn.h"

#include <span>

namespace ninfer::ops::detail {

void kvarn_compress_launch(const Tensor& rotated_k, const Tensor& rotated_v,
                           KvarnTileStorage storage, std::int32_t iterations,
                           cudaStream_t stream);
void kvarn_decompress_launch(const KvarnTileStorage& storage, Tensor& rotated_k, Tensor& rotated_v,
                             cudaStream_t stream);
void kvarn_hadamard_launch(const Tensor& source, Tensor& destination, cudaStream_t stream);

void kvarn_paged_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                               const Tensor& valid_columns, const Tensor& table_rows,
                               PagedKVBatchLayerView cache, cudaStream_t stream);

void kvarn_encode_full_tails_launch(std::span<const PagedKVBatchLayerView> layers,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, cudaStream_t stream);

void kvarn_flush_full_tails_launch(std::span<const PagedKVBatchLayerView> layers,
                                   const Tensor& positions, const Tensor& valid_columns,
                                   const Tensor& table_rows, cudaStream_t stream);

} // namespace ninfer::ops::detail
