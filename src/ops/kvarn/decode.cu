#include "ops/kvarn/decode.cuh"

#include "core/device.h"
#include "ops/kvarn/config.cuh"
#include "ops/kvarn/decode_kernel.cuh"
#include "ops/kvarn/streaming_prefill.cuh"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::kvarn {
namespace {

template <typename Geometry, bool Masked>
void launch_prefill(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                    const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                    Tensor& output, cudaStream_t stream) {
    using Metadata = CausalPromptBatchMetadata<Masked>;
    const Metadata metadata{
        .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
        .valid_columns = Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride  = cache.block_tables.ne[0],
    };
    const detail::PrefillCache input{
        static_cast<const std::uint8_t*>(cache.records.data),
        static_cast<const __nv_bfloat16*>(cache.tail_k.data),
        static_cast<const __nv_bfloat16*>(cache.tail_v.data),
        static_cast<const std::int32_t*>(cache.tail_logical_pages.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data),
        cache.block_tables.ne[0],
        cache.num_kv_heads,
    };
    static const cudaError_t attribute =
        cudaFuncSetAttribute(detail::attention_prefill_slab_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attribute);
    const int width         = query.ne[2];
    const int capacity      = static_cast<int>(envelope.max_visible_keys);
    const int slab_capacity = std::min(capacity, PrefillSlabTokens);
    auto scope              = workspace.scope();
    Tensor materialized_k   = workspace.alloc(DType::BF16, {D, slab_capacity, Geometry::KVHeads});
    Tensor materialized_v   = workspace.alloc(DType::BF16, {D, slab_capacity, Geometry::KVHeads});
    Tensor running_acc      = workspace.alloc(DType::FP32, {D, Geometry::QHeads, width});
    Tensor running_m        = workspace.alloc(DType::FP32, {Geometry::QHeads, width});
    Tensor running_l        = workspace.alloc(DType::FP32, {Geometry::QHeads, width});
    CUDA_CHECK(cudaMemsetAsync(running_l.data, 0, running_l.bytes(), stream));
    Tensor rotated_query = query;
    kvarn_hadamard(query, rotated_query, stream);
    const dim3 attention_grid(div_up(width, kCausalPromptBr), Geometry::QHeads, 1);
    for (int slab_begin = 0; slab_begin < capacity; slab_begin += slab_capacity) {
        const int slab_tokens = std::min(slab_capacity, capacity - slab_begin);
        const dim3 materialize_grid(div_up(slab_tokens, Group), Geometry::KVHeads, 1);
        detail::materialize_prefill_slab_kernel<Metadata><<<materialize_grid, D, 0, stream>>>(
            input, metadata, static_cast<const std::int32_t*>(positions.data), width, slab_begin,
            slab_capacity, static_cast<__nv_bfloat16*>(materialized_k.data),
            static_cast<__nv_bfloat16*>(materialized_v.data));
        const detail::MaterializedPrefillInput materialized{
            static_cast<const __nv_bfloat16*>(materialized_k.data),
            static_cast<const __nv_bfloat16*>(materialized_v.data),
            slab_begin,
            slab_capacity,
        };
        detail::attention_prefill_slab_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(rotated_query.data), materialized, metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<float*>(running_acc.data), static_cast<float*>(running_m.data),
                static_cast<float*>(running_l.data), width, slab_begin, slab_begin + slab_tokens);
    }
    const dim3 finalize_grid(Geometry::QHeads, width, 1);
    detail::finalize_prefill_slab_kernel<Geometry, Metadata><<<finalize_grid, D, 0, stream>>>(
        static_cast<const float*>(running_acc.data), static_cast<const float*>(running_l.data),
        metadata, width, static_cast<__nv_bfloat16*>(output.data));
    kvarn_hadamard(output, output, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool MultiBatch, bool Masked, int ColumnsPerBlock>
void launch_partial(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                    const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                    CausalAttentionExecutionEnvelope envelope, int column_begin, int width,
                    int splits, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                    Tensor& output, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits,
                    query.ne[3] * div_up(width + 2 * (ColumnsPerBlock - 1), ColumnsPerBlock));
    constexpr int query_groups = ColumnsPerBlock == 4 ? 2 : ColumnsPerBlock;
    constexpr std::size_t query_smem =
        ColumnsPerBlock >= 3
            ? static_cast<std::size_t>(query_groups) * detail::kDecodeBr * D * sizeof(__nv_bfloat16)
            : 0;
    detail::attention_decode_kernel<Geometry, MultiBatch, Masked, ColumnsPerBlock>
        <<<grid, detail::kDecodeWarps * query_groups * 32, query_smem, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data),
            static_cast<const std::uint8_t*>(cache.records.data),
            static_cast<const __nv_bfloat16*>(cache.tail_k.data),
            static_cast<const __nv_bfloat16*>(cache.tail_v.data),
            static_cast<const std::int32_t*>(cache.tail_logical_pages.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0], width,
            query.ne[2], column_begin, static_cast<std::int32_t>(envelope.max_visible_keys),
            cache.num_kv_heads, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 reduce_grid(Geometry::QHeads, 1, width * query.ne[3]);
    const auto reduce = [&]<bool PerTokenSplits>() {
        detail::reduce_output_hadamard_kernel<Geometry, MultiBatch, Masked, PerTokenSplits,
                                              ColumnsPerBlock><<<reduce_grid, D, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(partial_acc.data),
            static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
            static_cast<const std::int32_t*>(positions.data),
            Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr, width,
            query.ne[2], column_begin, query.ne[3], splits,
            static_cast<__nv_bfloat16*>(output.data));
    };
    // Preserve the measured W=1 reducer code while wider chunks select splits per query column.
    if (width == 1) {
        reduce.template operator()<false>();
    } else {
        reduce.template operator()<true>();
    }
}

} // namespace

void decode_attention(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                      const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                      CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                      Tensor& output, cudaStream_t stream) {
    if (query.ne[3] == 1 && query.ne[2] >= kCausalPromptBr) {
        const bool masked = valid_columns.data != nullptr;
        if (query.ne[1] == CausalD256H24Kv4::QHeads) {
            if (masked) {
                launch_prefill<CausalD256H24Kv4, true>(query, positions, valid_columns, table_rows,
                                                       scale, cache, envelope, workspace, output,
                                                       stream);
            } else {
                launch_prefill<CausalD256H24Kv4, false>(query, positions, valid_columns, table_rows,
                                                        scale, cache, envelope, workspace, output,
                                                        stream);
            }
        } else if (masked) {
            launch_prefill<CausalD256H16Kv2, true>(query, positions, valid_columns, table_rows,
                                                   scale, cache, envelope, workspace, output,
                                                   stream);
        } else {
            launch_prefill<CausalD256H16Kv2, false>(query, positions, valid_columns, table_rows,
                                                    scale, cache, envelope, workspace, output,
                                                    stream);
        }
        return;
    }
    Tensor rotated_query = query;
    kvarn_hadamard(query, rotated_query, stream);
    constexpr int kChunk = 6;
    for (int begin = 0; begin < query.ne[2]; begin += kChunk) {
        const int width = std::min(kChunk, query.ne[2] - begin);
        auto scope      = workspace.scope();
        int split_capacity =
            ops::detail::causal_attention_split_capacity(query.ne[1], width, DType::BF16, envelope);
        if (query.ne[1] == CausalD256H24Kv4::QHeads && envelope.max_visible_keys > 8198) {
            split_capacity = std::max(split_capacity, DecodeLongSplits);
        }
        const int split_scale = query.ne[1] == CausalD256H24Kv4::QHeads
                                    ? CausalD256H24Kv4::SmallTSplitScale
                                    : CausalD256H16Kv2::SmallTSplitScale;
        int split_limit       = DecodeLongSplits * split_scale;
        if (query.ne[1] == CausalD256H24Kv4::QHeads && envelope.min_visible_keys > 8198 &&
            envelope.max_visible_keys <= DecodeMidWindow) {
            split_limit = DecodeMidSplits;
        }
        const int splits = std::min(split_capacity, split_limit);
        Tensor acc = workspace.alloc(DType::BF16, {D, query.ne[1], width, splits * query.ne[3]});
        Tensor m   = workspace.alloc(DType::FP32, {query.ne[1], width, splits * query.ne[3]});
        Tensor l   = workspace.alloc(DType::FP32, {query.ne[1], width, splits * query.ne[3]});
        const bool multi    = query.ne[3] > 1;
        const bool masked   = valid_columns.data != nullptr;
        const auto dispatch = [&]<typename Geometry>() {
            const bool pair_columns = Geometry::QHeads == CausalD256H24Kv4::QHeads && width > 1 &&
                                      envelope.max_visible_keys > MtpPackedWindow;
            const auto launch = [&]<bool MultiBatch, bool Masked>() {
                if (pair_columns) {
                    // Narrow widths mask unused columns; widths five and six use a second CTA.
                    launch_partial<Geometry, MultiBatch, Masked, 4>(
                        query, positions, valid_columns, table_rows, scale, cache, envelope, begin,
                        width, splits, acc, m, l, output, stream);
                } else {
                    launch_partial<Geometry, MultiBatch, Masked, 1>(
                        query, positions, valid_columns, table_rows, scale, cache, envelope, begin,
                        width, splits, acc, m, l, output, stream);
                }
            };
            if (multi) {
                if (masked) {
                    launch.template operator()<true, true>();
                } else {
                    launch.template operator()<true, false>();
                }
            } else if (masked) {
                launch.template operator()<false, true>();
            } else {
                launch.template operator()<false, false>();
            }
        };
        if (query.ne[1] == CausalD256H24Kv4::QHeads) {
            dispatch.template operator()<CausalD256H24Kv4>();
        } else {
            dispatch.template operator()<CausalD256H16Kv2>();
        }
    }
}

} // namespace ninfer::ops::kvarn
