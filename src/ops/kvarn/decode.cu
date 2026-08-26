#include "ops/kvarn/decode.cuh"

#include "core/device.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kvarn/config.cuh"
#include "ops/kvarn/decode_kernel.cuh"
#include "ops/kvarn/prefill_kernel.cuh"
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::kvarn {
namespace {

constexpr int kDecodeSplitsPerScale = 82;

template <typename Geometry, bool Masked>
void launch_prefill(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                    const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                    Tensor& output, cudaStream_t stream) {
    using Metadata = GqaPrefillBatchMetadata<Masked>;
    const Metadata metadata{
        .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
        .valid_columns =
            Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        .table_rows = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride = cache.block_tables.ne[0],
    };
    const detail::PrefillCache input{
        static_cast<const std::uint8_t*>(cache.records.data),
        static_cast<const __nv_bfloat16*>(cache.tail_k.data),
        static_cast<const __nv_bfloat16*>(cache.tail_v.data),
        static_cast<const std::int32_t*>(cache.tail_logical_pages.data),
        static_cast<const std::int32_t*>(table_rows.data),
        cache.num_kv_heads,
    };
    static const cudaError_t attribute = cudaFuncSetAttribute(
        detail::attention_prefill_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, detail::kPrefillSmemBytes);
    CUDA_CHECK(attribute);
    const int width = query.ne[2];
    const dim3 grid(div_up(width, detail::kPrefillBr), Geometry::QHeads, 1);
    detail::attention_prefill_kernel<Geometry, Metadata>
        <<<grid, detail::kPrefillThreads, detail::kPrefillSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data), input, metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(output.data), width);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_partial(const Tensor& query, const Tensor& positions, const Tensor& valid_columns,
                    const Tensor& table_rows, float scale, KvarnPagedBatchLayerView cache,
                    GqaExecutionEnvelope envelope, int column_begin, int width, int splits,
                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                    Tensor& output, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits, query.ne[3] * width);
    detail::attention_decode_kernel<Geometry, MultiBatch, Masked>
        <<<grid, detail::kDecodeWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data),
            static_cast<const std::uint8_t*>(cache.records.data),
            static_cast<const __nv_bfloat16*>(cache.tail_k.data),
            static_cast<const __nv_bfloat16*>(cache.tail_v.data),
            static_cast<const std::int32_t*>(cache.tail_logical_pages.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0], width,
            query.ne[2], column_begin,
            static_cast<std::int32_t>(envelope.max_visible_keys), cache.num_kv_heads, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 reduce_grid(Geometry::QHeads, 1, width * query.ne[3]);
    const auto reduce = [&]<bool PerTokenSplits>() {
        detail::reduce_output_hadamard_kernel<Geometry, MultiBatch, Masked, PerTokenSplits>
            <<<reduce_grid, D, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
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

void decode_attention(const Tensor& query, const Tensor& positions,
                      const Tensor& valid_columns, const Tensor& table_rows, float scale,
                      KvarnPagedBatchLayerView cache, GqaExecutionEnvelope envelope,
                       WorkspaceArena& workspace, Tensor& output, cudaStream_t stream) {
    if (query.ne[3] == 1 && query.ne[2] >= kGqaPrefillBr) {
        const bool masked = valid_columns.data != nullptr;
        if (query.ne[1] == Gqa27Geometry::QHeads) {
            if (masked) {
                launch_prefill<Gqa27Geometry, true>(query, positions, valid_columns, table_rows,
                                                     scale, cache, output, stream);
            } else {
                launch_prefill<Gqa27Geometry, false>(query, positions, valid_columns, table_rows,
                                                      scale, cache, output, stream);
            }
        } else if (masked) {
            launch_prefill<Gqa35Geometry, true>(query, positions, valid_columns, table_rows, scale,
                                                 cache, output, stream);
        } else {
            launch_prefill<Gqa35Geometry, false>(query, positions, valid_columns, table_rows, scale,
                                                  cache, output, stream);
        }
        return;
    }
    Tensor rotated_query = query;
    kvarn_hadamard(query, rotated_query, stream);
    constexpr int kChunk = 6;
    for (int begin = 0; begin < query.ne[2]; begin += kChunk) {
        const int width = std::min(kChunk, query.ne[2] - begin);
        auto scope = workspace.scope();
        const int split_capacity = ops::detail::gqa_attention_split_capacity(
            query.ne[1], width, DType::BF16, envelope);
        const int split_scale = query.ne[1] == Gqa27Geometry::QHeads
                                    ? Gqa27Geometry::DecodeSplitScale
                                    : Gqa35Geometry::DecodeSplitScale;
        const int splits = std::min(split_capacity, kDecodeSplitsPerScale * split_scale);
        Tensor acc = workspace.alloc(DType::BF16,
                                     {D, query.ne[1], width, splits * query.ne[3]});
        Tensor m = workspace.alloc(DType::FP32,
                                   {query.ne[1], width, splits * query.ne[3]});
        Tensor l = workspace.alloc(DType::FP32,
                                   {query.ne[1], width, splits * query.ne[3]});
        const bool multi = query.ne[3] > 1;
        const bool masked = valid_columns.data != nullptr;
        const auto dispatch = [&]<typename Geometry>() {
            if (multi) {
                if (masked) {
                    launch_partial<Geometry, true, true>(query, positions, valid_columns, table_rows,
                                                         scale, cache, envelope, begin, width,
                                                          splits, acc, m, l, output, stream);
                } else {
                    launch_partial<Geometry, true, false>(query, positions, valid_columns,
                                                          table_rows, scale, cache, envelope, begin,
                                                          width, splits, acc, m, l, output, stream);
                }
            } else if (masked) {
                launch_partial<Geometry, false, true>(query, positions, valid_columns, table_rows,
                                                      scale, cache, envelope, begin, width, splits,
                                                      acc, m, l, output, stream);
            } else {
                launch_partial<Geometry, false, false>(query, positions, valid_columns, table_rows,
                                                       scale, cache, envelope, begin, width, splits,
                                                       acc, m, l, output, stream);
            }
        };
        if (query.ne[1] == Gqa27Geometry::QHeads) {
            dispatch.template operator()<Gqa27Geometry>();
        } else {
            dispatch.template operator()<Gqa35Geometry>();
        }
    }
}

} // namespace ninfer::ops::kvarn
