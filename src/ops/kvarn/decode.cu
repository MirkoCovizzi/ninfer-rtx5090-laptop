#include "ops/kvarn/decode.cuh"

#include "core/device.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kvarn/config.cuh"
#include "ops/kvarn/decode_kernel.cuh"
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::kvarn {
namespace {

struct KvarnPrefillInput {
    const std::uint8_t* records;
    const __nv_bfloat16* tail_k;
    const __nv_bfloat16* tail_v;
    const std::int32_t* markers;
    const std::int32_t* table_rows;
    int heads;

    template <bool Key>
    __device__ __forceinline__ void stage(__nv_bfloat16* destination, int head, int k0,
                                          int max_query_abs, int physical_page, int tid) const {
        constexpr int kVecPerRow = D / 8;
        const int table_row = table_rows[0];
        for (int chunk = tid; chunk < Group * kVecPerRow; chunk += kGqaPrefillThreads) {
            const int token = chunk / kVecPerRow;
            const int d = (chunk % kVecPerRow) * 8;
            __nv_bfloat16* output =
                destination + token * D + gqa_prefill_swz(token, d);
            const int position = k0 + token;
            if (position > max_query_abs) {
                store_vec(output, make_int4(0, 0, 0, 0));
                continue;
            }
            int tail_slot = -1;
#pragma unroll
            for (int slot = 0; slot < kKvarnTailSlots; ++slot) {
                if (markers[slot + kKvarnTailSlots * table_row] == position / Group) {
                    tail_slot = slot;
                }
            }
            if (tail_slot >= 0) {
                const auto* tail = Key ? tail_k : tail_v;
                const std::int64_t source =
                    static_cast<std::int64_t>(d) + static_cast<std::int64_t>(D) *
                                                       ((position & (Group - 1)) +
                                                        Group * (head + heads *
                                                                            (tail_slot +
                                                                             kKvarnTailSlots *
                                                                                 table_row)));
                store_vec(output, load_vec<int4>(tail + source));
                continue;
            }
            const std::uint8_t* record =
                records + (static_cast<std::int64_t>(physical_page) * heads + head) *
                              kKvarnRecordBytes;
            if constexpr (Key) {
                const auto* scale = reinterpret_cast<const __half*>(record + kKvarnKScaleOffset);
                const auto* zero = reinterpret_cast<const __half*>(record + kKvarnKZeroOffset);
                const auto* token_scale =
                    reinterpret_cast<const __half*>(record + kKvarnKTokenScaleOffset);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    const int dim = d + item;
                    const std::uint8_t packed =
                        record[kKvarnKPackedOffset + dim * (Group / 2) + token / 2];
                    const int code = (packed >> (4 * (token & 1))) & 15;
                    output[item] = __float2bfloat16_rn(
                        fmaf(static_cast<float>(code), __half2float(scale[dim]),
                             __half2float(zero[dim])) *
                        __half2float(token_scale[token]));
                }
            } else {
                const auto* channel =
                    reinterpret_cast<const __half*>(record + kKvarnVChannelScaleOffset);
                const auto* scale =
                    reinterpret_cast<const __half*>(record + kKvarnVTokenScaleOffset);
                const auto* zero = reinterpret_cast<const __half*>(record + kKvarnVTokenZeroOffset);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    const int dim = d + item;
                    const std::uint8_t packed =
                        record[kKvarnVPackedOffset + token * (D / 4) + dim / 4];
                    const int code = (packed >> (2 * (dim & 3))) & 3;
                    output[item] = __float2bfloat16_rn(
                        fmaf(static_cast<float>(code), __half2float(scale[token]),
                             __half2float(zero[token])) *
                        __half2float(channel[dim]));
                }
            }
        }
    }
};

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
    const KvarnPrefillInput input{
        static_cast<const std::uint8_t*>(cache.records.data),
        static_cast<const __nv_bfloat16*>(cache.tail_k.data),
        static_cast<const __nv_bfloat16*>(cache.tail_v.data),
        static_cast<const std::int32_t*>(cache.tail_logical_pages.data),
        static_cast<const std::int32_t*>(table_rows.data),
        cache.num_kv_heads,
    };
    static const cudaError_t attribute = cudaFuncSetAttribute(
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata, KvarnPrefillInput>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillSmemBytes);
    CUDA_CHECK(attribute);
    const int width = query.ne[2];
    const dim3 grid(div_up(width, kGqaPrefillBr), Geometry::QHeads, 1);
    gqa_attention_prefill_bf16_kernel<Geometry, Metadata, KvarnPrefillInput>
        <<<grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(
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

    constexpr int kDChunk = 64;
    const dim3 reduce_grid(Geometry::QHeads, D / kDChunk, width * query.ne[3]);
    gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, false, MultiBatch, Masked, true>
        <<<reduce_grid, 256, 0, stream>>>(
            partial_acc.data, static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data),
            static_cast<const std::int32_t*>(positions.data),
            Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr, width,
            query.ne[2], column_begin, query.ne[3], splits,
            static_cast<__nv_bfloat16*>(output.data));
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
    constexpr int kChunk = 6;
    for (int begin = 0; begin < query.ne[2]; begin += kChunk) {
        const int width = std::min(kChunk, query.ne[2] - begin);
        auto scope = workspace.scope();
        const int splits = ops::detail::gqa_attention_split_capacity(
            query.ne[1], width, DType::BF16, envelope);
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
