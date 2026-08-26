#include "ninfer/ops/kvarn_attention.h"

// Native CUDA execution of the fixed Huawei KVarN profile. Record decode and online-softmax follow
// commit 7586257f1c632e63187bfacbbe21ccb51540f7b3 triton_kvarn_decode.py; permanent FP16 sinks and
// provisional tail retirement follow kvarn_attn.py. NInfer supplies only its P=64 page translation
// and bounded lane/table-row ownership.

#include "core/device.h"
#include "ops/kvarn/config.cuh"
#include "ops/kvarn/decode.cuh"
#include "ops/kvarn/store.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kStoreSharedBytes =
    ((kvarn::D + 1) * kvarn::Group + 8 * kvarn::D + 16) * sizeof(float);

struct ViewPointers {
    std::uint8_t* records;
    __nv_bfloat16* tail_k;
    __nv_bfloat16* tail_v;
    std::int32_t* markers;
    const std::int32_t* block_tables;
    int physical_pages;
    int logical_pages;
    int table_rows;
    int heads;
};

int max_touched_pages(int width) { return (width + kvarn::Group - 2) / kvarn::Group + 1; }

void require_view(const KvarnPagedBatchLayerView& view) {
    const int record_slot = kKvarnRecordBytes / kvarn::Group;
    if (view.records.dtype != DType::U8 || view.records.ne[0] != record_slot ||
        view.records.ne[1] != kvarn::Group || view.records.ne[2] != view.num_kv_heads ||
        view.records.ne[3] <= 0 || !view.records.is_contiguous() ||
        view.tail_k.dtype != DType::BF16 || view.tail_v.dtype != DType::BF16 ||
        view.tail_k.ne[0] != kvarn::D || view.tail_k.ne[1] != kvarn::Group ||
        view.tail_k.ne[2] != view.num_kv_heads * kKvarnTailSlots ||
        view.tail_k.ne[3] <= 0 || view.tail_v.ne[0] != view.tail_k.ne[0] ||
        view.tail_v.ne[1] != view.tail_k.ne[1] || view.tail_v.ne[2] != view.tail_k.ne[2] ||
        view.tail_v.ne[3] != view.tail_k.ne[3] || !view.tail_k.is_contiguous() ||
        !view.tail_v.is_contiguous() || view.tail_logical_pages.dtype != DType::I32 ||
        view.tail_logical_pages.ne[0] != kKvarnTailSlots ||
        view.tail_logical_pages.ne[1] != view.tail_k.ne[3] ||
        !view.tail_logical_pages.is_contiguous() || view.block_tables.dtype != DType::I32 ||
        view.block_tables.ne[1] != view.tail_k.ne[3] || !view.block_tables.is_contiguous() ||
        view.records.data == nullptr || view.tail_k.data == nullptr || view.tail_v.data == nullptr ||
        view.tail_logical_pages.data == nullptr || view.block_tables.data == nullptr) {
        throw std::invalid_argument("KVarN attention: invalid cache view");
    }
}

ViewPointers pointers(KvarnPagedBatchLayerView view) {
    return {
        static_cast<std::uint8_t*>(view.records.data),
        static_cast<__nv_bfloat16*>(view.tail_k.data),
        static_cast<__nv_bfloat16*>(view.tail_v.data),
        static_cast<std::int32_t*>(view.tail_logical_pages.data),
        static_cast<const std::int32_t*>(view.block_tables.data),
        view.records.ne[3], view.block_tables.ne[0], view.block_tables.ne[1], view.num_kv_heads,
    };
}

__device__ int tail_slot(int logical_page, int first_page, int last_page) {
    if (logical_page < kKvarnSinkPages) return logical_page;
    if (logical_page == first_page) return 2;
    if (logical_page == last_page) return 3;
    return -1;
}

__device__ int mapped_tail_slot(const ViewPointers& cache, int row, int logical_page,
                                int first_page, int last_page) {
    for (int slot = 0; slot < kKvarnTailSlots; ++slot) {
        if (cache.markers[slot + kKvarnTailSlots * row] == logical_page) return slot;
    }
    const int preferred = tail_slot(logical_page, first_page, last_page);
    if (preferred < 0 || logical_page < kKvarnSinkPages) return preferred;
    if (cache.markers[preferred + kKvarnTailSlots * row] < 0) return preferred;
    const int alternate = preferred == 2 ? 3 : 2;
    return cache.markers[alternate + kKvarnTailSlots * row] < 0 ? alternate : -1;
}

__global__ void stage_kernel(const __nv_bfloat16* key, const __nv_bfloat16* value,
                             const std::int32_t* positions, const std::int32_t* valid_columns,
                             const std::int32_t* table_rows, ViewPointers cache, int width,
                             int batch, bool masked, bool provisional) {
    const int item = static_cast<int>(blockIdx.x);
    const int head = item % cache.heads;
    const int column = (item / cache.heads) % width;
    const int b = item / (cache.heads * width);
    const int count = masked ? valid_columns[b] : width;
    if (b >= batch || column >= count) return;
    const int start = positions[b * width];
    const int end = start + count;
    const int position = positions[b * width + column];
    const int page = position / kvarn::Group;
    const int first_page = start / kvarn::Group;
    const int last_page = (end - 1) / kvarn::Group;
    const int page_begin = page * kvarn::Group;
    const int intersection_begin = max(start, page_begin);
    const int intersection_end = min(end, page_begin + kvarn::Group);
    const bool full_direct = intersection_begin == page_begin &&
                             intersection_end == page_begin + kvarn::Group &&
                             page >= kKvarnSinkPages && !provisional;
    if (full_direct) return;
    const int row = table_rows[b];
    const int slot = mapped_tail_slot(cache, row, page, first_page, last_page);
    if (slot < 0) return;
    const int d = static_cast<int>(threadIdx.x);
    const std::int64_t source =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kvarn::D) *
                                              (head + cache.heads * (column + width * b));
    const int offset = position & (kvarn::Group - 1);
    const std::int64_t destination =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kvarn::D) *
                                              (offset + kvarn::Group *
                                                            (head + cache.heads *
                                                                        (slot + kKvarnTailSlots * row)));
    cache.tail_k[destination] = key[source];
    cache.tail_v[destination] = value[source];
    if (head == 0 && d == 0) cache.markers[slot + kKvarnTailSlots * row] = page;
}

__device__ kvarn::StorePointers record_pointers(std::uint8_t* record) {
    return {
        record + kKvarnKPackedOffset,
        reinterpret_cast<__half*>(record + kKvarnKScaleOffset),
        reinterpret_cast<__half*>(record + kKvarnKZeroOffset),
        reinterpret_cast<__half*>(record + kKvarnKTokenScaleOffset),
        record + kKvarnVPackedOffset,
        reinterpret_cast<__half*>(record + kKvarnVChannelScaleOffset),
        reinterpret_cast<__half*>(record + kKvarnVTokenScaleOffset),
        reinterpret_cast<__half*>(record + kKvarnVTokenZeroOffset),
    };
}

__global__ void encode_kernel(const __nv_bfloat16* key, const __nv_bfloat16* value,
                              const std::int32_t* positions, const std::int32_t* valid_columns,
                              const std::int32_t* table_rows, ViewPointers cache, int width,
                              int batch, int touched_pages, bool masked, bool from_tail_only) {
    extern __shared__ float shared[];
    const int encoded = static_cast<int>(blockIdx.x);
    const bool key_path = (encoded & 1) == 0;
    int task = encoded >> 1;
    const int head = task % cache.heads;
    task /= cache.heads;
    const int page_index = task % touched_pages;
    const int b = task / touched_pages;
    if (b >= batch) return;
    const int count = masked ? valid_columns[b] : width;
    if (count <= 0) return;
    const int start = positions[b * width];
    const int end = start + count;
    const int first_page = start / kvarn::Group;
    const int last_page = (end - 1) / kvarn::Group;
    const int page = first_page + page_index;
    if (page > last_page || page < kKvarnSinkPages) return;
    const int page_begin = page * kvarn::Group;
    const int intersection_begin = max(start, page_begin);
    const int intersection_end = min(end, page_begin + kvarn::Group);
    if (intersection_end != page_begin + kvarn::Group) return;
    const bool full_direct = intersection_begin == page_begin && !from_tail_only;
    const int row = table_rows[b];
    const int slot = mapped_tail_slot(cache, row, page, first_page, last_page);
    if (!full_direct &&
        (slot < 0 || cache.markers[slot + kKvarnTailSlots * row] != page)) return;
    const int physical = cache.block_tables[page + cache.logical_pages * row];
    if (physical < 0 || physical >= cache.physical_pages) return;
    const std::int64_t record_index =
        (static_cast<std::int64_t>(physical) * cache.heads + head) * kKvarnRecordBytes;
    const __nv_bfloat16* source = key_path ? key : value;
    for (int index = static_cast<int>(threadIdx.x); index < kvarn::D * kvarn::Group;
         index += static_cast<int>(blockDim.x)) {
        const int token = index / kvarn::D;
        const int d = index - token * kvarn::D;
        if (full_direct) {
            const int column = page_begin + token - start;
            const std::int64_t input =
                static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kvarn::D) *
                                                     (head + cache.heads * (column + width * b));
            shared[d + (kvarn::D + 1) * token] = __bfloat162float(source[input]);
        } else {
            const std::int64_t tail =
                static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kvarn::D) *
                                                     (token + kvarn::Group *
                                                                  (head + cache.heads *
                                                                              (slot + kKvarnTailSlots * row)));
            shared[d + (kvarn::D + 1) * token] =
                __bfloat162float(key_path ? cache.tail_k[tail] : cache.tail_v[tail]);
        }
    }
    __syncthreads();
    const kvarn::SinkhornWorkspace workspace = kvarn::workspace_after(shared);
    kvarn::StorePointers output = record_pointers(cache.records + record_index);
    if (key_path) {
        kvarn::store_k_tile(shared, 0, output, workspace);
    } else {
        kvarn::store_v_tile(shared, 0, output, workspace);
    }
}

__global__ void retire_kernel(const std::int32_t* positions,
                              const std::int32_t* valid_columns,
                              const std::int32_t* table_rows, ViewPointers cache, int width,
                              int batch, int touched_pages, bool masked) {
    const int task = static_cast<int>(blockIdx.x);
    const int page_index = task % touched_pages;
    const int b = task / touched_pages;
    if (b >= batch) return;
    const int count = masked ? valid_columns[b] : width;
    if (count <= 0) return;
    const int start = positions[b * width];
    const int end = start + count;
    const int first_page = start / kvarn::Group;
    const int last_page = (end - 1) / kvarn::Group;
    const int page = first_page + page_index;
    if (page > last_page || page < kKvarnSinkPages ||
        min(end, (page + 1) * kvarn::Group) != (page + 1) * kvarn::Group) {
        return;
    }
    const int row = table_rows[b];
    const int slot = mapped_tail_slot(cache, row, page, first_page, last_page);
    if (slot < 0) return;
    if (cache.markers[slot + kKvarnTailSlots * row] == page) {
        cache.markers[slot + kKvarnTailSlots * row] = -1;
    }
}

__global__ void prepare_restore_kernel(std::int32_t* markers, int page, int remainder) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (remainder == 0 || page < kKvarnSinkPages) {
        markers[2] = -1;
        markers[3] = -1;
        return;
    }
    if (markers[2] == page) {
        markers[3] = -1;
        return;
    }
    if (markers[3] == page) {
        markers[2] = -1;
        return;
    }
    markers[2] = -(page + 2);
    markers[3] = -1;
}

__global__ void restore_tail_kernel(const std::uint8_t* records, __nv_bfloat16* tail_k,
                                    __nv_bfloat16* tail_v, const std::int32_t* markers,
                                    const std::int32_t* block_table, int physical_pages,
                                    int heads, int page) {
    if (markers[2] != -(page + 2)) return;
    const int head = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    const int physical = block_table[page];
    if (physical < 0 || physical >= physical_pages) return;
    const std::uint8_t* record =
        records + (static_cast<std::int64_t>(physical) * heads + head) * kKvarnRecordBytes;
    const auto* k_scale = reinterpret_cast<const __half*>(record + kKvarnKScaleOffset);
    const auto* k_zero = reinterpret_cast<const __half*>(record + kKvarnKZeroOffset);
    const auto* k_token_scale =
        reinterpret_cast<const __half*>(record + kKvarnKTokenScaleOffset);
    const auto* v_channel =
        reinterpret_cast<const __half*>(record + kKvarnVChannelScaleOffset);
    const auto* v_scale = reinterpret_cast<const __half*>(record + kKvarnVTokenScaleOffset);
    const auto* v_zero = reinterpret_cast<const __half*>(record + kKvarnVTokenZeroOffset);
    for (int token = 0; token < kvarn::Group; ++token) {
        const std::uint8_t k_packed =
            record[kKvarnKPackedOffset + d * (kvarn::Group / 2) + token / 2];
        const int k_code = (k_packed >> (4 * (token & 1))) & 15;
        const std::uint8_t v_packed =
            record[kKvarnVPackedOffset + token * (kvarn::D / 4) + d / 4];
        const int v_code = (v_packed >> (2 * (d & 3))) & 3;
        const std::int64_t destination =
            static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kvarn::D) *
                                                 (token + kvarn::Group * (head + heads * 2));
        const float key =
            fmaf(static_cast<float>(k_code), __half2float(k_scale[d]), __half2float(k_zero[d])) *
            __half2float(k_token_scale[token]);
        const float value = fmaf(static_cast<float>(v_code), __half2float(v_scale[token]),
                                 __half2float(v_zero[token])) *
                            __half2float(v_channel[d]);
        tail_k[destination] = __float2bfloat16_rn(key);
        tail_v[destination] = __float2bfloat16_rn(value);
    }
}

__global__ void finalize_restore_kernel(std::int32_t* markers, int page) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && markers[2] == -(page + 2)) markers[2] = page;
}

void rotate_kv(Tensor key, Tensor value, cudaStream_t stream) {
    kvarn_hadamard(key, key, stream);
    kvarn_hadamard(value, value, stream);
}

void stage_and_encode(Tensor key, Tensor value, const Tensor& positions,
                      const Tensor& valid_columns, const Tensor& table_rows,
                      KvarnPagedBatchLayerView cache, bool provisional, cudaStream_t stream) {
    const int width = key.ne[2];
    const int batch = key.ne[3];
    const bool masked = valid_columns.data != nullptr;
    const ViewPointers view = pointers(cache);
    if (provisional && width > 16) {
        throw std::invalid_argument("KVarN provisional append width must be at most 16");
    }
    static const cudaError_t encode_attribute = cudaFuncSetAttribute(
        encode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kStoreSharedBytes));
    CUDA_CHECK(encode_attribute);
    stage_kernel<<<batch * width * cache.num_kv_heads, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<const std::int32_t*>(positions.data),
        masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(table_rows.data), view, width, batch, masked,
        provisional);
    CUDA_CHECK(cudaGetLastError());
    if (!provisional) {
        const int pages = max_touched_pages(width);
        encode_kernel<<<2 * batch * pages * cache.num_kv_heads, kThreads, kStoreSharedBytes,
                        stream>>>(
            static_cast<const __nv_bfloat16*>(key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const std::int32_t*>(positions.data),
            masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            static_cast<const std::int32_t*>(table_rows.data), view, width, batch, pages, masked,
            false);
        CUDA_CHECK(cudaGetLastError());
        retire_kernel<<<batch * pages, 1, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            static_cast<const std::int32_t*>(table_rows.data), view, width, batch, pages, masked);
        CUDA_CHECK(cudaGetLastError());
    }
}

void validate_inputs(const Tensor& query, const Tensor* key, const Tensor* value,
                     const Tensor& positions, const Tensor& valid_columns,
                     const Tensor& table_rows, const KvarnPagedBatchLayerView& cache,
                     const Tensor& output) {
    require_view(cache);
    if (query.dtype != DType::BF16 || query.ne[0] != kvarn::D || query.ne[2] <= 0 ||
        query.ne[3] <= 0 || query.ne[1] % cache.num_kv_heads != 0 || !query.is_contiguous() ||
        output.dtype != DType::BF16 || output.numel() != query.numel() ||
        !output.is_contiguous() || positions.dtype != DType::I32 ||
        positions.ne[0] != query.ne[2] || positions.ne[1] != query.ne[3] ||
        !positions.is_contiguous() || table_rows.dtype != DType::I32 ||
        table_rows.ne[0] != query.ne[3] || !table_rows.is_contiguous() ||
        (valid_columns.data != nullptr &&
         (valid_columns.dtype != DType::I32 || valid_columns.ne[0] != query.ne[3] ||
          !valid_columns.is_contiguous()))) {
        throw std::invalid_argument("KVarN attention: invalid input tensors");
    }
    if (key != nullptr &&
        (key->dtype != DType::BF16 || value->dtype != DType::BF16 || key->ne[0] != kvarn::D ||
         key->ne[1] != cache.num_kv_heads || key->ne[2] != query.ne[2] ||
         key->ne[3] != query.ne[3] || value->numel() != key->numel() ||
         !key->is_contiguous() || !value->is_contiguous())) {
        throw std::invalid_argument("KVarN attention: invalid K/V tensors");
    }
}

} // namespace

std::size_t kvarn_attention_workspace_capacity_bytes(std::int32_t query_heads,
                                                     GqaExecutionEnvelope envelope,
                                                     std::int32_t batch_size,
                                                     std::int32_t min_width,
                                                     std::int32_t max_width) {
    const std::int32_t decode_width = std::min(max_width, 6);
    const std::size_t decode = gqa_attention_workspace_capacity_bytes(
        query_heads, DType::BF16, envelope, batch_size, std::min(min_width, decode_width),
        decode_width);
    if (batch_size != 1 || max_width < kKvarnGroup) { return decode; }
    if (query_heads != 24 && query_heads != 16) {
        throw std::invalid_argument("KVarN workspace: unsupported query-head geometry");
    }
    const std::int32_t kv_heads = query_heads == 24 ? 4 : 2;
    const std::size_t slab_tokens =
        std::min<std::size_t>(envelope.max_visible_keys, kvarn::PrefillSlabTokens);
    const std::size_t materialized = 2 * static_cast<std::size_t>(kKvarnHeadDim) *
                                     slab_tokens * kv_heads * dtype_size(DType::BF16);
    const std::size_t rows = static_cast<std::size_t>(query_heads) * max_width;
    const std::size_t running = static_cast<std::size_t>(kKvarnHeadDim) * rows *
                                    dtype_size(DType::FP32) +
                                2 * rows * dtype_size(DType::FP32);
    return std::max(decode, materialized + running + 5 * 256);
}

void kvarn_attention(Tensor query, Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                     KvarnPagedBatchLayerView cache, bool provisional,
                     GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& output,
                     cudaStream_t stream) {
    (void)workspace;
    validate_inputs(query, &key, &value, positions, valid_columns, kv_table_rows, cache, output);
    if (envelope.max_visible_keys == 0) {
        throw std::invalid_argument("KVarN attention: empty execution envelope");
    }
    rotate_kv(key, value, stream);
    stage_and_encode(key, value, positions, valid_columns, kv_table_rows, cache, provisional,
                     stream);
    const int width = query.ne[2];
    const int batch = query.ne[3];
    (void)width;
    (void)batch;
    kvarn::decode_attention(query, positions, valid_columns, kv_table_rows, scale, cache, envelope,
                            workspace, output, stream);
}

void kvarn_attention_cached(Tensor query, const Tensor& positions, const Tensor& kv_table_rows,
                            float scale, const KvarnPagedBatchLayerView& cache,
                            GqaExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& output, cudaStream_t stream) {
    validate_inputs(query, nullptr, nullptr, positions, Tensor{}, kv_table_rows, cache, output);
    (void)workspace;
    if (envelope.max_visible_keys == 0) {
        throw std::invalid_argument("KVarN attention: empty execution envelope");
    }
    kvarn::decode_attention(query, positions, Tensor{}, kv_table_rows, scale, cache, envelope,
                            workspace, output, stream);
}

void kvarn_kv_append(Tensor key, Tensor value, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& kv_table_rows,
                     KvarnPagedBatchLayerView cache, bool provisional, cudaStream_t stream) {
    require_view(cache);
    kvarn_hadamard(key, key, stream);
    kvarn_hadamard(value, value, stream);
    stage_and_encode(key, value, positions, valid_columns, kv_table_rows, cache, provisional,
                     stream);
}

void kvarn_commit_pages(const Tensor& positions, const Tensor& accepted_columns,
                        const Tensor& kv_table_rows, KvarnPagedBatchLayerView cache,
                        cudaStream_t stream) {
    require_view(cache);
    if (positions.dtype != DType::I32 || accepted_columns.dtype != DType::I32 ||
        kv_table_rows.dtype != DType::I32 || positions.ne[1] != accepted_columns.ne[0] ||
        positions.ne[1] != kv_table_rows.ne[0]) {
        throw std::invalid_argument("KVarN commit: invalid metadata");
    }
    const int width = positions.ne[0];
    const int batch = positions.ne[1];
    const int pages = max_touched_pages(width);
    static const cudaError_t encode_attribute = cudaFuncSetAttribute(
        encode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kStoreSharedBytes));
    CUDA_CHECK(encode_attribute);
    encode_kernel<<<2 * batch * pages * cache.num_kv_heads, kThreads, kStoreSharedBytes, stream>>>(
        nullptr, nullptr, static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(accepted_columns.data),
        static_cast<const std::int32_t*>(kv_table_rows.data), pointers(cache), width, batch, pages,
        true, true);
    CUDA_CHECK(cudaGetLastError());
    retire_kernel<<<batch * pages, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(accepted_columns.data),
        static_cast<const std::int32_t*>(kv_table_rows.data), pointers(cache), width, batch, pages,
        true);
    CUDA_CHECK(cudaGetLastError());
}

void kvarn_restore_tail(std::int32_t frontier, KvarnPagedLayerView cache, cudaStream_t stream) {
    const int record_slot = kKvarnRecordBytes / kvarn::Group;
    if (frontier < 0 || cache.records.dtype != DType::U8 ||
        cache.records.ne[0] != record_slot || cache.records.ne[1] != kvarn::Group ||
        cache.records.ne[2] != cache.num_kv_heads || cache.tail_k.dtype != DType::BF16 ||
        cache.tail_v.dtype != DType::BF16 || cache.tail_k.ne[0] != kvarn::D ||
        cache.tail_k.ne[1] != kvarn::Group ||
        cache.tail_k.ne[2] != cache.num_kv_heads * kKvarnTailSlots ||
        cache.tail_v.numel() != cache.tail_k.numel() ||
        cache.tail_logical_pages.dtype != DType::I32 ||
        cache.tail_logical_pages.ne[0] != kKvarnTailSlots ||
        cache.block_table.dtype != DType::I32 || !cache.records.is_contiguous() ||
        !cache.tail_k.is_contiguous() || !cache.tail_v.is_contiguous() ||
        !cache.tail_logical_pages.is_contiguous() || !cache.block_table.is_contiguous()) {
        throw std::invalid_argument("KVarN restore: invalid cache view");
    }
    const int page = frontier / kvarn::Group;
    const int remainder = frontier % kvarn::Group;
    prepare_restore_kernel<<<1, 1, 0, stream>>>(
        static_cast<std::int32_t*>(cache.tail_logical_pages.data), page, remainder);
    CUDA_CHECK(cudaGetLastError());
    if (remainder != 0 && page >= kKvarnSinkPages) {
        restore_tail_kernel<<<cache.num_kv_heads, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(cache.records.data),
            static_cast<__nv_bfloat16*>(cache.tail_k.data),
            static_cast<__nv_bfloat16*>(cache.tail_v.data),
            static_cast<const std::int32_t*>(cache.tail_logical_pages.data),
            static_cast<const std::int32_t*>(cache.block_table.data), cache.records.ne[3],
            cache.num_kv_heads, page);
        CUDA_CHECK(cudaGetLastError());
        finalize_restore_kernel<<<1, 1, 0, stream>>>(
            static_cast<std::int32_t*>(cache.tail_logical_pages.data), page);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops
