#include "ops/launcher/kvarn.h"

#include "core/device.h"
#include "ops/kernel/kvarn.cuh"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int D, int G, int Iterations>
void launch_compress(const Tensor& k, const Tensor& v, KvarnTileStorage storage,
                     cudaStream_t stream) {
    constexpr int kThreads = 256;
    constexpr std::size_t kSharedFloats =
        static_cast<std::size_t>(D) * G + 7ULL * D + kThreads;
    constexpr std::size_t kSharedBytes = kSharedFloats * sizeof(float);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        kvarn_compress_kernel<D, G, Iterations>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    CUDA_CHECK(attribute);
    const int tiles = k.ne[2];
    kvarn_compress_kernel<D, G, Iterations><<<tiles * 2, kThreads, kSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<std::uint8_t*>(storage.k_codes.data),
        static_cast<std::uint8_t*>(storage.k_block_scales.data),
        static_cast<__half*>(storage.k_channel_scales.data),
        static_cast<std::uint8_t*>(storage.v_codes.data),
        static_cast<__half*>(storage.v_channel_scales.data),
        static_cast<__half*>(storage.v_token_scales.data),
        static_cast<__half*>(storage.v_token_zeros.data), tiles);
    CUDA_CHECK(cudaGetLastError());
}

template <int D, int G>
void launch_decompress(const KvarnTileStorage& storage, Tensor& k, Tensor& v,
                       cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks        = (D * G + kThreads - 1) / kThreads;
    const int tiles         = k.ne[2];
    kvarn_decompress_kernel<D, G><<<dim3(blocks, tiles), kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(storage.k_codes.data),
        static_cast<const std::uint8_t*>(storage.k_block_scales.data),
        static_cast<const __half*>(storage.k_channel_scales.data),
        static_cast<const std::uint8_t*>(storage.v_codes.data),
        static_cast<const __half*>(storage.v_channel_scales.data),
        static_cast<const __half*>(storage.v_token_scales.data),
        static_cast<const __half*>(storage.v_token_zeros.data), static_cast<float*>(k.data),
        static_cast<float*>(v.data), tiles);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked>
void launch_paged_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                          const Tensor& valid_columns, const Tensor& table_rows,
                          PagedKVBatchLayerView cache, cudaStream_t stream) {
    constexpr int D          = 256;
    constexpr int G          = 64;
    constexpr int Iterations = kKvarnIterations;
    constexpr int kThreads   = 256;
    constexpr std::size_t kSharedFloats =
        static_cast<std::size_t>(D) * G + 7ULL * D + kThreads;
    constexpr std::size_t kSharedBytes = kSharedFloats * sizeof(float);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        kvarn_paged_compress_direct_kernel<D, G, Iterations, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
    CUDA_CHECK(attribute);
    const auto* valid =
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr;
    const auto* rows = table_rows.data == nullptr
                           ? nullptr
                           : static_cast<const std::int32_t*>(table_rows.data);
    const dim3 edge_grid(cache.num_kv_heads, k.ne[3], 2);
    kvarn_paged_stage_edges_kernel<D, G, Masked><<<edge_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), valid, rows,
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0], k.ne[2],
        cache.num_kv_heads, cache.tail_k.ne[3], static_cast<std::uint8_t*>(cache.k_pages.data),
        static_cast<std::uint8_t*>(cache.k_scale_pages.data),
        static_cast<__half*>(cache.k_channel_scale_pages.data),
        static_cast<std::uint8_t*>(cache.v_pages.data),
        static_cast<__half*>(cache.v_channel_scale_pages.data),
        static_cast<__half*>(cache.v_scale_pages.data),
        static_cast<__half*>(cache.v_zero_pages.data),
        static_cast<__nv_bfloat16*>(cache.tail_k.data),
        static_cast<__nv_bfloat16*>(cache.tail_v.data),
        static_cast<std::int32_t*>(cache.tail_logical_pages.data));
    CUDA_CHECK(cudaGetLastError());

    const int page_candidates = (k.ne[2] + G - 1) / G + 1;
    const dim3 direct_grid(page_candidates * cache.num_kv_heads * 2, k.ne[3]);
    kvarn_paged_compress_direct_kernel<D, G, Iterations, Masked>
        <<<direct_grid, kThreads, kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), valid, rows,
            static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
            k.ne[2], cache.num_kv_heads, cache.tail_k.ne[3], page_candidates,
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.k_channel_scale_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<__half*>(cache.v_channel_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data),
            static_cast<__half*>(cache.v_zero_pages.data));
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked, bool Retire>
void launch_flush_full_tails(std::span<const PagedKVBatchLayerView> layer_views,
                             const Tensor& positions, const Tensor& valid_columns,
                             const Tensor& table_rows, cudaStream_t stream) {
    constexpr int D          = 256;
    constexpr int G          = 64;
    constexpr int Iterations = kKvarnIterations;
    constexpr int kThreads   = 256;
    constexpr std::size_t kSharedFloats =
        static_cast<std::size_t>(D) * G + 7ULL * D + kThreads;
    constexpr std::size_t kSharedBytes = kSharedFloats * sizeof(float);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        kvarn_flush_full_tails_kernel<D, G, Iterations, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
    CUDA_CHECK(attribute);

    KvarnPagedLayerPointers layers{};
    for (std::size_t layer = 0; layer < layer_views.size(); ++layer) {
        const PagedKVBatchLayerView& view = layer_views[layer];
        layers.k_codes[layer] = static_cast<std::uint8_t*>(view.k_pages.data);
        layers.k_block_scales[layer] = static_cast<std::uint8_t*>(view.k_scale_pages.data);
        layers.k_channel_scales[layer] = static_cast<__half*>(view.k_channel_scale_pages.data);
        layers.v_codes[layer] = static_cast<std::uint8_t*>(view.v_pages.data);
        layers.v_channel_scales[layer] = static_cast<__half*>(view.v_channel_scale_pages.data);
        layers.v_token_scales[layer] = static_cast<__half*>(view.v_scale_pages.data);
        layers.v_token_zeros[layer] = static_cast<__half*>(view.v_zero_pages.data);
        layers.tail_k[layer] = static_cast<__nv_bfloat16*>(view.tail_k.data);
        layers.tail_v[layer] = static_cast<__nv_bfloat16*>(view.tail_v.data);
        layers.tail_logical_pages[layer] =
            static_cast<std::int32_t*>(view.tail_logical_pages.data);
    }
    const PagedKVBatchLayerView& first = layer_views.front();
    const auto* valid =
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr;
    const auto* rows = table_rows.data == nullptr
                           ? nullptr
                           : static_cast<const std::int32_t*>(table_rows.data);
    const int batch = positions.ne[1];
    const int layer_count = static_cast<int>(layer_views.size());
    const dim3 grid(first.num_kv_heads, batch, layer_count * kKvarnTailSlots * 2);
    kvarn_flush_full_tails_kernel<D, G, Iterations, Masked>
        <<<grid, kThreads, kSharedBytes, stream>>>(
            layers, static_cast<const std::int32_t*>(positions.data), valid, rows,
            static_cast<const std::int32_t*>(first.block_tables.data), first.block_tables.ne[0],
            positions.ne[0], first.num_kv_heads, first.tail_k.ne[3], layer_count);
    CUDA_CHECK(cudaGetLastError());

    if constexpr (Retire) {
        const dim3 clear_grid(1, batch, layer_count * kKvarnTailSlots);
        kvarn_clear_flushed_tails_kernel<G, Masked><<<clear_grid, 32, 0, stream>>>(
            layers, static_cast<const std::int32_t*>(positions.data), valid, rows, positions.ne[0],
            first.num_kv_heads, first.tail_k.ne[3], layer_count);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void kvarn_compress_launch(const Tensor& k, const Tensor& v, KvarnTileStorage storage,
                           std::int32_t iterations, cudaStream_t stream) {
    if (k.ne[0] == 256 && k.ne[1] == 64) {
        if (iterations == 8) {
            launch_compress<256, 64, 8>(k, v, storage, stream);
        } else {
            launch_compress<256, 64, 16>(k, v, storage, stream);
        }
    } else if (k.ne[0] == 128 && k.ne[1] == 128) {
        if (iterations == 8) {
            launch_compress<128, 128, 8>(k, v, storage, stream);
        } else {
            launch_compress<128, 128, 16>(k, v, storage, stream);
        }
    } else {
        throw std::invalid_argument("kvarn_compress: unsupported geometry");
    }
}

void kvarn_decompress_launch(const KvarnTileStorage& storage, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    if (k.ne[0] == 256 && k.ne[1] == 64) {
        launch_decompress<256, 64>(storage, k, v, stream);
    } else if (k.ne[0] == 128 && k.ne[1] == 128) {
        launch_decompress<128, 128>(storage, k, v, stream);
    } else {
        throw std::invalid_argument("kvarn_decompress: unsupported geometry");
    }
}

void kvarn_hadamard_launch(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    const int vectors = static_cast<int>(source.numel() / source.ne[0]);
    if (source.ne[0] == 256) {
        kvarn_hadamard_kernel<256><<<vectors, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data),
            static_cast<__nv_bfloat16*>(destination.data), vectors);
    } else if (source.ne[0] == 128) {
        kvarn_hadamard_kernel<128><<<vectors, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data),
            static_cast<__nv_bfloat16*>(destination.data), vectors);
    } else {
        throw std::invalid_argument("kvarn_hadamard: unsupported head dimension");
    }
    CUDA_CHECK(cudaGetLastError());
}

void kvarn_paged_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                               const Tensor& valid_columns, const Tensor& table_rows,
                               PagedKVBatchLayerView cache, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_paged_append<false>(k, v, positions, valid_columns, table_rows, cache, stream);
    } else {
        launch_paged_append<true>(k, v, positions, valid_columns, table_rows, cache, stream);
    }
}

void kvarn_flush_full_tails_launch(std::span<const PagedKVBatchLayerView> layers,
                                   const Tensor& positions, const Tensor& valid_columns,
                                   const Tensor& table_rows, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_flush_full_tails<false, true>(layers, positions, valid_columns, table_rows, stream);
    } else {
        launch_flush_full_tails<true, true>(layers, positions, valid_columns, table_rows, stream);
    }
}

void kvarn_encode_full_tails_launch(std::span<const PagedKVBatchLayerView> layers,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_flush_full_tails<false, false>(layers, positions, valid_columns, table_rows, stream);
    } else {
        launch_flush_full_tails<true, false>(layers, positions, valid_columns, table_rows, stream);
    }
}

} // namespace ninfer::ops::detail
