#include <ninfer/targets/qwen3_6/decoder_state.h>

#include "core/device.h"

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                               std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                               std::int32_t quant_group, KvCacheStorage storage,
                               std::int32_t table_rows,
                               std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool quantized = storage == KvCacheStorage::Int8Group64;
    const bool kvarn     = storage == KvCacheStorage::KvarnK4V2Group64;
    if ((!quantized && !kvarn && (storage != KvCacheStorage::BFloat16 || dtype != DType::BF16 ||
                                  quant_group != 0)) ||
        (quantized && (dtype != DType::I8 || quant_group != kKvQuantGroup ||
                       head_dim % quant_group != 0)) ||
        (kvarn && (dtype != DType::U8 || quant_group != ops::kKvarnGroup ||
                   head_dim != ops::kKvarnHeadDim))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) *
                             (kvarn ? 1ULL : quantized ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (kvarn) {
            pool_spec.planes.push_back(
                {DType::U8, ops::kKvarnRecordBytes / kPagedKVPageSize, kv_heads, 256});
        } else {
            pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
            pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
        }
        if (!kvarn && quantized) {
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
        }
    }
    PagedKVCacheLayout layout{
        .pool        = plan_paged_kv_pool(builder, pool_spec),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
        .storage     = storage,
    };
    if (kvarn) {
        const std::int32_t row_heads = table_rows * kv_heads * ops::kKvarnTailSlots;
        layout.kvarn_tail_k = builder.add_tensor(
            DType::BF16, {head_dim, kPagedKVPageSize, row_heads, static_cast<std::int32_t>(layers)},
            256, "KVarN rotated K sink/tail");
        layout.kvarn_tail_v = builder.add_tensor(
            DType::BF16, {head_dim, kPagedKVPageSize, row_heads, static_cast<std::int32_t>(layers)},
            256, "KVarN rotated V sink/tail");
        layout.kvarn_tail_logical_pages = builder.add_tensor(
            DType::I32,
            {ops::kKvarnTailSlots, table_rows, static_cast<std::int32_t>(layers)}, 256,
            "KVarN sink/tail logical pages");
    }
    return layout;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
      quant_group_(layout.quant_group), storage_(layout.storage) {
    if (storage_ == KvCacheStorage::KvarnK4V2Group64) {
        kvarn_tail_k_ = layout.kvarn_tail_k.bind(backing);
        kvarn_tail_v_ = layout.kvarn_tail_v.bind(backing);
        kvarn_tail_logical_pages_ = layout.kvarn_tail_logical_pages.bind(backing);
        CUDA_CHECK(cudaMemset(kvarn_tail_logical_pages_.data, 0xff,
                              kvarn_tail_logical_pages_.bytes()));
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table,
                                   std::int32_t table_row) noexcept
    : cache_(&cache), block_table_(block_table), table_row_(table_row) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

ops::KvarnPagedLayerView PagedKVCacheView::kvarn_layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->kvarn_layer_view(layer, block_table_, table_row_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table(), allocation.bound_row());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    if (storage_ == KvCacheStorage::KvarnK4V2Group64) {
        throw std::logic_error("KVarN cache requires kvarn_layer_view");
    }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    if (storage_ == KvCacheStorage::KvarnK4V2Group64) {
        throw std::logic_error("KVarN cache requires kvarn_batch_layer_view");
    }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

ops::KvarnPagedLayerView PagedKVCache::kvarn_layer_view(std::uint32_t layer, Tensor block_table,
                                                        std::int32_t table_row) const {
    if (storage_ != KvCacheStorage::KvarnK4V2Group64 || layer >= layers_ || table_row < 0 ||
        table_row >= kvarn_tail_logical_pages_.ne[1]) {
        throw std::invalid_argument("invalid KVarN layer view");
    }
    const std::int32_t row_heads = kv_heads_ * ops::kKvarnTailSlots;
    const std::int32_t begin = table_row * row_heads;
    return {
        .records = pool_.plane(layer),
        .tail_k = kvarn_tail_k_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .slice(2, begin, row_heads)
                      .view({head_dim_, kPagedKVPageSize, row_heads}),
        .tail_v = kvarn_tail_v_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .slice(2, begin, row_heads)
                      .view({head_dim_, kPagedKVPageSize, row_heads}),
        .tail_logical_pages = kvarn_tail_logical_pages_
                                  .slice(2, static_cast<std::int32_t>(layer), 1)
                                  .slice(1, table_row, 1)
                                  .view({ops::kKvarnTailSlots}),
        .block_table = block_table,
        .num_kv_heads = kv_heads_,
    };
}

ops::KvarnPagedBatchLayerView PagedKVCache::kvarn_batch_layer_view(std::uint32_t layer) const {
    if (storage_ != KvCacheStorage::KvarnK4V2Group64 || layer >= layers_) {
        throw std::invalid_argument("invalid KVarN batch layer view");
    }
    const std::int32_t rows = kvarn_tail_logical_pages_.ne[1];
    const std::int32_t row_heads = kv_heads_ * ops::kKvarnTailSlots;
    return {
        .records = pool_.plane(layer),
        .tail_k = kvarn_tail_k_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .view({head_dim_, kPagedKVPageSize, row_heads, rows}),
        .tail_v = kvarn_tail_v_.slice(3, static_cast<std::int32_t>(layer), 1)
                      .view({head_dim_, kPagedKVPageSize, row_heads, rows}),
        .tail_logical_pages = kvarn_tail_logical_pages_
                                  .slice(2, static_cast<std::int32_t>(layer), 1)
                                  .view({ops::kKvarnTailSlots, rows}),
        .block_tables = pool_.block_tables(),
        .num_kv_heads = kv_heads_,
    };
}

void PagedKVCache::reset_kvarn_tail_row(std::int32_t table_row, cudaStream_t stream) const {
    if (storage_ != KvCacheStorage::KvarnK4V2Group64 || table_row < 0 ||
        table_row >= kvarn_tail_logical_pages_.ne[1]) {
        return;
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        Tensor markers = kvarn_tail_logical_pages_
                             .slice(2, static_cast<std::int32_t>(layer), 1)
                             .slice(1, table_row, 1);
        CUDA_CHECK(cudaMemsetAsync(markers.data, 0xff, markers.bytes(), stream));
    }
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
