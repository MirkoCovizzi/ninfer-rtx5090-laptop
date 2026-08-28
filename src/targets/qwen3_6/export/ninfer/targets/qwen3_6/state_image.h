#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/transfer_work.h"
#include "ninfer/ops/kvarn.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct DFlashLocalStateSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
};

struct KvarnContinuationStateSpec {
    std::uint32_t text_layers = 0;
    std::uint32_t mtp_layers  = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
};

struct KvarnContinuationImageLayout {
    KvarnContinuationStateSpec spec;
    std::size_t text_k_offset      = 0;
    std::size_t text_v_offset      = 0;
    std::size_t text_marker_offset = 0;
    std::size_t mtp_k_offset       = 0;
    std::size_t mtp_v_offset       = 0;
    std::size_t mtp_marker_offset  = 0;
    std::size_t tail_layer_bytes   = 0;
    std::size_t marker_layer_bytes = 0;
    std::size_t image_bytes        = 0;
};

struct StateImageSpec {
    LinearAttentionStatePoolSpec linear;
    std::int32_t hidden = 0;
    std::optional<DFlashLocalStateSpec> dflash_local;
    std::optional<KvarnContinuationStateSpec> kvarn;
};

struct StateImageHostLayout {
    StateImageSpec spec;
    LayoutRegion linear_conv;
    std::size_t linear_conv_layer_bytes = 0;
    LayoutRegion linear_recurrent;
    std::size_t linear_recurrent_layer_bytes = 0;
    LayoutRegion continuation_hidden;
    std::optional<LayoutRegion> dflash_local_k;
    std::optional<LayoutRegion> dflash_local_v;
    std::size_t dflash_local_layer_bytes = 0;
    std::optional<LayoutRegion> kvarn;
    std::optional<KvarnContinuationImageLayout> kvarn_layout;
    std::size_t image_bytes = 0;
};

struct KvarnContinuationStateLayout {
    TensorRegion images;
    KvarnContinuationImageLayout image;
};

struct StateImageDeviceLayout {
    LinearAttentionStatePoolLayout linear;
    TensorRegion continuation_hidden;
    std::optional<CyclicKVCacheLayout> dflash_local;
    std::optional<KvarnContinuationStateLayout> kvarn;
    StateImageHostLayout host;
};

[[nodiscard]] TransferWork state_image_transfer_work(const StateImageHostLayout& layout);
[[nodiscard]] TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout);

[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);

struct HostStateImageView {
    std::byte* data                    = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateImageConstView {
    const std::byte* data              = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateSlotHandle {
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;
};

/** Fixed-capacity pinned storage for complete physical StateImage payloads; owns no cache policy.
 */
class HostStatePool {
public:
    HostStatePool(StateImageHostLayout layout, std::uint32_t capacity);

    HostStatePool(const HostStatePool&)            = delete;
    HostStatePool& operator=(const HostStatePool&) = delete;
    HostStatePool(HostStatePool&&)                 = delete;
    HostStatePool& operator=(HostStatePool&&)      = delete;

    [[nodiscard]] std::optional<HostStateSlotHandle> allocate() noexcept;
    [[nodiscard]] bool release(HostStateSlotHandle handle) noexcept;

    [[nodiscard]] HostStateImageView writable_view(HostStateSlotHandle handle);
    [[nodiscard]] HostStateImageConstView view(HostStateSlotHandle handle) const;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }

    [[nodiscard]] const StateImageHostLayout& layout() const noexcept { return layout_; }

private:
    struct Slot {
        std::uint32_t generation = 1;
        bool occupied            = false;
    };

    [[nodiscard]] bool valid(HostStateSlotHandle handle) const noexcept;
    [[nodiscard]] std::byte* slot_data(std::uint32_t index) const noexcept;

    StateImageHostLayout layout_;
    std::optional<PinnedHostBuffer> backing_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t free_count_ = 0;
    std::uint32_t occupied_   = 0;
};

struct StateImageDeviceSlotView {
    LinearAttentionStateSlotView linear;
    Tensor continuation_hidden;
    std::optional<CyclicKVCacheSlotView> dflash_local;
};

/**
 * Caller-backed fixed storage for Qwen3.6 continuation state.
 *
 * Every absolute slot contains common GDN/hidden state and the selected backend's unpaged
 * continuation state, including DFlash local K/V or KVarN sink/tail K/V. The pool owns neither
 * slot roles nor logical checkpoint identity.
 */
class StateImageDevicePool {
public:
    StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout);

    StateImageDevicePool(const StateImageDevicePool&)            = delete;
    StateImageDevicePool& operator=(const StateImageDevicePool&) = delete;
    StateImageDevicePool(StateImageDevicePool&&)                 = delete;
    StateImageDevicePool& operator=(StateImageDevicePool&&)      = delete;

    [[nodiscard]] std::int32_t slot_count() const noexcept { return linear_.slot_count(); }

    [[nodiscard]] StateImageDeviceSlotView slot_view(std::int32_t slot) const;
    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;

    [[nodiscard]] LinearAttentionStatePool& linear() noexcept { return linear_; }

    [[nodiscard]] const LinearAttentionStatePool& linear() const noexcept { return linear_; }

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return continuation_hidden_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept {
        return continuation_hidden_;
    }

    [[nodiscard]] CyclicKVCache* dflash_local() noexcept;
    [[nodiscard]] const CyclicKVCache* dflash_local() const noexcept;

    [[nodiscard]] bool has_kvarn() const noexcept { return kvarn_images_.data != nullptr; }
    [[nodiscard]] std::uint32_t kvarn_text_layers() const noexcept {
        return kvarn_text_layers_;
    }
    [[nodiscard]] std::uint32_t kvarn_mtp_layers() const noexcept { return kvarn_mtp_layers_; }
    [[nodiscard]] ops::KvarnTailStateView kvarn_text_tail(std::uint32_t layer,
                                                           std::int32_t slot) const;
    [[nodiscard]] ops::KvarnTailStateView kvarn_mtp_tail(std::uint32_t layer,
                                                          std::int32_t slot) const;

    [[nodiscard]] const StateImageHostLayout& host_layout() const noexcept { return host_layout_; }

    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    void zero_all(cudaStream_t stream = nullptr);
    void copy_slot(std::int32_t source, std::int32_t destination, cudaStream_t stream = nullptr);
    void copy_dflash_local(std::int32_t source, std::int32_t destination,
                           cudaStream_t stream = nullptr);
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        cudaStream_t stream = nullptr);

private:
    void validate_host_layout(const StateImageHostLayout* layout, const std::byte* data) const;

    LinearAttentionStatePool linear_;
    Tensor continuation_hidden_;
    std::optional<CyclicKVCache> dflash_local_;
    Tensor kvarn_images_;
    std::optional<KvarnContinuationImageLayout> kvarn_layout_;
    std::uint32_t kvarn_text_layers_ = 0;
    std::uint32_t kvarn_mtp_layers_  = 0;
    StateImageHostLayout host_layout_;
};

} // namespace ninfer::targets::qwen3_6
