#include "core/layout.h"
#include <ninfer/targets/qwen3_6/state_image.h>

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::StateImageDeviceLayout plan(bool dflash, bool kvarn = false) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = 2,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local =
            q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    if (kvarn) {
        spec.kvarn = q36::KvarnContinuationStateSpec{
            .text_layers = 2, .mtp_layers = 1, .kv_heads = 2, .head_dim = 256};
    }
    ninfer::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

} // namespace

int main() {
    const q36::StateImageDeviceLayout common = plan(false);
    const auto common_work                   = q36::state_image_transfer_work(common.host);
    expect(common_work.payload_bytes == common.host.linear_conv.bytes +
                                            common.host.linear_recurrent.bytes +
                                            common.host.continuation_hidden.bytes,
           "common StateImage work payload includes layout padding");
    expect(common_work.copy_operations == 2 * common.host.spec.linear.layers + 1,
           "common StateImage work does not match physical CUDA copies");

    const q36::StateImageDeviceLayout dflash = plan(true);
    const auto full_work                     = q36::state_image_transfer_work(dflash.host);
    const auto local_work                    = q36::dflash_local_transfer_work(dflash.host);
    const std::uint64_t local_bytes =
        2ULL * dflash.host.dflash_local_layer_bytes * dflash.host.spec.dflash_local->layers;
    expect(local_work.payload_bytes == local_bytes &&
               local_work.copy_operations == 2 * dflash.host.spec.dflash_local->layers,
           "DFlash-local work does not match physical K/V layer copies");
    expect(full_work.payload_bytes == common_work.payload_bytes + local_work.payload_bytes &&
               full_work.copy_operations ==
                   common_work.copy_operations + local_work.copy_operations,
           "full DFlash StateImage work is not common plus local state");

    const q36::StateImageDeviceLayout kvarn = plan(false, true);
    const auto kvarn_work                   = q36::state_image_transfer_work(kvarn.host);
    expect(kvarn_work.payload_bytes ==
                   common_work.payload_bytes + kvarn.host.kvarn_layout->image_bytes &&
               kvarn_work.copy_operations == common_work.copy_operations + 1,
           "KVarN StateImage work includes exact sink/tail state");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
