#include "targets/qwen3_6/impl/runtime/mtp_adaptive.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>

namespace {

using ninfer::targets::qwen3_6::MtpAdaptiveCostPoint;
using ninfer::targets::qwen3_6::MtpAdaptiveCostProfile;
using ninfer::targets::qwen3_6::detail::MtpAdaptiveBatchController;
using ninfer::targets::qwen3_6::detail::MtpAdaptiveSignal;

MtpAdaptiveCostProfile profile(std::span<const MtpAdaptiveCostPoint> curve) {
    MtpAdaptiveCostProfile result;
    result.batch_curves.fill(curve);
    return result;
}

std::uint32_t select(MtpAdaptiveBatchController& controller, const MtpAdaptiveSignal& signal,
                     std::uint32_t available = 15, std::uint32_t room = 64,
                     std::uint64_t cohort = 1, std::uint32_t frontier = 0) {
    const std::array<const MtpAdaptiveSignal*, 1> signals{&signal};
    const std::array<std::uint32_t, 1> available_rows{available};
    const std::array<std::uint32_t, 1> rooms{room};
    const std::array<std::uint32_t, 1> frontiers{frontier};
    return controller.select(signals, available_rows, rooms, frontiers, cohort);
}

int require(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    static constexpr std::array<MtpAdaptiveCostPoint, 1> baseline{{
        {0U,
         {1.000F, 1.075F, 1.064F, 1.125F, 1.177F, 1.227F, 1.281F, 1.334F, 1.388F, 1.442F, 1.496F,
          1.550F, 1.604F, 1.658F, 1.712F}},
    }};
    int failures = 0;
    MtpAdaptiveSignal signal;
    MtpAdaptiveBatchController controller(profile(baseline));
    controller.reset(15);
    failures += require(select(controller, signal) == 3, "adaptive MTP did not start at K3");

    for (int round = 0; round < 64; ++round) {
        const std::uint32_t width = controller.selected_window();
        signal.observe(width, width);
        (void)select(controller, signal, width);
    }
    failures +=
        require(controller.selected_window() == 15, "high-survival evidence did not widen to K15");

    for (int round = 0; round < 64; ++round) {
        signal.observe(controller.selected_window(), 0);
        (void)select(controller, signal, controller.selected_window());
    }
    failures += require(controller.selected_window() == 3,
                        "populated proposal chain contracted below the K3 execution floor");

    controller.reset(15);
    MtpAdaptiveSignal fresh_signal;
    failures += require(select(controller, fresh_signal, 1, 64, 2) == 3,
                        "new compact cohort inherited the previous width");
    failures += require(!controller.transitioned(), "cohort reset was counted as a transition");

    MtpAdaptiveBatchController tail_controller(profile(baseline));
    tail_controller.reset(15);
    failures += require(select(tail_controller, signal, 0, 1, 3) == 1,
                        "one-token tail did not select the minimum physical width");

    static constexpr std::array<MtpAdaptiveCostPoint, 2> depth{{
        {256U,
         {1.000F, 1.088F, 1.067F, 1.128F, 1.184F, 1.241F, 1.287F, 1.348F, 1.409F, 1.470F, 1.531F,
          1.592F, 1.653F, 1.714F, 1.775F}},
        {65792U,
         {1.000F, 1.094F, 1.079F, 1.149F, 1.228F, 3.071F, 3.124F, 3.185F, 3.246F, 3.307F, 3.368F,
          3.429F, 3.490F, 3.551F, 3.612F}},
    }};
    MtpAdaptiveBatchController depth_controller(profile(depth));
    MtpAdaptiveSignal depth_signal;
    depth_controller.reset(15);
    failures += require(select(depth_controller, depth_signal, 8, 64, 3, 256) == 3,
                        "shallow context did not retain the K3 startup width");
    depth_controller.reset(15);
    failures += require(select(depth_controller, depth_signal, 8, 64, 4, 65792) == 3,
                        "deep context selected a dominated wide window");

    if (failures != 0) { return 1; }
    std::cout << "mtp_adaptive: PASS\n";
    return 0;
}
