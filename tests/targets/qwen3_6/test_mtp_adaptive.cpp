#include "targets/qwen3_6/impl/runtime/mtp_adaptive.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>

namespace {

using ninfer::targets::qwen3_6::detail::MtpAdaptiveBatchController;
using ninfer::targets::qwen3_6::detail::MtpAdaptiveSignal;
using ninfer::targets::qwen3_6::MtpAdaptiveCostPoint;
using ninfer::targets::qwen3_6::MtpAdaptiveCostProfile;

MtpAdaptiveCostProfile cost_profile(std::span<const MtpAdaptiveCostPoint> curve) {
    MtpAdaptiveCostProfile profile;
    profile.batch_curves.fill(curve);
    return profile;
}

MtpAdaptiveCostProfile default_cost_profile() {
    static constexpr std::array<MtpAdaptiveCostPoint, 1> points{{
        {0U, {1.000F, 1.075F, 1.064F, 1.125F, 1.177F, 1.227F, 1.281F, 1.334F}},
    }};
    return cost_profile(points);
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "mtp_adaptive: " << message << '\n';
    return 1;
}

std::uint32_t select(MtpAdaptiveBatchController& controller, const MtpAdaptiveSignal& signal,
                     std::uint32_t available = 8, std::uint32_t room = 64,
                     std::uint64_t cohort = 1, std::uint32_t frontier = 0) {
    const std::array<const MtpAdaptiveSignal*, 1> signals{&signal};
    const std::array<std::uint32_t, 1> extents{available};
    const std::array<std::uint32_t, 1> rooms{room};
    const std::array<std::uint32_t, 1> frontiers{frontier};
    return controller.select(signals, extents, rooms, frontiers, cohort);
}

} // namespace

int main() {
    int failures = 0;
    MtpAdaptiveSignal signal;
    MtpAdaptiveBatchController controller(default_cost_profile());
    controller.reset(8);
    failures += check(select(controller, signal) == 7, "Kmax=8 prior did not select K=7");

    std::uint32_t high_transitions = 0;
    for (int round = 0; round < 64; ++round) {
        const std::uint32_t window = controller.selected_window();
        signal.observe(window, window);
        (void)select(controller, signal, window);
        if (controller.transitioned()) {
            ++high_transitions;
            failures += check(controller.transition_from() == 7 &&
                                  controller.transition_to() == 8,
                              "high-survival transition endpoints are incorrect");
        }
    }
    failures += check(controller.selected_window() == 8,
                      "high-survival signal did not increase to K=8");
    failures += check(high_transitions == 1,
                      "high-survival transition count is incorrect");

    for (int round = 0; round < 64; ++round) {
        signal.observe(controller.selected_window(), 0);
        (void)select(controller, signal, controller.selected_window());
    }
    failures += check(controller.selected_window() == 1,
                      "collapsed-survival signal did not decrease to K=1");
    for (int round = 0; round < 64; ++round) {
        signal.observe(controller.selected_window(), controller.selected_window());
        (void)select(controller, signal, controller.selected_window());
    }
    failures += check(controller.selected_window() == 8,
                      "recovered signal did not increase from K=1");

    signal.reset();
    controller.reset(3);
    for (int round = 0; round < 12; ++round) {
        signal.observe(controller.selected_window(), controller.selected_window());
        (void)select(controller, signal, controller.selected_window());
    }
    failures += check(controller.selected_window() == 3,
                      "selection exceeded the configured maximum");

    signal.reset();
    controller.reset(8);
    for (int round = 0; round < 20; ++round) {
        const std::uint32_t window = select(controller, signal, 1, 1);
        signal.observe(window, std::min(window, 1U));
    }
    failures += check(controller.selected_window() == 1,
                      "limited row opportunity did not reduce the physical width");

    MtpAdaptiveSignal wide_prefix_signal;
    for (int round = 0; round < 24; ++round) { wide_prefix_signal.observe(7, 3); }
    controller.reset(8);
    for (int round = 0; round < 24; ++round) {
        (void)select(controller, wide_prefix_signal, controller.selected_window());
    }
    failures += check(controller.selected_window() >= 6,
                      "strong three-position survival contracted below K=6");

    MtpAdaptiveSignal high_signal;
    MtpAdaptiveSignal low_signal;
    for (int round = 0; round < 24; ++round) {
        high_signal.observe(8, 8);
        low_signal.observe(8, 0);
    }
    controller.reset(8);
    for (int round = 0; round < 20; ++round) {
        (void)select(controller, round % 2 == 0 ? high_signal : low_signal,
                     controller.selected_window());
    }
    failures += check(controller.selected_window() == 7,
                      "alternating one-round candidates defeated hysteresis");

    controller.reset(8);
    for (int round = 0; round < 64; ++round) {
        (void)select(controller, low_signal, controller.selected_window(), 64, 11);
    }
    failures += check(controller.selected_window() == 1,
                      "cohort reset test did not reach the low-width state");
    failures += check(select(controller, low_signal, 1, 64, 12) == 7,
                      "new request cohort inherited the prior request width");
    failures += check(!controller.transitioned(), "cohort reset was reported as a transition");

    MtpAdaptiveSignal score_recovery_signal;
    for (int round = 0; round < 12; ++round) { score_recovery_signal.observe(3, 3); }
    score_recovery_signal.observe(3, 0);
    controller.reset(8);
    for (int round = 0; round < 64; ++round) {
        (void)select(controller, low_signal, controller.selected_window());
    }
    failures += check(controller.selected_window() == 1,
                      "dominated-width test did not reach K=1");
    bool direct_k3_recovery = false;
    for (int round = 0; round < 16; ++round) {
        (void)select(controller, score_recovery_signal, controller.selected_window());
        failures += check(controller.selected_window() != 2,
                          "adaptive selection used dominated K=2 as a bridge");
        direct_k3_recovery = direct_k3_recovery ||
                             (controller.transitioned() && controller.transition_from() == 1 &&
                              controller.transition_to() == 3);
    }
    failures += check(direct_k3_recovery, "adaptive selection did not recover directly from K=1");

    controller.reset(8);
    for (int round = 0; round < 24; ++round) {
        (void)select(controller, high_signal, controller.selected_window());
    }
    failures += check(controller.selected_window() == 8,
                      "probation test did not reach its K=8 baseline");
    controller.observe_execution(1, 8);
    controller.observe_execution(1, 8);
    for (int round = 0; round < 4; ++round) {
        (void)select(controller, low_signal, controller.selected_window());
    }
    failures += check(controller.selected_window() < 8,
                      "K=8 probation did not permit a prompt retreat");

    MtpAdaptiveSignal bursty_signal;
    controller.reset(8);
    std::uint32_t bursty_transitions = 0;
    constexpr std::array<std::uint32_t, 8> bursty_accepts{3, 3, 3, 0, 1, 2, 1, 0};
    for (std::uint32_t round = 0; round < 512; ++round) {
        const std::uint32_t window = controller.selected_window();
        bursty_signal.observe(
            window, std::min(window, bursty_accepts[round % bursty_accepts.size()]));
        (void)select(controller, bursty_signal, window);
        bursty_transitions += controller.transitioned() ? 1U : 0U;
    }
    failures += check(bursty_transitions <= 2,
                      "stationary bursty survival caused repeated widening probes");

    static constexpr std::array<MtpAdaptiveCostPoint, 2> depth_points{{
        {256U, {1.000F, 1.088F, 1.067F, 1.128F, 1.184F, 1.241F, 1.287F, 1.348F}},
        {65792U, {1.000F, 1.094F, 1.079F, 1.149F, 1.228F, 3.071F, 3.124F, 3.185F}},
    }};
    MtpAdaptiveBatchController depth_controller(cost_profile(depth_points));
    MtpAdaptiveSignal depth_signal;
    depth_controller.reset(8);
    failures += check(select(depth_controller, depth_signal, 8, 64, 1, 256) == 7,
                      "shallow startup did not retain the wide exploration window");
    depth_controller.reset(8);
    failures += check(select(depth_controller, depth_signal, 8, 64, 2, 65792) == 3,
                      "deep startup selected a physically dominated wide window");
    depth_controller.reset(8);
    failures += check(select(depth_controller, depth_signal, 8, 64, 3, 200000) == 3,
                      "frontier beyond the calibrated range did not retain the endpoint policy");
    depth_controller.reset(8);
    const std::array<const MtpAdaptiveSignal*, 2> mixed_signals{&depth_signal, &depth_signal};
    const std::array<std::uint32_t, 2> mixed_extents{8, 8};
    const std::array<std::uint32_t, 2> mixed_rooms{64, 64};
    const std::array<std::uint32_t, 2> mixed_frontiers{256, 65792};
    failures += check(depth_controller.select(mixed_signals, mixed_extents, mixed_rooms,
                                              mixed_frontiers, 4) == 3,
                      "mixed-depth batch did not price the maximum frontier");
    for (int round = 0; round < 32; ++round) {
        (void)select(depth_controller, high_signal, depth_controller.selected_window(), 64, 5,
                     65792);
    }
    failures += check(depth_controller.selected_window() <= 5,
                      "deep strong-prefix probing selected a physically dominated width");

    MtpAdaptiveSignal k4_tail_signal;
    for (int round = 0; round < 12; ++round) { k4_tail_signal.observe(4, 4); }
    depth_controller.reset(8);
    failures += check(select(depth_controller, k4_tail_signal, 4, 5, 6, 65792) == 4,
                      "deep K5 fallback test did not establish K4");
    bool direct_k5_probe = false;
    for (int round = 0; round < 8; ++round) {
        (void)select(depth_controller, k4_tail_signal, 4, 64, 6, 65792);
        failures += check(depth_controller.selected_window() != 6,
                          "deep K4 tail probe selected dominated K6");
        direct_k5_probe = direct_k5_probe ||
                          (depth_controller.transitioned() &&
                           depth_controller.transition_from() == 4 &&
                           depth_controller.transition_to() == 5);
    }
    failures += check(direct_k5_probe,
                      "deep K4 tail evidence did not probe admissible K5");

    if (failures != 0) { return 1; }
    std::cout << "mtp_adaptive: PASS\n";
    return 0;
}
