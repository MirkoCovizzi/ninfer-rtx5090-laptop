#pragma once

#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/mtp_adaptive_cost.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

inline constexpr std::uint32_t kAdaptiveMtpMaximumDrafts = 8;
inline constexpr std::uint32_t kAdaptiveMtpMaximumBatch  = 8;

// Online accepted-prefix estimator. Each draft position is a Bernoulli signal for the event that
// the accepted prefix reaches that position. The fast/slow low-pass pair follows regime changes
// without making the selected physical width chatter on individual mismatches.
class MtpAdaptiveSignal final {
public:
    void reset() noexcept {
        fast_         = kPriorSurvival;
        slow_         = kPriorSurvival;
        observations_ = {};
        stale_rounds_ = {};
        success_streak_ = {};
    }

    void observe(std::uint32_t proposed, std::uint32_t accepted) noexcept {
        proposed = std::min(proposed, kAdaptiveMtpMaximumDrafts);
        accepted = std::min(accepted, proposed);
        for (std::uint32_t position = 0; position < kAdaptiveMtpMaximumDrafts; ++position) {
            if (position < proposed) {
                const float sample = position < accepted ? 1.0F : 0.0F;
                fast_[position] += kFastAlpha * (sample - fast_[position]);
                slow_[position] += kSlowAlpha * (sample - slow_[position]);
                if (observations_[position] != UINT16_MAX) { ++observations_[position]; }
                if (sample == 1.0F) {
                    if (success_streak_[position] != UINT8_MAX) { ++success_streak_[position]; }
                } else {
                    success_streak_[position] = 0;
                }
                stale_rounds_[position] = 0;
            } else if (stale_rounds_[position] != UINT8_MAX) {
                ++stale_rounds_[position];
            }
        }
    }

    [[nodiscard]] float expected_tokens(std::uint32_t window,
                                        std::uint32_t available) const noexcept {
        const std::uint32_t extent = std::min({window, available, kAdaptiveMtpMaximumDrafts});
        float expected             = 1.0F;
        float previous             = 1.0F;
        for (std::uint32_t position = 0; position < extent; ++position) {
            const float evidence = std::min(1.0F, static_cast<float>(observations_[position]) / 4.0F);
            float estimate = kPriorSurvival[position] +
                             evidence * (0.75F * fast_[position] + 0.25F * slow_[position] -
                                         kPriorSurvival[position]);
            if ((observations_[position] == 0 || stale_rounds_[position] >= 4) && position != 0) {
                // A wider position cannot be observed until the controller probes it. Continue
                // the measured prefix with a conservative conditional-survival prior so a strong
                // signal can earn one wider probe; direct observations replace this estimate.
                const float continuation =
                    success_streak_[position - 1U] >= 2 ? 0.85F : 0.60F;
                estimate = std::max(estimate, previous * continuation);
            }
            // Prefix-survival probability cannot increase with position. Enforcing that physical
            // constraint suppresses sparse high-position noise after a local regime change.
            estimate = std::clamp(estimate, 0.0F, previous);
            expected += estimate;
            previous = estimate;
        }
        return expected;
    }

    [[nodiscard]] bool confident_tail(std::uint32_t window) const noexcept {
        return window != 0 && window <= kAdaptiveMtpMaximumDrafts &&
               success_streak_[window - 1U] >= kWideningSuccessStreak;
    }

    [[nodiscard]] bool strong_wide_prefix() const noexcept {
        constexpr std::array<float, 3> thresholds{0.75F, 0.60F, 0.45F};
        for (std::size_t position = 0; position < thresholds.size(); ++position) {
            if (observations_[position] < 4 ||
                0.75F * fast_[position] + 0.25F * slow_[position] < thresholds[position]) {
                return false;
            }
        }
        return success_streak_[thresholds.size() - 1U] >= kWideningSuccessStreak;
    }

private:
    static constexpr float kFastAlpha = 0.375F;
    static constexpr float kSlowAlpha = 0.125F;
    static constexpr std::uint8_t kWideningSuccessStreak = 6;
    static constexpr std::array<float, kAdaptiveMtpMaximumDrafts> kPriorSurvival{
        0.54F, 0.16F, 0.08F, 0.04F, 0.02F, 0.01F, 0.005F, 0.0025F};

    std::array<float, kAdaptiveMtpMaximumDrafts> fast_ = kPriorSurvival;
    std::array<float, kAdaptiveMtpMaximumDrafts> slow_ = kPriorSurvival;
    std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts> observations_{};
    std::array<std::uint8_t, kAdaptiveMtpMaximumDrafts> stale_rounds_{};
    std::array<std::uint8_t, kAdaptiveMtpMaximumDrafts> success_streak_{};
};

class MtpAdaptiveBatchController final {
public:
    explicit MtpAdaptiveBatchController(qwen3_6::MtpAdaptiveCostProfile cost_profile) noexcept
        : cost_profile_(cost_profile) {}

    void reset(std::uint32_t maximum_window) noexcept {
        maximum_window_ = std::clamp(maximum_window, 1U, kAdaptiveMtpMaximumDrafts);
        selected_window_ = default_startup_window();
        candidate_window_ = selected_window_;
        rounds_at_window_ = 0;
        candidate_rounds_ = 0;
        execution_samples_ = {};
        last_width_execution_round_ = {};
        execution_round_ = 0;
        selection_cohort_key_ = 0;
        last_transition_from_ = selected_window_;
        last_transition_to_   = selected_window_;
    }

    void observe_execution(std::uint32_t batch_size, std::uint32_t window) noexcept {
        if (batch_size == 0 || batch_size > kAdaptiveMtpMaximumBatch || window == 0 ||
            window > maximum_window_) {
            return;
        }
        const std::size_t batch = batch_size - 1U;
        const std::size_t width = window - 1U;
        if (execution_round_ != UINT64_MAX) { ++execution_round_; }
        if (execution_samples_[batch][width] != UINT16_MAX) {
            ++execution_samples_[batch][width];
        }
        last_width_execution_round_[batch][width] = execution_round_;
    }

    [[nodiscard]] std::uint32_t
    select(std::span<const MtpAdaptiveSignal* const> signals,
           std::span<const std::uint32_t> available,
           std::span<const std::uint32_t> room, std::span<const std::uint32_t> frontiers,
           std::uint64_t cohort_key = 1) noexcept {
        last_transition_from_ = selected_window_;
        last_transition_to_   = selected_window_;
        if (signals.empty() || signals.size() != available.size() || signals.size() != room.size() ||
            signals.size() != frontiers.size()) {
            return selected_window_;
        }
        const std::uint32_t batch_frontier = maximum_frontier(frontiers);
        if (selection_cohort_key_ != cohort_key) {
            selected_window_    = startup_window(signals, available, room, frontiers);
            candidate_window_   = selected_window_;
            rounds_at_window_   = 0;
            candidate_rounds_   = 0;
            execution_samples_  = {};
            last_width_execution_round_ = {};
            execution_round_    = 0;
            selection_cohort_key_ = cohort_key;
        }
        const std::uint32_t round_window = selected_window_;

        std::uint32_t best_window = selected_window_;
        float best_score          = score(selected_window_, signals, available, room, frontiers);
        // K3's qualified W4A4 route is cheaper than K2 and has no lower token yield, so K2 is not
        // a valid steady-state bridge when recovering from K1.
        const std::uint32_t upward_limit =
            selected_window_ == 1 && maximum_window_ >= 3 ? 3U : selected_window_ + 1U;
        for (std::uint32_t window = 1; window <= maximum_window_; ++window) {
            if (window > upward_limit) { continue; }
            const float candidate = score(window, signals, available, room, frontiers);
            if (candidate > best_score) {
                best_score  = candidate;
                best_window = window;
            }
        }

        if (maximum_window_ >= 6 && best_window < 6) {
            bool all_prefixes_strong = true;
            for (const MtpAdaptiveSignal* signal : signals) {
                all_prefixes_strong = all_prefixes_strong && signal->strong_wide_prefix();
            }
            const std::uint32_t floor_window = widest_admissible_width(
                6, best_window, signals.size(), batch_frontier);
            if (all_prefixes_strong && floor_window != 0) {
                best_window = floor_window;
                best_score  = score(best_window, signals, available, room, frontiers);
            }
        }

        const float current_score = score(selected_window_, signals, available, room, frontiers);
        bool confident_probe      = false;
        if (selected_window_ < maximum_window_) {
            bool has_continuing_row = false;
            bool all_tails_confident = true;
            for (std::size_t row = 0; row < signals.size(); ++row) {
                if (available[row] < selected_window_ || room[row] <= selected_window_ + 1U) {
                    continue;
                }
                has_continuing_row = true;
                all_tails_confident =
                    all_tails_confident && signals[row]->confident_tail(selected_window_);
            }
            const std::uint32_t preferred_probe =
                std::min({selected_window_ + 2U, maximum_window_, 7U});
            const std::uint32_t probe_window = widest_admissible_width(
                preferred_probe, selected_window_, signals.size(), batch_frontier);
            if (probe_window != 0) {
                const std::size_t batch =
                    std::min(signals.size(), execution_samples_.size()) - 1U;
                const std::uint64_t observed_round =
                    last_width_execution_round_[batch][probe_window - 1U];
                const bool probe_due =
                    observed_round == 0 || execution_round_ - observed_round >= 32;
                if (selected_window_ < 7 && has_continuing_row && all_tails_confident &&
                    probe_due) {
                    best_window = probe_window;
                    best_score = std::max(score(best_window, signals, available, room, frontiers),
                                          current_score * 1.011F);
                    confident_probe = true;
                }
            }
        }
        const float margin = 1.01F;
        const bool material_gain = best_score > current_score * margin;
        if (best_window != selected_window_ && material_gain) {
            if (candidate_window_ == best_window) {
                if (candidate_rounds_ != UINT8_MAX) { ++candidate_rounds_; }
            } else {
                candidate_window_ = best_window;
                candidate_rounds_ = 1;
            }
            const bool high_width_contraction =
                best_window < selected_window_ && selected_window_ >= 6;
            const std::size_t batch = std::min(signals.size(), execution_samples_.size()) - 1U;
            const bool k8_probation = selected_window_ == 8 &&
                                      execution_samples_[batch][selected_window_ - 1U] <= 2;
            const std::uint8_t minimum_residency =
                confident_probe ? 2 : k8_probation ? 2 : high_width_contraction ? 12 : 3;
            const std::uint8_t required_candidate_rounds =
                confident_probe ? 2 : k8_probation ? 2 : 4;
            if (candidate_rounds_ >= required_candidate_rounds &&
                rounds_at_window_ >= minimum_residency) {
                selected_window_  = best_window;
                candidate_window_ = best_window;
                rounds_at_window_ = 0;
                candidate_rounds_ = 0;
            }
        } else {
            candidate_window_ = selected_window_;
            candidate_rounds_ = 0;
        }
        if (rounds_at_window_ != UINT8_MAX) {
            ++rounds_at_window_;
        }
        last_transition_from_ = round_window;
        last_transition_to_   = selected_window_;
        return selected_window_;
    }

    [[nodiscard]] std::uint32_t selected_window() const noexcept { return selected_window_; }
    [[nodiscard]] bool transitioned() const noexcept {
        return last_transition_from_ != last_transition_to_;
    }
    [[nodiscard]] std::uint32_t transition_from() const noexcept {
        return last_transition_from_;
    }
    [[nodiscard]] std::uint32_t transition_to() const noexcept { return last_transition_to_; }

private:
    [[nodiscard]] std::uint32_t default_startup_window() const noexcept {
        return std::min(maximum_window_ >= 6 ? 7U : 3U, maximum_window_);
    }

    [[nodiscard]] std::uint32_t
    startup_window(std::span<const MtpAdaptiveSignal* const> signals,
                   std::span<const std::uint32_t> available,
                   std::span<const std::uint32_t> room,
                   std::span<const std::uint32_t> frontiers) const noexcept {
        const std::uint32_t startup = default_startup_window();
        if (!physically_dominated(startup, signals.size(), maximum_frontier(frontiers))) {
            return startup;
        }
        std::uint32_t best_window = 1;
        float best_score = score(best_window, signals, available, room, frontiers);
        for (std::uint32_t window = 2; window < startup; ++window) {
            const float candidate = score(window, signals, available, room, frontiers);
            if (candidate > best_score) {
                best_score  = candidate;
                best_window = window;
            }
        }
        return best_window;
    }

    [[nodiscard]] float
    score(std::uint32_t window, std::span<const MtpAdaptiveSignal* const> signals,
           std::span<const std::uint32_t> available,
           std::span<const std::uint32_t> room,
           std::span<const std::uint32_t> frontiers) const noexcept {
        float expected        = 0.0F;
        float future_expected = 0.0F;
        float cost = round_cost(window, signals.size(), maximum_frontier(frontiers));
        std::size_t future_batch = 0;
        std::uint32_t future_frontier = 0;
        for (std::size_t row = 0; row < signals.size(); ++row) {
            expected += signals[row]->expected_tokens(window, available[row]);
            const std::uint32_t consumed = std::min(window, available[row]) + 1U;
            const std::uint32_t future_room = room[row] > consumed ? room[row] - consumed : 0U;
            const std::uint32_t future_extent = std::min(window, future_room);
            if (future_room != 0) {
                future_expected += signals[row]->expected_tokens(window, future_extent);
                ++future_batch;
                future_frontier = std::max(future_frontier, frontiers[row] + consumed);
            }
        }
        if (future_batch != 0) {
            expected += future_expected;
            cost += round_cost(window, future_batch, future_frontier);
        }
        return expected / cost;
    }

    [[nodiscard]] bool physically_dominated(std::uint32_t window, std::size_t batch_size,
                                            std::uint32_t frontier) const noexcept {
        const float candidate_cost = round_cost(window, batch_size, frontier);
        for (std::uint32_t narrower = 1; narrower < window; ++narrower) {
            const float narrower_cost = round_cost(narrower, batch_size, frontier);
            const float maximum_yield_ratio =
                static_cast<float>(window + 1U) / static_cast<float>(narrower + 1U);
            if (candidate_cost >= narrower_cost * maximum_yield_ratio) { return true; }
        }
        return false;
    }

    [[nodiscard]] std::uint32_t widest_admissible_width(std::uint32_t preferred,
                                                        std::uint32_t lower_exclusive,
                                                        std::size_t batch_size,
                                                        std::uint32_t frontier) const noexcept {
        for (std::uint32_t window = preferred; window > lower_exclusive; --window) {
            if (!physically_dominated(window, batch_size, frontier)) { return window; }
        }
        return 0;
    }

    [[nodiscard]] float round_cost(std::uint32_t window, std::size_t batch_size,
                                   std::uint32_t frontier) const noexcept {
        const std::size_t batch =
            std::clamp<std::size_t>(batch_size, 1, kAdaptiveMtpMaximumBatch) - 1U;
        const std::size_t width = std::clamp(window, 1U, kAdaptiveMtpMaximumDrafts) - 1U;
        const std::span<const qwen3_6::MtpAdaptiveCostPoint> curve =
            cost_profile_.batch_curves[batch];
        if (curve.empty()) { return 1.0F; }
        if (curve.size() == 1 || frontier <= curve.front().frontier) {
            return std::max(curve.front().round_costs[width], 0.001F);
        }
        if (frontier >= curve.back().frontier) {
            return std::max(curve.back().round_costs[width], 0.001F);
        }

        std::size_t upper = 1;
        while (upper < curve.size() && frontier > curve[upper].frontier) { ++upper; }
        const qwen3_6::MtpAdaptiveCostPoint& lo = curve[upper - 1U];
        const qwen3_6::MtpAdaptiveCostPoint& hi = curve[upper];
        const float extent = static_cast<float>(hi.frontier - lo.frontier);
        const float offset = static_cast<float>(frontier - lo.frontier);
        const float cost = lo.round_costs[width] +
                           offset / extent * (hi.round_costs[width] - lo.round_costs[width]);
        return std::max(cost, 0.001F);
    }

    [[nodiscard]] static std::uint32_t
    maximum_frontier(std::span<const std::uint32_t> frontiers) noexcept {
        std::uint32_t frontier = 0;
        for (const std::uint32_t row_frontier : frontiers) {
            frontier = std::max(frontier, row_frontier);
        }
        return frontier;
    }

    qwen3_6::MtpAdaptiveCostProfile cost_profile_;
    std::uint32_t maximum_window_ = 1;
    std::uint32_t selected_window_ = 1;
    std::uint32_t candidate_window_ = 1;
    std::uint8_t rounds_at_window_ = 0;
    std::uint8_t candidate_rounds_ = 0;
    std::array<std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        execution_samples_{};
    std::array<std::array<std::uint64_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        last_width_execution_round_{};
    std::uint64_t execution_round_ = 0;
    std::uint64_t selection_cohort_key_ = 0;
    std::uint32_t last_transition_from_ = 1;
    std::uint32_t last_transition_to_ = 1;
};

} // namespace ninfer::targets::qwen3_6::detail
