#pragma once

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
    void reset(std::uint32_t maximum_window) noexcept {
        maximum_window_ = std::clamp(maximum_window, 1U, kAdaptiveMtpMaximumDrafts);
        selected_window_ = startup_window();
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
           std::span<const std::uint32_t> room, std::uint64_t cohort_key = 1) noexcept {
        last_transition_from_ = selected_window_;
        last_transition_to_   = selected_window_;
        if (signals.empty() || signals.size() != available.size() || signals.size() != room.size()) {
            return selected_window_;
        }
        if (selection_cohort_key_ != cohort_key) {
            selected_window_    = startup_window();
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
        float best_score          = score(selected_window_, signals, available, room);
        // K3's qualified W4A4 route is cheaper than K2 and has no lower token yield, so K2 is not
        // a valid steady-state bridge when recovering from K1.
        const std::uint32_t upward_limit =
            selected_window_ == 1 && maximum_window_ >= 3 ? 3U : selected_window_ + 1U;
        for (std::uint32_t window = 1; window <= maximum_window_; ++window) {
            if (window > upward_limit) { continue; }
            const float candidate = score(window, signals, available, room);
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
            if (all_prefixes_strong) {
                best_window = 6;
                best_score  = score(best_window, signals, available, room);
            }
        }

        const float current_score = score(selected_window_, signals, available, room);
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
            const std::uint32_t probe_window =
                std::min({selected_window_ + 2U, maximum_window_, 7U});
            const std::size_t batch = std::min(signals.size(), execution_samples_.size()) - 1U;
            const std::uint64_t observed_round =
                last_width_execution_round_[batch][probe_window - 1U];
            const bool probe_due = observed_round == 0 || execution_round_ - observed_round >= 32;
            if (selected_window_ < 7 && has_continuing_row && all_tails_confident &&
                probe_due) {
                best_window = probe_window;
                best_score = std::max(score(best_window, signals, available, room),
                                      current_score * 1.011F);
                confident_probe = true;
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
    [[nodiscard]] std::uint32_t startup_window() const noexcept {
        return std::min(maximum_window_ >= 6 ? 7U : 3U, maximum_window_);
    }

    [[nodiscard]] float
    score(std::uint32_t window, std::span<const MtpAdaptiveSignal* const> signals,
          std::span<const std::uint32_t> available,
          std::span<const std::uint32_t> room) const noexcept {
        // Normalized fixed-width round costs measured on the supported Blackwell execution route.
        // K=3 is slightly cheaper than K=2 because it crosses the qualified W4A4 schedule boundary.
        float expected        = 0.0F;
        float future_expected = 0.0F;
        float cost            = round_cost(window, signals.size());
        std::size_t future_batch = 0;
        for (std::size_t row = 0; row < signals.size(); ++row) {
            expected += signals[row]->expected_tokens(window, available[row]);
            const std::uint32_t consumed = std::min(window, available[row]) + 1U;
            const std::uint32_t future_room = room[row] > consumed ? room[row] - consumed : 0U;
            const std::uint32_t future_extent = std::min(window, future_room);
            if (future_extent != 0) {
                future_expected += signals[row]->expected_tokens(window, future_extent);
                ++future_batch;
            }
        }
        if (future_batch != 0) {
            expected += future_expected;
            cost += round_cost(window, future_batch);
        }
        return expected / cost;
    }

    [[nodiscard]] float round_cost(std::uint32_t window, std::size_t) const noexcept {
        const std::size_t width = window - 1U;
        return kPriorRoundCost[width];
    }

    static constexpr std::array<float, kAdaptiveMtpMaximumDrafts> kPriorRoundCost{
        1.000F, 1.075F, 1.064F, 1.125F, 1.177F, 1.227F, 1.281F, 1.334F};

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
