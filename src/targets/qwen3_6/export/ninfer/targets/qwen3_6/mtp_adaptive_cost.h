#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::size_t kMtpAdaptiveCostWidths  = 8;
inline constexpr std::size_t kMtpAdaptiveCostBatches = 8;

struct MtpAdaptiveCostPoint {
    std::uint32_t frontier = 0;
    std::array<float, kMtpAdaptiveCostWidths> round_costs{};
};

struct MtpAdaptiveCostProfile {
    std::array<std::span<const MtpAdaptiveCostPoint>, kMtpAdaptiveCostBatches> batch_curves{};
};

} // namespace ninfer::targets::qwen3_6
