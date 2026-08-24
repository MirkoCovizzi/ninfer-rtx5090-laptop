#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/kvarn.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim       = 256;
constexpr std::int32_t kQuantGroup    = 64;
constexpr std::int32_t kQuantGroups   = kHeadDim / kQuantGroup;
constexpr float kAttentionScale       = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

// The Op has two registered compute profiles. A1 and A3 use the same criterion for a given
// profile; token count, geometry, execution envelope, and private launch route do not select it.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

constexpr ReductionCriterion kAttentionKvarnImplementationCriterion{
    // The production route stages exact represented K4/V2 values through BF16 before BF16 MMA.
    // Representation loss is excluded: the reference independently decodes the stored planes.
    /*relative_l2*/ 4.0e-3,
    /*gross_absolute*/ 1.5e-3,
    /*gross_relative_to_max_reference*/ 4.0e-3,
};

constexpr ReductionCriterion kAttentionKvarnQualityCriterion{
    // Separately records the admitted K4/V2 quality profile against uncompressed BF16 attention.
    /*relative_l2*/ 5.6e-1,
    /*gross_absolute*/ 1.1,
    /*gross_relative_to_max_reference*/ 1.12,
};

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"qwen3_6_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

void encode_group(const std::vector<float>& source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

HostCache make_cache(const Geometry& geometry, DType dtype, std::int32_t max_context,
                     std::uint32_t seed) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, dtype, max_context, logical_capacity};
    if (dtype == DType::BF16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_bf16 = to_bf16_bits(logical_v);
        return cache;
    }

    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    const std::size_t scales = scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
                encode_group(logical_v, code, cache.v_i8, code, cache.v_scale, scale);
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_bf16[target] = f32_to_bf16(v[source]);
                }
                continue;
            }

            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d     = group * kQuantGroup;
                const std::size_t source = kv_input_index(geometry, head, d, token);
                const std::size_t target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                encode_group(k, source, cache.k_i8, target, cache.k_scale, scale);
                encode_group(v, source, cache.v_i8, target, cache.v_scale, scale);
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code = cache_index(cache.geometry, cache.logical_capacity, head, position, d);
    if (cache.dtype == DType::BF16) {
        return static_cast<double>(bf16_to_f32(key ? cache.k_bf16[code] : cache.v_bf16[code]));
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.logical_capacity, head, position, d / kQuantGroup);
    const auto& codes   = key ? cache.k_i8 : cache.v_i8;
    const auto& scales  = key ? cache.k_scale : cache.v_scale;
    const float decoded = static_cast<float>(codes[code]) * f16_bits_to_f32(scales[scale]);
    return static_cast<double>(decoded);
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            double max_score           = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(geometry, q_head, d, token)]) *
                           cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                scores[static_cast<std::size_t>(position)] = score;
                max_score                                  = std::max(max_score, score);
            }

            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(position)] - max_score);
                probabilities[static_cast<std::size_t>(position)] = probability;
                sum += probability;
            }
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[static_cast<std::size_t>(position)] /= sum;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t position = 0; position < visible; ++position) {
                    value += probabilities[static_cast<std::size_t>(position)] *
                             cache_value(cache, false, kv_head, position, d);
                }
                output[q_index(geometry, q_head, d, token)] = value;
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), dtype_(cache.dtype), max_context_(cache.max_context),
          logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (dtype_ == DType::BF16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .dtype         = direct.dtype,
            .quant_group   = direct.quant_group,
        };
    }

    HostCache snapshot() const {
        HostCache cache{geometry_, dtype_, max_context_, logical_capacity_};
        if (dtype_ == DType::BF16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, code_elements_);
            cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_bf16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
            cache.k_i8             = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_i8             = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    DType dtype_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), dtype_(rows.front().dtype), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.dtype != dtype_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> expected_k(code_elements_, 0);
            std::vector<std::uint16_t> expected_v(code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, code_elements_), expected_v);
        } else {
            std::vector<std::int8_t> expected_k(code_elements_, 0);
            std::vector<std::int8_t> expected_v(code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::int8_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::int8_t>(v_, code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_),
                                     expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_bf16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> physical_k(code_elements_, 0);
            std::vector<std::uint16_t> physical_v(code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        std::vector<std::int8_t> physical_k(code_elements_, 0);
        std::vector<std::int8_t> physical_v(code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_ks);
            scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
    }

    Geometry geometry_;
    DType dtype_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.dtype == DType::BF16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_bf16, expected.v_bf16);
    } else {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "int8-g64"; }

ReductionCriterion attention_criterion(DType dtype) {
    return dtype == DType::BF16 ? kAttentionBf16Criterion : kAttentionInt8Criterion;
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::string case_label(const char* entry, const Geometry& geometry, DType dtype,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(dtype) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

int run_append_case(const Geometry& geometry, DType dtype, MappingPattern mapping,
                    std::uint32_t seed, std::int32_t tokens = 3, std::int32_t base = 63) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial, mapping);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_kv_append ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(mapping);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

int run_a1_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("gqa_attention_cached", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, DType dtype, const BatchAttentionCase& test_case) {
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());
    if (batch <= 0 || test_case.valid_columns.size() != static_cast<std::size_t>(batch) ||
        test_case.table_rows.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("invalid GQA batch test profile");
    }

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context       = maximum_visible + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    std::vector<HostCache> initial;
    initial.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial.push_back(
            make_cache(geometry, dtype, max_context, test_case.seed + 20u + 3u * row));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected[static_cast<std::size_t>(table_row)], row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected[static_cast<std::size_t>(table_row)], row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_attention batch ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], DType::I8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], DType::I8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], DType::I8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], DType::BF16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(geometry, dtype, mapping, 100u + geometry.q_heads);
            failures += run_a1_case(geometry, dtype, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, dtype, {1, 128, 129, 191u}, mapping);
        }
        if (dtype == DType::I8) {
            failures += run_append_case(geometry, dtype, MappingPattern::Fragmented,
                                        150u + geometry.q_heads, 129, 61);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, dtype, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, dtype, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

struct RepresentedKvarnCache {
    std::vector<double> k;
    std::vector<double> v;
};

RepresentedKvarnCache snapshot_kvarn_cache(
    const Geometry& geometry, std::int32_t logical_capacity,
    std::span<const std::int32_t> block_table, const GuardedDeviceBuffer& k_codes_device,
    const GuardedDeviceBuffer& v_codes_device, const GuardedDeviceBuffer& k_blocks_device,
    const GuardedDeviceBuffer& k_channels_device, const GuardedDeviceBuffer& v_channels_device,
    const GuardedDeviceBuffer& v_scales_device, const GuardedDeviceBuffer& v_zeros_device,
    const GuardedDeviceBuffer& tail_k_device, const GuardedDeviceBuffer& tail_v_device,
    const GuardedDeviceBuffer& markers_device) {
    const auto k_codes = copy_from_guarded<std::uint8_t>(k_codes_device, k_codes_device.bytes());
    const auto v_codes = copy_from_guarded<std::uint8_t>(v_codes_device, v_codes_device.bytes());
    const auto k_blocks =
        copy_from_guarded<std::uint8_t>(k_blocks_device, k_blocks_device.bytes());
    const auto k_channels = copy_from_guarded<std::uint16_t>(
        k_channels_device, k_channels_device.bytes() / sizeof(std::uint16_t));
    const auto v_channels = copy_from_guarded<std::uint16_t>(
        v_channels_device, v_channels_device.bytes() / sizeof(std::uint16_t));
    const auto v_scales = copy_from_guarded<std::uint16_t>(
        v_scales_device, v_scales_device.bytes() / sizeof(std::uint16_t));
    const auto v_zeros = copy_from_guarded<std::uint16_t>(
        v_zeros_device, v_zeros_device.bytes() / sizeof(std::uint16_t));
    const auto tail_k = copy_from_guarded<std::uint16_t>(
        tail_k_device, tail_k_device.bytes() / sizeof(std::uint16_t));
    const auto tail_v = copy_from_guarded<std::uint16_t>(
        tail_v_device, tail_v_device.bytes() / sizeof(std::uint16_t));
    const auto markers = copy_from_guarded<std::int32_t>(
        markers_device, markers_device.bytes() / sizeof(std::int32_t));

    RepresentedKvarnCache result{
        std::vector<double>(cache_elements(geometry, logical_capacity)),
        std::vector<double>(cache_elements(geometry, logical_capacity))};
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t logical_page = position / kPagedKVPageSize;
            const std::int32_t page_offset  = position & (kPagedKVPageSize - 1);
            std::int32_t tail_slot          = -1;
            for (std::int32_t slot = 0; slot < kKvarnTailSlots; ++slot) {
                if (markers[static_cast<std::size_t>(slot) * geometry.kv_heads + head] ==
                    logical_page) {
                    tail_slot = slot;
                }
            }
            const std::int32_t physical_page = block_table[logical_page];
            const std::size_t record =
                static_cast<std::size_t>(physical_page) * geometry.kv_heads + head;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t output =
                    cache_index(geometry, logical_capacity, head, position, d);
                if (tail_slot >= 0) {
                    const std::size_t tail =
                        ((static_cast<std::size_t>(tail_slot) * geometry.kv_heads + head) *
                             kPagedKVPageSize +
                         page_offset) *
                            kHeadDim +
                        d;
                    result.k[output] = bf16_to_f32(tail_k[tail]);
                    result.v[output] = bf16_to_f32(tail_v[tail]);
                    continue;
                }
                const std::size_t k_byte =
                    (record * kPagedKVPageSize + page_offset) * (kHeadDim / 2) + d / 2;
                const std::uint8_t k_code =
                    (k_codes[k_byte] >> (4 * (d & 1))) & 0x0fU;
                const std::size_t k_scale =
                    (record * kPagedKVPageSize + page_offset) * (kHeadDim / 16) + d / 16;
                result.k[output] = quantized_weight::detail::decode_e2m1(k_code) *
                                   quantized_weight::detail::decode_e4m3fn(k_blocks[k_scale]) *
                                   f16_bits_to_f32(k_channels[record * kHeadDim + d]);

                const std::size_t v_byte =
                    (record * kPagedKVPageSize + page_offset) * (kHeadDim / 4) + d / 4;
                const std::uint8_t v_code = (v_codes[v_byte] >> (2 * (d & 3))) & 3U;
                const float token_scale =
                    f16_bits_to_f32(v_scales[record * kPagedKVPageSize + page_offset]);
                const float token_zero =
                    f16_bits_to_f32(v_zeros[record * kPagedKVPageSize + page_offset]);
                result.v[output] =
                    (static_cast<float>(v_code) * token_scale + token_zero) *
                    f16_bits_to_f32(v_channels[record * kHeadDim + d]);
            }
        }
    }
    return result;
}

std::vector<double> ideal_attention_values(const std::vector<float>& q, const Geometry& geometry,
                                           std::int32_t logical_capacity,
                                           const std::vector<double>& k,
                                           const std::vector<double>& v,
                                           const std::vector<std::int32_t>& positions) {
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) * geometry.q_heads * tokens);
    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[token] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            double maximum             = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(geometry, q_head, d, token)]) *
                           k[cache_index(geometry, logical_capacity, kv_head, position, d)];
                }
                scores[position] = dot * static_cast<double>(kAttentionScale);
                maximum          = std::max(maximum, scores[position]);
            }
            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[position] = std::exp(scores[position] - maximum);
                sum += probabilities[position];
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t position = 0; position < visible; ++position) {
                    value += probabilities[position] / sum *
                             v[cache_index(geometry, logical_capacity, kv_head, position, d)];
                }
                output[q_index(geometry, q_head, d, token)] = value;
            }
        }
    }
    return output;
}

int run_kvarn_page_tail_case(const Geometry& geometry, std::int32_t tokens, bool flush_full_tail) {
    constexpr std::int32_t logical_pages  = 2;
    constexpr std::int32_t physical_pages = 3;
    constexpr std::int32_t table_rows     = 1;
    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.q_heads * tokens;
    const std::size_t kv_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens;
    std::vector<float> q = make_bf16_values(q_elements, 0x8101u + geometry.q_heads, -0.25F, 0.25F);
    std::vector<float> k = make_bf16_values(kv_elements, 0x8102u + geometry.q_heads, -0.25F, 0.25F);
    std::vector<float> v = make_bf16_values(kv_elements, 0x8103u + geometry.q_heads, -1.0F, 1.0F);
    std::vector<std::int32_t> positions(tokens);
    for (std::int32_t token = 0; token < tokens; ++token) { positions[token] = token; }

    HostCache expected = make_cache(geometry, DType::BF16, 128, 0x8120u);
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);

    const std::size_t records =
        static_cast<std::size_t>(physical_pages) * geometry.kv_heads;
    GuardedDeviceBuffer k_codes(records * kPagedKVPageSize * (kHeadDim / 2));
    GuardedDeviceBuffer v_codes(records * kPagedKVPageSize * (kHeadDim / 4));
    GuardedDeviceBuffer k_blocks(records * kPagedKVPageSize * (kHeadDim / 16));
    GuardedDeviceBuffer k_channels(records * kHeadDim * sizeof(std::uint16_t));
    GuardedDeviceBuffer v_channels(records * kHeadDim * sizeof(std::uint16_t));
    GuardedDeviceBuffer v_scales(records * kPagedKVPageSize * sizeof(std::uint16_t));
    GuardedDeviceBuffer v_zeros(records * kPagedKVPageSize * sizeof(std::uint16_t));
    GuardedDeviceBuffer tail_k(static_cast<std::size_t>(table_rows) * geometry.kv_heads *
                               kKvarnTailSlots *
                               kPagedKVPageSize * kHeadDim * sizeof(std::uint16_t));
    GuardedDeviceBuffer tail_v(tail_k.bytes());
    GuardedDeviceBuffer markers(static_cast<std::size_t>(table_rows) * geometry.kv_heads *
                                 kKvarnTailSlots *
                                 sizeof(std::int32_t));
    GuardedDeviceBuffer block_tables(logical_pages * table_rows * sizeof(std::int32_t));
    const std::vector<std::int32_t> marker_initial(geometry.kv_heads * kKvarnTailSlots, -1);
    const std::vector<std::int32_t> mapping{2, 0};
    markers.copy_from_host(marker_initial.data(), markers.bytes());
    block_tables.copy_from_host(mapping.data(), block_tables.bytes());

    PagedKVBatchLayerView cache{
        .k_pages = Tensor(k_codes.data(), DType::U8,
                          {kHeadDim / 2, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .v_pages = Tensor(v_codes.data(), DType::U8,
                          {kHeadDim / 4, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .k_scale_pages =
            Tensor(k_blocks.data(), DType::FP8_E4M3FN,
                   {kHeadDim / 16, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .v_scale_pages = Tensor(v_scales.data(), DType::FP16,
                                {1, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .k_channel_scale_pages =
            Tensor(k_channels.data(), DType::FP16,
                   {kHeadDim, 1, geometry.kv_heads, physical_pages}),
        .v_channel_scale_pages =
            Tensor(v_channels.data(), DType::FP16,
                   {kHeadDim, 1, geometry.kv_heads, physical_pages}),
        .v_zero_pages = Tensor(v_zeros.data(), DType::FP16,
                               {1, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .tail_k = Tensor(tail_k.data(), DType::BF16,
                          {kHeadDim, kPagedKVPageSize,
                           geometry.kv_heads * kKvarnTailSlots, table_rows}),
        .tail_v = Tensor(tail_v.data(), DType::BF16,
                          {kHeadDim, kPagedKVPageSize,
                           geometry.kv_heads * kKvarnTailSlots, table_rows}),
        .tail_logical_pages =
            Tensor(markers.data(), DType::I32,
                   {geometry.kv_heads * kKvarnTailSlots, table_rows}),
        .block_tables = Tensor(block_tables.data(), DType::I32, {logical_pages, table_rows}),
        .head_dim = kHeadDim,
        .num_kv_heads = geometry.kv_heads,
        .dtype = DType::U8,
        .quant_group = kQuantGroup,
    };

    const auto q_bits = to_bf16_bits(q);
    const auto k_bits = to_bf16_bits(k);
    const auto v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer drow(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), dq.bytes());
    dk.copy_from_host(k_bits.data(), dk.bytes());
    dv.copy_from_host(v_bits.data(), dv.bytes());
    dp.copy_from_host(positions.data(), dp.bytes());
    const std::int32_t row = 0;
    drow.copy_from_host(&row, sizeof(row));
    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens});
    Tensor tr(drow.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    ops::kvarn_hadamard(tq, tq, nullptr);
    ops::kvarn_hadamard(tk, tk, nullptr);
    ops::kvarn_hadamard(tv, tv, nullptr);

    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(tokens), 128};
    const std::size_t prompt_workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, DType::U8, envelope, 1, tokens, tokens);
    const std::size_t decode_workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, DType::U8, envelope, 1, 1, 1);
    const std::size_t workspace_bytes =
        std::max(prompt_workspace_bytes, decode_workspace_bytes);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, kAttentionScale, cache, envelope, workspace,
                       tout, nullptr);
    cuda_synchronize();

    const auto transformed_q_bits = copy_from_guarded<std::uint16_t>(dq, q_bits.size());
    std::vector<float> transformed_q(transformed_q_bits.size());
    for (std::size_t index = 0; index < transformed_q.size(); ++index) {
        transformed_q[index] = bf16_to_f32(transformed_q_bits[index]);
    }
    const RepresentedKvarnCache staged = snapshot_kvarn_cache(
        geometry, 128, mapping, k_codes, v_codes, k_blocks, k_channels, v_channels, v_scales,
        v_zeros, tail_k, tail_v, markers);
    const std::vector<double> staged_reference = ideal_attention_values(
        transformed_q, geometry, 128, staged.k, staged.v, positions);
    const std::string label = std::string("gqa_attention KVarN ") +
                              (flush_full_tail ? "full-tail flush " : "page+tail ") +
                              geometry.name;
    int failures = verify_attention(
        label + " represented staged", bf16_bits_to_double(copy_from_guarded<std::uint16_t>(
                                             dout, q_bits.size())),
        staged_reference, kAttentionKvarnImplementationCriterion);

    if (flush_full_tail) {
        const std::array<PagedKVBatchLayerView, 1> layers{cache};
        ops::gqa_kvarn_flush_full_pages(layers, tp.view({tokens, 1}), Tensor{}, tr, nullptr);
        PagedKVLayerView direct{
            .k_pages = cache.k_pages,
            .v_pages = cache.v_pages,
            .k_scale_pages = cache.k_scale_pages,
            .v_scale_pages = cache.v_scale_pages,
            .k_channel_scale_pages = cache.k_channel_scale_pages,
            .v_channel_scale_pages = cache.v_channel_scale_pages,
            .v_zero_pages = cache.v_zero_pages,
            .tail_k = cache.tail_k.view(
                {kHeadDim, kPagedKVPageSize, geometry.kv_heads * kKvarnTailSlots, 1}),
            .tail_v = cache.tail_v.view(
                {kHeadDim, kPagedKVPageSize, geometry.kv_heads * kKvarnTailSlots, 1}),
            .tail_logical_page = cache.tail_logical_pages.view(
                {geometry.kv_heads * kKvarnTailSlots}),
            .block_table = cache.block_tables.view({logical_pages}),
            .head_dim = kHeadDim,
            .num_kv_heads = geometry.kv_heads,
            .dtype = DType::U8,
            .quant_group = kQuantGroup,
        };
        ops::gqa_attention_cached(tq, tp, kAttentionScale, direct, envelope, workspace, tout,
                                  nullptr);
        cuda_synchronize();
        const RepresentedKvarnCache compressed = snapshot_kvarn_cache(
            geometry, 128, mapping, k_codes, v_codes, k_blocks, k_channels, v_channels, v_scales,
            v_zeros, tail_k, tail_v, markers);
        const std::vector<double> compressed_reference = ideal_attention_values(
            transformed_q, geometry, 128, compressed.k, compressed.v, positions);
        failures += verify_attention(
            label + " represented compressed prompt",
            bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, q_bits.size())),
            compressed_reference, kAttentionKvarnImplementationCriterion);

        Tensor last_q   = tq.slice(2, tokens - 1, 1);
        Tensor last_pos = tp.slice(0, tokens - 1, 1);
        Tensor last_out = tout.slice(2, tokens - 1, 1);
        ops::gqa_attention_cached(last_q, last_pos, kAttentionScale, direct, envelope, workspace,
                                  last_out, nullptr);
        cuda_synchronize();
        std::vector<float> last_q_host(static_cast<std::size_t>(kHeadDim) * geometry.q_heads);
        for (std::int32_t head = 0; head < geometry.q_heads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                last_q_host[q_index(geometry, head, d, 0)] =
                    transformed_q[q_index(geometry, head, d, tokens - 1)];
            }
        }
        const std::vector<double> decode_reference = ideal_attention_values(
            last_q_host, geometry, 128, compressed.k, compressed.v, {positions.back()});
        const auto decode_all =
            bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, q_bits.size()));
        std::vector<double> decode_actual(static_cast<std::size_t>(kHeadDim) * geometry.q_heads);
        for (std::int32_t head = 0; head < geometry.q_heads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                decode_actual[q_index(geometry, head, d, 0)] =
                    decode_all[q_index(geometry, head, d, tokens - 1)];
            }
        }
        failures += verify_attention(label + " represented compressed decode", decode_actual,
                                     decode_reference, kAttentionKvarnImplementationCriterion);
    }
    ops::kvarn_hadamard(tout, tout, nullptr);
    cuda_synchronize();

    const auto output_bits = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    const auto actual = bf16_bits_to_double(output_bits);
    failures += verify_attention(label + " compression quality", actual, reference,
                                 kAttentionKvarnQualityCriterion);
    const auto marker_result =
        copy_from_guarded<std::int32_t>(markers, geometry.kv_heads * kKvarnTailSlots);
    std::vector<std::int32_t> marker_expected(geometry.kv_heads * kKvarnTailSlots, -1);
    if (!flush_full_tail) { std::fill_n(marker_expected.begin(), geometry.kv_heads, 1); }
    failures +=
        verify_exact((label + " tail markers").c_str(), marker_result, marker_expected);
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    return failures;
}

int run_kvarn_long_context_split_case(const Geometry& geometry, std::int32_t position) {
    const std::int32_t visible        = position + 1;
    const std::int32_t late_begin     = std::max(0, visible - 769);
    const std::int32_t logical_pages  = (visible + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const std::int32_t physical_pages = logical_pages;
    const std::size_t records = static_cast<std::size_t>(physical_pages) * geometry.kv_heads;

    std::vector<std::uint8_t> k_codes(records * kPagedKVPageSize * (kHeadDim / 2), 0);
    std::vector<std::uint8_t> v_codes(records * kPagedKVPageSize * (kHeadDim / 4), 0);
    std::vector<std::uint8_t> k_blocks(records * kPagedKVPageSize * (kHeadDim / 16), 0);
    std::vector<std::uint16_t> k_channels(records * kHeadDim, f32_to_f16_bits(1.0F));
    std::vector<std::uint16_t> v_channels(records * kHeadDim, f32_to_f16_bits(1.0F));
    std::vector<std::uint16_t> v_scales(records * kPagedKVPageSize, f32_to_f16_bits(1.0F));
    std::vector<std::uint16_t> v_zeros(records * kPagedKVPageSize, f32_to_f16_bits(0.0F));
    for (std::int32_t token = late_begin; token < visible; ++token) {
        const std::int32_t page       = token / kPagedKVPageSize;
        const std::int32_t page_token = token & (kPagedKVPageSize - 1);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            const std::size_t record = static_cast<std::size_t>(page) * geometry.kv_heads + head;
            const std::size_t base =
                (record * kPagedKVPageSize + page_token) * (kHeadDim / 4);
            std::fill_n(v_codes.begin() + static_cast<std::ptrdiff_t>(base), kHeadDim / 4, 0xffU);
        }
    }
    std::vector<std::int32_t> mapping(logical_pages);
    std::iota(mapping.begin(), mapping.end(), 0);
    std::vector<std::int32_t> markers(geometry.kv_heads * kKvarnTailSlots, -1);

    GuardedDeviceBuffer dk_codes(k_codes.size());
    GuardedDeviceBuffer dv_codes(v_codes.size());
    GuardedDeviceBuffer dk_blocks(k_blocks.size());
    GuardedDeviceBuffer dk_channels(k_channels.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv_channels(v_channels.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv_scales(v_scales.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv_zeros(v_zeros.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer tail_k(static_cast<std::size_t>(geometry.kv_heads) * kKvarnTailSlots *
                               kPagedKVPageSize * kHeadDim * sizeof(std::uint16_t));
    GuardedDeviceBuffer tail_v(tail_k.bytes());
    GuardedDeviceBuffer dmarkers(markers.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable(mapping.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dq(static_cast<std::size_t>(geometry.q_heads) * kHeadDim *
                           sizeof(std::uint16_t));
    GuardedDeviceBuffer dpos(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(static_cast<std::size_t>(geometry.q_heads) * kHeadDim *
                             sizeof(std::uint16_t));
    dk_codes.copy_from_host(k_codes.data(), k_codes.size());
    dv_codes.copy_from_host(v_codes.data(), v_codes.size());
    dk_blocks.copy_from_host(k_blocks.data(), k_blocks.size());
    dk_channels.copy_from_host(k_channels.data(), dk_channels.bytes());
    dv_channels.copy_from_host(v_channels.data(), dv_channels.bytes());
    dv_scales.copy_from_host(v_scales.data(), dv_scales.bytes());
    dv_zeros.copy_from_host(v_zeros.data(), dv_zeros.bytes());
    dmarkers.copy_from_host(markers.data(), dmarkers.bytes());
    dtable.copy_from_host(mapping.data(), dtable.bytes());
    const std::vector<std::uint16_t> zero_q(static_cast<std::size_t>(geometry.q_heads) * kHeadDim,
                                             0);
    dq.copy_from_host(zero_q.data(), dq.bytes());
    dpos.copy_from_host(&position, sizeof(position));

    PagedKVLayerView cache{
        .k_pages = Tensor(dk_codes.data(), DType::U8,
                          {kHeadDim / 2, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .v_pages = Tensor(dv_codes.data(), DType::U8,
                          {kHeadDim / 4, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .k_scale_pages = Tensor(dk_blocks.data(), DType::FP8_E4M3FN,
                                {kHeadDim / 16, kPagedKVPageSize, geometry.kv_heads,
                                 physical_pages}),
        .v_scale_pages = Tensor(dv_scales.data(), DType::FP16,
                                {1, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .k_channel_scale_pages = Tensor(dk_channels.data(), DType::FP16,
                                        {kHeadDim, 1, geometry.kv_heads, physical_pages}),
        .v_channel_scale_pages = Tensor(dv_channels.data(), DType::FP16,
                                        {kHeadDim, 1, geometry.kv_heads, physical_pages}),
        .v_zero_pages = Tensor(dv_zeros.data(), DType::FP16,
                               {1, kPagedKVPageSize, geometry.kv_heads, physical_pages}),
        .tail_k = Tensor(tail_k.data(), DType::BF16,
                         {kHeadDim, kPagedKVPageSize, geometry.kv_heads * kKvarnTailSlots, 1}),
        .tail_v = Tensor(tail_v.data(), DType::BF16,
                         {kHeadDim, kPagedKVPageSize, geometry.kv_heads * kKvarnTailSlots, 1}),
        .tail_logical_page =
            Tensor(dmarkers.data(), DType::I32, {geometry.kv_heads * kKvarnTailSlots}),
        .block_table = Tensor(dtable.data(), DType::I32, {logical_pages}),
        .head_dim = kHeadDim,
        .num_kv_heads = geometry.kv_heads,
        .dtype = DType::U8,
        .quant_group = kQuantGroup,
    };
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(visible),
                                               static_cast<std::uint32_t>(visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, DType::U8, envelope, 1, 1, 1);
    GuardedDeviceBuffer workspace_buffer(workspace_bytes);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
    Tensor tp(dpos.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache, envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const double expected = 3.0 * static_cast<double>(visible - late_begin) / visible;
    const auto actual = bf16_bits_to_double(
        copy_from_guarded<std::uint16_t>(dout, zero_q.size()));
    int failures = 0;
    for (const double value : actual) {
        if (std::abs(value - expected) > 0.01) {
            std::cerr << "gqa_attention KVarN long-context split mismatch at visible=" << visible
                      << ": expected " << expected << ", got " << value << '\n';
            ++failures;
            break;
        }
    }
    failures += workspace_buffer.verify_guards(
        ("gqa_attention KVarN long-context workspace visible=" + std::to_string(visible)).c_str());
    return failures;
}

class KvarnLifecycleLayer {
public:
    static constexpr std::int32_t kLogicalPages  = 4;
    static constexpr std::int32_t kPhysicalPages = 4;

    explicit KvarnLifecycleLayer(const Geometry& geometry)
        : geometry_(geometry), records_(static_cast<std::size_t>(kPhysicalPages) * geometry.kv_heads),
          k_codes_(records_ * kPagedKVPageSize * (kHeadDim / 2)),
          v_codes_(records_ * kPagedKVPageSize * (kHeadDim / 4)),
          k_blocks_(records_ * kPagedKVPageSize * (kHeadDim / 16)),
          k_channels_(records_ * kHeadDim * sizeof(std::uint16_t)),
          v_channels_(records_ * kHeadDim * sizeof(std::uint16_t)),
          v_scales_(records_ * kPagedKVPageSize * sizeof(std::uint16_t)),
          v_zeros_(records_ * kPagedKVPageSize * sizeof(std::uint16_t)),
          tail_k_(static_cast<std::size_t>(geometry.kv_heads) * kKvarnTailSlots *
                  kPagedKVPageSize * kHeadDim * sizeof(std::uint16_t)),
          tail_v_(tail_k_.bytes()),
          markers_(static_cast<std::size_t>(geometry.kv_heads) * kKvarnTailSlots *
                   sizeof(std::int32_t)),
          block_table_(kLogicalPages * sizeof(std::int32_t)) {
        const std::vector<std::int32_t> markers(geometry.kv_heads * kKvarnTailSlots, -1);
        markers_.copy_from_host(markers.data(), markers_.bytes());
        block_table_.copy_from_host(mapping_.data(), block_table_.bytes());
    }

    PagedKVLayerView direct() {
        return {
            .k_pages = Tensor(k_codes_.data(), DType::U8,
                              {kHeadDim / 2, kPagedKVPageSize, geometry_.kv_heads, kPhysicalPages}),
            .v_pages = Tensor(v_codes_.data(), DType::U8,
                              {kHeadDim / 4, kPagedKVPageSize, geometry_.kv_heads, kPhysicalPages}),
            .k_scale_pages =
                Tensor(k_blocks_.data(), DType::FP8_E4M3FN,
                       {kHeadDim / 16, kPagedKVPageSize, geometry_.kv_heads, kPhysicalPages}),
            .v_scale_pages = Tensor(v_scales_.data(), DType::FP16,
                                    {1, kPagedKVPageSize, geometry_.kv_heads, kPhysicalPages}),
            .k_channel_scale_pages =
                Tensor(k_channels_.data(), DType::FP16,
                       {kHeadDim, 1, geometry_.kv_heads, kPhysicalPages}),
            .v_channel_scale_pages =
                Tensor(v_channels_.data(), DType::FP16,
                       {kHeadDim, 1, geometry_.kv_heads, kPhysicalPages}),
            .v_zero_pages = Tensor(v_zeros_.data(), DType::FP16,
                                   {1, kPagedKVPageSize, geometry_.kv_heads, kPhysicalPages}),
            .tail_k = Tensor(tail_k_.data(), DType::BF16,
                             {kHeadDim, kPagedKVPageSize,
                              geometry_.kv_heads * kKvarnTailSlots}),
            .tail_v = Tensor(tail_v_.data(), DType::BF16,
                             {kHeadDim, kPagedKVPageSize,
                              geometry_.kv_heads * kKvarnTailSlots}),
            .tail_logical_page =
                Tensor(markers_.data(), DType::I32,
                       {geometry_.kv_heads * kKvarnTailSlots}),
            .block_table = Tensor(block_table_.data(), DType::I32, {kLogicalPages}),
            .head_dim = kHeadDim,
            .num_kv_heads = geometry_.kv_heads,
            .dtype = DType::U8,
            .quant_group = kQuantGroup,
        };
    }

    PagedKVBatchLayerView batch() {
        PagedKVLayerView layer = direct();
        return {
            .k_pages = layer.k_pages,
            .v_pages = layer.v_pages,
            .k_scale_pages = layer.k_scale_pages,
            .v_scale_pages = layer.v_scale_pages,
            .k_channel_scale_pages = layer.k_channel_scale_pages,
            .v_channel_scale_pages = layer.v_channel_scale_pages,
            .v_zero_pages = layer.v_zero_pages,
            .tail_k = layer.tail_k.view(
                {kHeadDim, kPagedKVPageSize, geometry_.kv_heads * kKvarnTailSlots, 1}),
            .tail_v = layer.tail_v.view(
                {kHeadDim, kPagedKVPageSize, geometry_.kv_heads * kKvarnTailSlots, 1}),
            .tail_logical_pages = layer.tail_logical_page.view(
                {geometry_.kv_heads * kKvarnTailSlots, 1}),
            .block_tables = layer.block_table.view({kLogicalPages, 1}),
            .head_dim = kHeadDim,
            .num_kv_heads = geometry_.kv_heads,
            .dtype = DType::U8,
            .quant_group = kQuantGroup,
        };
    }

    std::vector<std::int32_t> markers() const {
        return copy_from_guarded<std::int32_t>(markers_, geometry_.kv_heads * kKvarnTailSlots);
    }

    RepresentedKvarnCache snapshot() const {
        return snapshot_kvarn_cache(geometry_, kLogicalPages * kPagedKVPageSize, mapping_, k_codes_,
                                    v_codes_, k_blocks_, k_channels_, v_channels_, v_scales_,
                                    v_zeros_, tail_k_, tail_v_, markers_);
    }

private:
    Geometry geometry_;
    std::size_t records_;
    const std::array<std::int32_t, kLogicalPages> mapping_{3, 1, 2, 0};
    GuardedDeviceBuffer k_codes_;
    GuardedDeviceBuffer v_codes_;
    GuardedDeviceBuffer k_blocks_;
    GuardedDeviceBuffer k_channels_;
    GuardedDeviceBuffer v_channels_;
    GuardedDeviceBuffer v_scales_;
    GuardedDeviceBuffer v_zeros_;
    GuardedDeviceBuffer tail_k_;
    GuardedDeviceBuffer tail_v_;
    GuardedDeviceBuffer markers_;
    GuardedDeviceBuffer block_table_;
};

int run_kvarn_page_transition_parity_case(const Geometry& geometry) {
    KvarnLifecycleLayer batched(geometry);
    KvarnLifecycleLayer sequential(geometry);

    const auto append = [&](KvarnLifecycleLayer& layer, const std::vector<std::uint16_t>& k,
                            const std::vector<std::uint16_t>& v,
                            const std::vector<std::int32_t>& positions) {
        GuardedDeviceBuffer dk(k.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dv(v.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
        dk.copy_from_host(k.data(), dk.bytes());
        dv.copy_from_host(v.data(), dv.bytes());
        dp.copy_from_host(positions.data(), dp.bytes());
        ops::gqa_kv_append(
            Tensor(dk.data(), DType::BF16,
                   {kHeadDim, geometry.kv_heads, static_cast<std::int32_t>(positions.size())}),
            Tensor(dv.data(), DType::BF16,
                   {kHeadDim, geometry.kv_heads, static_cast<std::int32_t>(positions.size())}),
            Tensor(dp.data(), DType::I32, {static_cast<std::int32_t>(positions.size())}),
            layer.direct(), nullptr);
        cuda_synchronize();
    };

    constexpr std::int32_t prefix_tokens = kPagedKVPageSize - 1;
    const std::size_t prefix_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * prefix_tokens;
    const auto prefix_k =
        to_bf16_bits(make_bf16_values(prefix_elements, 0xa100U + geometry.q_heads, -0.4F, 0.4F));
    const auto prefix_v =
        to_bf16_bits(make_bf16_values(prefix_elements, 0xa200U + geometry.q_heads, -1.0F, 1.0F));
    std::vector<std::int32_t> prefix_positions(prefix_tokens);
    for (std::int32_t token = 0; token < prefix_tokens; ++token) {
        prefix_positions[token] = token;
    }
    append(batched, prefix_k, prefix_v, prefix_positions);
    append(sequential, prefix_k, prefix_v, prefix_positions);

    constexpr std::int32_t tokens = 2;
    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.q_heads * tokens;
    const std::size_t kv_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens;
    const auto q =
        to_bf16_bits(make_bf16_values(q_elements, 0xa300U + geometry.q_heads, -0.3F, 0.3F));
    const auto k =
        to_bf16_bits(make_bf16_values(kv_elements, 0xa400U + geometry.q_heads, -0.4F, 0.4F));
    const auto v =
        to_bf16_bits(make_bf16_values(kv_elements, 0xa500U + geometry.q_heads, -1.0F, 1.0F));
    const std::array<std::int32_t, tokens> positions{kPagedKVPageSize - 1, kPagedKVPageSize};
    const std::int32_t row = 0;

    GuardedDeviceBuffer dq(q.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dr(sizeof(std::int32_t));
    GuardedDeviceBuffer batch_out(q.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer sequential_out(q.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q.data(), dq.bytes());
    dk.copy_from_host(k.data(), dk.bytes());
    dv.copy_from_host(v.data(), dv.bytes());
    dp.copy_from_host(positions.data(), dp.bytes());
    dr.copy_from_host(&row, sizeof(row));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens, 1});
    Tensor tr(dr.data(), DType::I32, {1});
    Tensor batch_result(batch_out.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor sequential_result(sequential_out.data(), DType::BF16,
                             {kHeadDim, geometry.q_heads, tokens});
    constexpr ops::GqaExecutionEnvelope envelope{1, 128};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, DType::U8, envelope, 1, 1, tokens);
    GuardedDeviceBuffer workspace_buffer(workspace_bytes);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    PagedKVBatchLayerView batch_cache = batched.batch();
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, kAttentionScale, batch_cache, envelope,
                       workspace, batch_result, nullptr);
    const std::array<PagedKVBatchLayerView, 1> batch_layers{batch_cache};
    ops::gqa_kvarn_flush_full_pages(batch_layers, tp, Tensor{}, tr, nullptr);

    PagedKVBatchLayerView sequential_cache = sequential.batch();
    const std::array<PagedKVBatchLayerView, 1> sequential_layers{sequential_cache};
    for (std::int32_t token = 0; token < tokens; ++token) {
        Tensor token_pos = tp.slice(0, token, 1);
        Tensor token_out = sequential_result.slice(2, token, 1);
        ops::gqa_attention(tq.slice(2, token, 1), tk.slice(2, token, 1), tv.slice(2, token, 1),
                           token_pos, Tensor{}, tr, kAttentionScale, sequential_cache, envelope,
                           workspace, token_out, nullptr);
        ops::gqa_kvarn_flush_full_pages(sequential_layers, token_pos, Tensor{}, tr, nullptr);
    }
    cuda_synchronize();

    const std::string label = std::string("KVarN page-transition T=2/T=1 parity ") + geometry.name;
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::uint16_t>(batch_out, q.size()),
                                copy_from_guarded<std::uint16_t>(sequential_out, q.size()));

    GuardedDeviceBuffer batch_cached_out(q_elements / tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer sequential_cached_out(q_elements / tokens * sizeof(std::uint16_t));
    Tensor last_q = tq.slice(2, 1, 1);
    Tensor last_pos = tp.slice(0, 1, 1);
    Tensor batch_cached(batch_cached_out.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
    Tensor sequential_cached(sequential_cached_out.data(), DType::BF16,
                             {kHeadDim, geometry.q_heads, 1});
    ops::gqa_attention_cached(last_q, last_pos, kAttentionScale, batched.direct(), envelope,
                              workspace, batch_cached, nullptr);
    ops::gqa_attention_cached(last_q, last_pos, kAttentionScale, sequential.direct(), envelope,
                              workspace, sequential_cached, nullptr);
    cuda_synchronize();
    failures += verify_exact(
        (label + " persisted cache").c_str(),
        copy_from_guarded<std::uint16_t>(batch_cached_out, q_elements / tokens),
        copy_from_guarded<std::uint16_t>(sequential_cached_out, q_elements / tokens));
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    return failures;
}

int run_kvarn_multilayer_lifecycle_case(const Geometry& geometry) {
    KvarnLifecycleLayer layer0(geometry);
    KvarnLifecycleLayer layer1(geometry);
    const auto append = [&](KvarnLifecycleLayer& layer, std::int32_t base, std::int32_t tokens,
                            std::uint32_t seed) {
        const std::size_t elements =
            static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens;
        const auto k = to_bf16_bits(make_bf16_values(elements, seed, -0.4F, 0.4F));
        const auto v = to_bf16_bits(make_bf16_values(elements, seed + 1U, -1.2F, 1.2F));
        std::vector<std::int32_t> positions(tokens);
        for (std::int32_t token = 0; token < tokens; ++token) { positions[token] = base + token; }
        GuardedDeviceBuffer dk(k.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dv(v.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
        dk.copy_from_host(k.data(), dk.bytes());
        dv.copy_from_host(v.data(), dv.bytes());
        dp.copy_from_host(positions.data(), dp.bytes());
        Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
        Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
        Tensor tp(dp.data(), DType::I32, {tokens});
        ops::gqa_kv_append(tk, tv, tp, layer.direct(), nullptr);
        cuda_synchronize();
    };

    append(layer0, 0, 63, 0x9200U);
    append(layer1, 0, 63, 0x9300U);
    std::vector<std::int32_t> first_expected(geometry.kv_heads * kKvarnTailSlots, -1);
    std::fill_n(first_expected.begin(), geometry.kv_heads, 0);
    int failures = verify_exact("KVarN first partial marker layer0", layer0.markers(),
                                first_expected);
    failures += verify_exact("KVarN first partial marker layer1", layer1.markers(),
                             first_expected);

    append(layer0, 63, 70, 0x9400U);
    append(layer1, 63, 70, 0x9500U);
    std::vector<std::int32_t> two_tail_expected(geometry.kv_heads * kKvarnTailSlots, 2);
    std::fill_n(two_tail_expected.begin(), geometry.kv_heads, 0);
    failures += verify_exact("KVarN two tails layer0", layer0.markers(), two_tail_expected);
    failures += verify_exact("KVarN two tails layer1", layer1.markers(), two_tail_expected);

    std::vector<std::int32_t> positions(70);
    for (std::int32_t token = 0; token < 70; ++token) { positions[token] = 63 + token; }
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dr(sizeof(std::int32_t));
    dp.copy_from_host(positions.data(), dp.bytes());
    const std::int32_t row = 0;
    dr.copy_from_host(&row, sizeof(row));
    Tensor tp(dp.data(), DType::I32, {70, 1});
    Tensor tr(dr.data(), DType::I32, {1});
    PagedKVBatchLayerView layer0_batch = layer0.batch();
    PagedKVBatchLayerView layer1_batch = layer1.batch();
    layer1_batch.block_tables          = layer0_batch.block_tables;
    const std::array<PagedKVBatchLayerView, 2> layers{layer0_batch, layer1_batch};
    ops::gqa_kvarn_flush_full_pages(layers, tp, Tensor{}, tr, nullptr);
    cuda_synchronize();
    std::vector<std::int32_t> flushed_expected(geometry.kv_heads * kKvarnTailSlots, 2);
    std::fill_n(flushed_expected.begin(), geometry.kv_heads, -1);
    failures += verify_exact("KVarN full marker cleared layer0", layer0.markers(),
                             flushed_expected);
    failures += verify_exact("KVarN full marker cleared layer1", layer1.markers(),
                             flushed_expected);

    // Re-appending after speculative rollback overwrites the reachable portion of the retained
    // partial tail; stale positions above the new frontier remain physically present but masked.
    append(layer0, 128, 5, 0x9600U);
    failures += verify_exact("KVarN retained partial after rewrite", layer0.markers(),
                             flushed_expected);

    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const auto q_bits = to_bf16_bits(make_bf16_values(q_elements, 0x9700U, -0.3F, 0.3F));
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpos(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), dq.bytes());
    const std::int32_t query_position = 132;
    dpos.copy_from_host(&query_position, sizeof(query_position));
    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
    Tensor tpos(dpos.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, 1});
    const ops::GqaExecutionEnvelope envelope{133, 133};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, DType::U8, envelope, 1, 1, 1);
    GuardedDeviceBuffer workspace_buffer(workspace_bytes);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    ops::gqa_attention_cached(tq, tpos, kAttentionScale, layer0.direct(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();
    const RepresentedKvarnCache represented = layer0.snapshot();
    std::vector<float> q(q_bits.size());
    for (std::size_t index = 0; index < q.size(); ++index) { q[index] = bf16_to_f32(q_bits[index]); }
    const auto reference = ideal_attention_values(q, geometry,
                                                  KvarnLifecycleLayer::kLogicalPages *
                                                      kPagedKVPageSize,
                                                  represented.k, represented.v, {query_position});
    failures += verify_attention(
        "KVarN fragmented compressed/tail lifecycle attention",
        bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, q_bits.size())), reference,
        kAttentionKvarnImplementationCriterion);
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8, DType::U8}) {
        constexpr ops::GqaExecutionEnvelope envelope{1, 1025};
        const std::size_t interval =
            ops::gqa_attention_workspace_capacity_bytes(16, dtype, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::gqa_attention_workspace_capacity_bytes(
                                            16, dtype, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "gqa_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "gqa_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += verify_workspace_capacity_contract();
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    for (const Geometry& geometry : kGeometries) {
        failures += run_kvarn_page_tail_case(geometry, 70, false);
        // W=97 selects the prompt kernel and crosses both halves of a 64-token encoded page.
        failures += run_kvarn_page_tail_case(geometry, 97, false);
        failures += run_kvarn_page_tail_case(geometry, 64, true);
        failures += run_kvarn_page_tail_case(geometry, 128, true);
        failures += run_kvarn_page_transition_parity_case(geometry);
        failures += run_kvarn_multilayer_lifecycle_case(geometry);
    }
    constexpr std::array<std::int32_t, 6> kLongContextPositions{
        8197,  // Last visible-key count before the KVarN-specific split policy.
        8198,  // First KVarN-specific split count.
        9000,  // Original reducer/producer mismatch reproduction.
        16390, // Upper split-planning tier transition.
        65535,
        79999,
    };
    for (const Geometry& geometry : kGeometries) {
        for (const std::int32_t position : kLongContextPositions) {
            failures += run_kvarn_long_context_split_case(geometry, position);
        }
    }
    failures += run_kvarn_long_context_split_case(
        kGeometries[0], static_cast<std::int32_t>(ops::kGqaAttentionMaximumVisibleKeys) - 1);
    failures += run_batch_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
