#include "ninfer/ops/kvarn.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct DeviceStorage {
    GuardedDeviceBuffer k_codes;
    GuardedDeviceBuffer k_block_scales;
    GuardedDeviceBuffer k_channel_scales;
    GuardedDeviceBuffer v_codes;
    GuardedDeviceBuffer v_channel_scales;
    GuardedDeviceBuffer v_token_scales;
    GuardedDeviceBuffer v_token_zeros;
    ops::KvarnTileStorage view;

    DeviceStorage(int d, int g, int tiles)
        : k_codes(static_cast<std::size_t>(d / 2) * g * tiles),
          k_block_scales(static_cast<std::size_t>(d / 16) * g * tiles),
          k_channel_scales(static_cast<std::size_t>(d) * tiles * sizeof(std::uint16_t)),
          v_codes(static_cast<std::size_t>(d / 4) * g * tiles),
          v_channel_scales(static_cast<std::size_t>(d) * tiles * sizeof(std::uint16_t)),
          v_token_scales(static_cast<std::size_t>(g) * tiles * sizeof(std::uint16_t)),
          v_token_zeros(static_cast<std::size_t>(g) * tiles * sizeof(std::uint16_t)),
          view{
              .k_codes = Tensor(k_codes.data(), DType::U8, {d / 2, g, tiles}),
              .k_block_scales =
                  Tensor(k_block_scales.data(), DType::FP8_E4M3FN, {d / 16, g, tiles}),
              .k_channel_scales =
                  Tensor(k_channel_scales.data(), DType::FP16, {d, tiles}),
              .v_codes = Tensor(v_codes.data(), DType::U8, {d / 4, g, tiles}),
              .v_channel_scales =
                  Tensor(v_channel_scales.data(), DType::FP16, {d, tiles}),
              .v_token_scales = Tensor(v_token_scales.data(), DType::FP16, {g, tiles}),
              .v_token_zeros = Tensor(v_token_zeros.data(), DType::FP16, {g, tiles}),
          } {}

    int verify_guards() const {
        int failures = 0;
        failures += k_codes.verify_guards("KVarN K codes");
        failures += k_block_scales.verify_guards("KVarN K block scales");
        failures += k_channel_scales.verify_guards("KVarN K channel scales");
        failures += v_codes.verify_guards("KVarN V codes");
        failures += v_channel_scales.verify_guards("KVarN V channel scales");
        failures += v_token_scales.verify_guards("KVarN V token scales");
        failures += v_token_zeros.verify_guards("KVarN V token zeros");
        return failures;
    }
};

struct StorageSnapshot {
    std::vector<std::uint8_t> k_codes;
    std::vector<std::uint8_t> k_block_scales;
    std::vector<std::uint16_t> k_channel_scales;
    std::vector<std::uint8_t> v_codes;
    std::vector<std::uint16_t> v_channel_scales;
    std::vector<std::uint16_t> v_token_scales;
    std::vector<std::uint16_t> v_token_zeros;
};

StorageSnapshot snapshot_storage(const DeviceStorage& storage) {
    return {
        .k_codes = from_device<std::uint8_t>(storage.k_codes.data(), storage.k_codes.bytes()),
        .k_block_scales = from_device<std::uint8_t>(storage.k_block_scales.data(),
                                                   storage.k_block_scales.bytes()),
        .k_channel_scales = from_device<std::uint16_t>(storage.k_channel_scales.data(),
                                                       storage.k_channel_scales.bytes() / 2),
        .v_codes = from_device<std::uint8_t>(storage.v_codes.data(), storage.v_codes.bytes()),
        .v_channel_scales = from_device<std::uint16_t>(storage.v_channel_scales.data(),
                                                       storage.v_channel_scales.bytes() / 2),
        .v_token_scales = from_device<std::uint16_t>(storage.v_token_scales.data(),
                                                     storage.v_token_scales.bytes() / 2),
        .v_token_zeros = from_device<std::uint16_t>(storage.v_token_zeros.data(),
                                                    storage.v_token_zeros.bytes() / 2),
    };
}

int verify_repeatable_storage(const char* label, const StorageSnapshot& first,
                              const StorageSnapshot& second) {
    int failures = 0;
    failures += verify_exact((std::string(label) + " K codes").c_str(), second.k_codes,
                             first.k_codes);
    failures += verify_exact((std::string(label) + " K block scales").c_str(),
                             second.k_block_scales, first.k_block_scales);
    failures += verify_exact((std::string(label) + " K channel scales").c_str(),
                             second.k_channel_scales, first.k_channel_scales);
    failures += verify_exact((std::string(label) + " V codes").c_str(), second.v_codes,
                             first.v_codes);
    failures += verify_exact((std::string(label) + " V channel scales").c_str(),
                             second.v_channel_scales, first.v_channel_scales);
    failures += verify_exact((std::string(label) + " V token scales").c_str(),
                             second.v_token_scales, first.v_token_scales);
    failures += verify_exact((std::string(label) + " V token zeros").c_str(),
                             second.v_token_zeros, first.v_token_zeros);
    return failures;
}

enum class InputProfile { Balanced, Gaussian, Outlier, DynamicRange, NearConstant };

const char* profile_name(InputProfile profile) {
    switch (profile) {
    case InputProfile::Balanced:
        return "balanced";
    case InputProfile::Gaussian:
        return "gaussian";
    case InputProfile::Outlier:
        return "outlier";
    case InputProfile::DynamicRange:
        return "dynamic-range";
    case InputProfile::NearConstant:
        return "near-constant";
    }
    return "unknown";
}

std::vector<std::uint16_t> represented_input(int d, int g, int tiles, std::uint32_t seed,
                                              float phase,
                                              InputProfile profile = InputProfile::Balanced) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 0.37F);
    std::vector<std::uint16_t> values(static_cast<std::size_t>(d) * g * tiles);
    for (int tile = 0; tile < tiles; ++tile) {
        for (int token = 0; token < g; ++token) {
            const float token_scale = 0.28F + 0.017F * static_cast<float>((token * 11 + 3) % 29);
            for (int channel = 0; channel < d; ++channel) {
                const float channel_scale =
                    0.35F + 0.013F * static_cast<float>((channel * 7 + tile * 5) % 31);
                const float signal =
                    std::sin(phase + 0.031F * static_cast<float>(channel) +
                             0.071F * static_cast<float>(token)) +
                    0.3F * std::cos(0.019F * static_cast<float>(channel * (token + 1)));
                float value = channel_scale * token_scale * (signal + normal(generator));
                if (profile == InputProfile::Gaussian) {
                    value = normal(generator);
                } else if (profile == InputProfile::Outlier) {
                    value = 0.08F * normal(generator);
                    if ((channel + 17 * token + 13 * tile) % 251 == 0) {
                        value += ((channel + token) & 1) == 0 ? 7.0F : -7.0F;
                    }
                } else if (profile == InputProfile::DynamicRange) {
                    const int exponent = ((channel * 5 + token * 3 + tile) % 17) - 8;
                    value = std::ldexp(signal + 0.15F * normal(generator), exponent);
                } else if (profile == InputProfile::NearConstant) {
                    value = 0.125F + 0.0005F * normal(generator);
                    if ((channel + token) % 97 == 0) { value -= 0.002F; }
                }
                values[(static_cast<std::size_t>(tile) * g + token) * d + channel] =
                    f32_to_bf16(value);
            }
        }
    }
    return values;
}

struct BalancedTile {
    std::vector<double> values;
    std::vector<double> row_scales;
    std::vector<double> col_scales;
};

BalancedTile variance_normalize(const std::vector<std::uint16_t>& represented, int d, int g,
                                int tile_index, bool key) {
    const int rows = key ? d : g;
    const int cols = key ? g : d;
    const auto input = [&](int row, int col) {
        const int channel = key ? row : col;
        const int token   = key ? col : row;
        const auto index = (static_cast<std::size_t>(tile_index) * g + token) * d + channel;
        return static_cast<double>(quantized_weight::detail::bf16_to_f32(represented[index]));
    };
    std::vector<double> log_row(rows, 0.0), log_col(cols, 0.0);
    std::vector<double> best_row = log_row, best_col = log_col;
    const auto stddev = [&](bool across_rows, int fixed) {
        const int count = across_rows ? rows : cols;
        double sum = 0.0, squares = 0.0;
        for (int index = 0; index < count; ++index) {
            const int row = across_rows ? index : fixed;
            const int col = across_rows ? fixed : index;
            const double x = input(row, col) * std::exp(-log_row[row] - log_col[col]);
            sum += x;
            squares += x * x;
        }
        return std::sqrt(std::max(0.0, (squares - sum * sum / count) / (count - 1)));
    };
    const auto imbalance = [&]() {
        std::vector<double> row_std(rows), col_std(cols);
        for (int row = 0; row < rows; ++row) row_std[row] = stddev(false, row);
        for (int col = 0; col < cols; ++col) col_std[col] = stddev(true, col);
        const auto [row_min, row_max] = std::minmax_element(row_std.begin(), row_std.end());
        const auto [col_min, col_max] = std::minmax_element(col_std.begin(), col_std.end());
        return *row_max / std::max(*row_min, 1.0e-8) +
               *col_max / std::max(*col_min, 1.0e-8);
    };

    double best = imbalance();
    for (int iteration = 0; iteration < ops::kKvarnIterations; ++iteration) {
        for (int col = 0; col < cols; ++col) {
            log_col[col] = std::clamp(log_col[col] + std::log(std::clamp(stddev(true, col), 1e-3,
                                                                         1e3)),
                                      -0.3, 10.0);
        }
        for (int row = 0; row < rows; ++row) {
            log_row[row] = std::clamp(log_row[row] + std::log(std::clamp(stddev(false, row), 1e-3,
                                                                         1e3)),
                                      -0.3, 10.0);
        }
        const double current = imbalance();
        if (current <= best) {
            best      = current;
            best_row = log_row;
            best_col = log_col;
        }
    }

    BalancedTile out;
    out.values.resize(static_cast<std::size_t>(rows) * cols);
    out.row_scales.resize(rows);
    out.col_scales.resize(cols);
    for (int row = 0; row < rows; ++row) out.row_scales[row] = std::exp(best_row[row]);
    for (int col = 0; col < cols; ++col) out.col_scales[col] = std::exp(best_col[col]);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            out.values[static_cast<std::size_t>(row) * cols + col] =
                input(row, col) / out.row_scales[row] / out.col_scales[col];
        }
    }
    return out;
}

double cosine(const std::vector<double>& expected, const std::vector<float>& actual,
              std::size_t offset) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        dot += expected[i] * actual[offset + i];
        aa += expected[i] * expected[i];
        bb += static_cast<double>(actual[offset + i]) * actual[offset + i];
    }
    if (aa == 0.0 || bb == 0.0) { return aa == bb ? 1.0 : 0.0; }
    return dot / std::sqrt(aa * bb);
}

struct QualityStats {
    double relative_l2 = 0.0;
    double cosine      = 0.0;
    double maximum_absolute = 0.0;
};

struct CodecQualityCriterion {
    double k_relative_l2;
    double v_relative_l2;
    double k_cosine;
    double v_cosine;
};

CodecQualityCriterion codec_quality_criterion(InputProfile profile) {
    switch (profile) {
    case InputProfile::Balanced:
        return {0.105, 0.52, 0.994, 0.89};
    case InputProfile::Gaussian:
        return {0.105, 0.57, 0.994, 0.87};
    case InputProfile::Outlier:
        return {0.035, 0.75, 0.9995, 0.80};
    case InputProfile::DynamicRange:
        return {0.08, 1.50, 0.997, 0.58};
    case InputProfile::NearConstant:
        return {0.035, 0.002, 0.9997, 0.99999};
    }
    throw std::invalid_argument("unknown KVarN codec input profile");
}

QualityStats quality_stats(const std::vector<std::uint16_t>& expected,
                           const std::vector<float>& actual, std::size_t offset,
                           std::size_t count) {
    long double squared_error = 0.0;
    long double squared_reference = 0.0;
    long double dot = 0.0;
    long double squared_actual = 0.0;
    double maximum_absolute = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double reference = quantized_weight::detail::bf16_to_f32(expected[offset + i]);
        const double got       = actual[offset + i];
        const double error     = got - reference;
        squared_error += error * error;
        squared_reference += reference * reference;
        squared_actual += got * got;
        dot += got * reference;
        maximum_absolute = std::max(maximum_absolute, std::abs(error));
    }
    const double denominator = std::sqrt(static_cast<double>(squared_reference));
    const double actual_norm = std::sqrt(static_cast<double>(squared_actual));
    return {
        .relative_l2 = denominator == 0.0
                           ? (squared_error == 0.0 ? 0.0
                                                   : std::numeric_limits<double>::infinity())
                           : std::sqrt(static_cast<double>(squared_error)) / denominator,
        .cosine = denominator == 0.0 || actual_norm == 0.0
                      ? (denominator == actual_norm ? 1.0 : 0.0)
                      : static_cast<double>(dot) / (denominator * actual_norm),
        .maximum_absolute = maximum_absolute,
    };
}

int verify_storage_decode(int d, int g, int tiles, const DeviceStorage& storage,
                          const std::vector<float>& decoded_k,
                          const std::vector<float>& decoded_v) {
    const auto k_codes = from_device<std::uint8_t>(storage.k_codes.data(), storage.k_codes.bytes());
    const auto k_blocks =
        from_device<std::uint8_t>(storage.k_block_scales.data(), storage.k_block_scales.bytes());
    const auto k_channels = from_device<std::uint16_t>(storage.k_channel_scales.data(),
                                                       storage.k_channel_scales.bytes() / 2);
    const auto v_codes = from_device<std::uint8_t>(storage.v_codes.data(), storage.v_codes.bytes());
    const auto v_channels = from_device<std::uint16_t>(storage.v_channel_scales.data(),
                                                       storage.v_channel_scales.bytes() / 2);
    const auto v_scales = from_device<std::uint16_t>(storage.v_token_scales.data(),
                                                     storage.v_token_scales.bytes() / 2);
    const auto v_zeros = from_device<std::uint16_t>(storage.v_token_zeros.data(),
                                                    storage.v_token_zeros.bytes() / 2);
    int failures = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        for (int token = 0; token < g; ++token) {
            for (int channel = 0; channel < d; ++channel) {
                const std::size_t output = (static_cast<std::size_t>(tile) * g + token) * d + channel;
                const std::size_t k_byte =
                    (static_cast<std::size_t>(tile) * g + token) * (d / 2) + channel / 2;
                const std::uint8_t k_word =
                    (k_codes[k_byte] >> (4 * (channel & 1))) & 0x0fU;
                const std::size_t k_scale =
                    (static_cast<std::size_t>(tile) * g + token) * (d / 16) + channel / 16;
                float expected_k = static_cast<float>(quantized_weight::detail::decode_e2m1(k_word));
                expected_k *=
                    static_cast<float>(quantized_weight::detail::decode_e4m3fn(k_blocks[k_scale]));
                expected_k *= quantized_weight::detail::f16_to_f32(
                    k_channels[static_cast<std::size_t>(tile) * d + channel]);
                const std::size_t v_byte =
                    (static_cast<std::size_t>(tile) * g + token) * (d / 4) + channel / 4;
                const int v_code = (v_codes[v_byte] >> (2 * (channel & 3))) & 3;
                float expected_v =
                    v_code * quantized_weight::detail::f16_to_f32(
                                 v_scales[static_cast<std::size_t>(tile) * g + token]) +
                    quantized_weight::detail::f16_to_f32(
                        v_zeros[static_cast<std::size_t>(tile) * g + token]);
                expected_v *= quantized_weight::detail::f16_to_f32(
                    v_channels[static_cast<std::size_t>(tile) * d + channel]);
                if (decoded_k[output] != expected_k || decoded_v[output] != expected_v) {
                    if (++failures <= 8) {
                        std::cerr << "KVarN exact decode mismatch at " << output << '\n';
                    }
                }
            }
        }
    }
    return failures;
}

int run_codec_case(int d, int g, int tiles, InputProfile profile) {
    const auto host_k = represented_input(d, g, tiles, 1741U + d, 0.17F, profile);
    const auto host_v = represented_input(d, g, tiles, 2917U + g, 0.53F, profile);
    const std::size_t elements = static_cast<std::size_t>(d) * g * tiles;
    GuardedDeviceBuffer device_k(elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_v(elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer decoded_k(elements * sizeof(float));
    GuardedDeviceBuffer decoded_v(elements * sizeof(float));
    device_k.copy_from_host(host_k.data(), device_k.bytes());
    device_v.copy_from_host(host_v.data(), device_v.bytes());
    DeviceStorage storage(d, g, tiles);

    Tensor k(device_k.data(), DType::BF16, {d, g, tiles});
    Tensor v(device_v.data(), DType::BF16, {d, g, tiles});
    Tensor dk(decoded_k.data(), DType::FP32, {d, g, tiles});
    Tensor dv(decoded_v.data(), DType::FP32, {d, g, tiles});
    ops::kvarn_compress(k, v, storage.view, nullptr);
    ops::kvarn_decompress(storage.view, dk, dv, nullptr);
    cuda_synchronize();
    const StorageSnapshot first_storage = snapshot_storage(storage);
    ops::kvarn_compress(k, v, storage.view, nullptr);
    ops::kvarn_decompress(storage.view, dk, dv, nullptr);
    cuda_synchronize();

    const auto actual_k = from_device<float>(decoded_k.data(), elements);
    const auto actual_v = from_device<float>(decoded_v.data(), elements);
    const CodecQualityCriterion criterion = codec_quality_criterion(profile);
    int failures = verify_repeatable_storage(
        (std::string("KVarN repeatability ") + profile_name(profile)).c_str(), first_storage,
        snapshot_storage(storage));
    failures += verify_storage_decode(d, g, tiles, storage, actual_k, actual_v);
    for (int tile = 0; tile < tiles; ++tile) {
        const BalancedTile oracle_k = variance_normalize(host_k, d, g, tile, true);
        const BalancedTile oracle_v = variance_normalize(host_v, d, g, tile, false);
        std::vector<double> represented_k(static_cast<std::size_t>(d) * g);
        std::vector<double> represented_v(static_cast<std::size_t>(d) * g);
        for (int token = 0; token < g; ++token) {
            for (int channel = 0; channel < d; ++channel) {
                represented_k[static_cast<std::size_t>(token) * d + channel] =
                    oracle_k.values[static_cast<std::size_t>(channel) * g + token] *
                    oracle_k.row_scales[channel] * oracle_k.col_scales[token];
                represented_v[static_cast<std::size_t>(token) * d + channel] =
                    oracle_v.values[static_cast<std::size_t>(token) * d + channel] *
                    oracle_v.row_scales[token] * oracle_v.col_scales[channel];
            }
        }
        const std::size_t offset = static_cast<std::size_t>(tile) * d * g;
        const double k_cos = cosine(represented_k, actual_k, offset);
        const double v_cos = cosine(represented_v, actual_v, offset);
        const QualityStats k_stats = quality_stats(host_k, actual_k, offset, d * g);
        const QualityStats v_stats = quality_stats(host_v, actual_v, offset, d * g);
        std::cout << "    codec D=" << d << " G=" << g << " profile="
                  << profile_name(profile) << " tile=" << tile << " K(rel_l2="
                  << k_stats.relative_l2 << ", cosine=" << k_stats.cosine << ", max_abs="
                  << k_stats.maximum_absolute << ") V(rel_l2=" << v_stats.relative_l2
                  << ", cosine=" << v_stats.cosine << ", max_abs="
                  << v_stats.maximum_absolute << ")\n";
        if (!std::isfinite(k_stats.relative_l2) || !std::isfinite(v_stats.relative_l2) ||
            std::abs(k_cos - k_stats.cosine) > 1.0e-12 ||
            std::abs(v_cos - v_stats.cosine) > 1.0e-12) {
            std::cerr << "KVarN non-finite/inconsistent quality statistics D=" << d
                      << " G=" << g << " profile=" << profile_name(profile) << '\n';
            ++failures;
        }
        if (k_stats.relative_l2 > criterion.k_relative_l2 ||
            v_stats.relative_l2 > criterion.v_relative_l2 ||
            k_stats.cosine < criterion.k_cosine || v_stats.cosine < criterion.v_cosine) {
            std::cerr << "KVarN codec quality regression D=" << d << " G=" << g
                      << " profile=" << profile_name(profile) << " tile=" << tile
                      << " limits K(rel_l2<=" << criterion.k_relative_l2
                      << ", cosine>=" << criterion.k_cosine << ") V(rel_l2<="
                      << criterion.v_relative_l2 << ", cosine>=" << criterion.v_cosine << ")\n";
            ++failures;
        }
    }
    failures += device_k.verify_guards("KVarN input K");
    failures += device_v.verify_guards("KVarN input V");
    failures += decoded_k.verify_guards("KVarN decoded K");
    failures += decoded_v.verify_guards("KVarN decoded V");
    failures += storage.verify_guards();
    return failures;
}

std::vector<std::uint16_t> hadamard_input(int d, int vectors) {
    auto input = represented_input(d, 1, vectors, 811U + d, 0.29F);
    for (int channel = 0; channel < d; ++channel) {
        input[channel] = f32_to_bf16(channel == 0 ? 1.0F : 0.0F);
        input[static_cast<std::size_t>(d) + channel] = f32_to_bf16(0.125F);
        input[static_cast<std::size_t>(2 * d) + channel] =
            f32_to_bf16((channel & 1) == 0 ? 0.25F : -0.25F);
        input[static_cast<std::size_t>(3 * d) + channel] =
            f32_to_bf16((static_cast<float>(channel) - d / 2.0F) / d);
    }
    return input;
}

std::vector<double> hadamard_oracle(const std::vector<std::uint16_t>& input, int d, int vectors) {
    std::vector<double> output(input.size());
    const double normalization = 1.0 / std::sqrt(static_cast<double>(d));
    for (int vector = 0; vector < vectors; ++vector) {
        std::vector<double> work(static_cast<std::size_t>(d));
        for (int channel = 0; channel < d; ++channel) {
            work[channel] = quantized_weight::detail::bf16_to_f32(
                input[static_cast<std::size_t>(vector) * d + channel]);
        }
        for (int stride = 1; stride < d; stride *= 2) {
            for (int base = 0; base < d; base += 2 * stride) {
                for (int lane = 0; lane < stride; ++lane) {
                    const double a = work[base + lane];
                    const double b = work[base + stride + lane];
                    work[base + lane]          = a + b;
                    work[base + stride + lane] = a - b;
                }
            }
        }
        for (int channel = 0; channel < d; ++channel) {
            output[static_cast<std::size_t>(vector) * d + channel] =
                work[channel] * normalization;
        }
    }
    return output;
}

int run_hadamard_case(int d, int vectors) {
    const auto input = hadamard_input(d, vectors);
    GuardedDeviceBuffer source(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer transformed(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer restored(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer in_place(input.size() * sizeof(std::uint16_t));
    source.copy_from_host(input.data(), source.bytes());
    in_place.copy_from_host(input.data(), in_place.bytes());
    Tensor a(source.data(), DType::BF16, {d, vectors});
    Tensor b(transformed.data(), DType::BF16, {d, vectors});
    Tensor c(restored.data(), DType::BF16, {d, vectors});
    ops::kvarn_hadamard(a, b, nullptr);
    ops::kvarn_hadamard(b, c, nullptr);
    Tensor inplace_tensor(in_place.data(), DType::BF16, {d, vectors});
    ops::kvarn_hadamard(inplace_tensor, inplace_tensor, nullptr);
    ops::kvarn_hadamard(inplace_tensor, inplace_tensor, nullptr);
    cuda_synchronize();
    const auto transformed_output =
        from_device<std::uint16_t>(transformed.data(), input.size());
    const auto output = from_device<std::uint16_t>(restored.data(), input.size());
    const auto inplace_output = from_device<std::uint16_t>(in_place.data(), input.size());
    double squared_error = 0.0, squared_input = 0.0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double expected = quantized_weight::detail::bf16_to_f32(input[i]);
        const double actual   = quantized_weight::detail::bf16_to_f32(output[i]);
        squared_error += (actual - expected) * (actual - expected);
        squared_input += expected * expected;
    }
    std::vector<double> transformed_actual(transformed_output.size());
    for (std::size_t i = 0; i < transformed_output.size(); ++i) {
        transformed_actual[i] = quantized_weight::detail::bf16_to_f32(transformed_output[i]);
    }
    constexpr ReductionCriterion kHadamardCriterion{
        /*relative_l2=*/3.5e-3,
        /*gross_absolute=*/1.0e-3,
        /*gross_relative_to_max_reference=*/3.5e-3,
    };
    int failures = verify_reduction(("KVarN Hadamard FP64 D=" + std::to_string(d)).c_str(),
                                    transformed_actual, hadamard_oracle(input, d, vectors),
                                    kHadamardCriterion);
    const double relative = std::sqrt(squared_error / squared_input);
    if (relative > 0.006) {
        std::cerr << "KVarN Hadamard round-trip D=" << d << " relative=" << relative << '\n';
        ++failures;
    }
    double inplace_error = 0.0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double expected = quantized_weight::detail::bf16_to_f32(input[i]);
        const double actual   = quantized_weight::detail::bf16_to_f32(inplace_output[i]);
        inplace_error += (actual - expected) * (actual - expected);
    }
    if (std::sqrt(inplace_error / squared_input) > 0.006) {
        std::cerr << "KVarN in-place Hadamard round-trip D=" << d << " failed\n";
        ++failures;
    }
    failures += source.verify_guards("KVarN Hadamard source");
    failures += transformed.verify_guards("KVarN Hadamard transformed");
    failures += restored.verify_guards("KVarN Hadamard restored");
    failures += in_place.verify_guards("KVarN Hadamard in-place");
    return failures;
}

std::vector<double> normalized_gaussian(std::mt19937& generator, int d, double norm) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> values(static_cast<std::size_t>(d));
    double squared = 0.0;
    for (double& value : values) {
        value = normal(generator);
        squared += value * value;
    }
    const double scale = norm / std::sqrt(squared);
    for (double& value : values) value *= scale;
    return values;
}

std::vector<double> correlated_key(std::mt19937& generator, const std::vector<double>& query,
                                   double rho) {
    const int d = static_cast<int>(query.size());
    double query_squared = 0.0;
    for (const double value : query) query_squared += value * value;
    std::vector<double> noise = normalized_gaussian(generator, d, std::sqrt(query_squared));
    double projection = 0.0;
    for (int index = 0; index < d; ++index) projection += noise[index] * query[index];
    projection /= query_squared;
    double orthogonal_squared = 0.0;
    for (int index = 0; index < d; ++index) {
        noise[index] -= projection * query[index];
        orthogonal_squared += noise[index] * noise[index];
    }
    const double orthogonal_scale = std::sqrt(query_squared / orthogonal_squared);
    const double residual = std::sqrt(1.0 - rho * rho);
    std::vector<double> key(static_cast<std::size_t>(d));
    for (int index = 0; index < d; ++index) {
        key[index] = rho * query[index] + residual * noise[index] * orthogonal_scale;
    }
    return key;
}

std::vector<float> int8_group64_decode(const std::vector<std::uint16_t>& represented, int d,
                                       int tokens) {
    std::vector<float> decoded(represented.size());
    for (int token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * d;
        for (int group = 0; group < d; group += 64) {
            float maximum = 0.0F;
            for (int lane = 0; lane < 64; ++lane) {
                maximum = std::max(maximum, std::abs(quantized_weight::detail::bf16_to_f32(
                                                    represented[base + group + lane])));
            }
            const std::uint16_t scale_bits = quantized_weight::detail::f32_to_f16(maximum / 127.0F);
            const float scale = quantized_weight::detail::f16_to_f32(scale_bits);
            for (int lane = 0; lane < 64; ++lane) {
                const float value =
                    quantized_weight::detail::bf16_to_f32(represented[base + group + lane]);
                const int code = scale == 0.0F
                                     ? 0
                                     : std::clamp(static_cast<int>(std::nearbyint(value / scale)),
                                                  -127, 127);
                decoded[base + group + lane] = static_cast<float>(code) * scale;
            }
        }
    }
    return decoded;
}

struct RankingResult {
    int rank = 0;
    int top  = -1;
    double relative_l2 = 0.0;
    double cosine      = 0.0;
};

RankingResult compare_scores(const std::vector<double>& reference,
                             const std::vector<double>& compared, int target) {
    long double dot = 0.0, reference_squared = 0.0, compared_squared = 0.0, error_squared = 0.0;
    int top = 0;
    int rank = 1;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        dot += reference[index] * compared[index];
        reference_squared += reference[index] * reference[index];
        compared_squared += compared[index] * compared[index];
        const double error = compared[index] - reference[index];
        error_squared += error * error;
        if (compared[index] > compared[static_cast<std::size_t>(top)]) {
            top = static_cast<int>(index);
        }
        if (compared[index] > compared[static_cast<std::size_t>(target)]) ++rank;
    }
    return {
        .rank = rank,
        .top = top,
        .relative_l2 = std::sqrt(static_cast<double>(error_squared / reference_squared)),
        .cosine = static_cast<double>(dot / std::sqrt(reference_squared * compared_squared)),
    };
}

int run_close_margin_retrieval_case() {
    constexpr int d = 256;
    constexpr int g = 64;
    constexpr int tokens = 32768;
    constexpr int queries = 5;
    constexpr int hard_distractors = 24;
    constexpr double component_sigma = 0.20;
    constexpr double target_rho = 0.94;
    constexpr std::array<double, queries> margins{0.08, 0.04, 0.02, 0.01, 0.005};
    constexpr std::array<double, queries> depths{0.05, 0.25, 0.50, 0.75, 0.95};

    std::mt19937 generator(0x4b564152U);
    std::normal_distribution<float> normal(0.0F, static_cast<float>(component_sigma));
    std::vector<std::uint16_t> keys(static_cast<std::size_t>(d) * tokens);
    for (std::uint16_t& value : keys) value = f32_to_bf16(normal(generator));
    std::vector<std::vector<double>> query_values;
    std::array<int, queries> targets{};
    const double vector_norm = std::sqrt(static_cast<double>(d)) * component_sigma;
    for (int query_index = 0; query_index < queries; ++query_index) {
        query_values.push_back(normalized_gaussian(generator, d, vector_norm));
        const int target = static_cast<int>(depths[query_index] * tokens);
        targets[query_index] = target;
        const auto store = [&](int token, const std::vector<double>& values) {
            for (int channel = 0; channel < d; ++channel) {
                keys[static_cast<std::size_t>(token) * d + channel] =
                    f32_to_bf16(static_cast<float>(values[channel]));
            }
        };
        store(target, correlated_key(generator, query_values.back(), target_rho));
        for (int distractor = 0; distractor < hard_distractors; ++distractor) {
            int token = target + distractor - hard_distractors / 2;
            if (token == target) token += hard_distractors + 1;
            token = std::clamp(token, 0, tokens - 1);
            store(token, correlated_key(generator, query_values.back(),
                                         target_rho - margins[query_index]));
        }
    }

    const std::size_t elements = keys.size();
    GuardedDeviceBuffer device_k(elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_v(elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer decoded_k(elements * sizeof(float));
    GuardedDeviceBuffer decoded_v(elements * sizeof(float));
    device_k.copy_from_host(keys.data(), device_k.bytes());
    device_v.copy_from_host(keys.data(), device_v.bytes());
    DeviceStorage storage(d, g, tokens / g);
    Tensor k(device_k.data(), DType::BF16, {d, g, tokens / g});
    Tensor v(device_v.data(), DType::BF16, {d, g, tokens / g});
    Tensor dk(decoded_k.data(), DType::FP32, {d, g, tokens / g});
    Tensor dv(decoded_v.data(), DType::FP32, {d, g, tokens / g});
    ops::kvarn_compress(k, v, storage.view, nullptr);
    ops::kvarn_decompress(storage.view, dk, dv, nullptr);
    cuda_synchronize();
    const std::vector<float> kvarn = from_device<float>(decoded_k.data(), elements);
    const std::vector<float> int8 = int8_group64_decode(keys, d, tokens);

    int failures = verify_storage_decode(d, g, tokens / g, storage, kvarn,
                                         from_device<float>(decoded_v.data(), elements));
    for (int query_index = 0; query_index < queries; ++query_index) {
        std::vector<double> query(static_cast<std::size_t>(d));
        for (int channel = 0; channel < d; ++channel) {
            query[channel] = quantized_weight::detail::bf16_to_f32(
                f32_to_bf16(static_cast<float>(query_values[query_index][channel])));
        }
        std::vector<double> bf16_scores(tokens), int8_scores(tokens), kvarn_scores(tokens);
        for (int token = 0; token < tokens; ++token) {
            const std::size_t base = static_cast<std::size_t>(token) * d;
            for (int channel = 0; channel < d; ++channel) {
                const double q = query[channel];
                bf16_scores[token] += q * quantized_weight::detail::bf16_to_f32(
                                                keys[base + channel]);
                int8_scores[token] += q * int8[base + channel];
                kvarn_scores[token] += q * kvarn[base + channel];
            }
        }
        const RankingResult int8_result =
            compare_scores(bf16_scores, int8_scores, targets[query_index]);
        const RankingResult kvarn_result =
            compare_scores(bf16_scores, kvarn_scores, targets[query_index]);
        const int bf16_top = static_cast<int>(
            std::max_element(bf16_scores.begin(), bf16_scores.end()) - bf16_scores.begin());
        std::cout << "    close-margin retrieval depth=" << depths[query_index] * 100.0
                  << "% rho_gap=" << margins[query_index] << " BF16(top=" << bf16_top
                  << ") INT8(top=" << int8_result.top << ", rank=" << int8_result.rank
                  << ", rel_l2=" << int8_result.relative_l2 << ", cosine=" << int8_result.cosine
                  << ") KVarN(top=" << kvarn_result.top << ", rank=" << kvarn_result.rank
                  << ", rel_l2=" << kvarn_result.relative_l2 << ", cosine="
                  << kvarn_result.cosine << ")\n";
        if (bf16_top != targets[query_index] || int8_result.top != targets[query_index]) {
            std::cerr << "BF16/INT8 close-margin fixture failed to retain its target at gap "
                      << margins[query_index] << '\n';
            ++failures;
        }
        if (margins[query_index] >= 0.01 && kvarn_result.top != targets[query_index]) {
            std::cerr << "KVarN lost a >=0.01 correlation-margin target at depth "
                      << depths[query_index] * 100.0 << "%\n";
            ++failures;
        }
    }
    failures += storage.verify_guards();
    failures += device_k.verify_guards("KVarN close-margin input K");
    failures += device_v.verify_guards("KVarN close-margin input V");
    failures += decoded_k.verify_guards("KVarN close-margin decoded K");
    failures += decoded_v.verify_guards("KVarN close-margin decoded V");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "KVarN: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    failures += run_hadamard_case(128, 8);
    failures += run_hadamard_case(256, 8);
    constexpr std::array profiles{
        InputProfile::Balanced,
        InputProfile::Gaussian,
        InputProfile::Outlier,
        InputProfile::DynamicRange,
        InputProfile::NearConstant,
    };
    for (const InputProfile profile : profiles) {
        failures += run_codec_case(128, 128, 2, profile);
        failures += run_codec_case(256, 64, 2, profile);
    }
    failures += run_close_margin_retrieval_case();
    if (failures != 0) {
        std::cerr << "KVarN failures=" << failures << '\n';
        return 1;
    }
    std::cout << "KVarN: PASS\n";
    return 0;
}
