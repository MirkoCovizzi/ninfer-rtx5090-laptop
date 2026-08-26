#include "ninfer/ops/kvarn.h"
#include "ninfer/ops/kvarn_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD     = ops::kKvarnHeadDim;
constexpr int kGroup = ops::kKvarnGroup;
constexpr int kTiles = 2;
constexpr int kTileElements = kD * kGroup;

std::uint16_t f32_to_f16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    std::uint32_t mantissa   = bits & 0x007fffffU;
    int exponent             = static_cast<int>((bits >> 23) & 0xffU) - 127;
    if (((bits >> 23) & 0xffU) == 0xffU) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
    }
    if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (exponent >= -14) {
        std::uint32_t half_exponent = static_cast<std::uint32_t>(exponent + 15);
        std::uint32_t rounded       = mantissa + 0x00000fffU + ((mantissa >> 13) & 1U);
        if ((rounded & 0x00800000U) != 0) {
            rounded = 0;
            if (++half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
        }
        return static_cast<std::uint16_t>(sign | (half_exponent << 10) | (rounded >> 13));
    }
    if (exponent < -24) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x00800000U;
    const int shift             = -exponent - 14;
    std::uint32_t half_mantissa = mantissa >> (shift + 13);
    const std::uint32_t remain  = mantissa & ((1U << (shift + 13)) - 1U);
    const std::uint32_t halfway = 1U << (shift + 12);
    if (remain > halfway || (remain == halfway && (half_mantissa & 1U) != 0)) {
        ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
}

float f16_to_f32(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
    const std::uint32_t exp  = (value >> 10) & 0x1fU;
    std::uint32_t mantissa   = value & 0x03ffU;
    std::uint32_t bits;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            bits = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                   ((mantissa & 0x03ffU) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exp + 112U) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double sample_std(const std::vector<double>& matrix, int rows, int cols, bool column, int index) {
    const int count = column ? rows : cols;
    double sum = 0.0;
    double sum_squared = 0.0;
    for (int item = 0; item < count; ++item) {
        const int row = column ? item : index;
        const int col = column ? index : item;
        const double value = matrix[static_cast<std::size_t>(row) * cols + col];
        sum += value;
        sum_squared += value * value;
    }
    const double variance = (sum_squared - sum * sum / count) / (count - 1);
    return std::sqrt(std::max(variance, 0.0));
}

double imbalance(const std::vector<double>& matrix, int rows, int cols) {
    double row_min = std::numeric_limits<double>::infinity();
    double row_max = 0.0;
    double col_min = std::numeric_limits<double>::infinity();
    double col_max = 0.0;
    for (int row = 0; row < rows; ++row) {
        const double deviation = sample_std(matrix, rows, cols, false, row);
        row_min = std::min(row_min, deviation);
        row_max = std::max(row_max, deviation);
    }
    for (int col = 0; col < cols; ++col) {
        const double deviation = sample_std(matrix, rows, cols, true, col);
        col_min = std::min(col_min, deviation);
        col_max = std::max(col_max, deviation);
    }
    return row_max / std::max(row_min, 1.0e-8) + col_max / std::max(col_min, 1.0e-8);
}

struct Balanced {
    std::vector<double> values;
    std::vector<double> column_scale;
    std::vector<double> row_scale;
};

Balanced balance(const std::vector<double>& input, int rows, int cols) {
    std::vector<double> log_column(cols, 0.0);
    std::vector<double> log_row(rows, 0.0);
    std::vector<double> best_column(log_column);
    std::vector<double> best_row(log_row);
    std::vector<double> current(input);
    double best = imbalance(current, rows, cols);

    const auto rebuild = [&] {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                current[static_cast<std::size_t>(row) * cols + col] =
                    input[static_cast<std::size_t>(row) * cols + col] /
                    std::exp(log_row[row] + log_column[col]);
            }
        }
    };
    for (int iteration = 0; iteration < ops::kKvarnIterations; ++iteration) {
        for (int col = 0; col < cols; ++col) {
            const double deviation = std::clamp(sample_std(current, rows, cols, true, col),
                                                1.0e-3, 1.0e3);
            log_column[col] = std::clamp(log_column[col] + std::log(deviation), -0.3, 10.0);
        }
        rebuild();
        for (int row = 0; row < rows; ++row) {
            const double deviation = std::clamp(sample_std(current, rows, cols, false, row),
                                                1.0e-3, 1.0e3);
            log_row[row] = std::clamp(log_row[row] + std::log(deviation), -0.3, 10.0);
        }
        rebuild();
        const double candidate = imbalance(current, rows, cols);
        if (candidate <= best) {
            best = candidate;
            best_column = log_column;
            best_row = log_row;
        }
    }

    Balanced result{input, std::vector<double>(cols), std::vector<double>(rows)};
    for (int col = 0; col < cols; ++col) result.column_scale[col] = std::exp(best_column[col]);
    for (int row = 0; row < rows; ++row) result.row_scale[row] = std::exp(best_row[row]);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            result.values[static_cast<std::size_t>(row) * cols + col] /=
                result.row_scale[row] * result.column_scale[col];
        }
    }
    return result;
}

std::vector<float> make_input(std::uint32_t seed, bool value) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    std::vector<float> result(static_cast<std::size_t>(kTileElements) * kTiles);
    for (int tile = 0; tile < kTiles; ++tile) {
        for (int token = 0; token < kGroup; ++token) {
            const float token_scale = std::exp((static_cast<float>((token * 17 + 5 * tile) % 23) -
                                                11.0F) * 0.055F);
            for (int d = 0; d < kD; ++d) {
                const float channel_scale =
                    std::exp((static_cast<float>((d * 13 + 7 * tile) % 31) - 15.0F) * 0.045F);
                float x = distribution(generator) * token_scale * channel_scale;
                if ((d + 19 * token + 11 * tile) % 509 == 0) { x *= value ? 4.0F : 6.0F; }
                const std::size_t index = static_cast<std::size_t>(tile) * kTileElements +
                                          static_cast<std::size_t>(token) * kD + d;
                result[index] = bf16_to_f32(f32_to_bf16(x));
            }
        }
    }
    return result;
}

std::vector<double> codec_oracle(const std::vector<float>& input, bool key) {
    std::vector<double> result(input.size());
    for (int tile = 0; tile < kTiles; ++tile) {
        const std::size_t base = static_cast<std::size_t>(tile) * kTileElements;
        const int rows = key ? kD : kGroup;
        const int cols = key ? kGroup : kD;
        std::vector<double> matrix(kTileElements);
        for (int token = 0; token < kGroup; ++token) {
            for (int d = 0; d < kD; ++d) {
                const int row = key ? d : token;
                const int col = key ? token : d;
                matrix[static_cast<std::size_t>(row) * cols + col] = input[base + token * kD + d];
            }
        }
        const Balanced balanced = balance(matrix, rows, cols);
        const int qmax = key ? 15 : 3;
        for (int row = 0; row < rows; ++row) {
            const auto first = balanced.values.begin() + static_cast<std::ptrdiff_t>(row) * cols;
            const auto last = first + cols;
            const auto [minimum_it, maximum_it] = std::minmax_element(first, last);
            const double minimum = *minimum_it;
            const double scale = std::max((*maximum_it - minimum) / qmax, 1.0e-10);
            const float absorbed_scale = f16_to_f32(f32_to_f16(
                static_cast<float>(balanced.row_scale[row] * scale)));
            const float absorbed_zero = f16_to_f32(f32_to_f16(
                static_cast<float>(balanced.row_scale[row] * minimum)));
            for (int col = 0; col < cols; ++col) {
                const int code = std::clamp(static_cast<int>(std::nearbyint(
                                                (balanced.values[static_cast<std::size_t>(row) *
                                                                     cols + col] - minimum) /
                                                scale)),
                                            0, qmax);
                const float other_scale = f16_to_f32(f32_to_f16(
                    static_cast<float>(balanced.column_scale[col])));
                const float decoded =
                    std::fma(static_cast<float>(code), absorbed_scale, absorbed_zero) * other_scale;
                const int token = key ? col : row;
                const int d = key ? row : col;
                result[base + token * kD + d] = decoded;
            }
        }
    }
    return result;
}

struct DeviceStorage {
    DeviceBuffer k_codes{static_cast<std::size_t>(kGroup / 2) * kD * kTiles};
    DeviceBuffer k_scales{static_cast<std::size_t>(kD) * kTiles * sizeof(std::uint16_t)};
    DeviceBuffer k_zeros{static_cast<std::size_t>(kD) * kTiles * sizeof(std::uint16_t)};
    DeviceBuffer k_token_scales{static_cast<std::size_t>(kGroup) * kTiles * sizeof(std::uint16_t)};
    DeviceBuffer v_codes{static_cast<std::size_t>(kD / 4) * kGroup * kTiles};
    DeviceBuffer v_channel_scales{static_cast<std::size_t>(kD) * kTiles * sizeof(std::uint16_t)};
    DeviceBuffer v_token_scales{static_cast<std::size_t>(kGroup) * kTiles * sizeof(std::uint16_t)};
    DeviceBuffer v_token_zeros{static_cast<std::size_t>(kGroup) * kTiles * sizeof(std::uint16_t)};

    ops::KvarnTileStorage view() {
        return {
            Tensor(k_codes.p, DType::U8, {kGroup / 2, kD, kTiles}),
            Tensor(k_scales.p, DType::FP16, {kD, kTiles}),
            Tensor(k_zeros.p, DType::FP16, {kD, kTiles}),
            Tensor(k_token_scales.p, DType::FP16, {kGroup, kTiles}),
            Tensor(v_codes.p, DType::U8, {kD / 4, kGroup, kTiles}),
            Tensor(v_channel_scales.p, DType::FP16, {kD, kTiles}),
            Tensor(v_token_scales.p, DType::FP16, {kGroup, kTiles}),
            Tensor(v_token_zeros.p, DType::FP16, {kGroup, kTiles}),
        };
    }
};

int compare_profile(const char* label, const std::vector<double>& actual,
                    const std::vector<double>& expected, double limit) {
    double error_squared = 0.0;
    double reference_squared = 0.0;
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = std::abs(actual[index] - expected[index]);
        error_squared += error * error;
        reference_squared += expected[index] * expected[index];
        maximum = std::max(maximum, error);
    }
    const double relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1.0e-30));
    if (relative_l2 > limit) {
        std::cerr << label << ": relative_l2=" << relative_l2 << " limit=" << limit
                  << " max_abs=" << maximum << '\n';
        return 1;
    }
    return 0;
}

int run_codec_case() {
    const std::vector<float> key = make_input(0x4b41524eU, false);
    const std::vector<float> value = make_input(0x56324736U, true);
    const std::vector<double> expected_key = codec_oracle(key, true);
    const std::vector<double> expected_value = codec_oracle(value, false);
    DeviceBuffer device_key = to_device_bf16(key);
    DeviceBuffer device_value = to_device_bf16(value);
    DeviceStorage storage;
    DeviceBuffer decoded_key(key.size() * sizeof(float));
    DeviceBuffer decoded_value(value.size() * sizeof(float));

    Tensor key_tensor(device_key.p, DType::BF16, {kD, kGroup, kTiles});
    Tensor value_tensor(device_value.p, DType::BF16, {kD, kGroup, kTiles});
    ops::KvarnTileStorage storage_view = storage.view();
    ops::kvarn_store(key_tensor, value_tensor, storage_view, nullptr);
    Tensor decoded_key_tensor(decoded_key.p, DType::FP32, {kD, kGroup, kTiles});
    Tensor decoded_value_tensor(decoded_value.p, DType::FP32, {kD, kGroup, kTiles});
    ops::kvarn_dequant(storage_view, decoded_key_tensor, decoded_value_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> actual_key = from_device_f32(decoded_key, key.size());
    const std::vector<double> actual_value = from_device_f32(decoded_value, value.size());
    int failures = 0;
    failures += compare_profile("KVarN K official oracle", actual_key, expected_key, 3.0e-4);
    failures += compare_profile("KVarN V official oracle", actual_value, expected_value, 3.0e-4);

    const auto k_codes = from_device<std::uint8_t>(storage.k_codes, storage.k_codes.bytes);
    const auto k_scales = from_device<std::uint16_t>(storage.k_scales, kD * kTiles);
    const auto k_zeros = from_device<std::uint16_t>(storage.k_zeros, kD * kTiles);
    const auto k_tokens = from_device<std::uint16_t>(storage.k_token_scales, kGroup * kTiles);
    const auto v_codes = from_device<std::uint8_t>(storage.v_codes, storage.v_codes.bytes);
    const auto v_channels = from_device<std::uint16_t>(storage.v_channel_scales, kD * kTiles);
    const auto v_scales = from_device<std::uint16_t>(storage.v_token_scales, kGroup * kTiles);
    const auto v_zeros = from_device<std::uint16_t>(storage.v_token_zeros, kGroup * kTiles);
    std::vector<double> represented_key(key.size());
    std::vector<double> represented_value(value.size());
    for (int tile = 0; tile < kTiles; ++tile) {
        for (int token = 0; token < kGroup; ++token) {
            for (int d = 0; d < kD; ++d) {
                const std::size_t output = static_cast<std::size_t>(tile) * kTileElements +
                                           static_cast<std::size_t>(token) * kD + d;
                const std::size_t kb = (static_cast<std::size_t>(tile) * kD + d) *
                                           (kGroup / 2) +
                                       token / 2;
                const int kc = (k_codes[kb] >> (4 * (token & 1))) & 15;
                represented_key[output] =
                    std::fma(static_cast<float>(kc), f16_to_f32(k_scales[tile * kD + d]),
                             f16_to_f32(k_zeros[tile * kD + d])) *
                    f16_to_f32(k_tokens[tile * kGroup + token]);
                const std::size_t vb = (static_cast<std::size_t>(tile) * kGroup + token) *
                                           (kD / 4) +
                                       d / 4;
                const int vc = (v_codes[vb] >> (2 * (d & 3))) & 3;
                represented_value[output] =
                    std::fma(static_cast<float>(vc), f16_to_f32(v_scales[tile * kGroup + token]),
                             f16_to_f32(v_zeros[tile * kGroup + token])) *
                    f16_to_f32(v_channels[tile * kD + d]);
            }
        }
    }
    failures += compare_profile("KVarN K stored-bit decode", actual_key, represented_key, 2.0e-7);
    failures +=
        compare_profile("KVarN V stored-bit decode", actual_value, represented_value, 2.0e-7);
    return failures;
}

int run_hadamard_case() {
    constexpr int kVectors = 11;
    std::vector<float> input(static_cast<std::size_t>(kD) * kVectors);
    fill_uniform(input, 0x48414441U, -3.0F, 3.0F);
    round_to_bf16(input);
    std::vector<double> expected(input.size());
    for (int vector = 0; vector < kVectors; ++vector) {
        for (int row = 0; row < kD; ++row) {
            double sum = 0.0;
            for (int col = 0; col < kD; ++col) {
                const bool negative = (__builtin_popcount(static_cast<unsigned>(row & col)) & 1) != 0;
                sum += (negative ? -1.0 : 1.0) * input[static_cast<std::size_t>(vector) * kD + col];
            }
            expected[static_cast<std::size_t>(vector) * kD + row] = sum / 16.0;
        }
    }
    DeviceBuffer source = to_device_bf16(input);
    DeviceBuffer destination(input.size() * sizeof(std::uint16_t));
    Tensor source_tensor(source.p, DType::BF16, {kD, kVectors});
    Tensor destination_tensor(destination.p, DType::BF16, {kD, kVectors});
    ops::kvarn_hadamard(source_tensor, destination_tensor, nullptr);
    cuda_synchronize();
    return compare_profile("KVarN Hadamard", from_device_bf16(destination, input.size()), expected,
                           2.0e-3);
}

std::vector<float> make_cache_values(int tokens, std::uint32_t seed, int heads = 1) {
    std::vector<float> values(static_cast<std::size_t>(kD) * heads * tokens);
    fill_uniform(values, seed, -2.0F, 2.0F);
    round_to_bf16(values);
    return values;
}

template <int Heads>
struct CacheFixture {
    static constexpr int kHeads = Heads;
    static constexpr int kPages = 4;
    DeviceBuffer records{static_cast<std::size_t>(ops::kKvarnRecordBytes) * kHeads * kPages};
    DeviceBuffer tail_k{static_cast<std::size_t>(kD) * kGroup * kHeads *
                        ops::kKvarnTailSlots * sizeof(std::uint16_t)};
    DeviceBuffer tail_v{static_cast<std::size_t>(kD) * kGroup * kHeads *
                        ops::kKvarnTailSlots * sizeof(std::uint16_t)};
    DeviceBuffer markers{ops::kKvarnTailSlots * sizeof(std::int32_t)};
    DeviceBuffer block_tables{kPages * sizeof(std::int32_t)};

    CacheFixture() {
        records.fill();
        tail_k.fill();
        tail_v.fill();
        markers.copy_from_host(std::vector<std::int32_t>(ops::kKvarnTailSlots, -1).data(),
                               markers.bytes);
        const std::vector<std::int32_t> table{0, 1, 2, 3};
        block_tables.copy_from_host(table.data(), block_tables.bytes);
    }

    ops::KvarnPagedBatchLayerView view() {
        return {
            .records = Tensor(records.p, DType::U8,
                              {ops::kKvarnRecordBytes / kGroup, kGroup, kHeads, kPages}),
            .tail_k = Tensor(tail_k.p, DType::BF16,
                             {kD, kGroup, kHeads * ops::kKvarnTailSlots, 1}),
            .tail_v = Tensor(tail_v.p, DType::BF16,
                             {kD, kGroup, kHeads * ops::kKvarnTailSlots, 1}),
            .tail_logical_pages =
                Tensor(markers.p, DType::I32, {ops::kKvarnTailSlots, 1}),
            .block_tables = Tensor(block_tables.p, DType::I32, {kPages, 1}),
            .num_kv_heads = kHeads,
        };
    }

    ops::KvarnPagedLayerView layer_view() {
        return {
            .records = Tensor(records.p, DType::U8,
                              {ops::kKvarnRecordBytes / kGroup, kGroup, kHeads, kPages}),
            .tail_k = Tensor(tail_k.p, DType::BF16,
                             {kD, kGroup, kHeads * ops::kKvarnTailSlots}),
            .tail_v = Tensor(tail_v.p, DType::BF16,
                             {kD, kGroup, kHeads * ops::kKvarnTailSlots}),
            .tail_logical_pages = Tensor(markers.p, DType::I32, {ops::kKvarnTailSlots}),
            .block_table = Tensor(block_tables.p, DType::I32, {kPages}),
            .num_kv_heads = kHeads,
        };
    }
};

template <int Heads>
struct BatchCacheFixture {
    static constexpr int kHeads = Heads;
    static constexpr int kRows = 2;
    static constexpr int kLogicalPages = 6;
    static constexpr int kPhysicalPages = kRows * kLogicalPages;
    DeviceBuffer records{static_cast<std::size_t>(ops::kKvarnRecordBytes) * kHeads *
                         kPhysicalPages};
    DeviceBuffer tail_k{static_cast<std::size_t>(kD) * kGroup * kHeads *
                        ops::kKvarnTailSlots * kRows * sizeof(std::uint16_t)};
    DeviceBuffer tail_v{static_cast<std::size_t>(kD) * kGroup * kHeads *
                        ops::kKvarnTailSlots * kRows * sizeof(std::uint16_t)};
    DeviceBuffer markers{ops::kKvarnTailSlots * kRows * sizeof(std::int32_t)};
    DeviceBuffer block_tables{kLogicalPages * kRows * sizeof(std::int32_t)};
    std::vector<std::int32_t> host_block_tables;

    BatchCacheFixture() : host_block_tables(kPhysicalPages) {
        records.fill();
        tail_k.fill();
        tail_v.fill();
        markers.copy_from_host(
            std::vector<std::int32_t>(ops::kKvarnTailSlots * kRows, -1).data(), markers.bytes);
        for (int physical = 0; physical < kPhysicalPages; ++physical) {
            host_block_tables[physical] = physical;
        }
        block_tables.copy_from_host(host_block_tables.data(), block_tables.bytes);
    }

    ops::KvarnPagedBatchLayerView view() {
        return {
            .records = Tensor(records.p, DType::U8,
                              {ops::kKvarnRecordBytes / kGroup, kGroup, kHeads, kPhysicalPages}),
            .tail_k = Tensor(tail_k.p, DType::BF16,
                             {kD, kGroup, kHeads * ops::kKvarnTailSlots, kRows}),
            .tail_v = Tensor(tail_v.p, DType::BF16,
                             {kD, kGroup, kHeads * ops::kKvarnTailSlots, kRows}),
            .tail_logical_pages =
                Tensor(markers.p, DType::I32, {ops::kKvarnTailSlots, kRows}),
            .block_tables = Tensor(block_tables.p, DType::I32, {kLogicalPages, kRows}),
            .num_kv_heads = kHeads,
        };
    }
};

float decode_cache_value(const std::vector<std::uint8_t>& records,
                           const std::vector<std::uint16_t>& tail,
                           const std::vector<std::int32_t>& markers, bool key, int position, int head,
                           int heads, int d, int table_row = 0, int logical_pages = 4,
                           const std::vector<std::int32_t>* block_tables = nullptr) {
    const int page = position / kGroup;
    const int token = position % kGroup;
    for (int slot = 0; slot < ops::kKvarnTailSlots; ++slot) {
        if (markers[slot + ops::kKvarnTailSlots * table_row] == page) {
            const std::size_t tail_index =
                static_cast<std::size_t>(d) + static_cast<std::size_t>(kD) *
                                                  (token + kGroup *
                                                               (head + heads *
                                                                           (slot +
                                                                            ops::kKvarnTailSlots *
                                                                                table_row)));
            return bf16_to_f32(tail[tail_index]);
        }
    }
    const int physical = block_tables == nullptr
                             ? page
                             : (*block_tables)[page + logical_pages * table_row];
    const std::uint8_t* record = records.data() +
                                 (static_cast<std::size_t>(physical) * heads + head) *
                                     ops::kKvarnRecordBytes;
    if (key) {
        const std::uint8_t packed =
            record[ops::kKvarnKPackedOffset + d * (kGroup / 2) + token / 2];
        const int code = (packed >> (4 * (token & 1))) & 15;
        std::uint16_t scale_bits;
        std::uint16_t zero_bits;
        std::uint16_t token_bits;
        std::memcpy(&scale_bits, record + ops::kKvarnKScaleOffset + 2 * d, 2);
        std::memcpy(&zero_bits, record + ops::kKvarnKZeroOffset + 2 * d, 2);
        std::memcpy(&token_bits, record + ops::kKvarnKTokenScaleOffset + 2 * token, 2);
        return std::fma(static_cast<float>(code), f16_to_f32(scale_bits),
                        f16_to_f32(zero_bits)) *
               f16_to_f32(token_bits);
    }
    const std::uint8_t packed =
        record[ops::kKvarnVPackedOffset + token * (kD / 4) + d / 4];
    const int code = (packed >> (2 * (d & 3))) & 3;
    std::uint16_t channel_bits;
    std::uint16_t scale_bits;
    std::uint16_t zero_bits;
    std::memcpy(&channel_bits, record + ops::kKvarnVChannelScaleOffset + 2 * d, 2);
    std::memcpy(&scale_bits, record + ops::kKvarnVTokenScaleOffset + 2 * token, 2);
    std::memcpy(&zero_bits, record + ops::kKvarnVTokenZeroOffset + 2 * token, 2);
    return std::fma(static_cast<float>(code), f16_to_f32(scale_bits), f16_to_f32(zero_bits)) *
           f16_to_f32(channel_bits);
}

template <int Heads>
void append_cache(CacheFixture<Heads>& cache, const std::vector<float>& key,
                   const std::vector<float>& value, int first_position, bool provisional) {
    const int width = static_cast<int>(key.size() / (kD * Heads));
    std::vector<std::int32_t> positions(width);
    for (int index = 0; index < width; ++index) positions[index] = first_position + index;
    DeviceBuffer device_key = to_device_bf16(key);
    DeviceBuffer device_value = to_device_bf16(value);
    DeviceBuffer device_positions = to_device(positions);
    DeviceBuffer rows = to_device(std::vector<std::int32_t>{0});
    Tensor key_tensor(device_key.p, DType::BF16, {kD, Heads, width, 1});
    Tensor value_tensor(device_value.p, DType::BF16, {kD, Heads, width, 1});
    Tensor position_tensor(device_positions.p, DType::I32, {width, 1});
    Tensor row_tensor(rows.p, DType::I32, {1});
    ops::kvarn_kv_append(key_tensor, value_tensor, position_tensor, Tensor{}, row_tensor,
                         cache.view(), provisional, nullptr);
    cuda_synchronize();
}

template <int Heads>
void append_cache_row(BatchCacheFixture<Heads>& cache, const std::vector<float>& key,
                      const std::vector<float>& value, int first_position, int table_row) {
    const int width = static_cast<int>(key.size() / (kD * Heads));
    std::vector<std::int32_t> host_positions(width);
    for (int index = 0; index < width; ++index) host_positions[index] = first_position + index;
    DeviceBuffer device_key = to_device_bf16(key);
    DeviceBuffer device_value = to_device_bf16(value);
    DeviceBuffer positions = to_device(host_positions);
    DeviceBuffer rows = to_device(std::vector<std::int32_t>{table_row});
    Tensor key_tensor(device_key.p, DType::BF16, {kD, Heads, width, 1});
    Tensor value_tensor(device_value.p, DType::BF16, {kD, Heads, width, 1});
    Tensor position_tensor(positions.p, DType::I32, {width, 1});
    Tensor rows_tensor(rows.p, DType::I32, {1});
    ops::kvarn_kv_append(key_tensor, value_tensor, position_tensor, Tensor{}, rows_tensor,
                         cache.view(), false, nullptr);
    cuda_synchronize();
}

template <int Heads>
int run_cached_attention_case(CacheFixture<Heads>& cache, int query_heads, int first_position,
                              int width, const char* label) {
    std::vector<float> query = make_cache_values(query_heads * width, 0x4001U + query_heads);
    DeviceBuffer device_query = to_device_bf16(query);
    DeviceBuffer output(query.size() * sizeof(std::uint16_t));
    std::vector<std::int32_t> query_positions(width);
    for (int column = 0; column < width; ++column) {
        query_positions[column] = first_position + column;
    }
    DeviceBuffer positions = to_device(query_positions);
    DeviceBuffer rows = to_device(std::vector<std::int32_t>{0});
    Tensor query_tensor(device_query.p, DType::BF16, {kD, query_heads, width, 1});
    Tensor output_tensor(output.p, DType::BF16, {kD, query_heads, width, 1});
    Tensor position_tensor(positions.p, DType::I32, {width, 1});
    Tensor rows_tensor(rows.p, DType::I32, {1});
    WorkspaceArena workspace(std::max<std::size_t>(
        1, ops::kvarn_attention_workspace_capacity_bytes(
               query_heads, {1, static_cast<std::uint32_t>(first_position + width)}, 1, width,
               width)));
    ops::kvarn_attention_cached(query_tensor, position_tensor, rows_tensor, 0.0625F, cache.view(),
                                {1, static_cast<std::uint32_t>(first_position + width)}, workspace,
                                output_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> rotated_query(query.size());
    for (int column = 0; column < width; ++column) {
        for (int head = 0; head < query_heads; ++head) {
            for (int row = 0; row < kD; ++row) {
                double sum = 0.0;
                for (int col = 0; col < kD; ++col) {
                    const bool negative =
                        (__builtin_popcount(static_cast<unsigned>(row & col)) & 1) != 0;
                    const std::size_t input_index =
                        static_cast<std::size_t>(col) + static_cast<std::size_t>(kD) *
                                                            (head + query_heads * column);
                    sum += (negative ? -1.0 : 1.0) * query[input_index];
                }
                const std::size_t output_index =
                    static_cast<std::size_t>(row) + static_cast<std::size_t>(kD) *
                                                        (head + query_heads * column);
                rotated_query[output_index] = f32_to_bf16(static_cast<float>(sum / 16.0));
            }
        }
    }
    const auto record_values = from_device<std::uint8_t>(cache.records, cache.records.bytes);
    const auto tail_key = from_device<std::uint16_t>(cache.tail_k, cache.tail_k.bytes / 2);
    const auto tail_value = from_device<std::uint16_t>(cache.tail_v, cache.tail_v.bytes / 2);
    const auto marker_values = from_device<std::int32_t>(cache.markers, ops::kKvarnTailSlots);
    std::vector<double> expected(query.size());
    for (int column = 0; column < width; ++column) {
        const int visible = query_positions[column] + 1;
        for (int head = 0; head < query_heads; ++head) {
            std::vector<double> scores(visible);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const std::size_t query_index =
                        static_cast<std::size_t>(d) + static_cast<std::size_t>(kD) *
                                                          (head + query_heads * column);
                    dot += bf16_to_f32(rotated_query[query_index]) *
                           decode_cache_value(record_values, tail_key, marker_values, true,
                                              position, head / (query_heads / Heads), Heads, d);
                }
                scores[position] = dot * 0.0625;
                maximum = std::max(maximum, scores[position]);
            }
            double denominator = 0.0;
            std::vector<double> rotated(kD, 0.0);
            for (int position = 0; position < visible; ++position) {
                const double probability = std::exp(scores[position] - maximum);
                denominator += probability;
                for (int d = 0; d < kD; ++d) {
                    rotated[d] += probability * decode_cache_value(
                                                        record_values, tail_value, marker_values,
                                                        false, position,
                                                        head / (query_heads / Heads), Heads, d);
                }
            }
            for (double& value : rotated) {
                value = bf16_to_f32(f32_to_bf16(static_cast<float>(value / denominator)));
            }
            for (int row = 0; row < kD; ++row) {
                double sum = 0.0;
                for (int col = 0; col < kD; ++col) {
                    const bool negative =
                        (__builtin_popcount(static_cast<unsigned>(row & col)) & 1) != 0;
                    sum += (negative ? -1.0 : 1.0) * rotated[col];
                }
                const std::size_t output_index =
                    static_cast<std::size_t>(row) + static_cast<std::size_t>(kD) *
                                                        (head + query_heads * column);
                expected[output_index] =
                    bf16_to_f32(f32_to_bf16(static_cast<float>(sum / 16.0)));
            }
        }
    }
    int failures =
        compare_profile(label, from_device_bf16(output, query.size()), expected, 8.0e-3);
    if ((width == 6 && Heads == 2) || width == 64) {
        DeviceBuffer original_query = to_device_bf16(query);
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        cuda_check(cudaStreamCreate(&stream), "create KVarN graph stream");
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "begin KVarN capture");
        ops::kvarn_attention_cached(query_tensor, position_tensor, rows_tensor, 0.0625F,
                                    cache.view(),
                                    {1, static_cast<std::uint32_t>(first_position + width)},
                                    workspace, output_tensor, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end KVarN capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                   "instantiate KVarN graph");
        cuda_check(cudaMemcpyAsync(device_query.p, original_query.p, original_query.bytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "restore KVarN graph query");
        cuda_check(cudaGraphLaunch(executable, stream), "launch KVarN graph");
        cuda_synchronize(stream);
        failures += compare_profile("KVarN CUDA Graph attention",
                                    from_device_bf16(output, query.size()), expected, 8.0e-3);
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    return failures;
}

template <int Heads>
int run_batched_attention_case(int query_heads, const char* label) {
    constexpr int Batch = 2;
    constexpr int Width = 6;
    constexpr int GuardBytes = 4096;
    const std::int32_t row_first[Batch] = {253, 70};
    const std::int32_t table_rows[Batch] = {1, 0};

    BatchCacheFixture<Heads> cache;
    append_cache_row(cache, make_cache_values(row_first[0] + Width, 0x8101U + Heads, Heads),
                     make_cache_values(row_first[0] + Width, 0x8201U + Heads, Heads), 0, 0);
    append_cache_row(cache, make_cache_values(row_first[1] + Width, 0x9101U + Heads, Heads),
                     make_cache_values(row_first[1] + Width, 0x9201U + Heads, Heads), 0, 1);
    BatchCacheFixture<Heads> graph_cache;
    append_cache_row(graph_cache,
                     make_cache_values(row_first[0] + Width, 0x8101U + Heads, Heads),
                     make_cache_values(row_first[0] + Width, 0x8201U + Heads, Heads), 0, 0);
    append_cache_row(graph_cache,
                     make_cache_values(row_first[1] + Width, 0x9101U + Heads, Heads),
                     make_cache_values(row_first[1] + Width, 0x9201U + Heads, Heads), 0, 1);

    std::vector<float> query =
        make_cache_values(query_heads * Width * Batch, 0xa001U + query_heads);
    DeviceBuffer device_query = to_device_bf16(query);
    DeviceBuffer original_query = to_device_bf16(query);
    DeviceBuffer output(query.size() * sizeof(std::uint16_t));
    std::vector<std::int32_t> host_positions(Width * Batch);
    for (int batch = 0; batch < Batch; ++batch) {
        for (int column = 0; column < Width; ++column) {
            host_positions[column + Width * batch] = row_first[table_rows[batch]] + column;
        }
    }
    DeviceBuffer positions = to_device(host_positions);
    DeviceBuffer rows = to_device(std::vector<std::int32_t>{table_rows[0], table_rows[1]});
    Tensor query_tensor(device_query.p, DType::BF16, {kD, query_heads, Width, Batch});
    Tensor output_tensor(output.p, DType::BF16, {kD, query_heads, Width, Batch});
    Tensor positions_tensor(positions.p, DType::I32, {Width, Batch});
    Tensor rows_tensor(rows.p, DType::I32, {Batch});
    const ops::GqaExecutionEnvelope envelope{1, 512};
    const std::size_t workspace_bytes = ops::kvarn_attention_workspace_capacity_bytes(
        query_heads, envelope, Batch, Width, Width);
    DeviceBuffer workspace_storage(workspace_bytes + 2 * GuardBytes);
    workspace_storage.fill(0xa5);
    auto* workspace_base = static_cast<std::uint8_t*>(workspace_storage.p) + GuardBytes;
    cuda_check(cudaMemset(workspace_base, 0x41, workspace_bytes),
               "poison B=2 KVarN workspace");
    WorkspaceArena workspace(DeviceSpan{workspace_base, workspace_bytes});

    ops::kvarn_attention_cached(query_tensor, positions_tensor, rows_tensor, 0.0625F, cache.view(),
                                envelope, workspace, output_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> rotated_query(query.size());
    for (int batch = 0; batch < Batch; ++batch) {
        for (int column = 0; column < Width; ++column) {
            for (int head = 0; head < query_heads; ++head) {
                for (int row = 0; row < kD; ++row) {
                    double sum = 0.0;
                    for (int col = 0; col < kD; ++col) {
                        const bool negative =
                            (__builtin_popcount(static_cast<unsigned>(row & col)) & 1) != 0;
                        const std::size_t index =
                            static_cast<std::size_t>(col) + static_cast<std::size_t>(kD) *
                                                                 (head + query_heads *
                                                                             (column +
                                                                              Width * batch));
                        sum += (negative ? -1.0 : 1.0) * query[index];
                    }
                    const std::size_t index =
                        static_cast<std::size_t>(row) + static_cast<std::size_t>(kD) *
                                                             (head + query_heads *
                                                                         (column + Width * batch));
                    rotated_query[index] = f32_to_bf16(static_cast<float>(sum / 16.0));
                }
            }
        }
    }

    const auto record_values = from_device<std::uint8_t>(cache.records, cache.records.bytes);
    const auto tail_key = from_device<std::uint16_t>(cache.tail_k, cache.tail_k.bytes / 2);
    const auto tail_value = from_device<std::uint16_t>(cache.tail_v, cache.tail_v.bytes / 2);
    const auto marker_values =
        from_device<std::int32_t>(cache.markers, ops::kKvarnTailSlots * Batch);
    std::vector<double> expected(query.size());
    for (int batch = 0; batch < Batch; ++batch) {
        for (int column = 0; column < Width; ++column) {
            const int visible = host_positions[column + Width * batch] + 1;
            for (int head = 0; head < query_heads; ++head) {
                std::vector<double> scores(visible);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int position = 0; position < visible; ++position) {
                    double dot = 0.0;
                    for (int d = 0; d < kD; ++d) {
                        const std::size_t query_index =
                            static_cast<std::size_t>(d) + static_cast<std::size_t>(kD) *
                                                               (head + query_heads *
                                                                           (column +
                                                                            Width * batch));
                        dot += bf16_to_f32(rotated_query[query_index]) *
                               decode_cache_value(
                                   record_values, tail_key, marker_values, true, position,
                                   head / (query_heads / Heads), Heads, d, table_rows[batch],
                                   BatchCacheFixture<Heads>::kLogicalPages,
                                   &cache.host_block_tables);
                    }
                    scores[position] = dot * 0.0625;
                    maximum = std::max(maximum, scores[position]);
                }
                double denominator = 0.0;
                std::vector<double> rotated(kD, 0.0);
                for (int position = 0; position < visible; ++position) {
                    const double probability = std::exp(scores[position] - maximum);
                    denominator += probability;
                    for (int d = 0; d < kD; ++d) {
                        rotated[d] += probability * decode_cache_value(
                                                            record_values, tail_value,
                                                            marker_values, false, position,
                                                            head / (query_heads / Heads), Heads, d,
                                                            table_rows[batch],
                                                            BatchCacheFixture<Heads>::kLogicalPages,
                                                            &cache.host_block_tables);
                    }
                }
                for (double& item : rotated) {
                    item = bf16_to_f32(f32_to_bf16(static_cast<float>(item / denominator)));
                }
                for (int row = 0; row < kD; ++row) {
                    double sum = 0.0;
                    for (int col = 0; col < kD; ++col) {
                        const bool negative =
                            (__builtin_popcount(static_cast<unsigned>(row & col)) & 1) != 0;
                        sum += (negative ? -1.0 : 1.0) * rotated[col];
                    }
                    const std::size_t output_index =
                        static_cast<std::size_t>(row) + static_cast<std::size_t>(kD) *
                                                             (head + query_heads *
                                                                         (column + Width * batch));
                    expected[output_index] =
                        bf16_to_f32(f32_to_bf16(static_cast<float>(sum / 16.0)));
                }
            }
        }
    }

    const auto check_canaries = [&]() {
        std::vector<std::uint8_t> guards(2 * GuardBytes);
        workspace_storage.copy_to_host(guards.data(), GuardBytes);
        workspace_storage.copy_to_host(guards.data() + GuardBytes, GuardBytes,
                                       GuardBytes + workspace_bytes);
        if (std::all_of(guards.begin(), guards.end(),
                        [](std::uint8_t value) { return value == 0xa5; })) {
            return 0;
        }
        std::cerr << label << " workspace canary overwritten\n";
        return 1;
    };

    int failures =
        compare_profile(label, from_device_bf16(output, query.size()), expected, 8.0e-3);
    failures += check_canaries();

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create B=2 KVarN graph stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin B=2 KVarN capture");
    ops::kvarn_attention_cached(query_tensor, positions_tensor, rows_tensor, 0.0625F,
                                graph_cache.view(), envelope, workspace, output_tensor, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end B=2 KVarN capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate B=2 KVarN graph");
    cuda_check(cudaMemcpyAsync(device_query.p, original_query.p, original_query.bytes,
                               cudaMemcpyDeviceToDevice, stream),
               "restore B=2 KVarN graph query");
    cuda_check(cudaGraphLaunch(executable, stream), "launch B=2 KVarN graph");
    cuda_synchronize(stream);
    failures += compare_profile((std::string(label) + " CUDA Graph").c_str(),
                                from_device_bf16(output, query.size()), expected, 8.0e-3);
    failures += check_canaries();
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return failures;
}

int run_cache_lifecycle_case() {
    CacheFixture<2> cache;
    append_cache(cache, make_cache_values(184, 0x1001U, 2),
                  make_cache_values(184, 0x1002U, 2), 0, false);
    append_cache(cache, make_cache_values(16, 0x2001U, 2),
                  make_cache_values(16, 0x2002U, 2), 184, true);

    std::vector<std::int32_t> provisional_positions(16);
    for (int index = 0; index < 16; ++index) provisional_positions[index] = 184 + index;
    DeviceBuffer positions = to_device(provisional_positions);
    DeviceBuffer accepted = to_device(std::vector<std::int32_t>{8});
    DeviceBuffer rows = to_device(std::vector<std::int32_t>{0});
    Tensor positions_tensor(positions.p, DType::I32, {16, 1});
    Tensor accepted_tensor(accepted.p, DType::I32, {1});
    Tensor rows_tensor(rows.p, DType::I32, {1});
    ops::kvarn_commit_pages(positions_tensor, accepted_tensor, rows_tensor, cache.view(), nullptr);
    cuda_synchronize();

    append_cache(cache, make_cache_values(8, 0x3001U, 2), make_cache_values(8, 0x3002U, 2), 192,
                 false);
    const auto marker_values =
        from_device<std::int32_t>(cache.markers, ops::kKvarnTailSlots);
    int failures = 0;
    const std::vector<std::int32_t> expected_markers{0, 1, 3, -1};
    failures += verify_exact("KVarN sink/tail lifecycle", marker_values, expected_markers);

    const auto record_values =
        from_device<std::uint8_t>(cache.records, cache.records.bytes);
    const auto tail_key = from_device<std::uint16_t>(cache.tail_k, cache.tail_k.bytes / 2);
    const auto tail_value = from_device<std::uint16_t>(cache.tail_v, cache.tail_v.bytes / 2);
    failures += run_cached_attention_case(cache, 16, 194, 6, "KVarN H16/KV2 width-6 attention");

    ops::kvarn_restore_tail(150, cache.layer_view(), nullptr);
    cuda_synchronize();
    const auto restored_markers =
        from_device<std::int32_t>(cache.markers, ops::kKvarnTailSlots);
    failures += verify_exact("KVarN restored tail markers", restored_markers,
                             std::vector<std::int32_t>{0, 1, 2, -1});
    const auto restored_key = from_device<std::uint16_t>(cache.tail_k, cache.tail_k.bytes / 2);
    const auto restored_value = from_device<std::uint16_t>(cache.tail_v, cache.tail_v.bytes / 2);
    for (int token = 0; token < kGroup; ++token) {
        for (int d = 0; d < kD; ++d) {
            const std::size_t index = static_cast<std::size_t>(d) +
                                      static_cast<std::size_t>(kD) *
                                          (token + kGroup * (2 * 2));
            const std::uint16_t expected_key = f32_to_bf16(decode_cache_value(
                record_values, tail_key, marker_values, true, 128 + token, 0, 2, d));
            const std::uint16_t expected_value = f32_to_bf16(decode_cache_value(
                record_values, tail_value, marker_values, false, 128 + token, 0, 2, d));
            if (restored_key[index] != expected_key || restored_value[index] != expected_value) {
                std::cerr << "KVarN restored tail mismatch at token=" << token << " d=" << d
                          << '\n';
                ++failures;
                token = kGroup;
                break;
            }
        }
    }
    return failures;
}

int run_27b_attention_case() {
    CacheFixture<4> cache;
    append_cache(cache, make_cache_values(200, 0x5001U, 4),
                 make_cache_values(200, 0x5002U, 4), 0, false);
    return run_cached_attention_case(cache, 24, 136, 64, "KVarN H24/KV4 tiled attention");
}

int run_35b_attention_case() {
    CacheFixture<2> cache;
    append_cache(cache, make_cache_values(200, 0x6001U, 2),
                 make_cache_values(200, 0x6002U, 2), 0, false);
    return run_cached_attention_case(cache, 16, 136, 64, "KVarN H16/KV2 tiled attention");
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_codec_case();
    failures += run_hadamard_case();
    failures += run_cache_lifecycle_case();
    failures += run_27b_attention_case();
    failures += run_35b_attention_case();
    failures += run_batched_attention_case<4>(24, "KVarN H24/KV4 B=2 attention");
    failures += run_batched_attention_case<2>(16, "KVarN H16/KV2 B=2 attention");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " kvarn correctness\n";
    return failures == 0 ? 0 : 1;
}
