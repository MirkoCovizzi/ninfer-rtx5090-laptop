#include "ninfer/ops/kvarn.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace ninfer;

namespace {

constexpr int kD = ops::kKvarnHeadDim;
constexpr int kGroup = ops::kKvarnGroup;

struct Options {
    int tiles = 64;
    int warmup = 5;
    int repeat = 30;
};

int parse_positive(const char* text, const char* label) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > 1'000'000) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--tiles" && index + 1 < argc) {
            options.tiles = parse_positive(argv[++index], "tiles");
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup = parse_positive(argv[++index], "warmup");
        } else if (argument == "--repeat" && index + 1 < argc) {
            options.repeat = parse_positive(argv[++index], "repeat");
        } else {
            throw std::invalid_argument("usage: ninfer_kvarn_codec_bench [--tiles N] [--warmup N] [--repeat N]");
        }
    }
    return options;
}

struct Storage {
    explicit Storage(int tiles)
        : k_codes(static_cast<std::size_t>(kGroup / 2) * kD * tiles),
          k_scales(static_cast<std::size_t>(kD) * tiles * 2),
          k_zeros(static_cast<std::size_t>(kD) * tiles * 2),
          k_token_scales(static_cast<std::size_t>(kGroup) * tiles * 2),
          v_codes(static_cast<std::size_t>(kD / 4) * kGroup * tiles),
          v_channel_scales(static_cast<std::size_t>(kD) * tiles * 2),
          v_token_scales(static_cast<std::size_t>(kGroup) * tiles * 2),
          v_token_zeros(static_cast<std::size_t>(kGroup) * tiles * 2), tiles(tiles) {}

    ops::KvarnTileStorage view() {
        return {
            Tensor(k_codes.p, DType::U8, {kGroup / 2, kD, tiles}),
            Tensor(k_scales.p, DType::FP16, {kD, tiles}),
            Tensor(k_zeros.p, DType::FP16, {kD, tiles}),
            Tensor(k_token_scales.p, DType::FP16, {kGroup, tiles}),
            Tensor(v_codes.p, DType::U8, {kD / 4, kGroup, tiles}),
            Tensor(v_channel_scales.p, DType::FP16, {kD, tiles}),
            Tensor(v_token_scales.p, DType::FP16, {kGroup, tiles}),
            Tensor(v_token_zeros.p, DType::FP16, {kGroup, tiles}),
        };
    }

    DeviceBuffer k_codes;
    DeviceBuffer k_scales;
    DeviceBuffer k_zeros;
    DeviceBuffer k_token_scales;
    DeviceBuffer v_codes;
    DeviceBuffer v_channel_scales;
    DeviceBuffer v_token_scales;
    DeviceBuffer v_token_zeros;
    int tiles;
};

void print_result(const char* operation, int tiles, double logical_bytes,
                  const bench::ColdTiming& timing) {
    const double gbps = logical_bytes / timing.median_us / 1.0e3;
    std::printf("operation=%s tiles=%d median_us=%.3f min_us=%.3f p95_us=%.3f logical_GBps=%.2f\n",
                operation, tiles, timing.median_us, timing.min_us, timing.p95_us, gbps);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::size_t elements = static_cast<std::size_t>(kD) * kGroup * options.tiles;
        DeviceBuffer key = bench::make_bf16(elements);
        DeviceBuffer value = bench::make_bf16(elements);
        DeviceBuffer decoded_key(elements * sizeof(float));
        DeviceBuffer decoded_value(elements * sizeof(float));
        Storage storage(options.tiles);
        Tensor key_tensor(key.p, DType::BF16, {kD, kGroup, options.tiles});
        Tensor value_tensor(value.p, DType::BF16, {kD, kGroup, options.tiles});
        Tensor decoded_key_tensor(decoded_key.p, DType::FP32, {kD, kGroup, options.tiles});
        Tensor decoded_value_tensor(decoded_value.p, DType::FP32, {kD, kGroup, options.tiles});
        ops::KvarnTileStorage storage_view = storage.view();
        cudaStream_t stream = nullptr;

        const auto store_timing = bench::measure_launch(
            [&](cudaStream_t active) {
                ops::kvarn_store(key_tensor, value_tensor, storage_view, active);
            }, stream, options.warmup, options.repeat);
        const double source_bytes = 2.0 * elements * sizeof(std::uint16_t);
        print_result("store", options.tiles, source_bytes, store_timing);

        const auto dequant_timing = bench::measure_launch(
            [&](cudaStream_t active) {
                ops::kvarn_dequant(storage_view, decoded_key_tensor, decoded_value_tensor, active);
            }, stream, options.warmup, options.repeat);
        const double output_bytes = 2.0 * elements * sizeof(float);
        print_result("dequant", options.tiles, output_bytes, dequant_timing);

        const auto hadamard_timing = bench::measure_launch(
            [&](cudaStream_t active) {
                ops::kvarn_hadamard(key_tensor, value_tensor, active);
            }, stream, options.warmup, options.repeat);
        print_result("hadamard", options.tiles * kGroup,
                     static_cast<double>(elements) * 2.0 * sizeof(std::uint16_t), hadamard_timing);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
