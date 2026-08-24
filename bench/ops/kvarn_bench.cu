// Operator evidence for KVarN normalization/compression, represented-value decode, and Hadamard
// costs. Production measurements use D256/G64 and sixteen normalization iterations; the
// D128/G128 section retains the paper geometry only as a descriptive comparison.

#include "ninfer/ops/kvarn.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

using namespace ninfer;

namespace {

constexpr int kD       = 128;
constexpr int kG       = 128;
constexpr int kKvHeads = 8;
constexpr int kLayers  = 36;
constexpr int kProductionD          = 256;
constexpr int kProductionG          = 64;
constexpr int kProductionKvHeads    = 4;
constexpr int kProductionFullLayers = 16;

struct Storage {
    DeviceBuffer k_codes;
    DeviceBuffer k_blocks;
    DeviceBuffer k_channels;
    DeviceBuffer v_codes;
    DeviceBuffer v_scales;
    DeviceBuffer v_zeros;
    DeviceBuffer v_channels;
    std::int32_t tiles;

    explicit Storage(std::int64_t tiles)
        : k_codes(static_cast<std::size_t>(tiles) * kG * kD / 2),
          k_blocks(static_cast<std::size_t>(tiles) * kG * kD / 16),
          k_channels(static_cast<std::size_t>(tiles) * kD * sizeof(std::uint16_t)),
          v_codes(static_cast<std::size_t>(tiles) * kG * kD / 4),
          v_scales(static_cast<std::size_t>(tiles) * kG * sizeof(std::uint16_t)),
          v_zeros(static_cast<std::size_t>(tiles) * kG * sizeof(std::uint16_t)),
          v_channels(static_cast<std::size_t>(tiles) * kD * sizeof(std::uint16_t)),
          tiles(static_cast<std::int32_t>(tiles)) {}

    ops::KvarnTileStorage view() {
        return {
            .k_codes = Tensor(k_codes.p, DType::U8, {kD / 2, kG, tiles}),
            .k_block_scales = Tensor(k_blocks.p, DType::FP8_E4M3FN, {kD / 16, kG, tiles}),
            .k_channel_scales = Tensor(k_channels.p, DType::FP16, {kD, tiles}),
            .v_codes = Tensor(v_codes.p, DType::U8, {kD / 4, kG, tiles}),
            .v_channel_scales = Tensor(v_channels.p, DType::FP16, {kD, tiles}),
            .v_token_scales = Tensor(v_scales.p, DType::FP16, {kG, tiles}),
            .v_token_zeros = Tensor(v_zeros.p, DType::FP16, {kG, tiles}),
        };
    }
};

struct ProductionStorage {
    DeviceBuffer k_codes;
    DeviceBuffer k_blocks;
    DeviceBuffer k_channels;
    DeviceBuffer v_codes;
    DeviceBuffer v_scales;
    DeviceBuffer v_zeros;
    DeviceBuffer v_channels;
    std::int32_t tiles;

    explicit ProductionStorage(std::int32_t count)
        : k_codes(static_cast<std::size_t>(count) * kProductionG * kProductionD / 2),
          k_blocks(static_cast<std::size_t>(count) * kProductionG * kProductionD / 16),
          k_channels(static_cast<std::size_t>(count) * kProductionD * sizeof(std::uint16_t)),
          v_codes(static_cast<std::size_t>(count) * kProductionG * kProductionD / 4),
          v_scales(static_cast<std::size_t>(count) * kProductionG * sizeof(std::uint16_t)),
          v_zeros(static_cast<std::size_t>(count) * kProductionG * sizeof(std::uint16_t)),
          v_channels(static_cast<std::size_t>(count) * kProductionD * sizeof(std::uint16_t)),
          tiles(count) {}

    ops::KvarnTileStorage view() {
        return {
            .k_codes = Tensor(k_codes.p, DType::U8,
                              {kProductionD / 2, kProductionG, tiles}),
            .k_block_scales = Tensor(k_blocks.p, DType::FP8_E4M3FN,
                                     {kProductionD / 16, kProductionG, tiles}),
            .k_channel_scales = Tensor(k_channels.p, DType::FP16, {kProductionD, tiles}),
            .v_codes = Tensor(v_codes.p, DType::U8,
                              {kProductionD / 4, kProductionG, tiles}),
            .v_channel_scales = Tensor(v_channels.p, DType::FP16, {kProductionD, tiles}),
            .v_token_scales = Tensor(v_scales.p, DType::FP16, {kProductionG, tiles}),
            .v_token_zeros = Tensor(v_zeros.p, DType::FP16, {kProductionG, tiles}),
        };
    }
};

__global__ void kivi_style_decompress(const std::uint8_t* k_codes,
                                      const std::uint8_t* k_block_scales,
                                      const std::uint8_t* v_codes,
                                      const std::uint16_t* v_token_scales,
                                      const std::uint16_t* v_token_zeros, float* k, float* v,
                                      std::int32_t tiles) {
    const std::int64_t tile = blockIdx.y;
    const std::int64_t local = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile < tiles && local < kD * kG) {
        const std::int64_t index   = tile * kD * kG + local;
        const std::int64_t token   = local / kD;
        const std::int64_t channel = local % kD;

        const std::uint8_t k_packed = k_codes[index >> 1];
        const float2 k_pair = ops::detail::decode_nvfp4_e2m1x2(k_packed);
        const float k_code = (index & 1) ? k_pair.y : k_pair.x;
        const std::int64_t k_scale = tile * kG * (kD / 16) + token * (kD / 16) + channel / 16;
        k[index] = k_code * ops::detail::decode_nvfp4_e4m3(k_block_scales[k_scale]);

        const std::uint8_t v_packed = v_codes[index >> 2];
        const std::int32_t v_code = static_cast<std::int32_t>((v_packed >> (2 * (index & 3))) & 3);
        const std::int64_t v_parameter = tile * kG + token;
        v[index] = static_cast<float>(v_code) *
                       __half2float(__ushort_as_half(v_token_scales[v_parameter])) +
                   __half2float(__ushort_as_half(v_token_zeros[v_parameter]));
    }
}

void launch_kivi(const ops::KvarnTileStorage& storage, float* k, float* v, cudaStream_t stream) {
    const std::int32_t tiles = static_cast<std::int32_t>(storage.k_codes.ne[2]);
    kivi_style_decompress<<<dim3((kD * kG + 255) / 256, tiles), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(storage.k_codes.data),
        static_cast<const std::uint8_t*>(storage.k_block_scales.data),
        static_cast<const std::uint8_t*>(storage.v_codes.data),
        static_cast<const std::uint16_t*>(storage.v_token_scales.data),
        static_cast<const std::uint16_t*>(storage.v_token_zeros.data), k, v, tiles);
    CUDA_CHECK(cudaGetLastError());
}

void benchmark_normalization(cudaStream_t stream) {
    constexpr std::int32_t tiles = kLayers * kKvHeads;
    constexpr std::size_t elements = static_cast<std::size_t>(tiles) * kD * kG;
    DeviceBuffer k = bench::make_bf16(elements);
    DeviceBuffer v = bench::make_bf16(elements);
    Storage storage(tiles);
    const Tensor k_tensor(k.p, DType::BF16, {kD, kG, tiles});
    const Tensor v_tensor(v.p, DType::BF16, {kD, kG, tiles});

    const auto timing = bench::measure_launch(
        [&](cudaStream_t measured_stream) {
            ops::kvarn_compress(k_tensor, v_tensor, storage.view(), measured_stream,
                                ops::kKvarnIterations);
        },
        stream, 3, 20);
    std::printf("normalization,qwen3-4b-36l-8kvh,d128-g128,16-iterations,%.3f ms,%.3f%% of "
                "1050 ms\n",
                timing.median_us / 1000.0, timing.median_us / 1050.0);
}

void benchmark_dequantization(cudaStream_t stream) {
    for (const int context : std::array<int, 4>{4096, 8192, 16384, 32768}) {
        const std::int32_t tiles = context / kG * kKvHeads;
        const std::size_t elements = static_cast<std::size_t>(tiles) * kD * kG;
        DeviceBuffer input_k = bench::make_bf16(elements);
        DeviceBuffer input_v = bench::make_bf16(elements);
        DeviceBuffer output_k(elements * sizeof(float));
        DeviceBuffer output_v(elements * sizeof(float));
        Storage storage(tiles);
        const Tensor k_tensor(input_k.p, DType::BF16, {kD, kG, tiles});
        const Tensor v_tensor(input_v.p, DType::BF16, {kD, kG, tiles});
        ops::kvarn_compress(k_tensor, v_tensor, storage.view(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto kivi = bench::measure_launch(
            [&](cudaStream_t measured_stream) {
                launch_kivi(storage.view(), static_cast<float*>(output_k.p),
                            static_cast<float*>(output_v.p), measured_stream);
            },
            stream, 3, 20);
        const auto kvarn = bench::measure_launch(
            [&](cudaStream_t measured_stream) {
                Tensor decoded_k(output_k.p, DType::FP32, {kD, kG, tiles});
                Tensor decoded_v(output_v.p, DType::FP32, {kD, kG, tiles});
                ops::kvarn_decompress(storage.view(), decoded_k, decoded_v, measured_stream);
            },
            stream, 3, 20);
        std::printf("dequantization,context=%d,kivi-style=%.3f us,kvarn=%.3f us,ratio=%.3fx\n",
                    context, kivi.median_us, kvarn.median_us, kvarn.median_us / kivi.median_us);
    }
}

void benchmark_production_compression(cudaStream_t stream) {
    for (const int batch : std::array<int, 5>{0, 1, 2, 4, 8}) {
        const std::int32_t tiles =
            batch == 0 ? 1 : batch * kProductionFullLayers * kProductionKvHeads;
        const std::size_t elements =
            static_cast<std::size_t>(tiles) * kProductionD * kProductionG;
        DeviceBuffer input_k = bench::make_bf16(elements);
        DeviceBuffer input_v = bench::make_bf16(elements);
        ProductionStorage storage(tiles);
        const Tensor k_tensor(input_k.p, DType::BF16, {kProductionD, kProductionG, tiles});
        const Tensor v_tensor(input_v.p, DType::BF16, {kProductionD, kProductionG, tiles});
        const auto timing = bench::measure_launch(
            [&](cudaStream_t measured_stream) {
                ops::kvarn_compress(k_tensor, v_tensor, storage.view(), measured_stream,
                                    ops::kKvarnIterations);
            },
            stream, 3, 20);
        const char* scope = batch == 0 ? "one-tile" : "qwen3.6-27b-main-full-page";
        const double amortized =
            batch == 0 ? timing.median_us / kProductionG
                       : timing.median_us / (static_cast<double>(batch) * kProductionG);
        std::printf("production-compression,scope=%s,d256-g64,B=%d,tiles=%d,iterations=%d,"
                    "median=%.3f us,p95=%.3f us,amortized=%.3f us/token/sequence\n",
                    scope, batch == 0 ? 1 : batch, tiles, ops::kKvarnIterations,
                    timing.median_us, timing.p95_us, amortized);
    }
}

void benchmark_production_hadamard(cudaStream_t stream) {
    for (const int width : std::array<int, 9>{1, 4, 6, 64, 96, 97, 128, 1024, 2048}) {
        DeviceBuffer q = bench::make_bf16(
            static_cast<std::size_t>(kProductionD) * 24 * width);
        DeviceBuffer k = bench::make_bf16(
            static_cast<std::size_t>(kProductionD) * kProductionKvHeads * width);
        DeviceBuffer v = bench::make_bf16(
            static_cast<std::size_t>(kProductionD) * kProductionKvHeads * width);
        DeviceBuffer out = bench::make_bf16(
            static_cast<std::size_t>(kProductionD) * 24 * width);
        Tensor tq(q.p, DType::BF16, {kProductionD, 24, width});
        Tensor tk(k.p, DType::BF16, {kProductionD, kProductionKvHeads, width});
        Tensor tv(v.p, DType::BF16, {kProductionD, kProductionKvHeads, width});
        Tensor tout(out.p, DType::BF16, {kProductionD, 24, width});
        const auto timing = bench::measure_launch(
            [&](cudaStream_t measured_stream) {
                ops::kvarn_hadamard(tq, tq, measured_stream);
                ops::kvarn_hadamard(tk, tk, measured_stream);
                ops::kvarn_hadamard(tv, tv, measured_stream);
                ops::kvarn_hadamard(tout, tout, measured_stream);
            },
            stream, 10, 61);
        std::printf("production-hadamard,d256-h24-kv4,W=%d,launches=4,median=%.3f us,p95=%.3f us\n",
                    width, timing.median_us, timing.p95_us);
    }
}

} // namespace

int main() {
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        benchmark_normalization(stream);
        benchmark_dequantization(stream);
        benchmark_production_compression(stream);
        benchmark_production_hadamard(stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVarN benchmark failed: %s\n", error.what());
        return 1;
    }
}
