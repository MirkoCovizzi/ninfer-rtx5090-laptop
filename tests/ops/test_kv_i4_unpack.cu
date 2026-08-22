#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kCases = 4096;

__global__ void unpack_kernel(const std::uint8_t* packed, std::int8_t* unpacked) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= kCases) { return; }
    ninfer::ops::gqa_kv_unpack_i4x16_vectorized(packed + 8 * row, unpacked + 16 * row);
}

std::int8_t unpack_nibble(std::uint8_t code) {
    return static_cast<std::int8_t>(static_cast<int>(code ^ 8U) - 8);
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) { return true; }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no usable CUDA device");
        return 77;
    }

    std::vector<std::uint8_t> packed(static_cast<std::size_t>(kCases) * 8);
    std::vector<std::int8_t> expected(static_cast<std::size_t>(kCases) * 16);
    for (int row = 0; row < kCases; ++row) {
        for (int byte = 0; byte < 8; ++byte) {
            const std::uint8_t value = static_cast<std::uint8_t>(37 * row + 53 * byte);
            packed[static_cast<std::size_t>(row) * 8 + byte] = value;
            expected[static_cast<std::size_t>(row) * 16 + 2 * byte] =
                unpack_nibble(value & 0x0fU);
            expected[static_cast<std::size_t>(row) * 16 + 2 * byte + 1] =
                unpack_nibble(value >> 4);
        }
    }

    std::uint8_t* device_packed = nullptr;
    std::int8_t* device_unpacked = nullptr;
    if (!cuda_ok(cudaMalloc(&device_packed, packed.size()), "cudaMalloc packed") ||
        !cuda_ok(cudaMalloc(&device_unpacked, expected.size()), "cudaMalloc unpacked") ||
        !cuda_ok(cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice),
                 "copy packed")) {
        cudaFree(device_packed);
        cudaFree(device_unpacked);
        return 1;
    }

    unpack_kernel<<<(kCases + 255) / 256, 256>>>(device_packed, device_unpacked);
    std::vector<std::int8_t> actual(expected.size());
    const bool copied = cuda_ok(cudaGetLastError(), "unpack kernel") &&
                        cuda_ok(cudaMemcpy(actual.data(), device_unpacked, actual.size(),
                                           cudaMemcpyDeviceToHost),
                                "copy unpacked");
    cudaFree(device_packed);
    cudaFree(device_unpacked);
    if (!copied) { return 1; }

    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr, "i4 unpack mismatch at %zu: got %d expected %d\n", index,
                         static_cast<int>(actual[index]), static_cast<int>(expected[index]));
            return 1;
        }
    }
    return 0;
}
