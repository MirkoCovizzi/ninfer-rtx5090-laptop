#include "ninfer/ops/kvarn.h"

#include "core/device.h"
#include "ops/kvarn/hadamard.cuh"
#include "ops/kvarn/store.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kStoreSharedBytes =
    ((kvarn::D + 1) * kvarn::Group + 8 * kvarn::D + 16) * sizeof(float);

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("KVarN: invalid ") + name);
    }
}

std::int32_t validate_storage(const KvarnTileStorage& storage) {
    const std::int32_t tiles = storage.k_codes.ne[2];
    if (tiles <= 0) { throw std::invalid_argument("KVarN: tile count must be positive"); }
    require_tensor(storage.k_codes, DType::U8, kvarn::Group / 2, kvarn::D, tiles, "K codes");
    require_tensor(storage.k_scales, DType::FP16, kvarn::D, tiles, 1, "K scales");
    require_tensor(storage.k_zeros, DType::FP16, kvarn::D, tiles, 1, "K zeros");
    require_tensor(storage.k_token_scales, DType::FP16, kvarn::Group, tiles, 1, "K token scales");
    require_tensor(storage.v_codes, DType::U8, kvarn::D / 4, kvarn::Group, tiles, "V codes");
    require_tensor(storage.v_channel_scales, DType::FP16, kvarn::D, tiles, 1, "V channel scales");
    require_tensor(storage.v_token_scales, DType::FP16, kvarn::Group, tiles, 1, "V token scales");
    require_tensor(storage.v_token_zeros, DType::FP16, kvarn::Group, tiles, 1, "V token zeros");
    return tiles;
}

__global__ void store_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                             kvarn::StorePointers output, int tiles) {
    extern __shared__ float shared[];
    float* tile             = shared;
    const int encoded       = static_cast<int>(blockIdx.x);
    const bool key          = encoded < tiles;
    const int record        = key ? encoded : encoded - tiles;
    const std::int64_t base = static_cast<std::int64_t>(record) * kvarn::D * kvarn::Group;
    for (int index = static_cast<int>(threadIdx.x); index < kvarn::D * kvarn::Group;
         index += static_cast<int>(blockDim.x)) {
        const int token = index / kvarn::D;
        const int d     = index - token * kvarn::D;
        tile[d + (kvarn::D + 1) * token] =
            __bfloat162float(key ? k[base + index] : v[base + index]);
    }
    __syncthreads();
    const kvarn::SinkhornWorkspace workspace = kvarn::workspace_after(tile);
    if (key) {
        kvarn::store_k_tile(tile, record, output, workspace);
    } else {
        kvarn::store_v_tile(tile, record, output, workspace);
    }
}

__global__ void dequant_kernel(const std::uint8_t* k_codes, const __half* k_scales,
                               const __half* k_zeros, const __half* k_token_scales,
                               const std::uint8_t* v_codes, const __half* v_channel_scales,
                               const __half* v_token_scales, const __half* v_token_zeros, float* k,
                               float* v, int tiles) {
    const int record = static_cast<int>(blockIdx.y);
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (record >= tiles || index >= kvarn::D * kvarn::Group) { return; }
    const int token = index / kvarn::D;
    const int d     = index - token * kvarn::D;
    const std::int64_t destination =
        static_cast<std::int64_t>(record) * kvarn::D * kvarn::Group + index;

    const std::int64_t k_byte =
        (static_cast<std::int64_t>(record) * kvarn::D + d) * (kvarn::Group / 2) + token / 2;
    const int k_code = (k_codes[k_byte] >> (4 * (token & 1))) & 15;
    k[destination]   = (static_cast<float>(k_code) * __half2float(k_scales[record * kvarn::D + d]) +
                      __half2float(k_zeros[record * kvarn::D + d])) *
                     __half2float(k_token_scales[record * kvarn::Group + token]);

    const std::int64_t v_byte =
        (static_cast<std::int64_t>(record) * kvarn::Group + token) * (kvarn::D / 4) + d / 4;
    const int v_code = (v_codes[v_byte] >> (2 * (d & 3))) & 3;
    v[destination] =
        (static_cast<float>(v_code) * __half2float(v_token_scales[record * kvarn::Group + token]) +
         __half2float(v_token_zeros[record * kvarn::Group + token])) *
        __half2float(v_channel_scales[record * kvarn::D + d]);
}

__global__ void hadamard_kernel(const __nv_bfloat16* source, __nv_bfloat16* destination,
                                int vectors) {
    __shared__ float stage[2][kvarn::D];
    const int vector = static_cast<int>(blockIdx.x);
    const int d      = static_cast<int>(threadIdx.x);
    if (vector >= vectors) { return; }

    float value = __bfloat162float(source[static_cast<std::int64_t>(vector) * kvarn::D + d]);
    value       = kvarn::detail::hadamard_block(value, stage, d);
    destination[static_cast<std::int64_t>(vector) * kvarn::D + d] = __float2bfloat16_rn(value);
}

} // namespace

void kvarn_store(const Tensor& rotated_k, const Tensor& rotated_v, KvarnTileStorage storage,
                 cudaStream_t stream) {
    const std::int32_t tiles = validate_storage(storage);
    require_tensor(rotated_k, DType::BF16, kvarn::D, kvarn::Group, tiles, "rotated K");
    require_tensor(rotated_v, DType::BF16, kvarn::D, kvarn::Group, tiles, "rotated V");
    static const cudaError_t attribute =
        cudaFuncSetAttribute(store_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(kStoreSharedBytes));
    CUDA_CHECK(attribute);
    const kvarn::StorePointers output{
        static_cast<std::uint8_t*>(storage.k_codes.data),
        static_cast<__half*>(storage.k_scales.data),
        static_cast<__half*>(storage.k_zeros.data),
        static_cast<__half*>(storage.k_token_scales.data),
        static_cast<std::uint8_t*>(storage.v_codes.data),
        static_cast<__half*>(storage.v_channel_scales.data),
        static_cast<__half*>(storage.v_token_scales.data),
        static_cast<__half*>(storage.v_token_zeros.data),
    };
    store_kernel<<<2 * tiles, kThreads, kStoreSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(rotated_k.data),
        static_cast<const __nv_bfloat16*>(rotated_v.data), output, tiles);
    CUDA_CHECK(cudaGetLastError());
}

void kvarn_dequant(const KvarnTileStorage& storage, Tensor& rotated_k, Tensor& rotated_v,
                   cudaStream_t stream) {
    const std::int32_t tiles = validate_storage(storage);
    require_tensor(rotated_k, DType::FP32, kvarn::D, kvarn::Group, tiles, "dequantized K");
    require_tensor(rotated_v, DType::FP32, kvarn::D, kvarn::Group, tiles, "dequantized V");
    const dim3 grid(static_cast<unsigned>((kvarn::D * kvarn::Group + kThreads - 1) / kThreads),
                    static_cast<unsigned>(tiles));
    dequant_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(storage.k_codes.data),
        static_cast<const __half*>(storage.k_scales.data),
        static_cast<const __half*>(storage.k_zeros.data),
        static_cast<const __half*>(storage.k_token_scales.data),
        static_cast<const std::uint8_t*>(storage.v_codes.data),
        static_cast<const __half*>(storage.v_channel_scales.data),
        static_cast<const __half*>(storage.v_token_scales.data),
        static_cast<const __half*>(storage.v_token_zeros.data), static_cast<float*>(rotated_k.data),
        static_cast<float*>(rotated_v.data), tiles);
    CUDA_CHECK(cudaGetLastError());
}

void kvarn_hadamard(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    bool matching_shape = true;
    for (int dim = 0; dim < 4; ++dim) {
        matching_shape = matching_shape && source.ne[dim] == destination.ne[dim];
    }
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 || !matching_shape ||
        source.ne[0] != kvarn::D || !source.is_contiguous() || !destination.is_contiguous() ||
        source.data == nullptr || destination.data == nullptr) {
        throw std::invalid_argument(
            "kvarn_hadamard: expected matching contiguous BF16 D256 tensors");
    }
    const int vectors = static_cast<int>(source.numel() / kvarn::D);
    hadamard_kernel<<<vectors, kvarn::D, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source.data),
        static_cast<__nv_bfloat16*>(destination.data), vectors);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
