#include "ninfer/ops/kvarn.h"

#include "ops/launcher/kvarn.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != 1 || tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("KVarN invalid ") + name);
    }
}

void validate_storage(const KvarnTileStorage& storage, std::int32_t d, std::int32_t g,
                      std::int32_t tiles) {
    require_tensor(storage.k_codes, DType::U8, d / 2, g, tiles, "K codes");
    require_tensor(storage.k_block_scales, DType::FP8_E4M3FN, d / 16, g, tiles,
                   "K block scales");
    require_tensor(storage.k_channel_scales, DType::FP16, d, tiles, 1, "K channel scales");
    require_tensor(storage.v_codes, DType::U8, d / 4, g, tiles, "V codes");
    require_tensor(storage.v_channel_scales, DType::FP16, d, tiles, 1, "V channel scales");
    require_tensor(storage.v_token_scales, DType::FP16, g, tiles, 1, "V token scales");
    require_tensor(storage.v_token_zeros, DType::FP16, g, tiles, 1, "V token zeros");
}

} // namespace

void kvarn_compress(const Tensor& rotated_k, const Tensor& rotated_v, KvarnTileStorage storage,
                    cudaStream_t stream, std::int32_t iterations) {
    if (rotated_k.dtype != DType::BF16 || rotated_v.dtype != DType::BF16 ||
        rotated_k.ne[0] != rotated_v.ne[0] || rotated_k.ne[1] != rotated_v.ne[1] ||
        rotated_k.ne[2] != rotated_v.ne[2] || rotated_k.ne[3] != 1 || rotated_v.ne[3] != 1 ||
        rotated_k.data == nullptr || rotated_v.data == nullptr || !rotated_k.is_contiguous() ||
        !rotated_v.is_contiguous() || rotated_k.data == rotated_v.data || rotated_k.ne[2] <= 0) {
        throw std::invalid_argument("kvarn_compress: invalid represented inputs");
    }
    const bool geometry = (rotated_k.ne[0] == 256 && rotated_k.ne[1] == 64) ||
                          (rotated_k.ne[0] == 128 && rotated_k.ne[1] == 128);
    if (!geometry) { throw std::invalid_argument("kvarn_compress: unsupported geometry"); }
    if (iterations != 8 && iterations != 16) {
        throw std::invalid_argument("kvarn_compress: iterations must be 8 or 16");
    }
    validate_storage(storage, rotated_k.ne[0], rotated_k.ne[1], rotated_k.ne[2]);
    detail::kvarn_compress_launch(rotated_k, rotated_v, storage, iterations, stream);
}

void kvarn_decompress(const KvarnTileStorage& storage, Tensor& rotated_k, Tensor& rotated_v,
                      cudaStream_t stream) {
    if (rotated_k.dtype != DType::FP32 || rotated_v.dtype != DType::FP32 ||
        rotated_k.ne[0] != rotated_v.ne[0] || rotated_k.ne[1] != rotated_v.ne[1] ||
        rotated_k.ne[2] != rotated_v.ne[2] || rotated_k.ne[3] != 1 || rotated_v.ne[3] != 1 ||
        rotated_k.data == nullptr || rotated_v.data == nullptr || !rotated_k.is_contiguous() ||
        !rotated_v.is_contiguous() || rotated_k.data == rotated_v.data || rotated_k.ne[2] <= 0) {
        throw std::invalid_argument("kvarn_decompress: invalid outputs");
    }
    const bool geometry = (rotated_k.ne[0] == 256 && rotated_k.ne[1] == 64) ||
                          (rotated_k.ne[0] == 128 && rotated_k.ne[1] == 128);
    if (!geometry) { throw std::invalid_argument("kvarn_decompress: unsupported geometry"); }
    validate_storage(storage, rotated_k.ne[0], rotated_k.ne[1], rotated_k.ne[2]);
    detail::kvarn_decompress_launch(storage, rotated_k, rotated_v, stream);
}

void kvarn_hadamard(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 ||
        source.ne[0] != destination.ne[0] || source.ne[1] != destination.ne[1] ||
        source.ne[2] != destination.ne[2] || source.ne[3] != destination.ne[3] ||
        (source.ne[0] != 128 && source.ne[0] != 256) || source.data == nullptr ||
        destination.data == nullptr || !source.is_contiguous() ||
        !destination.is_contiguous()) {
        throw std::invalid_argument("kvarn_hadamard: invalid tensors");
    }
    detail::kvarn_hadamard_launch(source, destination, stream);
}

} // namespace ninfer::ops
