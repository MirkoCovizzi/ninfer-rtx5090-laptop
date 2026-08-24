#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kKvarnIterations = 16;

/**
 * KVarN's variance-normalized NVFP4/V2 tile representation.
 *
 * The represented inputs are BF16 K/V values after the normalized Hadamard rotation. Inputs use
 * the engine's d-contiguous `[D,G,N]` layout. K is normalized as `[D,G]` (channel rows, token
 * columns); V is normalized as `[G,D]` (token rows, channel columns). Both run sixteen alternating
 * sample-standard-deviation normalization passes in FP32, with standard deviations clamped to
 * `[1e-3,1e3]`, accumulated log-scales clamped to `[-0.3,10]`, and the least-imbalanced visited
 * scale pair retained.
 *
 * K uses native NVFP4 E2M1 codes in channel-contiguous K16 blocks. The per-token K16 scale and
 * KVarN token scale are multiplied and rounded once to E4M3FN; the channel scale is rounded to
 * FP16. Its exact decode is:
 *
 *   K[d,t] = E2M1(k_code[d,t]) * FP32(k_block_scale[d/16,t])
 *                                  * FP32(k_channel_scale[d])
 *
 * V uses asymmetric two-bit RTN per token. RTN scale/zero are absorbed into the KVarN token scale
 * and rounded to FP16; the channel scale is rounded to FP16. Its exact decode is:
 *
 *   V[d,t] = (v_code[d,t] * FP32(v_token_scale[t]) + FP32(v_token_zero[t]))
 *                                  * FP32(v_channel_scale[d])
 *
 * Codes are packed least-significant element first. One K byte contains d=(2b,2b+1); one V byte
 * contains d=(4b..4b+3). The admitted production geometry is D=256,G=64. D=128,G=128 is also
 * admitted so the paper's dequantization workload can be reproduced directly.
 */
struct KvarnTileStorage {
    Tensor k_codes;          // U8  [D/2,  G, N]
    Tensor k_block_scales;   // FP8 [D/16, G, N]
    Tensor k_channel_scales; // FP16[D,     N]
    Tensor v_codes;          // U8  [D/4,  G, N]
    Tensor v_channel_scales; // FP16[D,     N]
    Tensor v_token_scales;   // FP16[G,     N]
    Tensor v_token_zeros;    // FP16[G,     N]
};

void kvarn_compress(const Tensor& rotated_k, const Tensor& rotated_v, KvarnTileStorage storage,
                    cudaStream_t stream, std::int32_t iterations = kKvarnIterations);

// Independent observable decode of the stored representation. Outputs are contiguous FP32
// `[D,G,N]`; this entry is used by conformance tests and by non-hot diagnostic paths.
void kvarn_decompress(const KvarnTileStorage& storage, Tensor& rotated_k, Tensor& rotated_v,
                      cudaStream_t stream);

// Applies the orthonormal Sylvester-Hadamard transform independently to every d-vector in a
// contiguous BF16 `[D,...]` tensor. D is 128 or 256. Exact in-place operation is supported.
void kvarn_hadamard(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops
