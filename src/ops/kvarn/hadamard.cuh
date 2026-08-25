#pragma once

#include "ops/kvarn/config.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::kvarn::detail {

__device__ __forceinline__ void hadamard_warp(float (&values)[D / 32], int lane) {
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int segment = 0; segment < D / 32; ++segment) {
            const float other = __shfl_xor_sync(0xffffffffU, values[segment], span);
            values[segment] =
                (lane & span) == 0 ? values[segment] + other : other - values[segment];
        }
    }
#pragma unroll
    for (int span = 1; span < D / 32; span <<= 1) {
#pragma unroll
        for (int block = 0; block < D / 32; block += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float left = values[block + offset];
                const float right = values[block + span + offset];
                values[block + offset] = left + right;
                values[block + span + offset] = left - right;
            }
        }
    }
#pragma unroll
    for (float& value : values) { value *= 0.0625F; }
}

} // namespace ninfer::ops::kvarn::detail
