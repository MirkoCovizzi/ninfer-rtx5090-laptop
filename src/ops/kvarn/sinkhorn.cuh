#pragma once

// CUDA translation of Huawei KVarN's sinkhorn.py / triton_kvarn_sinkhorn.py contract:
// alternating log-domain sample-standard-deviation normalization with per-tile best-state
// selection by row/column standard-deviation imbalance.

#include "ops/kvarn/config.cuh"

#include <math_constants.h>

namespace ninfer::ops::kvarn {

struct Range {
    float minimum;
    float maximum;
};

template <int Count>
__device__ Range block_range(float value, float* scratch) {
    constexpr unsigned mask = 0xffffffffU;
    const int tid            = static_cast<int>(threadIdx.x);
    const int lane           = tid & 31;
    const int warp           = tid >> 5;
    float minimum            = tid < Count ? value : CUDART_INF_F;
    float maximum            = tid < Count ? value : -CUDART_INF_F;
#pragma unroll
    for (int delta = 16; delta != 0; delta >>= 1) {
        minimum = fminf(minimum, __shfl_down_sync(mask, minimum, delta));
        maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, delta));
    }
    if (lane == 0) {
        scratch[2 * warp]     = minimum;
        scratch[2 * warp + 1] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        minimum = lane < 8 ? scratch[2 * lane] : CUDART_INF_F;
        maximum = lane < 8 ? scratch[2 * lane + 1] : -CUDART_INF_F;
#pragma unroll
        for (int delta = 16; delta != 0; delta >>= 1) {
            minimum = fminf(minimum, __shfl_down_sync(mask, minimum, delta));
            maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, delta));
        }
        if (lane == 0) {
            scratch[0] = minimum;
            scratch[1] = maximum;
        }
    }
    __syncthreads();
    const Range result{scratch[0], scratch[1]};
    __syncthreads();
    return result;
}

template <bool Key>
struct Orientation {
    static constexpr int Rows  = Key ? D : Group;
    static constexpr int Cols  = Key ? Group : D;
    static constexpr int Pitch = D + 1;

    __device__ static float get(const float* tile, int row, int col) {
        if constexpr (Key) {
            return tile[row + Pitch * col];
        } else {
            return tile[col + Pitch * row];
        }
    }
};

template <bool Key>
__device__ float row_std(const float* tile, const float* row_inverse,
                         const float* column_inverse, int row) {
    using O = Orientation<Key>;
    float sum = 0.0F;
    float sum_squared = 0.0F;
    for (int col = 0; col < O::Cols; ++col) {
        const float x = O::get(tile, row, col) * row_inverse[row] * column_inverse[col];
        sum += x;
        sum_squared = fmaf(x, x, sum_squared);
    }
    constexpr float count = static_cast<float>(O::Cols);
    return sqrtf(fmaxf((sum_squared - sum * sum / count) / (count - 1.0F), 0.0F));
}

template <bool Key>
__device__ float column_std(const float* tile, const float* row_inverse,
                            const float* column_inverse, int col) {
    using O = Orientation<Key>;
    float sum = 0.0F;
    float sum_squared = 0.0F;
    for (int row = 0; row < O::Rows; ++row) {
        const float x = O::get(tile, row, col) * row_inverse[row] * column_inverse[col];
        sum += x;
        sum_squared = fmaf(x, x, sum_squared);
    }
    constexpr float count = static_cast<float>(O::Rows);
    return sqrtf(fmaxf((sum_squared - sum * sum / count) / (count - 1.0F), 0.0F));
}

template <bool Key>
__device__ float measure_imbalance(const float* tile, const float* row_inverse,
                                   const float* column_inverse, float* row_deviation,
                                   float* column_deviation, float* scratch) {
    using O       = Orientation<Key>;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < O::Rows) { row_deviation[tid] = row_std<Key>(tile, row_inverse, column_inverse, tid); }
    if (tid < O::Cols) {
        column_deviation[tid] = column_std<Key>(tile, row_inverse, column_inverse, tid);
    }
    __syncthreads();
    const Range rows = block_range<O::Rows>(tid < O::Rows ? row_deviation[tid] : 0.0F, scratch);
    const Range cols =
        block_range<O::Cols>(tid < O::Cols ? column_deviation[tid] : 0.0F, scratch);
    return rows.maximum / fmaxf(rows.minimum, 1.0e-8F) +
           cols.maximum / fmaxf(cols.minimum, 1.0e-8F);
}

template <bool Key>
__device__ void variance_normalize(const float* tile, float* log_column, float* log_row,
                                   float* best_column, float* best_row, float* row_deviation,
                                   float* column_deviation, float* row_inverse,
                                   float* column_inverse, float* scratch) {
    using O       = Orientation<Key>;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < O::Cols) {
        log_column[tid] = best_column[tid] = 0.0F;
        column_inverse[tid]                = 1.0F;
    }
    if (tid < O::Rows) {
        log_row[tid] = best_row[tid] = 0.0F;
        row_inverse[tid]             = 1.0F;
    }
    __syncthreads();

    float best = measure_imbalance<Key>(tile, row_inverse, column_inverse, row_deviation,
                                        column_deviation, scratch);
    for (int iteration = 0; iteration < Iterations; ++iteration) {
        if (tid < O::Cols) {
            const float update = logf(fminf(StdMax, fmaxf(StdMin, column_deviation[tid])));
            log_column[tid] = fminf(LogMax, fmaxf(LogMin, log_column[tid] + update));
            column_inverse[tid] = expf(-log_column[tid]);
        }
        __syncthreads();
        if (tid < O::Rows) {
            row_deviation[tid] = row_std<Key>(tile, row_inverse, column_inverse, tid);
        }
        __syncthreads();
        if (tid < O::Rows) {
            const float update = logf(fminf(StdMax, fmaxf(StdMin, row_deviation[tid])));
            log_row[tid] = fminf(LogMax, fmaxf(LogMin, log_row[tid] + update));
            row_inverse[tid] = expf(-log_row[tid]);
        }
        __syncthreads();
        const float current = measure_imbalance<Key>(tile, row_inverse, column_inverse,
                                                     row_deviation, column_deviation, scratch);
        if (current <= best) {
            best = current;
            if (tid < O::Cols) { best_column[tid] = log_column[tid]; }
            if (tid < O::Rows) { best_row[tid] = log_row[tid]; }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::kvarn
