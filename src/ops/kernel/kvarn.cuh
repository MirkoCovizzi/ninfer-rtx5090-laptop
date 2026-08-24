#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr float kKvarnStdMinimum = 1.0e-3F;
inline constexpr float kKvarnStdMaximum = 1.0e3F;
inline constexpr float kKvarnLogMinimum = -0.3F;
inline constexpr float kKvarnLogMaximum = 10.0F;

template <int Count, bool Maximum>
__device__ float kvarn_block_extreme(float value, float* scratch) {
    constexpr int WarpSize = 32;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & (WarpSize - 1);
    const int warp = tid / WarpSize;
    value = tid < Count ? value : (Maximum ? -CUDART_INF_F : CUDART_INF_F);
#pragma unroll
    for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(FullMask, value, offset);
        value             = Maximum ? fmaxf(value, other) : fminf(value, other);
    }
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        constexpr int Warps = 256 / WarpSize;
        value = lane < Warps ? scratch[lane] : (Maximum ? -CUDART_INF_F : CUDART_INF_F);
#pragma unroll
        for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(FullMask, value, offset);
            value             = Maximum ? fmaxf(value, other) : fminf(value, other);
        }
        if (lane == 0) { scratch[0] = value; }
    }
    __syncthreads();
    const float result = scratch[0];
    // Every thread must consume the result before the next reduction reuses the same scratch.
    __syncthreads();
    return result;
}

template <int D, int G, bool Key>
struct KvarnTileGeometry {
    static constexpr int Rows = Key ? D : G;
    static constexpr int Cols = Key ? G : D;

    __device__ static float element(const float* tile, int row, int col) {
        if constexpr (Key) {
            return tile[row + D * col];
        } else {
            return tile[col + D * row];
        }
    }
};

template <int D, int G, bool Key>
__device__ float kvarn_row_std(const float* tile, const float* row_inverse,
                                const float* col_inverse,
                                int row) {
    using Geometry = KvarnTileGeometry<D, G, Key>;
    float sum      = 0.0F;
    float squares  = 0.0F;
    const float rs = row_inverse[row];
#pragma unroll 1
    for (int col = 0; col < Geometry::Cols; ++col) {
        const float x = Geometry::element(tile, row, col) * rs * col_inverse[col];
        sum += x;
        squares = fmaf(x, x, squares);
    }
    constexpr float count = static_cast<float>(Geometry::Cols);
    const float variance  = fmaxf((squares - sum * sum / count) / (count - 1.0F), 0.0F);
    return sqrtf(variance);
}

template <int D, int G, bool Key>
__device__ float kvarn_col_std(const float* tile, const float* row_inverse,
                                const float* col_inverse,
                                int col) {
    using Geometry = KvarnTileGeometry<D, G, Key>;
    float sum      = 0.0F;
    float squares  = 0.0F;
    const float cs = col_inverse[col];
#pragma unroll 1
    for (int row = 0; row < Geometry::Rows; ++row) {
        const float x = Geometry::element(tile, row, col) * row_inverse[row] * cs;
        sum += x;
        squares = fmaf(x, x, squares);
    }
    constexpr float count = static_cast<float>(Geometry::Rows);
    const float variance  = fmaxf((squares - sum * sum / count) / (count - 1.0F), 0.0F);
    return sqrtf(variance);
}

template <int D, int G, bool Key>
__device__ float kvarn_imbalance(const float* tile, const float* row_inverse,
                                  const float* col_inverse,
                                  float* row_std, float* col_std, float* scratch) {
    using Geometry = KvarnTileGeometry<D, G, Key>;
    const int tid  = static_cast<int>(threadIdx.x);
    if (tid < Geometry::Rows) {
        row_std[tid] = kvarn_row_std<D, G, Key>(tile, row_inverse, col_inverse, tid);
    }
    if (tid < Geometry::Cols) {
        col_std[tid] = kvarn_col_std<D, G, Key>(tile, row_inverse, col_inverse, tid);
    }
    __syncthreads();
    const float row_value = tid < Geometry::Rows ? row_std[tid] : 0.0F;
    const float row_max = kvarn_block_extreme<Geometry::Rows, true>(row_value, scratch);
    const float row_min = kvarn_block_extreme<Geometry::Rows, false>(row_value, scratch);
    const float col_value = tid < Geometry::Cols ? col_std[tid] : 0.0F;
    const float col_max = kvarn_block_extreme<Geometry::Cols, true>(col_value, scratch);
    const float col_min = kvarn_block_extreme<Geometry::Cols, false>(col_value, scratch);
    return row_max / fmaxf(row_min, 1.0e-8F) + col_max / fmaxf(col_min, 1.0e-8F);
}

// Immediately after a row update, each row's new standard deviation is exactly its pre-update
// deviation multiplied by exp(old_log-new_log). The caller has stored that residual in row_std,
// so only columns need another reduction to score the newly visited scale pair.
template <int D, int G, bool Key>
__device__ float kvarn_imbalance_after_row_update(const float* tile,
                                                   const float* row_inverse,
                                                   const float* col_inverse, float* row_std,
                                                   float* col_std, float* scratch) {
    using Geometry = KvarnTileGeometry<D, G, Key>;
    const int tid  = static_cast<int>(threadIdx.x);
    if (tid < Geometry::Cols) {
        col_std[tid] = kvarn_col_std<D, G, Key>(tile, row_inverse, col_inverse, tid);
    }
    __syncthreads();
    const float row_value = tid < Geometry::Rows ? row_std[tid] : 0.0F;
    const float row_max = kvarn_block_extreme<Geometry::Rows, true>(row_value, scratch);
    const float row_min = kvarn_block_extreme<Geometry::Rows, false>(row_value, scratch);
    const float col_value = tid < Geometry::Cols ? col_std[tid] : 0.0F;
    const float col_max = kvarn_block_extreme<Geometry::Cols, true>(col_value, scratch);
    const float col_min = kvarn_block_extreme<Geometry::Cols, false>(col_value, scratch);
    return row_max / fmaxf(row_min, 1.0e-8F) + col_max / fmaxf(col_min, 1.0e-8F);
}

template <int D, int G, bool Key, int Iterations>
__device__ void kvarn_balance(float* tile, float* log_row, float* log_col, float* best_row,
                               float* best_col, float* row_std, float* col_std,
                               float* row_inverse, float* col_inverse, float* scratch) {
    using Geometry = KvarnTileGeometry<D, G, Key>;
    const int tid  = static_cast<int>(threadIdx.x);
    if (tid < Geometry::Rows) {
        log_row[tid]  = 0.0F;
        best_row[tid] = 0.0F;
        row_inverse[tid] = 1.0F;
    }
    if (tid < Geometry::Cols) {
        log_col[tid]  = 0.0F;
        best_col[tid] = 0.0F;
        col_inverse[tid] = 1.0F;
    }
    __syncthreads();

    float best =
        kvarn_imbalance<D, G, Key>(tile, row_inverse, col_inverse, row_std, col_std, scratch);
    for (int iteration = 0; iteration < Iterations; ++iteration) {
        // The initial score and each preceding post-row score leave col_std evaluated at the
        // current scale pair. Recomputing it here would repeat the same reduction exactly.
        if (tid < Geometry::Cols) {
            const float stddev = col_std[tid];
            log_col[tid] = fminf(kKvarnLogMaximum,
                                   fmaxf(kKvarnLogMinimum,
                                         log_col[tid] + logf(fminf(kKvarnStdMaximum,
                                                                   fmaxf(kKvarnStdMinimum, stddev)))));
            col_inverse[tid] = expf(-log_col[tid]);
        }
        __syncthreads();
        if (tid < Geometry::Rows) {
            row_std[tid] =
                kvarn_row_std<D, G, Key>(tile, row_inverse, col_inverse, tid);
        }
        __syncthreads();
        if (tid < Geometry::Rows) {
            const float stddev = row_std[tid];
            const float old_log = log_row[tid];
            const float new_log = fminf(
                kKvarnLogMaximum,
                fmaxf(kKvarnLogMinimum,
                      old_log + logf(fminf(kKvarnStdMaximum,
                                           fmaxf(kKvarnStdMinimum, stddev)))));
            log_row[tid] = new_log;
            row_std[tid] = stddev * expf(old_log - new_log);
            row_inverse[tid] = expf(-new_log);
        }
        __syncthreads();
        const float imbalance = kvarn_imbalance_after_row_update<D, G, Key>(
            tile, row_inverse, col_inverse, row_std, col_std, scratch);
        if (imbalance <= best) {
            best = imbalance;
            if (tid < Geometry::Rows) { best_row[tid] = log_row[tid]; }
            if (tid < Geometry::Cols) { best_col[tid] = log_col[tid]; }
        }
        __syncthreads();
    }
}

template <int D, int G, int Iterations>
__device__ void kvarn_compress_loaded_tile(float* shared, bool key_path, std::int64_t record,
                                            std::uint8_t* k_codes,
                                            std::uint8_t* k_block_scales,
                                            __half* k_channel_scales, std::uint8_t* v_codes,
                                            __half* v_channel_scales, __half* v_token_scales,
                                            __half* v_token_zeros) {
    float* tile      = shared;
    float* log_row   = tile + D * G;
    float* log_col   = log_row + D;
    float* best_row  = log_col + D;
    float* best_col  = best_row + D;
    float* row_std   = best_col + D;
    float* col_std   = row_std + D;
    float* row_inverse = col_std + D;
    float* col_inverse = row_inverse + D;
    float* scratch   = col_inverse + D;

    const int tid          = static_cast<int>(threadIdx.x);

    if (key_path) {
        kvarn_balance<D, G, true, Iterations>(tile, log_row, log_col, best_row, best_col, row_std,
                                               col_std, row_inverse, col_inverse, scratch);
        if (tid < D) {
            k_channel_scales[record * D + tid] =
                __float2half_rn(expf(best_row[tid]));
        }
        constexpr int Blocks = D / 16;
        for (int task = tid; task < G * Blocks; task += static_cast<int>(blockDim.x)) {
            const int token = task / Blocks;
            const int block = task - token * Blocks;
            float2 values[8];
            float max_abs = 0.0F;
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const int d   = block * 16 + pair * 2;
                const float s0 = expf(-best_row[d] - best_col[token]);
                const float s1 = expf(-best_row[d + 1] - best_col[token]);
                values[pair]   = make_float2(tile[d + D * token] * s0,
                                             tile[d + 1 + D * token] * s1);
                max_abs = fmaxf(max_abs, fabsf(values[pair].x));
                max_abs = fmaxf(max_abs, fabsf(values[pair].y));
            }
            const float raw_scale = max_abs / 6.0F;
            const float combined  = raw_scale * expf(best_col[token]);
            const std::uint8_t encoded_scale =
                __nv_cvt_float_to_fp8(combined, __NV_SATFINITE, __NV_E4M3);
            const std::int64_t scale_index =
                (record * G + token) * Blocks + block;
            k_block_scales[scale_index] = encoded_scale;
            std::uint32_t lo = 0;
            std::uint32_t hi = 0;
            if (raw_scale > 0.0F && encoded_scale != 0) {
#pragma unroll
                for (float2& value : values) {
                    value.x /= raw_scale;
                    value.y /= raw_scale;
                }
                pack_nvfp4_e2m1x16(values, lo, hi);
            }
            const std::int64_t code_index =
                (record * G + token) * (D / 2) + block * 8;
            *reinterpret_cast<std::uint32_t*>(k_codes + code_index)     = lo;
            *reinterpret_cast<std::uint32_t*>(k_codes + code_index + 4) = hi;
        }
    } else {
        // V is logically [G,D], while the represented tensor remains d-contiguous [D,G].
        kvarn_balance<D, G, false, Iterations>(tile, log_row, log_col, best_row, best_col, row_std,
                                                col_std, row_inverse, col_inverse, scratch);
        if (tid < D) {
            v_channel_scales[record * D + tid] =
                __float2half_rn(expf(best_col[tid]));
        }
        for (int token = tid; token < G; token += static_cast<int>(blockDim.x)) {
            float lo = CUDART_INF_F;
            float hi = -CUDART_INF_F;
            const float token_inverse = expf(-best_row[token]);
            for (int d = 0; d < D; ++d) {
                const float x = tile[d + D * token] * token_inverse * expf(-best_col[d]);
                lo            = fminf(lo, x);
                hi            = fmaxf(hi, x);
            }
            const float scale = fmaxf((hi - lo) / 3.0F, 1.0e-10F);
            const float token_scale = expf(best_row[token]);
            v_token_scales[record * G + token] =
                __float2half_rn(scale * token_scale);
            v_token_zeros[record * G + token] =
                __float2half_rn(lo * token_scale);
            const std::int64_t code_base =
                (record * G + token) * (D / 4);
            for (int byte = 0; byte < D / 4; ++byte) {
                std::uint8_t packed = 0;
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    const int d = byte * 4 + item;
                    const float x = tile[d + D * token] * token_inverse * expf(-best_col[d]);
                    const int code = max(0, min(3, __float2int_rn((x - lo) / scale)));
                    packed |= static_cast<std::uint8_t>(code << (2 * item));
                }
                v_codes[code_base + byte] = packed;
            }
        }
    }
}

template <int D, int G, int Iterations>
__global__ void kvarn_compress_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                       std::uint8_t* k_codes, std::uint8_t* k_block_scales,
                                       __half* k_channel_scales, std::uint8_t* v_codes,
                                       __half* v_channel_scales, __half* v_token_scales,
                                       __half* v_token_zeros, int tiles) {
    extern __shared__ float shared[];
    const int encoded_tile = static_cast<int>(blockIdx.x);
    const bool key_path    = encoded_tile < tiles;
    const int tile_index   = key_path ? encoded_tile : encoded_tile - tiles;
    const int tid          = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(tile_index) * D * G;
    for (int index = tid; index < D * G; index += static_cast<int>(blockDim.x)) {
        shared[index] = __bfloat162float(key_path ? k[base + index] : v[base + index]);
    }
    __syncthreads();
    kvarn_compress_loaded_tile<D, G, Iterations>(
        shared, key_path, tile_index, k_codes, k_block_scales, k_channel_scales, v_codes,
        v_channel_scales, v_token_scales, v_token_zeros);
}

template <int D, int G, bool Masked>
__global__ void kvarn_paged_stage_edges_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    const std::int32_t* valid_columns, const std::int32_t* table_rows,
    const std::int32_t* block_tables, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t heads, std::int32_t rows,
    std::uint8_t* k_codes, std::uint8_t* k_block_scales, __half* k_channel_scales,
    std::uint8_t* v_codes, __half* v_channel_scales, __half* v_token_scales,
    __half* v_token_zeros, __nv_bfloat16* tail_k, __nv_bfloat16* tail_v,
    std::int32_t* tail_logical_pages) {
    const int head  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int edge  = static_cast<int>(blockIdx.z);
    const int tid   = static_cast<int>(threadIdx.x);
    const int valid = Masked ? valid_columns[batch] : full_width;
    if (head >= heads || valid <= 0) { return; }

    const std::int64_t position_base = static_cast<std::int64_t>(batch) * full_width;
    const int first_position         = positions[position_base];
    const int last_position          = positions[position_base + valid - 1];
    const int first_page             = first_position / G;
    const int last_page              = last_position / G;
    if (edge != 0 && first_page == last_page) { return; }
    if (edge == 0 && first_page != last_page && (first_position % G) == 0) { return; }
    const int logical_page           = edge == 0 ? first_page : last_page;
    const int page_begin             = logical_page * G;
    const int begin                  = max(first_position, page_begin);
    const int end                    = min(last_position + 1, page_begin + G);
    if (begin >= end) { return; }

    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (table_row < 0 || table_row >= rows) { return; }
    const int physical_page = block_tables[static_cast<std::int64_t>(table_row) * table_stride +
                                                logical_page];
    const std::int64_t record = static_cast<std::int64_t>(physical_page) * heads + head;
    const std::int64_t row_base =
        static_cast<std::int64_t>(table_row) * heads * kKvarnTailSlots;
    int slot = -1;
#pragma unroll
    for (int candidate = 0; candidate < kKvarnTailSlots; ++candidate) {
        if (tail_logical_pages[row_base + candidate * heads + head] == logical_page) {
            slot = candidate;
        }
    }
    if (slot < 0) {
        if (edge == 0 || (first_position % G) == 0) {
            slot = 0;
        } else {
            int first_slot = 0;
#pragma unroll
            for (int candidate = 0; candidate < kKvarnTailSlots; ++candidate) {
                if (tail_logical_pages[row_base + candidate * heads + head] == first_page) {
                    first_slot = candidate;
                }
            }
            slot = 1 - first_slot;
        }
    }
    const std::int64_t tail_record = row_base + slot * heads + head;
    std::int32_t* marker           = tail_logical_pages + tail_record;

    if (*marker != logical_page && begin > page_begin) {
        for (int index = tid; index < D * G; index += static_cast<int>(blockDim.x)) {
            const int token = index / D;
            const int d     = index - token * D;
            const std::int64_t tail_index = tail_record * D * G + index;
            const std::int64_t k_byte = (record * G + token) * (D / 2) + d / 2;
            __nv_fp4x2_e2m1 pair;
            pair.__x             = k_codes[k_byte];
            const float2 decoded = static_cast<float2>(pair);
            const float k_code   = (d & 1) == 0 ? decoded.x : decoded.y;
            const std::int64_t k_scale = (record * G + token) * (D / 16) + d / 16;
            tail_k[tail_index] = __float2bfloat16(
                k_code * decode_nvfp4_e4m3(k_block_scales[k_scale]) *
                __half2float(k_channel_scales[record * D + d]));

            const std::int64_t v_byte = (record * G + token) * (D / 4) + d / 4;
            const int v_code          = (v_codes[v_byte] >> (2 * (d & 3))) & 3;
            tail_v[tail_index]        = __float2bfloat16(
                (static_cast<float>(v_code) * __half2float(v_token_scales[record * G + token]) +
                 __half2float(v_token_zeros[record * G + token])) *
                __half2float(v_channel_scales[record * D + d]));
        }
        __syncthreads();
    }

    for (int item = tid; item < (end - begin) * D; item += static_cast<int>(blockDim.x)) {
        const int token_delta = item / D;
        const int d           = item - token_delta * D;
        const int position    = begin + token_delta;
        const int source_token = position - first_position;
        const std::int64_t source =
            ((static_cast<std::int64_t>(batch) * full_width + source_token) * heads + head) *
                D +
            d;
        const std::int64_t destination =
            tail_record * D * G + static_cast<std::int64_t>(position - page_begin) * D + d;
        tail_k[destination] = k[source];
        tail_v[destination] = v[source];
    }
    __syncthreads();
    if (tid == 0) { *marker = logical_page; }
}

template <int D, int G, int Iterations, bool Masked>
__global__ void kvarn_paged_compress_direct_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    const std::int32_t* valid_columns, const std::int32_t* table_rows,
    const std::int32_t* block_tables, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t heads, std::int32_t rows, std::int32_t page_candidates,
    std::uint8_t* k_codes, std::uint8_t* k_block_scales, __half* k_channel_scales,
    std::uint8_t* v_codes, __half* v_channel_scales, __half* v_token_scales,
    __half* v_token_zeros) {
    extern __shared__ float shared[];
    const int encoded = static_cast<int>(blockIdx.x);
    const int batch   = static_cast<int>(blockIdx.y);
    const bool key_path = (encoded & 1) == 0;
    const int record_task = encoded >> 1;
    const int head      = record_task % heads;
    const int candidate = record_task / heads;
    const int tid       = static_cast<int>(threadIdx.x);
    if (candidate >= page_candidates) { return; }

    const int valid = Masked ? valid_columns[batch] : full_width;
    if (valid <= 0) { return; }
    const std::int64_t position_base = static_cast<std::int64_t>(batch) * full_width;
    const int first_position = positions[position_base];
    const int last_position  = positions[position_base + valid - 1];
    const int logical_page   = first_position / G + candidate;
    const int last_page      = last_position / G;
    const int page_begin     = logical_page * G;
    if (logical_page >= last_page || page_begin < first_position ||
        page_begin + G > last_position + 1) {
        return;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (table_row < 0 || table_row >= rows) { return; }
    const int physical_page = block_tables[static_cast<std::int64_t>(table_row) * table_stride +
                                                logical_page];
    const std::int64_t record = static_cast<std::int64_t>(physical_page) * heads + head;
    const int source_token     = page_begin - first_position;
    for (int index = tid; index < D * G; index += static_cast<int>(blockDim.x)) {
        const int token = index / D;
        const int d     = index - token * D;
        const std::int64_t source =
            ((position_base + source_token + token) * heads + head) * D + d;
        shared[index] = __bfloat162float(key_path ? k[source] : v[source]);
    }
    __syncthreads();
    kvarn_compress_loaded_tile<D, G, Iterations>(
        shared, key_path, record, k_codes, k_block_scales, k_channel_scales, v_codes,
        v_channel_scales, v_token_scales, v_token_zeros);
}

inline constexpr int kKvarnMaximumFlushLayers = 32;

struct KvarnPagedLayerPointers {
    std::uint8_t* k_codes[kKvarnMaximumFlushLayers];
    std::uint8_t* k_block_scales[kKvarnMaximumFlushLayers];
    __half* k_channel_scales[kKvarnMaximumFlushLayers];
    std::uint8_t* v_codes[kKvarnMaximumFlushLayers];
    __half* v_channel_scales[kKvarnMaximumFlushLayers];
    __half* v_token_scales[kKvarnMaximumFlushLayers];
    __half* v_token_zeros[kKvarnMaximumFlushLayers];
    __nv_bfloat16* tail_k[kKvarnMaximumFlushLayers];
    __nv_bfloat16* tail_v[kKvarnMaximumFlushLayers];
    std::int32_t* tail_logical_pages[kKvarnMaximumFlushLayers];
};

template <int D, int G, int Iterations, bool Masked>
__global__ void kvarn_flush_full_tails_kernel(
    KvarnPagedLayerPointers layers, const std::int32_t* positions,
    const std::int32_t* valid_columns, const std::int32_t* table_rows,
    const std::int32_t* block_tables, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t heads, std::int32_t rows, std::int32_t layer_count) {
    extern __shared__ float shared[];
    const int head  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int encoded = static_cast<int>(blockIdx.z);
    const bool key_path = (encoded & 1) == 0;
    const int slot       = (encoded >> 1) % kKvarnTailSlots;
    const int layer      = (encoded >> 1) / kKvarnTailSlots;
    const int tid        = static_cast<int>(threadIdx.x);
    if (layer >= layer_count) { return; }
    const int valid = Masked ? valid_columns[batch] : full_width;
    if (valid <= 0) { return; }
    const std::int64_t position_base = static_cast<std::int64_t>(batch) * full_width;
    const int last_position = positions[position_base + valid - 1];
    const int table_row      = table_rows == nullptr ? 0 : table_rows[batch];
    if (table_row < 0 || table_row >= rows) { return; }
    const std::int64_t tail_record =
        (static_cast<std::int64_t>(table_row) * kKvarnTailSlots + slot) * heads + head;
    const int logical_page = layers.tail_logical_pages[layer][tail_record];
    if (logical_page < 0 || (logical_page + 1) * G > last_position + 1) { return; }
    const int physical_page = block_tables[static_cast<std::int64_t>(table_row) * table_stride +
                                                logical_page];
    const std::int64_t record = static_cast<std::int64_t>(physical_page) * heads + head;
    const __nv_bfloat16* source =
        (key_path ? layers.tail_k[layer] : layers.tail_v[layer]) + tail_record * D * G;
    for (int index = tid; index < D * G; index += static_cast<int>(blockDim.x)) {
        shared[index] = __bfloat162float(source[index]);
    }
    __syncthreads();
    kvarn_compress_loaded_tile<D, G, Iterations>(
        shared, key_path, record, layers.k_codes[layer], layers.k_block_scales[layer],
        layers.k_channel_scales[layer], layers.v_codes[layer], layers.v_channel_scales[layer],
        layers.v_token_scales[layer], layers.v_token_zeros[layer]);
}

template <int G, bool Masked>
__global__ void kvarn_clear_flushed_tails_kernel(
    KvarnPagedLayerPointers layers, const std::int32_t* positions,
    const std::int32_t* valid_columns, const std::int32_t* table_rows, std::int32_t full_width,
    std::int32_t heads, std::int32_t rows, std::int32_t layer_count) {
    const int head  = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int slot  = static_cast<int>(blockIdx.z) % kKvarnTailSlots;
    const int layer = static_cast<int>(blockIdx.z) / kKvarnTailSlots;
    if (head >= heads || layer >= layer_count) { return; }
    const int valid = Masked ? valid_columns[batch] : full_width;
    if (valid <= 0) { return; }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (table_row < 0 || table_row >= rows) { return; }
    const std::int64_t position_base = static_cast<std::int64_t>(batch) * full_width;
    const int last_position = positions[position_base + valid - 1];
    const std::int64_t tail_record =
        (static_cast<std::int64_t>(table_row) * kKvarnTailSlots + slot) * heads + head;
    std::int32_t* marker = layers.tail_logical_pages[layer] + tail_record;
    const int logical_page = *marker;
    if (logical_page >= 0 && (logical_page + 1) * G <= last_position + 1) { *marker = -1; }
}

template <int D, int G>
__global__ void kvarn_decompress_kernel(const std::uint8_t* k_codes,
                                        const std::uint8_t* k_block_scales,
                                        const __half* k_channel_scales,
                                        const std::uint8_t* v_codes,
                                        const __half* v_channel_scales,
                                        const __half* v_token_scales,
                                        const __half* v_token_zeros, float* k, float* v,
                                        int tiles) {
    const int tile = static_cast<int>(blockIdx.y);
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (tile >= tiles || index >= D * G) { return; }
    const int token = index / D;
    const int d     = index - token * D;
    const std::int64_t output = static_cast<std::int64_t>(tile) * D * G + index;

    const std::int64_t k_byte =
        (static_cast<std::int64_t>(tile) * G + token) * (D / 2) + d / 2;
    const std::uint8_t packed_k = k_codes[k_byte];
    __nv_fp4x2_e2m1 pair;
    pair.__x             = packed_k;
    const float2 decoded = static_cast<float2>(pair);
    const float k_code   = (d & 1) == 0 ? decoded.x : decoded.y;
    const std::int64_t k_scale_index =
        (static_cast<std::int64_t>(tile) * G + token) * (D / 16) + d / 16;
    k[output] = k_code * decode_nvfp4_e4m3(k_block_scales[k_scale_index]) *
                __half2float(k_channel_scales[static_cast<std::int64_t>(tile) * D + d]);

    const std::int64_t v_byte =
        (static_cast<std::int64_t>(tile) * G + token) * (D / 4) + d / 4;
    const int v_code = (v_codes[v_byte] >> (2 * (d & 3))) & 3;
    v[output] =
        (static_cast<float>(v_code) *
             __half2float(v_token_scales[static_cast<std::int64_t>(tile) * G + token]) +
         __half2float(v_token_zeros[static_cast<std::int64_t>(tile) * G + token])) *
        __half2float(v_channel_scales[static_cast<std::int64_t>(tile) * D + d]);
}

template <int D>
__global__ void kvarn_hadamard_kernel(const __nv_bfloat16* source, __nv_bfloat16* destination,
                                       int vectors) {
    __shared__ float values[D];
    const int vector = static_cast<int>(blockIdx.x);
    const int d      = static_cast<int>(threadIdx.x);
    if (vector >= vectors || d >= D) { return; }
    float value = __bfloat162float(source[static_cast<std::int64_t>(vector) * D + d]);
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffU, value, span);
        value             = (d & span) == 0 ? value + other : other - value;
    }
    values[d] = value;
    __syncthreads();
    for (int span = 32; span < D; span <<= 1) {
        const int group = d / (span << 1);
        const int lane  = d & (span - 1);
        const int left  = group * (span << 1) + lane;
        const int right = left + span;
        const float a   = values[left];
        const float b   = values[right];
        __syncthreads();
        value     = (d & span) == 0 ? a + b : a - b;
        values[d] = value;
        __syncthreads();
    }
    constexpr float normalization = D == 256 ? 0.0625F : 0.08838834764831845F;
    destination[static_cast<std::int64_t>(vector) * D + d] =
        __float2bfloat16_rn(value * normalization);
}

} // namespace ninfer::ops::detail
