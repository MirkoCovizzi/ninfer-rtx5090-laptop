#pragma once

#include "ninfer/ops/kvarn.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/kvarn/config.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::kvarn::detail {

struct PrefillCache {
    const std::uint8_t* records;
    const __nv_bfloat16* tail_k;
    const __nv_bfloat16* tail_v;
    const std::int32_t* markers;
    const std::int32_t* block_tables;
    const std::int32_t* table_rows;
    int table_stride;
    int heads;
};

template <typename Metadata>
__launch_bounds__(D) __global__
    void materialize_prefill_slab_kernel(PrefillCache cache, Metadata metadata,
                                         const std::int32_t* positions, int width, int slab_begin,
                                         int slab_tokens, __nv_bfloat16* materialized_k,
                                         __nv_bfloat16* materialized_v) {
    const int logical_page = slab_begin / Group + static_cast<int>(blockIdx.x);
    const int head         = static_cast<int>(blockIdx.y);
    const int d            = static_cast<int>(threadIdx.x);
    const int tokens       = metadata.valid_tokens(width);
    if (tokens <= 0) { return; }
    const int visible    = positions[tokens - 1] + 1;
    const int page_begin = logical_page * Group;
    if (page_begin >= visible) { return; }

    const int table_row = cache.table_rows[0];
    int tail_slot       = -1;
#pragma unroll
    for (int slot = 0; slot < kKvarnTailSlots; ++slot) {
        if (cache.markers[slot + kKvarnTailSlots * table_row] == logical_page) { tail_slot = slot; }
    }

    if (tail_slot >= 0) {
        for (int token = 0; token < Group && page_begin + token < visible; ++token) {
            const std::int64_t source =
                static_cast<std::int64_t>(d) +
                static_cast<std::int64_t>(D) *
                    (token +
                     Group * (head + cache.heads * (tail_slot + kKvarnTailSlots * table_row)));
            const std::int64_t destination =
                static_cast<std::int64_t>(d) +
                static_cast<std::int64_t>(D) * (page_begin + token - slab_begin +
                                                static_cast<std::int64_t>(slab_tokens) * head);
            materialized_k[destination] = cache.tail_k[source];
            materialized_v[destination] = cache.tail_v[source];
        }
        return;
    }

    const int physical_page =
        cache
            .block_tables[logical_page + static_cast<std::int64_t>(cache.table_stride) * table_row];
    const std::uint8_t* record =
        cache.records +
        (static_cast<std::int64_t>(physical_page) * cache.heads + head) * kKvarnRecordBytes;
    const auto* k_scale       = reinterpret_cast<const __half*>(record + kKvarnKScaleOffset);
    const auto* k_zero        = reinterpret_cast<const __half*>(record + kKvarnKZeroOffset);
    const auto* k_token_scale = reinterpret_cast<const __half*>(record + kKvarnKTokenScaleOffset);
    const auto* v_channel     = reinterpret_cast<const __half*>(record + kKvarnVChannelScaleOffset);
    const auto* v_scale       = reinterpret_cast<const __half*>(record + kKvarnVTokenScaleOffset);
    const auto* v_zero        = reinterpret_cast<const __half*>(record + kKvarnVTokenZeroOffset);
    const auto* k_words =
        reinterpret_cast<const std::uint32_t*>(record + kKvarnKPackedOffset + d * (Group / 2));
    std::uint32_t packed_k[Group / 8];
#pragma unroll
    for (int word = 0; word < Group / 8; ++word) packed_k[word] = k_words[word];
    const float column_scale  = __half2float(k_scale[d]);
    const float column_zero   = __half2float(k_zero[d]);
    const float channel_scale = __half2float(v_channel[d]);
    for (int token = 0; token < Group && page_begin + token < visible; ++token) {
        const int k_code            = (packed_k[token / 8] >> (4 * (token & 7))) & 15;
        const std::uint8_t v_packed = record[kKvarnVPackedOffset + token * (D / 4) + d / 4];
        const int v_code            = (v_packed >> (2 * (d & 3))) & 3;
        const float key             = fmaf(static_cast<float>(k_code), column_scale, column_zero) *
                          __half2float(k_token_scale[token]);
        const float value = fmaf(static_cast<float>(v_code), __half2float(v_scale[token]),
                                 __half2float(v_zero[token])) *
                            channel_scale;
        const std::int64_t destination =
            static_cast<std::int64_t>(d) +
            static_cast<std::int64_t>(D) *
                (page_begin + token - slab_begin + static_cast<std::int64_t>(slab_tokens) * head);
        materialized_k[destination] = __float2bfloat16_rn(key);
        materialized_v[destination] = __float2bfloat16_rn(value);
    }
}

struct MaterializedPrefillInput {
    const __nv_bfloat16* key;
    const __nv_bfloat16* value;
    int slab_begin;
    int slab_tokens;

    template <bool Key>
    __device__ __forceinline__ void stage(__nv_bfloat16* destination, int kv_head, int k0,
                                          int max_query_abs, int, int tid) const {
        constexpr int Bc        = kCausalPromptBc;
        constexpr int Threads   = kCausalPromptThreads;
        constexpr int VecPerRow = D / 8;
        const auto* source_block =
            (Key ? key : value) +
            static_cast<std::int64_t>(D) *
                (static_cast<std::int64_t>(slab_tokens) * kv_head + k0 - slab_begin);
        if (k0 + Bc - 1 <= max_query_abs) {
#pragma unroll
            for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
                const int token = chunk >> 5;
                const int d     = (chunk & 31) << 3;
                auto* output    = destination + token * D + causal_prompt_swz(token, d);
                cp_async<16, Cache::cg>(output, source_block + token * D + d);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
                const int token = chunk >> 5;
                const int d     = (chunk & 31) << 3;
                auto* output    = destination + token * D + causal_prompt_swz(token, d);
                if (k0 + token <= max_query_abs) {
                    cp_async<16, Cache::cg>(output, source_block + token * D + d);
                } else {
                    store_vec(output, make_int4(0, 0, 0, 0));
                }
            }
        }
    }
};

} // namespace ninfer::ops::kvarn::detail
