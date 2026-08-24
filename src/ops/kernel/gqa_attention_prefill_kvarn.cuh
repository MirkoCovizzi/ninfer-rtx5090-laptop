#pragma once

// KVarN K4/V2 storage adapter for the shared BF16-MMA prompt kernel. A 64-key attention tile is
// exactly one KVarN page, so each represented value is decoded once per query tile and reused by
// all 64 query rows. Partial pages are read losslessly from either staged tail slot.

#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

struct GqaPrefillKvarnStorage {
    const std::uint8_t* k_codes;
    const std::uint8_t* k_block_scales;
    const __half* k_channel_scales;
    const std::uint8_t* v_codes;
    const __half* v_channel_scales;
    const __half* v_token_scales;
    const __half* v_token_zeros;
    const __nv_bfloat16* tail_k;
    const __nv_bfloat16* tail_v;
    const std::int32_t* tail_logical_pages;
    const std::int32_t* cached_tail_pages;
    const float2* cached_k_channels;
    const float2* cached_v_channels;
    std::int32_t heads;
};

template <bool Key>
__device__ __forceinline__ float2 gqa_prefill_kvarn_decode_pair(
    const GqaPrefillKvarnStorage& storage, std::int64_t record, int token, int d) {
    if constexpr (Key) {
        const std::uint8_t packed =
            storage.k_codes[(record * kPagedKVPageSize + token) * (kGqaPrefillHeadDim / 2) + d / 2];
        const float2 codes = detail::decode_nvfp4_e2m1x2(packed);
        const std::int64_t scale =
            (record * kPagedKVPageSize + token) * (kGqaPrefillHeadDim / 16) + d / 16;
        const float block    = detail::decode_nvfp4_e4m3(storage.k_block_scales[scale]);
        const float2 channel = storage.cached_k_channels[((d & 7) / 2) * 32 + d / 8];
        return make_float2(codes.x * block * channel.x, codes.y * block * channel.y);
    }
    const std::uint8_t packed =
        storage.v_codes[(record * kPagedKVPageSize + token) * (kGqaPrefillHeadDim / 4) + d / 4];
    const float scale = __half2float(storage.v_token_scales[record * kPagedKVPageSize + token]);
    const float zero  = __half2float(storage.v_token_zeros[record * kPagedKVPageSize + token]);
    const int shift   = 2 * (d & 3);
    const float c0    = static_cast<float>((packed >> shift) & 3);
    const float c1    = static_cast<float>((packed >> (shift + 2)) & 3);
    const float2 channel = storage.cached_v_channels[((d & 7) / 2) * 32 + d / 8];
    return make_float2((c0 * scale + zero) * channel.x,
                       (c1 * scale + zero) * channel.y);
}

template <typename Geometry, int Bc, int Threads, bool Key>
__device__ __forceinline__ void gqa_prefill_stage_kv(
    __nv_bfloat16* dst, const GqaPrefillKvarnStorage& storage, int kv_head, int k0,
    int max_query_abs, int physical_page, int table_row, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int VecPerRow = D / 8;
    const int logical_page  = k0 >> kPagedKVPageShift;
    int tail_slot           = -1;
#pragma unroll
    for (int candidate = 0; candidate < kKvarnTailSlots; ++candidate) {
        if (storage.cached_tail_pages[candidate] == logical_page) { tail_slot = candidate; }
    }
    const std::int64_t record = static_cast<std::int64_t>(physical_page) * storage.heads + kv_head;
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l = chunk / VecPerRow;
        const int d     = (chunk - key_l * VecPerRow) * 8;
        const int token = k0 + key_l;
        const int page_token = token & kPagedKVPageMask;
        __nv_bfloat16* destination = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
        if (token > max_query_abs) {
            store_vec(destination, make_int4(0, 0, 0, 0));
            continue;
        }
        if (tail_slot >= 0) {
            const auto* tail = Key ? storage.tail_k : storage.tail_v;
            const std::int64_t source =
                (((static_cast<std::int64_t>(table_row) * kKvarnTailSlots + tail_slot) *
                      storage.heads +
                  kv_head) *
                     kPagedKVPageSize +
                 page_token) *
                    D +
                d;
            store_vec(destination, load_vec<int4>(tail + source));
            continue;
        }
        __nv_bfloat16 values[8];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded =
                gqa_prefill_kvarn_decode_pair<Key>(storage, record, page_token, d + 2 * pair);
            values[2 * pair]     = __float2bfloat16(decoded.x);
            values[2 * pair + 1] = __float2bfloat16(decoded.y);
        }
        store_vec(destination, load_vec<int4>(values));
    }
}

} // namespace ninfer::ops
