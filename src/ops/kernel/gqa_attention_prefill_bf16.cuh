#include "hip/hip_runtime.h"
#pragma once

// BF16-only GQA prompt kernel. INT8 has an independent kernel body and resource
// policy in gqa_attention_prefill_i8.cuh.
//
//   * Br = 64 query rows and Bc = 64 key columns per CTA tile.
//   * 4 warps / 128 threads; each warp owns 16 query rows of the tile.
//   * Q, K, V staged in 96 KiB of dynamic shared memory (single-buffered), with
//     the cp.async of the next K/V tile overlapped against the current
//     QK / PV tensor-core work (exactly FA's single-buffer overlap pattern).
//   * m16n8k16 bf16 MMA for both S = Q Kᵀ and O += P V, online softmax in exp2.
//
// The op first writes the new chunk K/V into absolute positions in the paged cache,
// then computes causal GQA attention for
// every chunk token over all cached history using bottom-right causal alignment
// (query row i attends to keys [0, base_pos + i]).

#include <hip/hip_math_constants.h>

#include "ops/kernel/gqa_attention_prefill_common.cuh"

namespace ninfer::ops {

template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_fill_bf16_kernel(
    const __hip_bfloat16* __restrict__ k, const __hip_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    __hip_bfloat16* __restrict__ cache_k, __hip_bfloat16* __restrict__ cache_v, std::int32_t width) {
    constexpr int VecElems = 8; // 8 bf16 == 16 B, matching the cache row alignment.
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (kGqaPrefillHeadDim / VecElems);
    if (idx >= n) { return; }

    const int vec                   = static_cast<int>(idx % (kGqaPrefillHeadDim / VecElems));
    const int tmp                   = static_cast<int>(idx / (kGqaPrefillHeadDim / VecElems));
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int d                     = vec * VecElems;
    const int position              = positions[0] + token;
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kGqaPrefillHeadDim) * (kv_head + Geometry::KVHeads * token);
    const int4 k_value = load_vec<int4>(&k[src_off]);
    const int4 v_value = load_vec<int4>(&v[src_off]);

    physical_page = __shfl_sync(0xffffffffffffffffull, physical_page, 0);

    const std::int64_t cache_off = paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
        physical_page, kv_head, position & kPagedKVPageMask, d);
    store_vec(&cache_k[cache_off], k_value);
    store_vec(&cache_v[cache_off], v_value);
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.

// Stage one [Bc, D] V tile transposed: v_s[d][key] with the XOR swizzle keyed on
// the dimension row. The WMMA B-fragment for O += P V holds column n = d and
// elements k = key, and reads v_s[n * Bc + swz(n, k)] — contiguous per 8-key
// group, matching the fragment-load pattern (the untransposed [key][d] layout
// would scatter the fragment over one element per smem row).
template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_stage_kv_t(__hip_bfloat16* dst, const __hip_bfloat16* cache,
                                                       int kv_head, int k0, int max_query_abs,
                                                       int physical_page, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kGqaPrefillBc;
    constexpr int Threads   = kGqaPrefillThreads;
    const __hip_bfloat16* cache_block =
        cache + paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    // XOR-swizzled transposed layout: dst[d * Bc + swz(d, key)]. The swizzle's
    // 64-value range exceeds the 32-wide Bc row, so OOB keys (16..31) would
    // collide with valid d=7/d=8 columns if zero-filled in place; instead the
    // OOB chunks are skipped entirely and the tile is pre-zeroed once before the
    // key loop (P for OOB keys is masked to zero, so the zero V slots are inert).
    // The swizzled addresses also keep the 8 stores un-pairable, avoiding the
    // toolchain's broken ds_store_b16_d16_hi pairing on gfx1151.
#pragma unroll 1
    for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
        const int key = chunk / (D / 8);
        const int d8  = chunk - key * (D / 8);
        if ((k0 + key) <= max_query_abs) {
            const __hip_bfloat16* src = &cache_block[key * D + d8 * 8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = d8 * 8 + i;
                dst[d * Bc + gqa_prefill_swz_identity(d, key)] = src[i];
            }
        }
    }
}
template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_stage_kv(__hip_bfloat16* dst, const __hip_bfloat16* cache,
                                                     int kv_head, int k0, int max_query_abs,
                                                     int physical_page, int tid) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kGqaPrefillBc;
    constexpr int Threads   = kGqaPrefillThreads;
    constexpr int VecPerRow = D / 8; // 8 bf16 per 16B cp.async
    const bool full_tile    = (k0 + Bc - 1) <= max_query_abs;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    const __hip_bfloat16* cache_block =
        cache + paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __hip_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __hip_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            if ((k0 + key_l) <= max_query_abs) {
                cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
template <typename Geometry, typename Metadata>
__launch_bounds__(kGqaPrefillThreads, 1) __global__
    void gqa_attention_prefill_bf16_kernel(const __hip_bfloat16* __restrict__ q,
                                           const __hip_bfloat16* __restrict__ cache_k,
                                           const __hip_bfloat16* __restrict__ cache_v,
                                           Metadata metadata,
                                           const std::int32_t* __restrict__ positions, float scale,
                                           __hip_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim; // 256
    constexpr int Br            = kGqaPrefillBr;      // 64 query rows
    constexpr int Bc            = kGqaPrefillBc;      // 64 key cols
    constexpr int Threads       = kGqaPrefillThreads; // 128
    constexpr int QKNt          = Bc / 8;             // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;             // 16 QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;              // 32 PV output n-tiles
    constexpr int PVKs          = Bc / 16;            // 4  PV contraction steps over keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned long long FullMask = 0xffffffffull;

    static_assert(Threads == 128);

    extern __shared__ __align__(16) __hip_bfloat16 gqa_smem[];
    __hip_bfloat16* q_s = gqa_smem;     // [Br, D] swizzled
    __hip_bfloat16* k_s = q_s + Br * D; // [Bc, D] swizzled
    __hip_bfloat16* v_s = k_s + Bc * D; // [Bc, D] swizzled

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (extra 4096 bytes), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop. Global Q rows are 256 bf16
    // contiguous, with a token stride of 256*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __hip_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __hip_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __hip_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    // gfx1151 WMMA conversion: atoms are 16x16, so the QK tile has Bc/16 n-atoms
    // and the PV tile has D/16 n-atoms; each atom carries 8 accumulator registers.
    constexpr int WQKNt = QKNt / 2; // Bc / 16
    constexpr int WPVNt = PVNt / 2; // D / 16
    static_assert(WQKNt * 16 == Bc);
    static_assert(WPVNt * 16 == D);

    float acc[WPVNt][8];
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
    }
    // Running row max/sum per row. Lanes 0-15 track rows 0..7 (register r = row r);
    // lanes 16-31 track rows 8..15 (register r = row r + 8).
    float m0[8], m1[8], l0[8], l1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        m0[r] = -HIP_INF_F;
        m1[r] = -HIP_INF_F;
        l0[r] = 0.0f;
        l1[r] = 0.0f;
    }
    // HIP's __shfl_*_sync requires the mask to name every active lane (it asserts
    // otherwise), so the 16-lane row reductions use the full mask and rely on the
    // width=16 segmentation to keep the two row halves independent.
    const bool lane_hi = (lane >> 4) != 0;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1; // n_block_min == 0

    // Fold softmax_scale into the exp2 (FA-style): scores stay raw, so the
    // per-element "* scale" multiply drops out of the QK epilogue entirely.
    const float scale_l2 = scale * Log2E;
    int physical_page    = block_table[0];

    // Prologue: commit Q, then kick off K(0). The loop's wait<0> below drains both.
    ninfer::ops::cp_commit();
    gqa_prefill_stage_kv<Geometry>(k_s, cache_k, kv_head, 0, max_query_abs, physical_page, tid);
    ninfer::ops::cp_commit();

    // Zero the V tile once (OOB keys are skipped by the per-block staging; the
    // causal mask zeroes their P, so the zero V slots are inert).
#pragma unroll 1
    for (int i = tid; i < Bc * D; i += Threads) v_s[i] = __hip_bfloat16(0);



    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0                 = kb * Bc;
        // Bc (32) is half the 64-key cache page: the key block at k0 lives in the
        // page of absolute position k0, not page index kb.
        const int next_physical_page =
            (kb + 1 < n_block_max) ? paged_kv_physical_page(block_table, k0 + Bc) : physical_page;

        ninfer::ops::cp_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
        __syncthreads();

        // Overlap V(kb) load against the QK MMA below. V is staged transposed
        // ([d][key]) so the PV B-fragments load contiguously.
        gqa_prefill_stage_kv_t<Geometry>(v_s, cache_v, kv_head, k0, max_query_abs, physical_page,
                                         tid);
        ninfer::ops::cp_commit();

        // S = Q K^T for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined: issue the next contraction step's fragment loads
        float score[WQKNt][8];
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { score[nt][i] = 0.0f; }
        }
        unsigned af[2][8];
        unsigned bf[2][WQKNt][8];
        {
            const int arow = warp_row0 + (lane >> 1);
            if (wmma_a_lane_active(lane)) {
                wmma_load_a_bf16(af[0], q_s, arow, 0, D, gqa_prefill_swz);
            }
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
                const int kcol = nt * 16 + (lane & 15);
                wmma_load_b_bf16(bf[0][nt], k_s, kcol, 0, D, gqa_prefill_swz);
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const int k0n = (k + 1) * 16;
                const int arow = warp_row0 + (lane >> 1);
                if (wmma_a_lane_active(lane)) {
                    wmma_load_a_bf16(af[nxt], q_s, arow, k0n, D, gqa_prefill_swz);
                }
#pragma unroll
                for (int nt = 0; nt < WQKNt; ++nt) {
                    const int kcol = nt * 16 + (lane & 15);
                    wmma_load_b_bf16(bf[nxt][nt], k_s, kcol, k0n, D, gqa_prefill_swz);
                }
            }
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
                WmmaC8& c = *reinterpret_cast<WmmaC8*>(score[nt]);
                WmmaA16I a = *reinterpret_cast<WmmaA16I*>(af[cur]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bf[cur][nt]);
                c = wmma_bf16(a, b, c);
            }
        }

        const int qrow_off = q0 + warp_row0;
        const bool full_score_tile = (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0));

        // Block row-max on raw (unscaled) scores; scale is folded into exp2 below.
        // With the WMMA C layout each lane holds one column of all 16 rows, so the
        // per-row max reduces across the 16 lanes of the row's half.
        float bm0[8], bm1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bm0[r] = -HIP_INF_F; bm1[r] = -HIP_INF_F; }
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    bm0[r] = fmaxf(bm0[r], score[nt][r]);
                    bm1[r] = fmaxf(bm1[r], score[nt][r]);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < WQKNt; ++nt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int row   = r + (lane_hi ? 8 : 0);
                    const int key   = k0 + nt * 16 + (lane & 15);
                    const int qrow  = qrow_off + row;
                    const int qabs  = (qrow < tokens) ? base_pos + qrow : -1;
                    const bool keep = (qrow < tokens) && (key <= qabs);
                    score[nt][r]    = keep ? score[nt][r] : -HIP_INF_F;
                    bm0[r]          = fmaxf(bm0[r], score[nt][r]);
                    bm1[r]          = fmaxf(bm1[r], score[nt][r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            bm0[r] = warp_max<16>(bm0[r], FullMask);
            bm1[r] = warp_max<16>(bm1[r], FullMask);
        }

        float nm0[8], nm1[8], alpha0[8], alpha1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            nm0[r]        = fmaxf(m0[r], bm0[r]);
            nm1[r]        = fmaxf(m1[r], bm1[r]);
            alpha0[r]     = exp2_approx(__fmaf_rn(m0[r], scale_l2, -nm0[r] * scale_l2));
            alpha1[r]     = exp2_approx(__fmaf_rn(m1[r], scale_l2, -nm1[r] * scale_l2));
        }

        // P = exp2(S - m), packed directly into the PV A-fragment layout via lane
        // shuffles (the WMMA A operand for row q = lane>>1 needs row q's elements,
        // which live in lanes 0..15/16..31), plus local block row-sum.
        float bl0[8], bl1[8];
        unsigned pv_a[PVKs][8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { bl0[r] = 0.0f; bl1[r] = 0.0f; }
        const int arow = lane >> 1;
        const int a_rr = arow & 7;
#pragma unroll
        for (int nt = 0; nt < WQKNt; ++nt) {
            float p[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float s = score[nt][r];
                const float nm = (lane_hi ? nm1[r] : nm0[r]);
                p[r] = (s > -HIP_INF_F) ? exp2_approx(__fmaf_rn(s, scale_l2, -nm * scale_l2))
                                        : 0.0f;
                bl0[r] += p[r];
                bl1[r] += p[r];
            }
            // A-fragment element i holds key (nt*16 + i) of row (lane>>1). Element
            // (row q, key i) of the P tile lives in lane (i & 15) + 16*(q >= 8) at
            // register (q & 7) = a_rr.
            //
            // A naive __shfl_sync(FullMask, p[a_rr], src) is WRONG: each *source*
            // lane evaluates p[a_rr] with ITS OWN (lane>>1)&7, so the dest receives
            // the source's p[(src>>1)&7] instead of p[(q)&7] -- corrupting every P
            // weight except the key where (src>>1)&7 happens to equal (q)&7. Because
            // __shfl sends a uniform register per call, we must read the dest row's
            // register a_rr by iterating all 8 registers and keeping the one that
            // matches. (gfx1151 WMMA. See .hipdebug/full_wmma_repro.cu.)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int src0 = 2 * j + (lane_hi ? 16 : 0);
                const int src1 = 2 * j + 1 + (lane_hi ? 16 : 0);
                float e0 = 0.0f, e1 = 0.0f;
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const float v0 = __shfl_sync(FullMask, p[rr], src0);
                    const float v1 = __shfl_sync(FullMask, p[rr], src1);
                    if (rr == a_rr) { e0 = v0; e1 = v1; }
                }
                pv_a[nt][j] = pack_bf16x2(e0, e1);
            }
        }


#pragma unroll
        for (int r = 0; r < 8; ++r) {
            l0[r] = __fmaf_rn(l0[r], alpha0[r], bl0[r]);
            l1[r] = __fmaf_rn(l1[r], alpha1[r], bl1[r]);
            m0[r] = nm0[r];
            m1[r] = nm1[r];
        }
#pragma unroll
        for (int n = 0; n < WPVNt; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[n][r] *= (lane_hi ? alpha1[r] : alpha0[r]);
            }
        }

        ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s
        __syncthreads();

        // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
        if (kb + 1 < n_block_max) {
            physical_page = next_physical_page;
            gqa_prefill_stage_kv<Geometry>(k_s, cache_k, kv_head, (kb + 1) * Bc, max_query_abs,
                                           physical_page, tid);
            ninfer::ops::cp_commit();
        }

        // O += P V, contracting over the Bc keys.
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            WmmaA16I a = *reinterpret_cast<WmmaA16I*>(pv_a[k]);
#pragma unroll
            for (int n = 0; n < WPVNt; ++n) {
                unsigned bfrag[8];
                const int dcol = n * 16 + (lane & 15);
                wmma_load_b_bf16(bfrag, v_s, dcol, k * 16, Bc, gqa_prefill_swz_identity);
                WmmaC8& c = *reinterpret_cast<WmmaC8*>(acc[n]);
                WmmaA16I b = *reinterpret_cast<WmmaA16I*>(bfrag);
                c = wmma_bf16(a, b, c);
            }
        }

    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        l0[r] = warp_sum<16>(l0[r], FullMask);
        l1[r] = warp_sum<16>(l1[r], FullMask);
    }

    // Normalize once per row via reciprocal-multiply instead of 128 IEEE divides.
    float inv_l0[8], inv_l1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        inv_l0[r] = (l0[r] > 0.0f) ? __frcp_rn(l0[r]) : 0.0f;
        inv_l1[r] = (l1[r] > 0.0f) ? __frcp_rn(l1[r]) : 0.0f;
    }
#pragma unroll
    for (int n = 0; n < WPVNt; ++n) {
        const int d = n * 16 + (lane & 15);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int qrow = q0 + warp_row0 + r + (lane_hi ? 8 : 0);
            if (qrow < tokens) {
                out[gqa_prefill_q_index<Geometry>(q_head, d, qrow)] = __float2bfloat16_rn(
                    acc[n][r] * (lane_hi ? inv_l1[r] : inv_l0[r]));
            }
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Threads);
}

} // namespace ninfer::ops