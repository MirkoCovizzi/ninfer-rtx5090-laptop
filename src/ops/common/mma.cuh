#pragma once

// gfx1151 (RDNA 3.5) WMMA tensor-core layer.
//
// RDNA3 has no mma.sync/ldmatrix; the tensor path is the wave32 WMMA instruction
// set. The fragment layouts below were probed and validated on Strix Halo:
//
//   A (16x16x16 f16/bf16): lane l with (l & 1) == ((l >> 4) & 1) holds row m = l >> 1;
//                          vector element i is k = i.
//   B: every lane l holds column n = l & 15; element i is k = i. The same column
//      is duplicated at lane n + 16 (the instruction reads both copies).
//   C/D (f32 accumulate): lane l holds element (m, n) = (r + 8 * (l >= 16), l & 15)
//                          at register r.
//   D = A * B + C, all 16x16x16 tiles.
//
// i8 variant: identical distribution with k = byte index within the 16 bytes per
// lane (4 v4i32 registers), C/D are i32.

#include "ops/common/memory.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace ninfer::ops {

using WmmaA16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using WmmaA16I = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using WmmaC8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using WmmaA8I = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using WmmaC8I = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ bool wmma_a_lane_active(int lane) {
    return (lane & 1) == ((lane >> 4) & 1);
}

__device__ __forceinline__ WmmaC8 wmma_bf16(const WmmaA16I& a, const WmmaA16I& b, const WmmaC8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

__device__ __forceinline__ WmmaC8 wmma_f16(const WmmaA16& a, const WmmaA16& b, const WmmaC8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ WmmaC8I wmma_i8(const WmmaA8I& a, const WmmaA8I& b, const WmmaC8I& c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}


// ---------------------------------------------------------------------------
// Shared-memory fragment loads for the WMMA 16x16x16 atoms.
//
// All NInfer GEMM/attention tiles stage their operands in shared memory with the
// Xor64 swizzle (8-element groups permuted by the low 3 row bits), so a lane's
// 16 K-wide fragment is two 16-byte ds reads at swizzled addresses. On gfx11 a
// ds_read with a VGPR address requires an explicit s_waitcnt lgkmcnt(0) after it.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void wmma_ds_load_b128(unsigned* out, unsigned addr, int offset_dwords) {
    uint4 v;
    // Leading s_waitcnt lgkmcnt(0): drains this wave's LDS pipe after the staging
    // barrier. gfx1151's s_barrier does not always make other waves' ds_stores
    // visible to a subsequent ds_read (observed as rare stale-fragment garbage with
    // many resident CTAs); the wait forces the read to hit committed LDS. The
    // trailing wait makes the value ready before use. The "memory" clobber is
    // required: without it the read is not a memory operation at IR level and LLVM
    // hoists it across __syncthreads(), racing the staging stores.
    asm volatile("s_waitcnt lgkmcnt(0);\n ds_read_b128 %0, %1 offset:%2;\n s_waitcnt lgkmcnt(0);\n"
                 : "=v"(v)
                 : "v"(addr), "n"(offset_dwords)
                 : "memory");
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}

// A fragment for row `row`, K window [k0, k0+16): participating lanes only; the
// swizzle keeps each 8-element K group contiguous, so two 16-byte loads cover
// k0..k0+7 and k0+8..k0+15.
template <class Swizzle>
__device__ __forceinline__ void wmma_load_a_bf16(unsigned (&frag)[8], const __hip_bfloat16* base,
                                                 int row, int k0, int stride_k, Swizzle swizzle_col) {
    const unsigned chunk1 = smem_addr(&base[row * stride_k + swizzle_col(row, k0)]);
    const unsigned chunk2 = smem_addr(&base[row * stride_k + swizzle_col(row, k0 + 8)]);
    wmma_ds_load_b128(frag, chunk1, 0);
    wmma_ds_load_b128(frag + 4, chunk2, 0);
}

// B fragment for column `col` (B stored [col][k]): every lane holds column n,
// element i is k = k0 + i.
template <class Swizzle>
__device__ __forceinline__ void wmma_load_b_bf16(unsigned (&frag)[8], const __hip_bfloat16* base,
                                                 int col, int k0, int stride_k, Swizzle swizzle_col) {
    const unsigned chunk1 = smem_addr(&base[col * stride_k + swizzle_col(col, k0)]);
    const unsigned chunk2 = smem_addr(&base[col * stride_k + swizzle_col(col, k0 + 8)]);
    wmma_ds_load_b128(frag, chunk1, 0);
    wmma_ds_load_b128(frag + 4, chunk2, 0);
}

} // namespace ninfer::ops