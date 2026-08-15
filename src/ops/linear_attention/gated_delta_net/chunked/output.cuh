#include "hip/hip_runtime.h"
#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"

#include <cmath>

// Stage 4: chunk_output.
//
//   A[t,s] = (s <= t) ? dot(q[t], k[s]) * exp(g[t] - g[s]) : 0
//   out    = scale * (exp(g) * q @ h_chunk^T + A @ v_new)
//
// Q/K/H/V are represented BF16 inputs. Q @ K^T and Q @ H^T therefore use
// native BF16 MMA with FP32 accumulation. The decayed A matrix is FP32 and
// remains on the existing TF32-MMA/FP32-accumulation path for A @ V; no FP32
// intermediate is down-cast to BF16.
//
// Shared memory: persistent BF16 Q (16 KiB) and two 4 KiB BF16 staging
// buffers, for 24 KiB total. FP32 g reuses the K buffer that becomes dead
// while the final K panel is consumed.

namespace ninfer::ops::detail::gated_delta_net::chunked::output {

using ninfer::ops::Cache;
using ninfer::ops::cp_async;
using ninfer::ops::cp_commit;
using ninfer::ops::cp_wait;
using ninfer::ops::exp2_approx;


static_assert(kChunkSize == 64,
              "stage_chunk_output: kChunkSize must be 64 (kernel hard-codes BT=64)");
static_assert(kStateDim == 128);

constexpr int N_WARPS = 4;
constexpr int THREADS = N_WARPS * ninfer::ops::kWarpSize;

static_assert(BT == N_WARPS * MMA_M,
              "kernel assigns one 16-row strip per warp; BT must equal N_WARPS * MMA_M");

constexpr int BF16_MMA_K = 16;
constexpr int K_PANEL     = 32;
constexpr int D_PANEL     = 16;
constexpr int N_K_PANELS  = kStateDim / K_PANEL;
constexpr int N_D_PANELS  = kStateDim / D_PANEL;
constexpr int N_TILES_BT  = BT / MMA_N;
constexpr int K_TILES_BT  = BT / MMA_K;

static_assert(K_PANEL % BF16_MMA_K == 0);
static_assert(D_PANEL % MMA_N == 0);

struct kernel_dims {
    static constexpr int Q_BF16       = BT * kStateDim;
    static constexpr int K_PANEL_BF16 = BT * K_PANEL;
    static constexpr int H_PANEL_BF16 = D_PANEL * kStateDim;
    static constexpr int V_PANEL_BF16 = BT * D_PANEL;
    static constexpr int STAGE_BF16 =
        K_PANEL_BF16 > H_PANEL_BF16 ? K_PANEL_BF16 : H_PANEL_BF16;
    static_assert(STAGE_BF16 >= V_PANEL_BF16);

    static constexpr int BF16_SMEM_ELEMS = Q_BF16 + 2 * STAGE_BF16;
    // FP32 A = decayed Q@K^T (64x64) materialized in smem for the scalar A@V
    // fallback (gfx1151 has no fp32/tf32 WMMA, so A must be readable by every
    // thread across the 64-token contraction).
    static constexpr int A_FLOATS       = BT * BT;
    static constexpr int SMEM_BYTES =
        BF16_SMEM_ELEMS * static_cast<int>(sizeof(__hip_bfloat16)) +
        A_FLOATS * static_cast<int>(sizeof(float));
};

template <int STRIDE>
struct Bf16SmemTile {
    __hip_bfloat16* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE == 32 || STRIDE == 128);

    __device__ __forceinline__ int swizzled_col(int row, int col) const {
        return col ^ ((row & (STRIDE / 8 - 1)) << 3);
    }

    __device__ __forceinline__ __hip_bfloat16* ptr(int row, int col) const {
        return base + row * STRIDE + swizzled_col(row, col);
    }
};

template <int ROWS, int COLS, int BLOCK_THREADS>
__device__ __forceinline__ void
issue_cp_bf16(Bf16SmemTile<COLS> dst, const __hip_bfloat16* __restrict__ src_row0,
              std::int64_t src_row_stride, int tid) {
    static_assert(COLS % 8 == 0);
    constexpr int VECS_PER_ROW = COLS / 8;
    constexpr int N_VECS       = ROWS * VECS_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VECS; v += BLOCK_THREADS) {
        const int row  = v / VECS_PER_ROW;
        const int col8 = (v - row * VECS_PER_ROW) * 8;
        cp_async<16, Cache::cg>(dst.ptr(row, col8),
                                src_row0 + static_cast<std::int64_t>(row) * src_row_stride + col8);
    }
}

// Scalar bf16 panel GEMM D += A(row-strip) @ B(output-tile). Each lane owns the
// m16n8-style (row pair, col pair) element (D[nt][e]); the BF16 operands are
// read straight from shared memory over the 16-wide k window (BF16_MMA_K).
// Scalar bf16 panel GEMM D += A(row-tile) @ B(output-tile, contraction). A is
// q[row][state]; the contraction is A's column (state). B stores the same
// contraction on its column for both QK (K[token][state]) and QH
// (h_chunk[d][state]), so a single uniform read applies.
template <int N_TILES, int K_TILES, int A_STRIDE, int B_STRIDE>
__device__ __forceinline__ void
mma_bf16_panel(float (&D)[N_TILES][4], Bf16SmemTile<A_STRIDE> A,
               Bf16SmemTile<B_STRIDE> B, int a_row_base, int a_col_base, int b_col_base,
               int lane) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int row0   = a_row_base + lane_g;
    const int row1   = row0 + 8;

#pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        const int k_off = kt * BF16_MMA_K;
#pragma unroll
        for (int dk = 0; dk < BF16_MMA_K; ++dk) {
            const int a_dcol = a_col_base + k_off + dk;  // A (state-dim) column = contraction
            const int b_dcol = b_col_base + k_off + dk;  // B contraction index (row or col)
            const float q0   = __bfloat162float(*A.ptr(row0, a_dcol));
            const float q1   = __bfloat162float(*A.ptr(row1, a_dcol));
#pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                const int col0 = nt * MMA_N + 2 * lane_t;
                // B[row = output-tile col, col = contraction (state)]. Both QK
                // (B = K[token][state]) and QH (B = h_chunk[d][state]) store the
                // contraction on B's column, so the read is uniform.
                const float bvl0 = __bfloat162float(*B.ptr(col0, b_dcol));
                const float bvl1 = __bfloat162float(*B.ptr(col0 + 1, b_dcol));
                D[nt][0] += q0 * bvl0;
                D[nt][1] += q0 * bvl1;
                D[nt][2] += q1 * bvl0;
                D[nt][3] += q1 * bvl1;
            }
        }
    }
}

// Scalar A@V: D(row, d) += sum_s A[row, s] * V[s, d] over the full 64-token
// contraction. A is the decayed attention matrix materialized in smem, so each
// lane reads the full row A[row0, s] it needs (there is no fp32/tf32 WMMA).
template <int N_TILES>
__device__ __forceinline__ void mma_av_panel(float (&D)[N_TILES][4], const float* A_smem,
                                              Bf16SmemTile<D_PANEL> V, int warp_row,
                                              int lane_g, int lane_t) {
    const int row0 = warp_row + lane_g;
    const int row1 = row0 + 8;
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int r     = row0 + (e >= 2 ? 8 : 0);
            const int d     = nt * MMA_N + 2 * lane_t + (e & 1);
            float acc       = D[nt][e];
#pragma unroll
            for (int s = 0; s < BT; ++s) {
                acc += A_smem[r * BT + s] * __bfloat162float(*V.ptr(s, d));
            }
            D[nt][e] = acc;
        }
    }
}

__device__ __forceinline__ void
output_job(const __hip_bfloat16* __restrict__ q_in,
           const __hip_bfloat16* __restrict__ k_in,
           const __hip_bfloat16* __restrict__ v_new_in,
           const float* __restrict__ g_cumsum_in,
           const __hip_bfloat16* __restrict__ h_chunk_in,
           __hip_bfloat16* __restrict__ attn_out, head_map qk_map, float scale, int chunk, int h_v,
           float* smem) {
    auto* const bf16_smem = reinterpret_cast<__hip_bfloat16*>(smem);
    auto* const q_smem    = bf16_smem;
    auto* const stage0    = q_smem + kernel_dims::Q_BF16;
    auto* const stage1    = stage0 + kernel_dims::STAGE_BF16;
    float* const g_smem =
        reinterpret_cast<float*>(stage0 + kernel_dims::V_PANEL_BF16);
    // FP32 decayed A, placed after the BF16 region. Written once after the
    // decay, read by every lane across the A@V token contraction.
    float* const A_smem = smem + (kernel_dims::BF16_SMEM_ELEMS / 2);

    Bf16SmemTile<kStateDim> q_view{q_smem};
    Bf16SmemTile<K_PANEL> k_stage0{stage0};
    Bf16SmemTile<K_PANEL> k_stage1{stage1};
    Bf16SmemTile<kStateDim> h_view{stage0};
    Bf16SmemTile<D_PANEL> v_view{stage1};

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & (kWarpSize - 1);
    const int warp   = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const std::int64_t cs          = static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t H_v         = qk_map.H_v;
    const std::int64_t qk_stride_t = static_cast<std::int64_t>(qk_map.H_qk) * kStateDim;
    const std::int64_t qk_head_idx = static_cast<std::int64_t>(qk_map.qk_head(h_v)) * kStateDim;
    const std::int64_t q_base      = cs * qk_stride_t + qk_head_idx;
    const std::int64_t k_base      = cs * qk_stride_t + qk_head_idx;
    const std::int64_t vn_base =
        cs * H_v * kStateDim + static_cast<std::int64_t>(h_v) * kStateDim;
    const std::int64_t hc_base =
        (static_cast<std::int64_t>(chunk) * H_v + h_v) * kStateDim * kStateDim;

    const std::int64_t value_row_stride = H_v * kStateDim;

    // Q is permanent. K uses two 64x32 BF16 buffers and is prefetched one
    // panel ahead while the current panel feeds BF16 MMA.
    issue_cp_bf16<BT, kStateDim, THREADS>(q_view, q_in + q_base, qk_stride_t, tid);
    issue_cp_bf16<BT, K_PANEL, THREADS>(k_stage0, k_in + k_base, qk_stride_t, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float A_strip[N_TILES_BT][4] = {};

#pragma unroll
    for (int panel = 0; panel < N_K_PANELS; ++panel) {
        Bf16SmemTile<K_PANEL> current = (panel & 1) == 0 ? k_stage0 : k_stage1;
        if (panel + 1 < N_K_PANELS) {
            Bf16SmemTile<K_PANEL> next = (panel & 1) == 0 ? k_stage1 : k_stage0;
            issue_cp_bf16<BT, K_PANEL, THREADS>(
                next, k_in + k_base + static_cast<std::int64_t>(panel + 1) * K_PANEL, qk_stride_t,
                tid);
            cp_commit();
        } else if (tid < BT) {
            // stage0 was consumed by panel 2 and is now dead. Reuse its
            // otherwise-idle tail for g while panel 3 consumes stage1.
            g_smem[tid] =
                g_cumsum_in[(cs + static_cast<std::int64_t>(tid)) * H_v + h_v];
        }

        mma_bf16_panel<N_TILES_BT, K_PANEL / BF16_MMA_K>(
            A_strip, q_view, current, warp * MMA_M, panel * K_PANEL, 0, lane);

        if (panel + 1 < N_K_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
    __syncthreads();

    // Apply causal decay in FP32. Upper-triangle exp2 values may overflow, so
    // the conditional select must replace the entire product rather than
    // multiply by a zero mask.
    const int row_g0   = warp * MMA_M + lane_g;
    const int row_g1   = row_g0 + 8;
    const float g_r0   = g_smem[row_g0];
    const float g_r1   = g_smem[row_g1];
    const float gamma0 = exp2_approx(g_r0 * kLog2E);
    const float gamma1 = exp2_approx(g_r1 * kLog2E);

#pragma unroll
    for (int nt = 0; nt < N_TILES_BT; ++nt) {
        const int s0     = nt * MMA_N + 2 * lane_t;
        const int s1     = s0 + 1;
        const float g_s0 = g_smem[s0];
        const float g_s1 = g_smem[s1];

        const float dec00 = exp2_approx((g_r0 - g_s0) * kLog2E);
        const float dec01 = exp2_approx((g_r0 - g_s1) * kLog2E);
        const float dec10 = exp2_approx((g_r1 - g_s0) * kLog2E);
        const float dec11 = exp2_approx((g_r1 - g_s1) * kLog2E);

        A_strip[nt][0] = (s0 <= row_g0) ? A_strip[nt][0] * dec00 : 0.0f;
        A_strip[nt][1] = (s1 <= row_g0) ? A_strip[nt][1] * dec01 : 0.0f;
        A_strip[nt][2] = (s0 <= row_g1) ? A_strip[nt][2] * dec10 : 0.0f;
        A_strip[nt][3] = (s1 <= row_g1) ? A_strip[nt][3] * dec11 : 0.0f;

        // Materialize the decayed attention matrix in shared memory so the
        // A@V fallback can read the full row over the 64-token contraction.
        A_smem[(row_g0) * BT + (nt * MMA_N + 2 * lane_t)]     = A_strip[nt][0];
        A_smem[(row_g0) * BT + (nt * MMA_N + 2 * lane_t + 1)] = A_strip[nt][1];
        A_smem[(row_g1) * BT + (nt * MMA_N + 2 * lane_t)]     = A_strip[nt][2];
        A_smem[(row_g1) * BT + (nt * MMA_N + 2 * lane_t + 1)] = A_strip[nt][3];
    }
    __syncthreads();

    // All K reads must finish before stage0 becomes H panel 0.
    __syncthreads();
    issue_cp_bf16<D_PANEL, kStateDim, THREADS>(h_view, h_chunk_in + hc_base, kStateDim, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // H and V use disjoint buffers. V[c] is fetched while Q @ H[c]^T runs;
    // H[c+1] is fetched while the FP32 A @ V[c] path runs.
#pragma unroll 1
    for (int panel = 0; panel < N_D_PANELS; ++panel) {
        const int d_off = panel * D_PANEL;

        issue_cp_bf16<BT, D_PANEL, THREADS>(
            v_view, v_new_in + vn_base + static_cast<std::int64_t>(d_off), value_row_stride, tid);
        cp_commit();

        float D_frag[D_PANEL / MMA_N][4] = {};
        mma_bf16_panel<D_PANEL / MMA_N, kStateDim / BF16_MMA_K>(
            D_frag, q_view, h_view, warp * MMA_M, 0, 0, lane);

#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            D_frag[nt][0] *= gamma0;
            D_frag[nt][1] *= gamma0;
            D_frag[nt][2] *= gamma1;
            D_frag[nt][3] *= gamma1;
        }

        cp_wait<0>();
        __syncthreads();

        if (panel + 1 < N_D_PANELS) {
            issue_cp_bf16<D_PANEL, kStateDim, THREADS>(
                h_view, h_chunk_in + hc_base +
                            static_cast<std::int64_t>(panel + 1) * D_PANEL * kStateDim,
                kStateDim, tid);
            cp_commit();
        }

        mma_av_panel(D_frag, A_smem, v_view, warp * MMA_M, lane_g, lane_t);

#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            const int d_global = d_off + nt * MMA_N + 2 * lane_t;
            const __hip_bfloat162 out0 =
                __floats2bfloat162_rn(scale * D_frag[nt][0], scale * D_frag[nt][1]);
            const __hip_bfloat162 out1 =
                __floats2bfloat162_rn(scale * D_frag[nt][2], scale * D_frag[nt][3]);
            store_vec(&attn_out[vn_base + static_cast<std::int64_t>(row_g0) * value_row_stride +
                                d_global],
                      out0);
            store_vec(&attn_out[vn_base + static_cast<std::int64_t>(row_g1) * value_row_stride +
                                d_global],
                      out1);
        }

        if (panel + 1 < N_D_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
}

template <bool MULTI_JOB>
__launch_bounds__(THREADS, 4) __global__
    void output_kernel(const __hip_bfloat16* __restrict__ q_in,
                       const __hip_bfloat16* __restrict__ k_in,
                       const __hip_bfloat16* __restrict__ v_new_in,
                       const float* __restrict__ g_cumsum_in,
                       const __hip_bfloat16* __restrict__ h_chunk_in,
                       __hip_bfloat16* __restrict__ attn_out, head_map qk_map, float scale,
                       int chunks) {
    extern __shared__ float smem[];

    const int h_v = static_cast<int>(blockIdx.y);
    if constexpr (MULTI_JOB) {
        const int chunk_stride = static_cast<int>(gridDim.x);
        for (int chunk = static_cast<int>(blockIdx.x); chunk < chunks; chunk += chunk_stride) {
            output_job(q_in, k_in, v_new_in, g_cumsum_in, h_chunk_in, attn_out, qk_map, scale,
                       chunk, h_v, smem);
            if (chunk + chunk_stride < chunks) { __syncthreads(); }
        }
    } else {
        output_job(q_in, k_in, v_new_in, g_cumsum_in, h_chunk_in, attn_out, qk_map, scale,
                   static_cast<int>(blockIdx.x), h_v, smem);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::output