#include "hip/hip_runtime.h"
#include "core/device.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Small-T (2..24) batched forward: one weight stream per kTt-column slice, at the
// memory wall instead of the split SIMT GEMM paths (which are 10-50x below DRAM on gfx1151).
template <int kN, int kK>
void launch_q5_gemv_t4(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    q5_rowsplit_gemv_t_launch_kernel<kN, kK, 4>(
        static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__hip_bfloat16*>(out.data), x.ne[1], stream);
    HIP_CHECK(hipGetLastError());
}

void launch_q5_gemv_t4_6144(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    launch_q5_gemv_t4<6144, 5120>(x, w, out, stream);
}
void launch_q5_gemv_t4_7168(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    launch_q5_gemv_t4<7168, 5120>(x, w, out, stream);
}
void launch_q5_gemv_t4_5120x6144(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    launch_q5_gemv_t4<5120, 6144>(x, w, out, stream);
}
void launch_q5_gemv_t4_5120x17408(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    launch_q5_gemv_t4<5120, 17408>(x, w, out, stream);
}

void launch_q5_gemv_t4_residual_5120x17408(const Tensor& x, const Weight& w, Tensor& residual_out,
                                           hipStream_t stream) {
    q5_rowsplit_gemv_t_residual_launch_kernel<5120, 17408, 4>(
        static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__hip_bfloat16*>(residual_out.data), x.ne[1], stream);
    HIP_CHECK(hipGetLastError());
}

void launch_q5_gemv_t4_residual_5120x6144(const Tensor& x, const Weight& w, Tensor& residual_out,
                                          hipStream_t stream) {
    q5_rowsplit_gemv_t_residual_launch_kernel<5120, 6144, 4>(
        static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__hip_bfloat16*>(residual_out.data), x.ne[1], stream);
    HIP_CHECK(hipGetLastError());
}

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    constexpr int kK = 5120;
    if (w.n == 6144) {
        q5_rowsplit_gemv_launch_kernel<6144, kK, 16, 2, true>(
            static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__hip_bfloat16*>(out.data), stream);
    } else if (w.n == 7168) {
        q5_rowsplit_gemv_launch_kernel<7168, kK, 16, 2, true>(
            static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__hip_bfloat16*>(out.data), stream);
    } else {
        throw std::invalid_argument("q5 GEMV R16/S2/X: unsupported exact N");
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
