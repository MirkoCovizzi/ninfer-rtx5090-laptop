#include "hip/hip_runtime.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_gemv_t.cuh"

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/hip_compat.cuh"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

template <int kN, int kK>
void launch_t4(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    q4_rowsplit_gemv_t_launch_kernel<kN, kK, 4>(
        static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__hip_bfloat16*>(out.data),
        x.ne[1], stream);
}

} // namespace

void launch_q4_gemv_t4_4096x5120(const Tensor& x, const Weight& w, Tensor& out,
                                 hipStream_t stream) {
    launch_t4<4096, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_q4_gemv_t4_6144x5120(const Tensor& x, const Weight& w, Tensor& out,
                                 hipStream_t stream) {
    launch_t4<6144, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_q4_gemv_t4_7168x5120(const Tensor& x, const Weight& w, Tensor& out,
                                 hipStream_t stream) {
    launch_t4<7168, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_q4_gemv_t4_34816x5120(const Tensor& x, const Weight& w, Tensor& out,
                                  hipStream_t stream) {
    launch_t4<34816, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
