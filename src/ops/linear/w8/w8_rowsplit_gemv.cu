#include "hip/hip_runtime.h"
#include "ops/linear/w8/w8_rowsplit_gemv.cuh"
#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "core/tensor.h"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

template <int kN, int kK>
void launch_t4(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream) {
    w8_rowsplit_gemv_t_launch_kernel<kN, kK, 4>(
        static_cast<const __hip_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__hip_bfloat16*>(out.data),
        x.ne[1], stream);
}

} // namespace

void launch_w8_gemv_t4_248320x5120(const Tensor& x, const Weight& w, Tensor& out,
                                   hipStream_t stream) {
    launch_t4<248320, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_w8_gemv_t4_34816x5120(const Tensor& x, const Weight& w, Tensor& out,
                                  hipStream_t stream) {
    launch_t4<34816, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_w8_gemv_t4_14336x5120(const Tensor& x, const Weight& w, Tensor& out,
                                  hipStream_t stream) {
    launch_t4<14336, 5120>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_w8_gemv_t4_5120x10240(const Tensor& x, const Weight& w, Tensor& out,
                                  hipStream_t stream) {
    launch_t4<5120, 10240>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_w8_gemv_t4_5120x6144(const Tensor& x, const Weight& w, Tensor& out,
                                 hipStream_t stream) {
    launch_t4<5120, 6144>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}
void launch_w8_gemv_t4_5120x17408(const Tensor& x, const Weight& w, Tensor& out,
                                  hipStream_t stream) {
    launch_t4<5120, 17408>(x, w, out, stream);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
