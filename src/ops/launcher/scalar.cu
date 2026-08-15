#include "ops/launcher/scalar.h"

#include "core/device.h"
#include "ops/kernel/scalar.cuh"

namespace ninfer::ops::detail {

void set_i32_scalar_launch(Tensor& destination, std::int32_t value, hipStream_t stream) {
    set_i32_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(destination.data), value);
    HIP_CHECK(hipGetLastError());
}

void assign_i32_scalar_launch(const Tensor& source, Tensor& destination, hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(destination.data, source.data, sizeof(std::int32_t),
                               hipMemcpyDeviceToDevice, stream));
}

void add_i32_scalars_launch(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                            hipStream_t stream) {
    add_i32_scalars_kernel<<<1, 1, 0, stream>>>(static_cast<const std::int32_t*>(lhs.data),
                                                static_cast<const std::int32_t*>(rhs.data),
                                                static_cast<std::int32_t*>(destination.data));
    HIP_CHECK(hipGetLastError());
}

void increment_i32_scalar_launch(Tensor& scalar, hipStream_t stream) {
    increment_i32_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(scalar.data));
    HIP_CHECK(hipGetLastError());
}

void increment_i64_scalar_launch(Tensor& scalar, hipStream_t stream) {
    increment_i64_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int64_t*>(scalar.data));
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
