#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops::detail {

using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&, hipStream_t);

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_gemv_t4_6144(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_gemv_t4_7168(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_gemv_t4_5120x6144(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_gemv_t4_5120x17408(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_gemv_t4_residual_5120x17408(const Tensor& x, const Weight& w, Tensor& residual_out,
                                           hipStream_t stream);
void launch_q5_gemv_t4_residual_5120x6144(const Tensor& x, const Weight& w, Tensor& residual_out,
                                          hipStream_t stream);
void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_simt_split2_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 hipStream_t stream);
void launch_q5_simt_split4_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 hipStream_t stream);
void launch_q5_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);
void launch_q5_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops::detail
