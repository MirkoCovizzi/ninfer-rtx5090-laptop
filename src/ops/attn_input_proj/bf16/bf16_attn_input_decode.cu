#include "hip/hip_runtime.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <hip/hip_bf16.h>
#include "ops/common/hip_compat.cuh"

namespace ninfer::ops::detail {
namespace {

struct Bf16AttentionInputOutput {
    __hip_bfloat16* query;
    __hip_bfloat16* key;
    __hip_bfloat16* gate;
    __hip_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, __hip_bfloat16 result) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        if (parent_row < kKeyBegin) {
            query[parent_row] = result;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result;
        } else {
            value[parent_row - kValueBegin] = result;
        }
    }
};

} // namespace

void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, hipStream_t stream) {
    using Geometry = Bf16GemvGeometry<14336, 5120>;
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;
    static_assert((6144 % Schedule::kRowsPerCta) == 0);
    static_assert((1024 % Schedule::kRowsPerCta) == 0);

    const Bf16AttentionInputOutput output{
        static_cast<__hip_bfloat16*>(q.data),
        static_cast<__hip_bfloat16*>(k.data),
        static_cast<__hip_bfloat16*>(gate.data),
        static_cast<__hip_bfloat16*>(v.data),
    };
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __hip_bfloat16*>(x.data), static_cast<const __hip_bfloat16*>(weight.qdata),
        output);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail
