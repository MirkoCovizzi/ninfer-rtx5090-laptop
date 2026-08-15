// HIP has no NVFP4 tensor hardware (the e4m3/fp4 MMA and TMA paths are Blackwell
// SM120 features), so the NVFP4 weight profile is not registered on HIP. The
// family dispatchers still reference the NVFP4 plan/dispatch entry points; these
// stubs keep them linkable and fail loudly if a stray NVFP4 weight ever reaches
// them. The real implementations live in the NVFP4 .cu/.cpp sources, which the
// HIP build excludes.
#if defined(__HIP_PLATFORM_AMD__)

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
[[noreturn]] void nvfp4_unsupported_on_hip(const char* operation) {
    throw std::invalid_argument(std::string(operation) + ": NVFP4 weights are not supported on HIP");
}
} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 linear workspace");
}

void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                    hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 linear");
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 attn_input_proj workspace");
}

void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                               Tensor&, LinearPolicy, WorkspaceArena*, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 attn_input_proj");
}

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn_input_proj workspace");
}

void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                              WorkspaceArena*, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn_input_proj");
}

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn conv snapshot");
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn snapshot workspace");
}

void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                 Tensor&, Tensor&, LinearPolicy, WorkspaceArena&, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn snapshot");
}

void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn record");
}

void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 gdn record post");
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                      std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 linear_add workspace");
}

void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                               WorkspaceArena&, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 linear_add");
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported_on_hip("nvfp4 linear_swiglu workspace");
}

void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                  WorkspaceArena&, hipStream_t) {
    nvfp4_unsupported_on_hip("nvfp4 linear_swiglu");
}

} // namespace ninfer::ops::detail

#endif // __HIP_PLATFORM_AMD__
