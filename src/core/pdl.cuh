#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    hipStream_t stream            = nullptr;
};

// HIP has no Programmatic Dependent Launch; the consumer kernel is launched with
// normal stream ordering, which is the conservative fallback semantics.
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline hipError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    hipLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    return hipLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// No-op: ordinary stream ordering already guarantees producer completion before the
// dependent kernel starts.
__device__ __forceinline__ void trigger_dependents() {}

// No-op: see trigger_dependents().
__device__ __forceinline__ void wait_for_dependencies() {}

} // namespace ninfer::pdl
