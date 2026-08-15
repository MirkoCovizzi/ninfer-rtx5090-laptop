#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace ninfer {

void hip_check(hipError_t err, const char* expr, const char* file, int line);

#define HIP_CHECK(expr) ::ninfer::hip_check((expr), #expr, __FILE__, __LINE__)

struct DeviceContext {
    int device               = 0;
    hipStream_t stream      = nullptr;
    hipStream_t load_stream = nullptr;
    hipDeviceProp_t props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

class HipEventTimer {
public:
    explicit HipEventTimer(const DeviceContext& ctx);
    ~HipEventTimer();

    HipEventTimer(const HipEventTimer&)            = delete;
    HipEventTimer& operator=(const HipEventTimer&) = delete;
    HipEventTimer(HipEventTimer&& other) noexcept;
    HipEventTimer& operator=(HipEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    hipStream_t stream_ = nullptr;
    hipEvent_t start_   = nullptr;
    hipEvent_t stop_    = nullptr;
};

} // namespace ninfer
