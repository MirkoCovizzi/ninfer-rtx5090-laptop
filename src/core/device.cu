#include "core/device.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string hip_error_message(const char* prefix, hipError_t err) {
    return std::string(prefix) + ": " + hipGetErrorName(err) + ": " + hipGetErrorString(err);
}

void log_hip_error(const char* op, hipError_t err) noexcept {
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP cleanup failed during %s: %s: %s\n", op, hipGetErrorName(err),
                     hipGetErrorString(err));
    }
}

void destroy_stream(hipStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_hip_error("hipStreamDestroy", hipStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(hipEvent_t& event) noexcept {
    if (event != nullptr) {
        log_hip_error("hipEventDestroy", hipEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void hip_check(hipError_t err, const char* expr, const char* file, int line) {
    if (err == hipSuccess) { return; }
    std::fprintf(stderr, "%s:%d: HIP_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 hipGetErrorName(err), hipGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no HIP devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid HIP device id"); }

    err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipSetDevice failed", err));
    }

    err = hipGetDeviceProperties(&props, device_id);
    if (err != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipGetDeviceProperties failed", err));
    }

    hipStream_t compute = nullptr;
    hipStream_t load    = nullptr;
    err                  = hipStreamCreateWithFlags(&compute, hipStreamNonBlocking);
    if (err != hipSuccess) {
        throw std::runtime_error(
            hip_error_message("hipStreamCreateWithFlags(stream) failed", err));
    }

    err = hipStreamCreateWithFlags(&load, hipStreamNonBlocking);
    if (err != hipSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            hip_error_message("hipStreamCreateWithFlags(load_stream) failed", err));
    }

    stream      = compute;
    load_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || load_stream != nullptr) {
        log_hip_error("hipSetDevice", hipSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), load_stream(other.load_stream),
      props(other.props) {
    other.stream      = nullptr;
    other.load_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || load_stream != nullptr) {
        log_hip_error("hipSetDevice", hipSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);

    device      = other.device;
    props       = other.props;
    stream      = other.stream;
    load_stream = other.load_stream;

    other.stream      = nullptr;
    other.load_stream = nullptr;
    return *this;
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const { HIP_CHECK(hipStreamSynchronize(stream)); }

HipEventTimer::HipEventTimer(const DeviceContext& ctx) : stream_(ctx.stream) {
    hipError_t err = hipSetDevice(ctx.device);
    if (err != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipSetDevice(timer) failed", err));
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    err               = hipEventCreate(&start);
    if (err != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipEventCreate(start) failed", err));
    }

    err = hipEventCreate(&stop);
    if (err != hipSuccess) {
        destroy_event(start);
        throw std::runtime_error(hip_error_message("hipEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

HipEventTimer::~HipEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

HipEventTimer::HipEventTimer(HipEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

HipEventTimer& HipEventTimer::operator=(HipEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void HipEventTimer::start() { HIP_CHECK(hipEventRecord(start_, stream_)); }

void HipEventTimer::record_stop() { HIP_CHECK(hipEventRecord(stop_, stream_)); }

float HipEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float HipEventTimer::stop_ms() {
    record_stop();
    HIP_CHECK(hipEventSynchronize(stop_));
    return elapsed_ms();
}

} // namespace ninfer
