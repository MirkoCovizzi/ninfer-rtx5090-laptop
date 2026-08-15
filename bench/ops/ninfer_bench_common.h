#include "hip/hip_runtime.h"
#pragma once
//
// ninfer_bench_common.h — shared bench harness for L1 op performance binaries.
//
// Adapted from ~/chunked_gdn/bench/bench_common.h. Timing uses CUDA events
// with inner-iter batching to amortize the per-sample host sync; throughput is
// reported from the MEDIAN per-launch time (robust to host scheduling spikes).
//
// IMPORTANT (docs/op-development.md §9): the GB/s printed here is a convenience
// readout, not a universal acceptance gate. Interpret it with cache conditions,
// a same-topology payload control, and only the profiler evidence needed for the
// concrete kernel question.

#include "core/arena.h"
#include "core/device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::bench {

constexpr double kRooflineGBs = 1792.0; // RTX 5090 GDDR7 bandwidth roofline.

inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return std::uint16_t(u >> 16);
}

// Device bf16 buffer filled with a small varied ramp (avoids all-zero special
// paths; exact values are irrelevant to bandwidth). Returns an owning DeviceBuffer.
inline DeviceBuffer make_bf16(std::size_t n) {
    std::vector<std::uint16_t> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(0.5f - float(i % 251) / 250.0f);
    DeviceBuffer d(n * 2);
    d.copy_from_host(h.data(), d.bytes);
    return d;
}

inline DeviceBuffer make_zeros(std::size_t bytes) {
    DeviceBuffer d(bytes);
    d.fill();
    return d;
}

// hipDeviceProp_t memory-clock fields were removed in CUDA 13; the in-process
// GB/s is informational anyway (ncu is the acceptance gate), so report against
// the known RTX 5090 roofline constant.
inline double device_peak_bw_gbs(int /*dev*/ = 0) { return kRooflineGBs; }

struct ColdTiming {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

class TimedGraph {
public:
    TimedGraph() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~TimedGraph() {
        if (exec_ != nullptr) { hipGraphExecDestroy(exec_); }
        if (graph_ != nullptr) { hipGraphDestroy(graph_); }
        if (start_ != nullptr) { hipEventDestroy(start_); }
        if (stop_ != nullptr) { hipEventDestroy(stop_); }
    }

    TimedGraph(const TimedGraph&)            = delete;
    TimedGraph& operator=(const TimedGraph&) = delete;

    template <class Body>
    void capture(hipStream_t stream, Body&& body) {
        if (graph_ != nullptr || exec_ != nullptr) {
            throw std::logic_error("benchmark graph is already captured");
        }
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        try {
            body(stream);
        } catch (...) {
            hipGraph_t discard = nullptr;
            hipStreamEndCapture(stream, &discard);
            if (discard != nullptr) { hipGraphDestroy(discard); }
            throw;
        }
        HIP_CHECK(hipStreamEndCapture(stream, &graph_));
        HIP_CHECK(hipGraphInstantiate(&exec_, graph_, 0));
        HIP_CHECK(hipGraphGetNodes(graph_, nullptr, &nodes_));
        if (nodes_ == 0) { throw std::runtime_error("captured benchmark graph is empty"); }
    }

    void launch(hipStream_t stream) const { HIP_CHECK(hipGraphLaunch(exec_, stream)); }

    double launch_timed(hipStream_t stream) const {
        HIP_CHECK(hipEventRecord(start_, stream));
        launch(stream);
        HIP_CHECK(hipEventRecord(stop_, stream));
        HIP_CHECK(hipEventSynchronize(stop_));
        float milliseconds = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

    [[nodiscard]] std::size_t nodes() const noexcept { return nodes_; }

private:
    hipGraph_t graph_    = nullptr;
    hipGraphExec_t exec_ = nullptr;
    hipEvent_t start_    = nullptr;
    hipEvent_t stop_     = nullptr;
    std::size_t nodes_    = 0;
};

inline ColdTiming summarize_timings(std::vector<double> samples) {
    if (samples.empty()) { throw std::invalid_argument("cannot summarize empty timings"); }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double fraction) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

template <class Launch>
ColdTiming measure_launch(Launch&& launch, hipStream_t stream, int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("benchmark requires nonnegative warmup and positive repeat");
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for (int index = 0; index < warmup; ++index) { launch(stream); }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        HIP_CHECK(hipEventRecord(start, stream));
        launch(stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return summarize_timings(std::move(samples));
}

inline ColdTiming measure_graph(const TimedGraph& graph, hipStream_t stream, int warmup,
                                int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument("benchmark requires nonnegative warmup and positive repeat");
    }
    for (int index = 0; index < warmup; ++index) { graph.launch(stream); }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) { samples.push_back(graph.launch_timed(stream)); }
    return summarize_timings(std::move(samples));
}

inline void flush_l2(DeviceBuffer& flush, hipStream_t stream) {
    HIP_CHECK(hipMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
}

template <class Launch>
ColdTiming measure_cold_launch(Launch&& launch, DeviceBuffer& flush, hipStream_t stream,
                               int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "cold benchmark requires nonnegative warmup and positive repeat");
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int index = 0; index < warmup; ++index) {
        flush_l2(flush, stream);
        launch(stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        flush_l2(flush, stream);
        HIP_CHECK(hipEventRecord(start, stream));
        launch(stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return {
        samples[samples.size() / 2],
        samples.front(),
        samples[std::min(samples.size() - 1,
                         static_cast<std::size_t>(0.95 * static_cast<double>(samples.size())))],
    };
}

inline ColdTiming measure_cold_graph(const TimedGraph& graph, DeviceBuffer& flush,
                                     hipStream_t stream, int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) {
        throw std::invalid_argument(
            "cold benchmark requires nonnegative warmup and positive repeat");
    }
    for (int index = 0; index < warmup; ++index) {
        flush_l2(flush, stream);
        graph.launch(stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        flush_l2(flush, stream);
        samples.push_back(graph.launch_timed(stream));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double fraction) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

struct Result {
    int n_runs       = 0;
    int inner_iters  = 1;
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
    double mean_us   = 0.0;
    double gbs       = 0.0; // from median
};

using launch_fn = std::function<void(hipStream_t)>;

inline Result bench_loop(const launch_fn& launch, double bytes_moved, int warmup = 20,
                         int repeat = 100, int min_time_ms = 500) {
    hipStream_t stream = nullptr;
    hipEvent_t a, b;
    hipEventCreate(&a);
    hipEventCreate(&b);

    for (int i = 0; i < warmup; ++i) launch(stream);
    hipStreamSynchronize(stream);

    // Auto-size inner_iters so each timed batch is ~500us (amortize sync wait).
    int inner = 0;
    {
        constexpr int probe = 4;
        hipEventRecord(a, stream);
        for (int i = 0; i < probe; ++i) launch(stream);
        hipEventRecord(b, stream);
        hipEventSynchronize(b);
        float ms = 0.f;
        hipEventElapsedTime(&ms, a, b);
        const double per_us = double(ms) * 1000.0 / probe;
        inner = std::max(1, std::min(1024, int(std::ceil(500.0 / std::max(per_us, 1.0)))));
    }

    std::vector<double> samples;
    long long total_us = 0;
    while (int(samples.size()) < repeat || total_us < (long long)min_time_ms * 1000) {
        hipEventRecord(a, stream);
        for (int i = 0; i < inner; ++i) launch(stream);
        hipEventRecord(b, stream);
        hipEventSynchronize(b);
        float ms = 0.f;
        hipEventElapsedTime(&ms, a, b);
        const double batch_us = double(ms) * 1000.0;
        samples.push_back(batch_us / inner);
        total_us += (long long)batch_us;
        if (samples.size() > 100000) break;
    }
    hipEventDestroy(a);
    hipEventDestroy(b);

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        const std::size_t idx = std::min(sorted.size() - 1, std::size_t(q * sorted.size()));
        return sorted[idx];
    };
    double sum = 0.0;
    for (double v : samples) sum += v;

    Result r;
    r.n_runs         = int(samples.size());
    r.inner_iters    = inner;
    r.median_us      = pct(0.50);
    r.min_us         = sorted.front();
    r.p95_us         = pct(0.95);
    r.mean_us        = sum / samples.size();
    const double sec = r.median_us * 1e-6;
    r.gbs            = (sec > 0.0) ? bytes_moved / sec / 1e9 : 0.0;
    return r;
}

inline void print_result(const char* tag, const Result& r) {
    std::printf(
        "%-32s median=%8.2f us  min=%8.2f us  p95=%8.2f us  %8.1f GB/s  (%.1f%% of %.0f GB/s "
        "roofline)\n",
        tag, r.median_us, r.min_us, r.p95_us, r.gbs, r.gbs / kRooflineGBs * 100.0, kRooflineGBs);
}

} // namespace ninfer::bench
