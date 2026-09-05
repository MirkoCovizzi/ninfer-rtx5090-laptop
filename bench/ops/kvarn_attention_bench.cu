#include "ninfer/ops/kvarn_attention.h"
#include "ninfer_bench_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kD          = ops::kKvarnHeadDim;
constexpr int kGroup      = ops::kKvarnGroup;
constexpr int kKvHeads    = 4;
constexpr int kQueryHeads = 24;

} // namespace

int main(int argc, char** argv) {
    try {
        int context        = 8192;
        int batch          = 1;
        int selected_width = 0;
        std::string_view selected_phase;
        bool tight_envelope = false;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--tight-envelope") {
                tight_envelope = true;
                continue;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing benchmark option value");
            }
            const char* value = argv[++index];
            if (argument == "--phase") {
                selected_phase = value;
                if (selected_phase != "cached" && selected_phase != "append" &&
                    selected_phase != "provisional" && selected_phase != "prefill") {
                    throw std::invalid_argument("unknown benchmark phase");
                }
                continue;
            }
            char* end         = nullptr;
            const long number = std::strtol(value, &end, 10);
            if (end == value || *end != '\0') {
                throw std::invalid_argument("invalid benchmark integer");
            }
            if (argument == "--context" && number >= 128 && number <= 262144) {
                context = static_cast<int>(number);
            } else if (argument == "--batch" && number >= 1 && number <= 8) {
                batch = static_cast<int>(number);
            } else if (argument == "--width" && number >= 1 && number <= 6) {
                selected_width = static_cast<int>(number);
            } else {
                throw std::invalid_argument(
                    "usage: ninfer_kvarn_attention_bench [--context 128..262144] "
                    "[--batch 1..8] [--width 1..6] "
                    "[--phase cached|append|provisional|prefill] "
                    "[--tight-envelope]");
            }
        }
        if (batch > 1 && selected_phase == "prefill") {
            throw std::invalid_argument("prefill benchmark requires --batch 1");
        }
        const int pages = (context + kGroup - 1) / kGroup + 1;
        DeviceBuffer records(static_cast<std::size_t>(pages) * batch * kKvHeads *
                             ops::kKvarnRecordBytes);
        DeviceBuffer tail_k(static_cast<std::size_t>(kD) * kGroup * kKvHeads *
                            ops::kKvarnTailSlots * batch * 2);
        DeviceBuffer tail_v(tail_k.bytes);
        DeviceBuffer markers(batch * ops::kKvarnTailSlots * sizeof(std::int32_t));
        DeviceBuffer block_table(static_cast<std::size_t>(pages) * batch * sizeof(std::int32_t));
        DeviceBuffer row(batch * sizeof(std::int32_t));
        std::vector<std::int32_t> table_host(pages * batch), row_host(batch);
        for (int page = 0; page < pages * batch; ++page) { table_host[page] = page; }
        for (int index = 0; index < batch; ++index) { row_host[index] = index; }
        block_table.copy_from_host(table_host.data(), block_table.bytes);
        row.copy_from_host(row_host.data(), row.bytes);
        ops::KvarnPagedBatchLayerView cache{
            .records = Tensor(records.p, DType::U8,
                              {ops::kKvarnRecordBytes / kGroup, kGroup, kKvHeads, pages * batch}),
            .tail_k =
                Tensor(tail_k.p, DType::BF16, {kD, kGroup, kKvHeads * ops::kKvarnTailSlots, batch}),
            .tail_v =
                Tensor(tail_v.p, DType::BF16, {kD, kGroup, kKvHeads * ops::kKvarnTailSlots, batch}),
            .tail_logical_pages = Tensor(markers.p, DType::I32, {ops::kKvarnTailSlots, batch}),
            .block_tables       = Tensor(block_table.p, DType::I32, {pages, batch}),
            .num_kv_heads       = kKvHeads,
        };
        Tensor row_tensor(row.p, DType::I32, {batch});

        const auto populate = [&](int count) {
            records.fill();
            tail_k.fill();
            tail_v.fill();
            markers.fill(0xff);
            for (int begin = 0; begin < count; begin += 1024) {
                const int width = std::min(1024, count - begin);
                DeviceBuffer key =
                    bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width * batch);
                DeviceBuffer value =
                    bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width * batch);
                std::vector<std::int32_t> host_positions(width * batch);
                for (int column = 0; column < width * batch; ++column) {
                    host_positions[column] = begin + column % width;
                }
                DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
                positions.copy_from_host(host_positions.data(), positions.bytes);
                Tensor kt(key.p, DType::BF16, {kD, kKvHeads, width, batch});
                Tensor vt(value.p, DType::BF16, {kD, kKvHeads, width, batch});
                Tensor pt(positions.p, DType::I32, {width, batch});
                ops::kvarn_kv_append(kt, vt, pt, Tensor{}, row_tensor, cache, false, nullptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        int measured_cases = 0;
        const auto measure = [&](const char* phase, int width, bool cached, bool append_only,
                                 bool provisional) {
            if (!selected_phase.empty() && selected_phase != phase) { return; }
            if (selected_width != 0 && selected_width != width) { return; }
            ++measured_cases;
            const int first = context - width;
            populate(cached ? context : first);
            DeviceBuffer query =
                bench::make_bf16(static_cast<std::size_t>(kD) * kQueryHeads * width * batch);
            DeviceBuffer key =
                bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width * batch);
            DeviceBuffer value = bench::make_bf16(key.bytes / 2);
            DeviceBuffer output(query.bytes);
            DeviceBuffer original_q(query.bytes), original_k(key.bytes), original_v(value.bytes);
            DeviceBuffer original_tail_k(tail_k.bytes), original_tail_v(tail_v.bytes),
                original_markers(markers.bytes);
            const std::size_t record_offset =
                static_cast<std::size_t>(first / kGroup) * kKvHeads * ops::kKvarnRecordBytes;
            const std::size_t touched_bytes =
                static_cast<std::size_t>((context + kGroup - 1) / kGroup - first / kGroup) *
                kKvHeads * ops::kKvarnRecordBytes;
            DeviceBuffer original_records(touched_bytes * batch);
            auto* touched_records = static_cast<std::uint8_t*>(records.p) + record_offset;
            const std::size_t row_record_bytes =
                static_cast<std::size_t>(pages) * kKvHeads * ops::kKvarnRecordBytes;
            const auto copy = [](void* destination, const void* source, std::size_t bytes) {
                CUDA_CHECK(
                    cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, nullptr));
            };
            copy(original_q.p, query.p, query.bytes);
            copy(original_k.p, key.p, key.bytes);
            copy(original_v.p, value.p, value.bytes);
            copy(original_tail_k.p, tail_k.p, tail_k.bytes);
            copy(original_tail_v.p, tail_v.p, tail_v.bytes);
            copy(original_markers.p, markers.p, markers.bytes);
            for (int index = 0; index < batch; ++index) {
                copy(static_cast<std::uint8_t*>(original_records.p) + index * touched_bytes,
                     touched_records + index * row_record_bytes, touched_bytes);
            }
            std::vector<std::int32_t> host_positions(width * batch), host_valid(batch, width);
            for (int column = 0; column < width * batch; ++column) {
                host_positions[column] = first + column % width;
            }
            DeviceBuffer valid(batch * sizeof(std::int32_t));
            valid.copy_from_host(host_valid.data(), valid.bytes);
            DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
            positions.copy_from_host(host_positions.data(), positions.bytes);
            Tensor qt(query.p, DType::BF16, {kD, kQueryHeads, width, batch});
            Tensor kt(key.p, DType::BF16, {kD, kKvHeads, width, batch});
            Tensor vt(value.p, DType::BF16, {kD, kKvHeads, width, batch});
            Tensor ot(output.p, DType::BF16, {kD, kQueryHeads, width, batch});
            Tensor pt(positions.p, DType::I32, {width, batch});
            Tensor ct(valid.p, DType::I32, {batch});
            const ops::CausalAttentionExecutionEnvelope envelope{
                batch == 1 || tight_envelope ? static_cast<std::uint32_t>(first + 1) : 1U,
                static_cast<std::uint32_t>(context)};
            WorkspaceArena workspace(ops::kvarn_attention_workspace_capacity_bytes(
                kQueryHeads, envelope, batch, width, width));
            cudaEvent_t start = nullptr, stop = nullptr;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            std::vector<double> samples;
            const int repeats = width >= 64 ? 3 : 30;
            for (int sample = -3; sample < repeats; ++sample) {
                // Restore represented inputs and the exact pre-append state outside the timed
                // interval.
                copy(query.p, original_q.p, query.bytes);
                copy(key.p, original_k.p, key.bytes);
                copy(value.p, original_v.p, value.bytes);
                copy(tail_k.p, original_tail_k.p, tail_k.bytes);
                copy(tail_v.p, original_tail_v.p, tail_v.bytes);
                copy(markers.p, original_markers.p, markers.bytes);
                for (int index = 0; index < batch; ++index) {
                    copy(touched_records + index * row_record_bytes,
                         static_cast<std::uint8_t*>(original_records.p) + index * touched_bytes,
                         touched_bytes);
                }
                CUDA_CHECK(cudaEventRecord(start));
                if (cached) {
                    ops::kvarn_attention_cached(qt, pt, row_tensor, 0.0625F, cache, envelope,
                                                workspace, ot, nullptr);
                } else if (append_only) {
                    ops::kvarn_kv_append(kt, vt, pt, Tensor{}, row_tensor, cache, false, nullptr);
                } else {
                    ops::kvarn_attention(qt, kt, vt, pt, provisional ? ct : Tensor{}, row_tensor,
                                         0.0625F, cache, provisional, envelope, workspace, ot,
                                         nullptr);
                }
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float milliseconds = 0;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                if (sample >= 0) { samples.push_back(milliseconds * 1000.0); }
            }
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            const auto timing = bench::summarize_timings(std::move(samples));
            std::printf("phase=%s context=%d width=%d batch=%d min_visible=%u closes_page=%d "
                        "median_us=%.3f min_us=%.3f "
                        "p95_us=%.3f\n",
                        phase, context, width, batch, envelope.min_visible_keys,
                        !cached && context / kGroup > first / kGroup, timing.median_us,
                        timing.min_us, timing.p95_us);
        };
        measure("cached", 1, true, false, false);
        for (int width : {1, 4, 6}) { measure("append", width, false, true, false); }
        for (int width = 1; width <= 6; ++width) {
            measure("provisional", width, false, false, true);
        }
        if (batch == 1) { measure("prefill", std::min(context, 1024), false, false, false); }
        if (measured_cases == 0) {
            throw std::invalid_argument("phase and width selectors matched no benchmark case");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
