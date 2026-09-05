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

int parse_context(int argc, char** argv) {
    if (argc == 1) { return 8192; }
    if (argc != 3 || std::string_view(argv[1]) != "--context") {
        throw std::invalid_argument("usage: ninfer_kvarn_attention_bench [--context N]");
    }
    char* end        = nullptr;
    const long value = std::strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value < 128 || value > 262144) {
        throw std::invalid_argument("context must be in [128,262144]");
    }
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int context = parse_context(argc, argv);
        const int pages   = (context + kGroup - 1) / kGroup + 1;
        DeviceBuffer records(static_cast<std::size_t>(pages) * kKvHeads * ops::kKvarnRecordBytes);
        DeviceBuffer tail_k(static_cast<std::size_t>(kD) * kGroup * kKvHeads *
                            ops::kKvarnTailSlots * 2);
        DeviceBuffer tail_v(tail_k.bytes);
        DeviceBuffer markers(ops::kKvarnTailSlots * sizeof(std::int32_t));
        DeviceBuffer block_table(static_cast<std::size_t>(pages) * sizeof(std::int32_t));
        DeviceBuffer row(sizeof(std::int32_t));
        std::vector<std::int32_t> table_host(pages);
        for (int page = 0; page < pages; ++page) { table_host[page] = page; }
        block_table.copy_from_host(table_host.data(), block_table.bytes);
        row.fill();
        ops::KvarnPagedBatchLayerView cache{
            .records = Tensor(records.p, DType::U8,
                              {ops::kKvarnRecordBytes / kGroup, kGroup, kKvHeads, pages}),
            .tail_k =
                Tensor(tail_k.p, DType::BF16, {kD, kGroup, kKvHeads * ops::kKvarnTailSlots, 1}),
            .tail_v =
                Tensor(tail_v.p, DType::BF16, {kD, kGroup, kKvHeads * ops::kKvarnTailSlots, 1}),
            .tail_logical_pages = Tensor(markers.p, DType::I32, {ops::kKvarnTailSlots, 1}),
            .block_tables       = Tensor(block_table.p, DType::I32, {pages, 1}),
            .num_kv_heads       = kKvHeads,
        };
        Tensor row_tensor(row.p, DType::I32, {1});

        const auto populate = [&](int count) {
            records.fill();
            tail_k.fill();
            tail_v.fill();
            markers.fill(0xff);
            for (int begin = 0; begin < count; begin += 1024) {
                const int width = std::min(1024, count - begin);
                DeviceBuffer key =
                    bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
                DeviceBuffer value =
                    bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
                std::vector<std::int32_t> host_positions(width);
                for (int column = 0; column < width; ++column) {
                    host_positions[column] = begin + column;
                }
                DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
                positions.copy_from_host(host_positions.data(), positions.bytes);
                Tensor kt(key.p, DType::BF16, {kD, kKvHeads, width, 1});
                Tensor vt(value.p, DType::BF16, {kD, kKvHeads, width, 1});
                Tensor pt(positions.p, DType::I32, {width, 1});
                ops::kvarn_kv_append(kt, vt, pt, Tensor{}, row_tensor, cache, false, nullptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        };

        const auto measure = [&](const char* phase, int width, bool cached, bool append_only,
                                 bool provisional) {
            const int first = context - width;
            populate(cached ? context : first);
            DeviceBuffer query =
                bench::make_bf16(static_cast<std::size_t>(kD) * kQueryHeads * width);
            DeviceBuffer key   = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
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
            DeviceBuffer original_records(touched_bytes);
            auto* touched_records = static_cast<std::uint8_t*>(records.p) + record_offset;
            const auto copy       = [](void* destination, const void* source, std::size_t bytes) {
                CUDA_CHECK(
                    cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, nullptr));
            };
            copy(original_q.p, query.p, query.bytes);
            copy(original_k.p, key.p, key.bytes);
            copy(original_v.p, value.p, value.bytes);
            copy(original_tail_k.p, tail_k.p, tail_k.bytes);
            copy(original_tail_v.p, tail_v.p, tail_v.bytes);
            copy(original_markers.p, markers.p, markers.bytes);
            copy(original_records.p, touched_records, touched_bytes);
            std::vector<std::int32_t> host_positions(width);
            for (int column = 0; column < width; ++column) {
                host_positions[column] = first + column;
            }
            DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
            positions.copy_from_host(host_positions.data(), positions.bytes);
            Tensor qt(query.p, DType::BF16, {kD, kQueryHeads, width, 1});
            Tensor kt(key.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor vt(value.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor ot(output.p, DType::BF16, {kD, kQueryHeads, width, 1});
            Tensor pt(positions.p, DType::I32, {width, 1});
            const ops::CausalAttentionExecutionEnvelope envelope{
                static_cast<std::uint32_t>(first + 1), static_cast<std::uint32_t>(context)};
            WorkspaceArena workspace(ops::kvarn_attention_workspace_capacity_bytes(
                kQueryHeads, envelope, 1, width, width));
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
                copy(touched_records, original_records.p, touched_bytes);
                CUDA_CHECK(cudaEventRecord(start));
                if (cached) {
                    ops::kvarn_attention_cached(qt, pt, row_tensor, 0.0625F, cache, envelope,
                                                workspace, ot, nullptr);
                } else if (append_only) {
                    ops::kvarn_kv_append(kt, vt, pt, Tensor{}, row_tensor, cache, false, nullptr);
                } else {
                    ops::kvarn_attention(qt, kt, vt, pt, Tensor{}, row_tensor, 0.0625F, cache,
                                         provisional, envelope, workspace, ot, nullptr);
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
            std::printf("phase=%s context=%d width=%d closes_page=%d median_us=%.3f min_us=%.3f "
                        "p95_us=%.3f\n",
                        phase, context, width, !cached && context / kGroup > first / kGroup,
                        timing.median_us, timing.min_us, timing.p95_us);
        };
        measure("cached", 1, true, false, false);
        for (int width : {1, 4, 6}) { measure("append", width, false, true, false); }
        for (int width = 1; width <= 6; ++width) {
            measure("provisional", width, false, false, true);
        }
        measure("prefill", std::min(context, 1024), false, false, false);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
