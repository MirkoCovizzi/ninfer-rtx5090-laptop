#include "ninfer/ops/kvarn_attention.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

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
    if (argc == 1) return 8192;
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
        records.fill();
        tail_k.fill();
        tail_v.fill();
        const std::vector<std::int32_t> marker_host(ops::kKvarnTailSlots, -1);
        markers.copy_from_host(marker_host.data(), markers.bytes);
        std::vector<std::int32_t> table_host(pages);
        for (int page = 0; page < pages; ++page) table_host[page] = page;
        block_table.copy_from_host(table_host.data(), block_table.bytes);
        const std::int32_t zero = 0;
        row.copy_from_host(&zero, sizeof(zero));

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

        constexpr int kAppendChunk = 1024;
        for (int begin = 0; begin < context; begin += kAppendChunk) {
            const int width    = std::min(kAppendChunk, context - begin);
            DeviceBuffer key   = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
            DeviceBuffer value = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
            std::vector<std::int32_t> host_positions(width);
            for (int column = 0; column < width; ++column) host_positions[column] = begin + column;
            DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
            positions.copy_from_host(host_positions.data(), positions.bytes);
            Tensor key_tensor(key.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor value_tensor(value.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor position_tensor(positions.p, DType::I32, {width, 1});
            ops::kvarn_kv_append(key_tensor, value_tensor, position_tensor, Tensor{}, row_tensor,
                                 cache, false, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto measure_append = [&](int width) {
            DeviceBuffer key   = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
            DeviceBuffer value = bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
            std::vector<std::int32_t> host_positions(width);
            for (int column = 0; column < width; ++column) {
                host_positions[column] = context + column;
            }
            DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
            positions.copy_from_host(host_positions.data(), positions.bytes);
            Tensor key_tensor(key.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor value_tensor(value.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor position_tensor(positions.p, DType::I32, {width, 1});
            const auto timing = bench::measure_launch(
                [&](cudaStream_t stream) {
                    ops::kvarn_kv_append(key_tensor, value_tensor, position_tensor, Tensor{},
                                         row_tensor, cache, false, stream);
                },
                nullptr, 5, 100);
            std::printf("append=%d median_us=%.3f min_us=%.3f p95_us=%.3f\n", width,
                        timing.median_us, timing.min_us, timing.p95_us);
        };
        measure_append(1);
        measure_append(4);
        measure_append(6);

        DeviceBuffer query = bench::make_bf16(static_cast<std::size_t>(kD) * kQueryHeads);
        DeviceBuffer output(static_cast<std::size_t>(kD) * kQueryHeads * 2);
        DeviceBuffer position(sizeof(std::int32_t));
        const std::int32_t last = context - 1;
        position.copy_from_host(&last, sizeof(last));
        Tensor query_tensor(query.p, DType::BF16, {kD, kQueryHeads, 1, 1});
        Tensor output_tensor(output.p, DType::BF16, {kD, kQueryHeads, 1, 1});
        Tensor position_tensor(position.p, DType::I32, {1, 1});
        WorkspaceArena workspace(ops::kvarn_attention_workspace_capacity_bytes(
            kQueryHeads, {static_cast<std::uint32_t>(context), static_cast<std::uint32_t>(context)},
            1, 1, 1));
        const auto timing = bench::measure_launch(
            [&](cudaStream_t stream) {
                ops::kvarn_attention_cached(
                    query_tensor, position_tensor, row_tensor, 0.0625F, cache,
                    {static_cast<std::uint32_t>(context), static_cast<std::uint32_t>(context)},
                    workspace, output_tensor, stream);
            },
            nullptr, 5, 30);
        std::printf("context=%d median_us=%.3f min_us=%.3f p95_us=%.3f\n", context,
                    timing.median_us, timing.min_us, timing.p95_us);

        const auto measure_mtp = [&](int width) {
            DeviceBuffer mtp_query =
                bench::make_bf16(static_cast<std::size_t>(kD) * kQueryHeads * width);
            DeviceBuffer mtp_key =
                bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
            DeviceBuffer mtp_value =
                bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * width);
            DeviceBuffer mtp_output(static_cast<std::size_t>(kD) * kQueryHeads * width * 2);
            std::vector<std::int32_t> mtp_positions(width);
            for (int column = 0; column < width; ++column) {
                mtp_positions[column] = context - width + column;
            }
            DeviceBuffer mtp_position(mtp_positions.size() * sizeof(std::int32_t));
            mtp_position.copy_from_host(mtp_positions.data(), mtp_position.bytes);
            DeviceBuffer mtp_valid(sizeof(std::int32_t));
            mtp_valid.copy_from_host(&width, sizeof(width));
            Tensor mq(mtp_query.p, DType::BF16, {kD, kQueryHeads, width, 1});
            Tensor mk(mtp_key.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor mv(mtp_value.p, DType::BF16, {kD, kKvHeads, width, 1});
            Tensor mo(mtp_output.p, DType::BF16, {kD, kQueryHeads, width, 1});
            Tensor mp(mtp_position.p, DType::I32, {width, 1});
            Tensor mc(mtp_valid.p, DType::I32, {1});
            WorkspaceArena mtp_workspace(ops::kvarn_attention_workspace_capacity_bytes(
                kQueryHeads,
                {static_cast<std::uint32_t>(context), static_cast<std::uint32_t>(context)}, 1,
                width, width));
            const auto mtp_timing = bench::measure_launch(
                [&](cudaStream_t stream) {
                    ops::kvarn_attention(
                        mq, mk, mv, mp, mc, row_tensor, 0.0625F, cache, true,
                        {static_cast<std::uint32_t>(context), static_cast<std::uint32_t>(context)},
                        mtp_workspace, mo, stream);
                },
                nullptr, 5, 30);
            std::printf("mtp%d_context=%d width=%d median_us=%.3f min_us=%.3f p95_us=%.3f\n",
                        width - 1, context, width, mtp_timing.median_us, mtp_timing.min_us,
                        mtp_timing.p95_us);
        };
        measure_mtp(4);
        measure_mtp(6);

        const int prefill = std::min(context, 1024);
        DeviceBuffer prefill_query =
            bench::make_bf16(static_cast<std::size_t>(kD) * kQueryHeads * prefill);
        DeviceBuffer prefill_key =
            bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * prefill);
        DeviceBuffer prefill_value =
            bench::make_bf16(static_cast<std::size_t>(kD) * kKvHeads * prefill);
        DeviceBuffer prefill_output(static_cast<std::size_t>(kD) * kQueryHeads * prefill * 2);
        std::vector<std::int32_t> prefill_positions(prefill);
        for (int column = 0; column < prefill; ++column) {
            prefill_positions[column] = context - prefill + column;
        }
        DeviceBuffer prefill_position(prefill_positions.size() * sizeof(std::int32_t));
        prefill_position.copy_from_host(prefill_positions.data(), prefill_position.bytes);
        Tensor pq(prefill_query.p, DType::BF16, {kD, kQueryHeads, prefill, 1});
        Tensor pk(prefill_key.p, DType::BF16, {kD, kKvHeads, prefill, 1});
        Tensor pv(prefill_value.p, DType::BF16, {kD, kKvHeads, prefill, 1});
        Tensor po(prefill_output.p, DType::BF16, {kD, kQueryHeads, prefill, 1});
        Tensor pp(prefill_position.p, DType::I32, {prefill, 1});
        WorkspaceArena prefill_workspace(ops::kvarn_attention_workspace_capacity_bytes(
            kQueryHeads, {1, static_cast<std::uint32_t>(context)}, 1, prefill, prefill));
        const auto prefill_timing = bench::measure_launch(
            [&](cudaStream_t stream) {
                ops::kvarn_attention(pq, pk, pv, pp, Tensor{}, row_tensor, 0.0625F, cache, false,
                                     {1, static_cast<std::uint32_t>(context)}, prefill_workspace,
                                     po, stream);
            },
            nullptr, 1, 3);
        std::printf("prefill=%d median_us=%.3f tok_per_s=%.1f\n", prefill, prefill_timing.median_us,
                    prefill * 1.0e6 / prefill_timing.median_us);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
