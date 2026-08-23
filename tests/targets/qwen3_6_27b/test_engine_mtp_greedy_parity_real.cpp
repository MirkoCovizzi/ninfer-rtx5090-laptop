#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutputTokens = 128;
constexpr std::array<std::uint32_t, 5> kMtpDraftCounts{1, 2, 3, 4, 5};

struct KvProfile {
    std::string_view name;
    ninfer::KvCacheStorage storage;
};

constexpr std::array kKvProfiles{
    KvProfile{"bf16", ninfer::KvCacheStorage::BFloat16},
    KvProfile{"int8", ninfer::KvCacheStorage::Int8Group64},
};

ninfer::EngineOptions engine_options(const char* artifact, ninfer::KvCacheStorage kv_storage,
                                     std::uint32_t mtp_draft_tokens) {
    const bool mtp = mtp_draft_tokens != 0;
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 512;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.max_concurrency      = 1;
    options.prefill_chunk        = 128;
    options.kv_cache             = kv_storage;
    options.use_cuda_graph       = false;
    options.speculative.backend  = mtp ? ninfer::SpeculativeBackend::Mtp
                                       : ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp_draft_tokens;
    options.speculative.proposal_head =
        mtp ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
    return options;
}

ninfer::PromptInput prompt() {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Write snake in Python, but in a code block. Do not use tools.",
        .media = {},
    });
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = true;
    return input;
}

std::vector<ninfer::TokenId> generate(const char* artifact, ninfer::KvCacheStorage kv_storage,
                                       std::uint32_t mtp_draft_tokens) {
    const std::string route = mtp_draft_tokens == 0
                                  ? "ordinary"
                                  : "MTP k=" + std::to_string(mtp_draft_tokens);
    try {
        ninfer::Engine engine(engine_options(artifact, kv_storage, mtp_draft_tokens));
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = kOutputTokens;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.sampling.seed           = 424242;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;
        ninfer::GenerationResult result = engine.generate(engine.prepare(prompt()), request);
        if (result.generated_token_ids.size() != kOutputTokens ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            throw std::runtime_error("did not reach its fixed output limit");
        }
        return result.generated_token_ids;
    } catch (const std::exception& error) {
        throw std::runtime_error(route + ": " + error.what());
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_MTP_GREEDY_PARITY_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_MTP_GREEDY_PARITY_WEIGHTS is not set\n";
        return 77;
    }

    try {
        for (const KvProfile profile : kKvProfiles) {
            const std::vector<ninfer::TokenId> ordinary =
                generate(artifact, profile.storage, 0);
            for (const std::uint32_t draft_tokens : kMtpDraftCounts) {
                const std::vector<ninfer::TokenId> mtp =
                    generate(artifact, profile.storage, draft_tokens);
                const auto [ordinary_mismatch, mtp_mismatch] =
                    std::mismatch(ordinary.begin(), ordinary.end(), mtp.begin(), mtp.end());
                if (ordinary_mismatch != ordinary.end()) {
                    const std::size_t index =
                        static_cast<std::size_t>(ordinary_mismatch - ordinary.begin());
                    std::cerr << "greedy MTP parity mismatch for " << profile.name << " KV at k="
                              << draft_tokens << ", generated token " << index
                              << ": ordinary=" << *ordinary_mismatch
                              << " mtp=" << *mtp_mismatch << '\n';
                    return 1;
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "greedy MTP parity test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}
