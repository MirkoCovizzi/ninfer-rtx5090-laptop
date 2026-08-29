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
constexpr std::uint32_t kMaximumConcurrency = 8;
constexpr std::array<std::uint32_t, 8> kConcurrencyFrontiers{1, 2, 3, 4,
                                                             5, 6, 7, kMaximumConcurrency};

struct KvProfile {
    std::string_view name;
    ninfer::KvCacheStorage storage;
};

constexpr std::array kKvProfiles{
    KvProfile{"bf16", ninfer::KvCacheStorage::BFloat16},
    KvProfile{"int8", ninfer::KvCacheStorage::Int8Group64},
};

ninfer::EngineOptions engine_options(const char* artifact, ninfer::KvCacheStorage kv_storage,
                                     std::uint32_t mtp_draft_tokens,
                                     std::uint32_t max_concurrency = 1) {
    const bool mtp = mtp_draft_tokens != 0;
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 512;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(512 * max_concurrency);
    options.max_concurrency = max_concurrency;
    options.prefill_chunk   = 128;
    options.kv_cache        = kv_storage;
    options.use_cuda_graph  = false;
    options.speculative.backend =
        mtp ? ninfer::SpeculativeBackend::Mtp : ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp_draft_tokens;
    options.speculative.proposal_head =
        mtp ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
    return options;
}

ninfer::RequestOptions greedy_request() {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = kOutputTokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 424242;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    return request;
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
    const std::string route =
        mtp_draft_tokens == 0 ? "ordinary" : "MTP k=" + std::to_string(mtp_draft_tokens);
    try {
        ninfer::Engine engine(engine_options(artifact, kv_storage, mtp_draft_tokens));
        ninfer::GenerationResult result =
            engine.generate(engine.prepare(prompt()), greedy_request());
        if (result.generated_token_ids.size() != kOutputTokens ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            throw std::runtime_error("did not reach its fixed output limit");
        }
        return result.generated_token_ids;
    } catch (const std::exception& error) { throw std::runtime_error(route + ": " + error.what()); }
}

void verify_result(std::string_view label, const ninfer::GenerationResult& result,
                   const std::vector<ninfer::TokenId>& expected) {
    const auto& actual = result.generated_token_ids;
    if (actual.size() != expected.size() ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        throw std::runtime_error(std::string(label) + " did not reach its fixed output limit");
    }
    const auto [expected_mismatch, actual_mismatch] =
        std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (expected_mismatch != expected.end()) {
        const std::size_t index = static_cast<std::size_t>(expected_mismatch - expected.begin());
        throw std::runtime_error(std::string(label) + " mismatch at token " +
                                 std::to_string(index) +
                                 ": expected=" + std::to_string(*expected_mismatch) +
                                 " actual=" + std::to_string(*actual_mismatch));
    }
}

void verify_batch(ninfer::Engine& engine, KvProfile profile, std::uint32_t draft_tokens,
                  std::uint32_t concurrency, std::uint32_t iteration,
                  const std::vector<ninfer::TokenId>& expected) {
    std::vector<ninfer::PreparedPrompt> prompts;
    prompts.reserve(concurrency);
    for (std::uint32_t row = 0; row < concurrency; ++row) {
        prompts.push_back(engine.prepare(prompt()));
    }
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(concurrency);
    for (std::uint32_t row = 0; row < concurrency; ++row) {
        handles.push_back(engine.submit(std::move(prompts[row]), greedy_request()));
    }
    for (std::uint32_t row = 0; row < concurrency; ++row) {
        const std::string label =
            std::string(profile.name) + " MTP k=" + std::to_string(draft_tokens) +
            " C=" + std::to_string(concurrency) + " iteration=" + std::to_string(iteration) +
            " row=" + std::to_string(row);
        verify_result(label, handles[row].wait(), expected);
    }
}

void verify_route(const char* artifact, KvProfile profile, std::uint32_t draft_tokens,
                  const std::vector<ninfer::TokenId>& expected) {
    ninfer::Engine engine(
        engine_options(artifact, profile.storage, draft_tokens, kMaximumConcurrency));
    for (const std::uint32_t concurrency : kConcurrencyFrontiers) {
        verify_batch(engine, profile, draft_tokens, concurrency, 0, expected);
        if (concurrency == kMaximumConcurrency) {
            verify_batch(engine, profile, draft_tokens, concurrency, 1, expected);
        }
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
            const std::vector<ninfer::TokenId> ordinary = generate(artifact, profile.storage, 0);
            verify_route(artifact, profile, 0, ordinary);
            for (const std::uint32_t draft_tokens : kMtpDraftCounts) {
                verify_route(artifact, profile, draft_tokens, ordinary);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "greedy MTP parity test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}
