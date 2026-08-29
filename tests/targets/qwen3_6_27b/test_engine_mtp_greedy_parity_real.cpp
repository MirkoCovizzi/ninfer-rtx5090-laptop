#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutputTokens = 128;
constexpr std::array<std::uint32_t, 15> kMtpDraftCounts{1, 2,  3,  4,  5,  6,  7, 8,
                                                        9, 10, 11, 12, 13, 14, 15};
constexpr std::uint32_t kMaximumConcurrency = 8;

struct KvProfile {
    std::string_view name;
    ninfer::KvCacheStorage storage;
};

constexpr std::array kKvProfiles{
    KvProfile{"bf16", ninfer::KvCacheStorage::BFloat16},
    KvProfile{"int8", ninfer::KvCacheStorage::Int8Group64},
};

ninfer::EngineOptions
engine_options(const char* artifact, ninfer::KvCacheStorage kv_storage,
               std::uint32_t mtp_draft_tokens, bool use_cuda_graph,
               std::uint32_t max_concurrency     = 1,
               ninfer::MtpDraftPolicy mtp_policy = ninfer::MtpDraftPolicy::Fixed) {
    const bool mtp = mtp_draft_tokens != 0;
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 512;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(512 * max_concurrency);
    options.max_concurrency = max_concurrency;
    options.prefill_chunk   = 128;
    options.kv_cache        = kv_storage;
    options.use_cuda_graph  = use_cuda_graph;
    options.speculative.backend =
        mtp ? ninfer::SpeculativeBackend::Mtp : ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp_draft_tokens;
    options.speculative.proposal_head =
        mtp ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
    options.speculative.mtp_policy = mtp_policy;
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
                                      std::uint32_t mtp_draft_tokens, bool use_cuda_graph) {
    const std::string route =
        mtp_draft_tokens == 0 ? "ordinary" : "MTP k=" + std::to_string(mtp_draft_tokens);
    try {
        ninfer::Engine engine(
            engine_options(artifact, kv_storage, mtp_draft_tokens, use_cuda_graph));
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
                   const std::vector<ninfer::TokenId>& expected, std::uint32_t draft_tokens,
                   ninfer::MtpDraftPolicy mtp_policy) {
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
    const ninfer::SpeculativeStats& stats = result.speculative;
    if (draft_tokens == 0) {
        if (stats.enabled) {
            throw std::runtime_error(std::string(label) + " unexpectedly reported MTP statistics");
        }
        return;
    }
    const bool adaptive = mtp_policy == ninfer::MtpDraftPolicy::Adaptive;
    const std::uint64_t physical_rounds =
        std::accumulate(stats.rounds_per_window.begin(), stats.rounds_per_window.end(), 0ULL);
    const std::uint64_t transitions = std::accumulate(stats.window_transition_counts.begin(),
                                                      stats.window_transition_counts.end(), 0ULL);
    const std::uint64_t window_fallbacks =
        std::accumulate(stats.fallbacks_per_window.begin(), stats.fallbacks_per_window.end(), 0ULL);
    const std::uint64_t window_drafted = std::accumulate(
        stats.drafted_tokens_per_window.begin(), stats.drafted_tokens_per_window.end(), 0ULL);
    const std::uint64_t window_accepted = std::accumulate(
        stats.accepted_tokens_per_window.begin(), stats.accepted_tokens_per_window.end(), 0ULL);
    const std::uint64_t window_committed = std::accumulate(
        stats.committed_tokens_per_window.begin(), stats.committed_tokens_per_window.end(), 0ULL);
    if (!stats.enabled || stats.draft_window != draft_tokens || stats.adaptive != adaptive ||
        stats.rounds_per_window.size() != draft_tokens ||
        stats.fallbacks_per_window.size() != draft_tokens ||
        stats.drafted_tokens_per_window.size() != draft_tokens ||
        stats.accepted_tokens_per_window.size() != draft_tokens ||
        stats.committed_tokens_per_window.size() != draft_tokens ||
        stats.decode_seconds_per_window.size() != draft_tokens ||
        stats.window_transition_counts.size() != draft_tokens * draft_tokens ||
        physical_rounds != stats.rounds + stats.fallback_steps ||
        window_fallbacks != stats.fallback_steps || window_drafted != stats.drafted_tokens ||
        window_accepted != stats.accepted_tokens ||
        window_committed != stats.rounds + stats.fallback_steps + stats.accepted_tokens ||
        transitions != stats.window_transitions || (!adaptive && stats.window_transitions != 0)) {
        throw std::runtime_error(std::string(label) + " reported inconsistent MTP statistics");
    }
}

void verify_batch(ninfer::Engine& engine, KvProfile profile, std::uint32_t draft_tokens,
                  std::uint32_t concurrency, std::uint32_t iteration,
                  const std::vector<ninfer::TokenId>& expected, bool use_cuda_graph,
                  ninfer::MtpDraftPolicy mtp_policy) {
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
            std::string(profile.name) + " " + (use_cuda_graph ? "graph" : "eager") + " " +
            (mtp_policy == ninfer::MtpDraftPolicy::Adaptive ? "adaptive" : "fixed") +
            " MTP k=" + std::to_string(draft_tokens) + " C=" + std::to_string(concurrency) +
            " iteration=" + std::to_string(iteration) + " row=" + std::to_string(row);
        verify_result(label, handles[row].wait(), expected, draft_tokens, mtp_policy);
    }
}

void verify_route(const char* artifact, KvProfile profile, std::uint32_t draft_tokens,
                  const std::vector<ninfer::TokenId>& expected, bool use_cuda_graph,
                  std::uint32_t maximum_concurrency,
                  ninfer::MtpDraftPolicy mtp_policy = ninfer::MtpDraftPolicy::Fixed) {
    ninfer::Engine engine(engine_options(artifact, profile.storage, draft_tokens, use_cuda_graph,
                                         maximum_concurrency, mtp_policy));
    for (std::uint32_t concurrency = 1; concurrency <= maximum_concurrency; ++concurrency) {
        verify_batch(engine, profile, draft_tokens, concurrency, 0, expected, use_cuda_graph,
                     mtp_policy);
        if (concurrency == maximum_concurrency) {
            verify_batch(engine, profile, draft_tokens, concurrency, 1, expected, use_cuda_graph,
                         mtp_policy);
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
    const char* eager_value = std::getenv("NINFER_MTP_GREEDY_PARITY_EAGER");
    const bool use_cuda_graph =
        eager_value == nullptr || *eager_value == '\0' || std::string_view(eager_value) == "0";
    const char* concurrency_value = std::getenv("NINFER_MTP_GREEDY_PARITY_MAX_CONCURRENCY");
    const std::uint32_t maximum_concurrency =
        concurrency_value == nullptr ? kMaximumConcurrency
                                     : static_cast<std::uint32_t>(std::stoul(concurrency_value));
    if (maximum_concurrency == 0 || maximum_concurrency > kMaximumConcurrency) {
        std::cerr << "NINFER_MTP_GREEDY_PARITY_MAX_CONCURRENCY must be in [1,8]\n";
        return 2;
    }
    const char* draft_value = std::getenv("NINFER_MTP_GREEDY_PARITY_K");
    const std::uint32_t selected_drafts =
        draft_value == nullptr ? 0U : static_cast<std::uint32_t>(std::stoul(draft_value));
    if (selected_drafts > kMtpDraftCounts.back()) {
        std::cerr << "NINFER_MTP_GREEDY_PARITY_K must be in [1,15]\n";
        return 2;
    }
    const char* kv_value = std::getenv("NINFER_MTP_GREEDY_PARITY_KV");
    if (kv_value != nullptr && std::string_view(kv_value) != "bf16" &&
        std::string_view(kv_value) != "int8") {
        std::cerr << "NINFER_MTP_GREEDY_PARITY_KV must be bf16 or int8\n";
        return 2;
    }
    const char* policy_value = std::getenv("NINFER_MTP_GREEDY_PARITY_POLICY");
    if (policy_value != nullptr && std::string_view(policy_value) != "fixed" &&
        std::string_view(policy_value) != "adaptive") {
        std::cerr << "NINFER_MTP_GREEDY_PARITY_POLICY must be fixed or adaptive\n";
        return 2;
    }

    try {
        for (const KvProfile profile : kKvProfiles) {
            if (kv_value != nullptr && profile.name != kv_value) { continue; }
            const std::vector<ninfer::TokenId> ordinary =
                generate(artifact, profile.storage, 0, use_cuda_graph);
            verify_route(artifact, profile, 0, ordinary, use_cuda_graph, maximum_concurrency);
            for (const std::uint32_t draft_tokens : kMtpDraftCounts) {
                if (selected_drafts != 0 && draft_tokens != selected_drafts) { continue; }
                if (policy_value == nullptr || std::string_view(policy_value) == "fixed") {
                    verify_route(artifact, profile, draft_tokens, ordinary, use_cuda_graph,
                                 maximum_concurrency);
                }
                if (policy_value == nullptr || std::string_view(policy_value) == "adaptive") {
                    verify_route(artifact, profile, draft_tokens, ordinary, use_cuda_graph,
                                 maximum_concurrency, ninfer::MtpDraftPolicy::Adaptive);
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
