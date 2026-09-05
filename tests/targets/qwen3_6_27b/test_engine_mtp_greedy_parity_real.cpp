#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

ninfer::RequestOptions greedy_request(std::uint32_t output_tokens = kOutputTokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = output_tokens;
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

struct KvarnCases {
    std::uint32_t output_tokens = 8192;
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t concurrency   = 1;
    int sample                  = -1;
    int depth                   = -1;
    bool graphs                 = true;
    bool prefix_reuse           = false;
    bool full_proposal_head     = false;
    std::vector<ninfer::TokenId> corpus;
};

void verify_kvarn(const char* artifact, const KvarnCases& cases) {
    constexpr std::array<std::uint32_t, 5> long_contexts{8190, 32799, 122879, 196607, 245743};
    const int samples = cases.corpus.empty() ? 3 : 8;
    if (cases.sample >= samples) { throw std::invalid_argument("long samples require --corpus"); }
    std::array<std::array<std::vector<ninfer::TokenId>, kMaximumConcurrency>, 8> expected;
    for (std::uint32_t depth : {0, 3, 1, 2, 4, 5}) {
        if (cases.depth >= 0 && depth != 0 && depth != cases.depth) { continue; }
        auto options = engine_options(artifact, ninfer::KvCacheStorage::KvarnK4V2Group64, depth,
                                      cases.concurrency);
        const auto prompt_capacity =
            cases.sample < 0 ? (samples == 8 ? long_contexts.back() : 4096U)
                             : (cases.sample >= 3 ? long_contexts[cases.sample - 3] : 4096U);
        options.max_context = cases.output_tokens + prompt_capacity + 16;
        options.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(cases.concurrency * options.max_context);
        options.prefill_chunk  = cases.prefill_chunk;
        options.use_cuda_graph = cases.graphs;
        if (cases.full_proposal_head) {
            options.speculative.proposal_head = ninfer::ProposalHead::Full;
        }
        if (cases.prefix_reuse) {
            options.context_cache.device_state_slots        = cases.concurrency;
            options.context_cache.max_private_continuations = cases.concurrency;
        }
        ninfer::Engine engine(options);
        auto request                         = greedy_request(cases.output_tokens);
        request.execution.allow_prefix_reuse = cases.prefix_reuse;
        for (int sample = 0; sample < samples; ++sample) {
            if (cases.sample >= 0 && sample != cases.sample) { continue; }
            const auto prepare = [&](std::uint32_t row) {
                if (sample >= 3) {
                    const auto length = long_contexts[sample - 3] - 7 * row;
                    if (cases.corpus.size() < length) {
                        throw std::invalid_argument("corpus is shorter than the selected prompt");
                    }
                    return engine.prepare_tokens(std::vector<ninfer::TokenId>(
                        cases.corpus.begin(), cases.corpus.begin() + length));
                }
                if (sample == 2) {
                    return engine.prepare_tokens(
                        std::vector<ninfer::TokenId>(2110 + 31 * row, 198));
                }
                auto input = prompt();
                if (sample == 1) {
                    std::string text;
                    for (std::uint32_t index = 0; index < 190 + 11 * row; ++index) { text += "x "; }
                    text += "\nWrite a numbered list of 200 distinct fictional identifiers. "
                            "Do not explain the task and do not stop before the list is complete.";
                    input.messages[0].parts[0].text = std::move(text);
                    input.options.enable_thinking   = false;
                } else if (row > 0) {
                    input.messages[0].parts[0].text +=
                        "\nNumber the comments starting from " + std::to_string(row * 10) + ".";
                }
                return engine.prepare(std::move(input));
            };
            // Establish real reusable state, not a second measured generation of the same case.
            std::vector<ninfer::TokenId> warm_tokens;
            if (cases.prefix_reuse) {
                auto warm_request                              = request;
                warm_request.execution.requested_output_tokens = 1;
                for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                    const auto warm = engine.generate(prepare(row), warm_request);
                    if (warm.generated_token_ids.size() != 1) {
                        throw std::runtime_error("KVarN prefix prewarm failed");
                    }
                    warm_tokens.push_back(warm.generated_token_ids.front());
                }
            }
            const auto before = engine.runtime_stats();
            std::vector<std::uint32_t> prompt_tokens;
            std::vector<ninfer::GenerationHandle> handles;
            std::cout << "starting kvarn k=" << depth << " sample=" << sample
                      << " C=" << cases.concurrency << " output=" << cases.output_tokens
                      << " prefill=" << cases.prefill_chunk << " graphs=" << cases.graphs
                      << " prefix=" << cases.prefix_reuse << std::endl;
            for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                auto prepared = prepare(row);
                prompt_tokens.push_back(prepared.summary().prompt_tokens);
                auto row_request = request;
                row_request.execution.requested_output_tokens -= row;
                handles.push_back(engine.submit(std::move(prepared), row_request));
            }
            for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                const auto result = handles[row].wait();
                const std::string label =
                    "kvarn k=" + std::to_string(depth) + " sample=" + std::to_string(sample) +
                    " row=" + std::to_string(row) + " prompt=" + std::to_string(prompt_tokens[row]);
                if (result.generated_token_ids.size() != cases.output_tokens - row) {
                    throw std::runtime_error(label + " did not reach the requested decode length");
                }
                if (cases.prefix_reuse &&
                    (result.reused_prompt_tokens != prompt_tokens[row] ||
                     result.generated_token_ids.front() != warm_tokens[row])) {
                    throw std::runtime_error(label +
                                             " did not restore the prewarmed frontier: reused=" +
                                             std::to_string(result.reused_prompt_tokens));
                }
                if (depth == 0) { expected[sample][row] = result.generated_token_ids; }
                verify_result(label, result, expected[sample][row]);
                std::cout << label << " matched " << expected[sample][row].size()
                          << " tokens reused=" << result.reused_prompt_tokens << std::endl;
            }
            const auto after = engine.runtime_stats();
            if (cases.concurrency > 1 && after.decode_row_rounds - before.decode_row_rounds <=
                                             after.decode_rounds - before.decode_rounds) {
                throw std::runtime_error("KVarN concurrent case did not execute a compact batch");
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_MTP_GREEDY_PARITY_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_MTP_GREEDY_PARITY_WEIGHTS is not set\n";
        return 77;
    }

    try {
        bool kvarn_only = false;
        KvarnCases cases;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--kvarn-only") {
                kvarn_only = true;
            } else if ((argument == "--output-tokens" || argument == "--sample" ||
                        argument == "--draft-tokens" || argument == "--prefill-chunk" ||
                        argument == "--concurrency") &&
                       index + 1 < argc) {
                const std::string_view value(argv[++index]);
                std::uint32_t number = 0;
                const auto parsed =
                    std::from_chars(value.data(), value.data() + value.size(), number);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
                    throw std::invalid_argument("invalid integer for " + std::string(argument));
                }
                if (argument == "--output-tokens" && number >= 128 && number <= 16384) {
                    cases.output_tokens = number;
                } else if (argument == "--sample" && number < 8) {
                    cases.sample = static_cast<int>(number);
                } else if (argument == "--draft-tokens" && number >= 1 && number <= 5) {
                    cases.depth = static_cast<int>(number);
                } else if (argument == "--prefill-chunk" && number >= 1 && number <= 4096) {
                    cases.prefill_chunk = number;
                } else if (argument == "--concurrency" && number >= 1 &&
                           number <= kMaximumConcurrency) {
                    cases.concurrency = number;
                } else {
                    throw std::invalid_argument("out of range: " + std::string(argument));
                }
            } else if (argument == "--no-cuda-graph") {
                cases.graphs = false;
            } else if (argument == "--prefix-reuse") {
                cases.prefix_reuse = true;
            } else if (argument == "--full-proposal-head") {
                cases.full_proposal_head = true;
            } else if (argument == "--corpus" && index + 1 < argc) {
                std::ifstream input(argv[++index]);
                ninfer::TokenId token;
                while (input >> token) { cases.corpus.push_back(token); }
                if (!input.eof() || cases.corpus.empty()) {
                    throw std::invalid_argument("cannot read token corpus");
                }
            } else {
                throw std::invalid_argument("usage: mtp_greedy_parity_real_test [--kvarn-only] "
                                            "[--output-tokens 128..16384] [--sample 0..7] "
                                            "[--draft-tokens 1..5] [--prefill-chunk 1..4096] "
                                            "[--concurrency 1..8] [--full-proposal-head] "
                                            "[--no-cuda-graph] [--prefix-reuse] [--corpus PATH]");
            }
        }
        if (!kvarn_only)
            for (const KvProfile profile : kKvProfiles) {
                const std::vector<ninfer::TokenId> ordinary =
                    generate(artifact, profile.storage, 0);
                verify_route(artifact, profile, 0, ordinary);
                for (const std::uint32_t draft_tokens : kMtpDraftCounts) {
                    verify_route(artifact, profile, draft_tokens, ordinary);
                }
            }
        verify_kvarn(artifact, cases);
    } catch (const std::exception& error) {
        std::cerr << "greedy MTP parity test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}
