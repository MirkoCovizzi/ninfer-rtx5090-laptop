#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_27b::detail

namespace ninfer::targets::qwen3_6_27b {
namespace {

// General-task presets published with each exact model. Keep the registrations separate even
// while their values agree so an upstream model-specific change has one obvious owner.
constexpr ModelSamplingDefaults kQwen3_6Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6Defaults; }
    if (model == qwen3_8_model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

bool endpoint_matches(const artifact::Reader& reader, std::string_view name,
                      artifact::NumericFormat format, artifact::StorageLayout layout) {
    const auto* object = reader.find(name);
    if (object == nullptr) { return false; }
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
    return tensor != nullptr && tensor->format == format && tensor->layout == layout &&
           tensor->shape == std::vector<std::uint64_t>{248320, 5120};
}

Package::WeightsProfile Package::resolve_weights(const artifact::Reader& reader) {
    const auto& identity = reader.identity();
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen36GroupwiseInt;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen38GroupwiseInt;
    }
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen36Nvfp4;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "nvfp4") {
        const bool legacy_w8 =
            endpoint_matches(reader, "text/token_embedding", artifact::NumericFormat::W8G32_F16S,
                             artifact::StorageLayout::RowSplitK128V1) &&
            endpoint_matches(reader, "text/output_head", artifact::NumericFormat::W8G32_F16S,
                             artifact::StorageLayout::RowSplitK128V1);
        const bool current_fp8 = endpoint_matches(reader, "text/token_embedding",
                                                  artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
                                                  artifact::StorageLayout::RowScaleV1) &&
                                 endpoint_matches(reader, "text/output_head",
                                                  artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
                                                  artifact::StorageLayout::RowScaleV1);
        if (legacy_w8) { return WeightsProfile::Qwen38Nvfp4LegacyW8; }
        if (current_fp8) { return WeightsProfile::Qwen38Nvfp4; }
        throw std::runtime_error("unsupported qwen3.8-27b/nvfp4 endpoint storage profile");
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(
        model.impl_->data.frontend,
        qwen3_6::FrontendOptions{
            .vision_enabled                = model.impl_->data.runtime.features.vision,
            .max_context                   = options.max_context,
            .media_cache_bytes             = options.media_cache_bytes,
            .media_live_bytes              = options.media_live_bytes,
            .media_preprocess_threads      = options.media_preprocess_threads,
            .max_cache_markers_per_request = *options.context_cache.max_cache_markers_per_request,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace ninfer::targets::qwen3_6_27b
