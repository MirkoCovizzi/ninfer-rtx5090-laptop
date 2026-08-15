#pragma once

#include <roctracer/roctx.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::nvtx {

enum class Category : std::uint32_t {
    Runtime = 1,
    Prefill,
    Decode,
    Mtp,
    DFlash,
    Attention,
    Gdn,
    PostMixer,
    Moe,
    Control,
};

enum class Name : std::size_t {
    Generate,
    Prefill,
    Decode,
    DecodeMtpRound,
    DecodeOrdinaryRound,
    DecodeMtpSubmit,
    DecodeMtpWait,
    DecodeDFlashRound,
    DecodeDFlashSubmit,
    DecodeDFlashWait,
    DecodeOrdinarySubmit,
    DecodeOrdinaryWait,
    PrefillMtpChunk,
    PrefillLayerFull,
    VerifyLayerFull,
    PrefillAttention,
    VerifyAttention,
    PrefillPostMixer,
    VerifyPostMixer,
    PrefillLayerGdn,
    VerifyLayerGdn,
    PrefillGdn,
    VerifyGdn,
    PrefillChunk,
    SparseMoePrefill,
    SparseMoeSmallT,
    SparseMoeDecode,
    Count,
};

[[nodiscard]] inline std::uint32_t color(Category category) noexcept {
    switch (category) {
    case Category::Runtime:
        return 0xff4c78a8u;
    case Category::Prefill:
        return 0xff59a14fu;
    case Category::Decode:
        return 0xfff28e2bu;
    case Category::Mtp:
        return 0xffb279a2u;
    case Category::DFlash:
        return 0xffaf7aa1u;
    case Category::Attention:
        return 0xff76b7b2u;
    case Category::Gdn:
        return 0xffe15759u;
    case Category::PostMixer:
        return 0xffedc948u;
    case Category::Moe:
        return 0xffb07aa1u;
    case Category::Control:
        return 0xff9c9c9cu;
    }
    return 0xff9c9c9cu;
}

// roctx has no domain or registered-string concepts; these handles exist only to
// keep the calling interface stable. Every range is pushed on the global trace.
using nvtxDomainHandle_t = int;
using nvtxStringHandle_t = const char*;

[[nodiscard]] inline nvtxDomainHandle_t domain() noexcept { return 0; }

[[nodiscard]] inline nvtxStringHandle_t registered_message(Name name) noexcept {
    static constexpr std::array<const char*, static_cast<std::size_t>(Name::Count)> names{
        "generate",
        "prefill",
        "decode",
        "decode.mtp_round",
        "decode.ordinary_round",
        "decode.mtp.submit",
        "decode.mtp.wait",
        "decode.dflash_round",
        "decode.dflash.submit",
        "decode.dflash.wait",
        "decode.ordinary.submit",
        "decode.ordinary.wait",
        "prefill.mtp_chunk",
        "prefill.layer.full",
        "verify.layer.full",
        "prefill.attention",
        "verify.attention",
        "prefill.post_mixer",
        "verify.post_mixer",
        "prefill.layer.gdn",
        "verify.layer.gdn",
        "prefill.gdn",
        "verify.gdn",
        "prefill.chunk",
        "sparse_moe.prefill",
        "sparse_moe.small_t",
        "sparse_moe.decode",
    };
    return names[static_cast<std::size_t>(name)];
}

class ScopedRange {
public:
    explicit ScopedRange(Name name, Category, std::uint64_t = 0) noexcept {
        roctxRangePushA(registered_message(name));
    }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&)                 = delete;
    ScopedRange& operator=(ScopedRange&&)      = delete;

    ~ScopedRange() noexcept { roctxRangePop(); }
};

} // namespace ninfer::nvtx
