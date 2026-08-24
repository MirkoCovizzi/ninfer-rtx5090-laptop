from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bench.run_kv_cache_quality import (
    MAX_NEEDLE_CONTEXT_MARGIN,
    MAX_NEEDLE_OUTPUT_TOKENS,
    build_fixtures,
    build_fixture,
    build_max_needle_fixture,
    calibrate_max_needle_fixture,
    common_prefix_length,
    fixture_identity,
    needle_recall_stats,
    profile_identity,
    repetition_stats,
    unwrapped_payload,
)


def document_lines(fixture: object) -> list[str]:
    content = fixture.messages[1]["content"]
    document = content.split("<document>\n", 1)[1].split("\n</document>", 1)[0]
    return document.splitlines()


def test_ledger_needle_matches_distractor_distribution_and_depth() -> None:
    fixture = build_fixture(32768, 50, "single")
    lines = document_lines(fixture)
    target = (
        "date 2041-07-19 | clerk Mara Ilyan | depot North Quay | shipment 483 crates | "
        "material cedar | destination Red Harbor | revision 2."
    )
    matches = [index for index, line in enumerate(lines) if target in line]
    assert len(matches) == 1
    assert abs(matches[0] / (len(lines) - 1) - 0.5) < 0.01

    near_matches = [
        line
        for line in lines
        if sum(
            field in line
            for field in ("date 2041-07-19", "clerk Mara Ilyan", "depot North Quay")
        )
        >= 2
    ]
    assert len(near_matches) >= 10
    assert all(line.startswith("Entry ") and "shipment " in line for line in near_matches)
    assert "ORCHID" not in fixture.messages[1]["content"]


def test_negative_composite_key_is_absent_but_in_distribution() -> None:
    fixture = build_fixture(8192, 90, "negative")
    lines = document_lines(fixture)
    assert not any(
        "date 2046-02-28" in line
        and "clerk Mara Ilyan" in line
        and "depot North Quay" in line
        for line in lines
    )
    assert fixture.expected == "NOT FOUND"


def test_fixture_identity_covers_content_and_output_budget() -> None:
    first = build_fixture(8192, 50, "single")
    second = build_fixture(8192, 50, "single")
    assert fixture_identity(first) == fixture_identity(second)
    second = dataclasses.replace(second, max_tokens=second.max_tokens + 1)
    assert fixture_identity(first) != fixture_identity(second)


def test_profile_identity_covers_capacity_and_concurrency() -> None:
    args = SimpleNamespace(
        kv_capacity=98304,
        max_context=98304,
        device=0,
        max_concurrency=1,
        prefill_chunk=2048,
    )
    first = profile_identity(args, Path("serve"), Path("model.ninfer"), "kvarn", "mtp3")
    args.max_concurrency = 2
    assert first != profile_identity(
        args, Path("serve"), Path("model.ninfer"), "kvarn", "mtp3"
    )


def test_repetition_and_pairwise_helpers() -> None:
    text = "prefix:" + "I just cannot continue. " * 40
    stats = repetition_stats(text)
    assert stats["period_chars"] == len("I just cannot continue. ")
    assert stats["repeated_fraction"] > 0.9
    assert repetition_stats("ordinary nonperiodic response")["period_chars"] == 0
    assert common_prefix_length("abcdef", "abcxyz") == 3
    assert unwrapped_payload("```svg\n<svg/>\n```") == "<svg/>"
    assert unwrapped_payload("```svg\n<svg") == "<svg"


def test_maximum_context_needles_are_unique_stratified_and_deterministic() -> None:
    fixture = build_max_needle_fixture(64, 1, 259_968, record_count=4096)
    repeated = build_max_needle_fixture(64, 1, 259_968, record_count=4096)
    assert fixture_identity(fixture) == fixture_identity(repeated)
    assert len(fixture.needles) == 64
    assert len({needle.label for needle in fixture.needles}) == 64
    assert min(needle.depth_percent for needle in fixture.needles) < 2.0
    assert max(needle.depth_percent for needle in fixture.needles) > 98.0
    assert len(fixture.expected.splitlines()) == 64


def test_maximum_context_needle_recall_scores_content_and_depth() -> None:
    fixture = build_max_needle_fixture(32, 0, 259_968, record_count=4096)
    perfect = needle_recall_stats(fixture.expected, fixture)
    assert perfect is not None
    assert perfect["correct"] == 32
    assert perfect["all_recalled"]
    assert sum(bucket["total"] for bucket in perfect["depth_buckets"].values()) == 32

    lines = fixture.expected.splitlines()
    label = lines[7].split("=", 1)[0]
    lines[7] = f"{label}=999|glass|North Mill|1"
    degraded = needle_recall_stats("\n".join(lines), fixture)
    assert degraded is not None
    assert degraded["correct"] == 31
    assert degraded["missed_labels"] == [label]
    assert degraded["incorrect_labels"] == [label]


def test_maximum_context_preset_reserves_output_and_safety_margin() -> None:
    max_context = 262_144
    fixtures = build_fixtures("max-needles", False, 65_536, max_context)
    assert len(fixtures) == 6
    assert {len(fixture.needles) for fixture in fixtures} == {32, 64}
    assert {
        fixture.nominal_tokens + fixture.max_tokens for fixture in fixtures
    } == {max_context - MAX_NEEDLE_CONTEXT_MARGIN}
    assert {fixture.max_tokens for fixture in fixtures} == {MAX_NEEDLE_OUTPUT_TOKENS}

    selected = build_fixtures(
        "max-needles", False, 65_536, max_context, needle_counts=(64,), needle_layouts=(2,)
    )
    assert [fixture.name for fixture in selected] == ["niah-max-n64-s2"]


def test_maximum_context_calibration_selects_largest_fitting_document() -> None:
    fixture = build_max_needle_fixture(32, 2, 259_968, record_count=4096)
    with patch(
        "tools.bench.run_kv_cache_quality.count_prompt_tokens",
        side_effect=lambda _connection, candidate: candidate.record_count * 50 + 17,
    ):
        calibrated, tokens = calibrate_max_needle_fixture(None, fixture)
    assert tokens <= fixture.target_prompt_tokens
    assert tokens + 50 > fixture.target_prompt_tokens
    assert calibrated.record_count * 50 + 17 == tokens


def test_reported_creative_reasoning_loop_is_detected() -> None:
    text = "The composition is ready. " + 'GoT "sage". ' * 80
    stats = repetition_stats(text)
    assert stats["period_chars"] == len('GoT "sage". ')
    assert stats["repeated_fraction"] > 0.9
