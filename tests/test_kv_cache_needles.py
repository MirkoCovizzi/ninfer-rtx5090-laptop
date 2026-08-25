from __future__ import annotations

import dataclasses
from unittest.mock import patch

from tools.bench.run_kv_cache_needles import (
    NEEDLE_COUNT,
    build_fixture,
    calibrate_fixture,
    fixture_identity,
    recall_stats,
)


def document_lines(fixture: object) -> list[str]:
    content = fixture.messages[1]["content"]
    return content.split("<document>\n", 1)[1].split("\n</document>", 1)[0].splitlines()


def test_needles_are_unique_stratified_and_deterministic() -> None:
    fixture = build_fixture(3, 63_360, record_count=4096)
    repeated = build_fixture(3, 63_360, record_count=4096)
    assert fixture_identity(fixture) == fixture_identity(repeated)
    assert len(fixture.needles) == NEEDLE_COUNT
    assert len({needle.signature for needle in fixture.needles}) == NEEDLE_COUNT
    assert min(needle.depth_percent for needle in fixture.needles) < 2.0
    assert max(needle.depth_percent for needle in fixture.needles) > 98.0
    lines = document_lines(fixture)
    for needle in fixture.needles:
        assert sum(all(field in line for field in needle.signature) for line in lines) == 1


def test_recall_scores_wrong_and_duplicate_lines() -> None:
    fixture = build_fixture(0, 63_360, record_count=4096)
    perfect = recall_stats(fixture.expected, fixture)
    assert perfect["correct"] == NEEDLE_COUNT
    lines = fixture.expected.splitlines()
    label = lines[7].split("=", 1)[0]
    lines[7] = f"{label}=999|glass|North Mill|1"
    lines.append(lines[8])
    degraded = recall_stats("\n".join(lines), fixture)
    assert degraded["correct"] == NEEDLE_COUNT - 1
    assert degraded["missed_labels"] == [label]
    assert degraded["incorrect_labels"] == [label]
    assert degraded["duplicate_labels"] == [lines[8].split("=", 1)[0]]


def test_fixture_identity_includes_output_content() -> None:
    fixture = build_fixture(1, 63_360, record_count=4096)
    changed = dataclasses.replace(fixture, expected=fixture.expected + "\n")
    assert fixture_identity(fixture) != fixture_identity(changed)


def test_calibration_selects_largest_fitting_document() -> None:
    target = 63_360
    with patch(
        "tools.bench.run_kv_cache_needles.count_tokens",
        side_effect=lambda _connection, _model, fixture: fixture.record_count * 15 + 17,
    ):
        fixture, tokens = calibrate_fixture(None, "model", 2, target)
    assert tokens <= target
    assert tokens + 15 > target
    assert fixture.record_count * 15 + 17 == tokens
