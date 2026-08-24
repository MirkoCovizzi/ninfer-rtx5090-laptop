#!/usr/bin/env python3
"""Compare BF16, INT8, and KVarN KV quality on deterministic long-context tasks."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_corpus as corpus


MODEL_ID = "qwen3.8-27b"
KV_DTYPES = ("bf16", "int8", "kvarn")
MODES = {"mtp0": ("none", 0), "mtp3": ("mtp", 3)}
KV_LOG_NAMES = {
    "bf16": "bf16",
    "int8": "int8-group64",
    "kvarn": "kvarn-nvfp4-v2-group64",
}
PRESETS = {
    "smoke": ((8192, 65536), (50,), ("single", "multi", "conflict", "negative")),
    "qualification": (
        (32768, 81920),
        (10, 90),
        ("single", "multi", "conflict", "negative"),
    ),
    "standard": (
        (8192, 32768, 65536, 80000),
        (10, 50, 90),
        ("single", "multi", "conflict", "negative"),
    ),
    "full": (
        (8192, 16384, 32768, 65536, 80000),
        (0, 10, 25, 50, 75, 90, 100),
        ("single", "multi", "conflict", "negative"),
    ),
}
PRESET_NAMES = (*PRESETS, "max-needles")
MAX_NEEDLE_COUNTS = (32, 64)
MAX_NEEDLE_LAYOUTS = (0, 1, 2)
MAX_NEEDLE_OUTPUT_TOKENS = 2048
MAX_NEEDLE_CONTEXT_MARGIN = 128

FIRST_NAMES = ("Mara", "Ilan", "Soren", "Talia", "Neris", "Oren", "Lina", "Caro")
LAST_NAMES = ("Venn", "Ilyan", "Sorel", "Tarin", "Kest", "Orin", "Vale", "Daro")
DEPOTS = (
    "North Quay",
    "South Quay",
    "East Annex",
    "West Annex",
    "Upper Yard",
    "Lower Yard",
    "Old Harbor",
    "New Harbor",
)
MATERIALS = ("cedar", "copper", "linen", "granite", "barley", "glass", "wool", "ash")
DESTINATIONS = (
    "Red Harbor",
    "Blue Harbor",
    "Green Market",
    "White Market",
    "Stone Gate",
    "River Gate",
    "North Mill",
    "South Mill",
)


class QualityError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Needle:
    label: str
    query: str
    signature: tuple[str, str, str]
    answer: str
    depth_percent: float


@dataclasses.dataclass(frozen=True)
class Fixture:
    name: str
    nominal_tokens: int
    depth_percent: int
    variant: str
    messages: list[dict[str, str]]
    expected: str
    max_tokens: int = 96
    needles: tuple[Needle, ...] = ()
    record_count: int = 0
    target_prompt_tokens: int = 0
    layout_seed: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--serve", type=Path, default=REPO_ROOT / "build/apps/ninfer-serve"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESET_NAMES, default="standard")
    parser.add_argument("--kv-dtype", action="append", choices=KV_DTYPES)
    parser.add_argument("--mode", action="append", choices=tuple(MODES))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=98304)
    parser.add_argument("--kv-capacity", type=int)
    parser.add_argument("--include-loop-probe", action="store_true")
    parser.add_argument("--loop-only", action="store_true")
    parser.add_argument("--loop-output-tokens", type=int, default=65536)
    parser.add_argument("--needle-count", action="append", type=int, choices=MAX_NEEDLE_COUNTS)
    parser.add_argument("--needle-layout", action="append", type=int, choices=MAX_NEEDLE_LAYOUTS)
    return parser.parse_args(argv)


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def ledger_values(index: int) -> dict[str, Any]:
    value = mix64(index + 0xC0FFEE)
    year = 2029 + value % 19
    month = 1 + (value >> 8) % 12
    day = 1 + (value >> 16) % 28
    return {
        "date": f"{year:04d}-{month:02d}-{day:02d}",
        "clerk": f"{FIRST_NAMES[(value >> 24) % len(FIRST_NAMES)]} "
        f"{LAST_NAMES[(value >> 29) % len(LAST_NAMES)]}",
        "depot": DEPOTS[(value >> 35) % len(DEPOTS)],
        "shipment": 100 + (value >> 40) % 900,
        "material": MATERIALS[(value >> 50) % len(MATERIALS)],
        "destination": DESTINATIONS[(value >> 54) % len(DESTINATIONS)],
        "revision": 1 + (value >> 61) % 3,
    }


def ledger_line(entry: int, values: dict[str, Any]) -> str:
    return (
        f"Entry {entry:06d} | date {values['date']} | clerk {values['clerk']} | "
        f"depot {values['depot']} | shipment {values['shipment']} crates | "
        f"material {values['material']} | destination {values['destination']} | "
        f"revision {values['revision']}."
    )


def answer_for(values: dict[str, Any]) -> str:
    return (
        f"SHIPMENT={values['shipment']}; MATERIAL={values['material']}; "
        f"DESTINATION={values['destination']}; REVISION={values['revision']}"
    )


def query_for(values: dict[str, Any]) -> str:
    return (
        f"date {values['date']}, clerk {values['clerk']}, depot {values['depot']}"
    )


def hard_distractors(target: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    distractors: list[dict[str, Any]] = []
    for index in range(12):
        candidate = dict(ledger_values(seed + index))
        selector = index % 3
        if selector != 0:
            candidate["date"] = target["date"]
        if selector != 1:
            candidate["clerk"] = target["clerk"]
        if selector != 2:
            candidate["depot"] = target["depot"]
        distractors.append(candidate)
    return distractors


def build_fixture(nominal_tokens: int, depth_percent: int, variant: str) -> Fixture:
    if nominal_tokens <= 0 or not 0 <= depth_percent <= 100:
        raise QualityError("invalid fixture length/depth")
    # The fixed-width records average roughly 52 Qwen tokens. Every candidate has the same schema,
    # numeric density, and vocabulary as the target; only the composite key distinguishes it.
    record_count = max(256, nominal_tokens // 52)
    lines = [ledger_line(index, ledger_values(index)) for index in range(record_count)]
    insertion = min(record_count - 1, int((record_count - 1) * depth_percent / 100))
    target = {
        "date": "2041-07-19",
        "clerk": "Mara Ilyan",
        "depot": "North Quay",
        "shipment": 483,
        "material": "cedar",
        "destination": "Red Harbor",
        "revision": 2,
    }
    for offset, values in enumerate(hard_distractors(target, nominal_tokens + depth_percent)):
        index = max(0, min(record_count - 1, insertion - 6 + offset))
        lines[index] = ledger_line(index, values)

    if variant == "single":
        lines[insertion] = ledger_line(insertion, target)
        expected = answer_for(target)
        question = f"Find the unique entry with {query_for(target)}. Return exactly: {expected}"
    elif variant == "multi":
        targets = [target, dict(ledger_values(900_001)), dict(ledger_values(900_002))]
        span = max(20, record_count // 20)
        positions: list[int] = []
        for candidate in (insertion - span, insertion, insertion + span):
            bounded = max(0, min(record_count - 1, candidate))
            if bounded not in positions:
                positions.append(bounded)
        for candidate in range(record_count):
            if len(positions) == 3:
                break
            if candidate not in positions:
                positions.append(candidate)
        positions.sort()
        for position, values in zip(positions, targets, strict=True):
            lines[position] = ledger_line(position, values)
        expected = " || ".join(answer_for(values) for values in targets)
        keys = " || ".join(query_for(values) for values in targets)
        question = f"Find these three entries in order: {keys}. Return exactly: {expected}"
    elif variant == "conflict":
        old = dict(target)
        old.update(shipment=481, material="copper", destination="Blue Harbor", revision=1)
        lines[max(0, insertion - max(32, record_count // 8))] = ledger_line(insertion, old)
        lines[insertion] = ledger_line(insertion, target)
        expected = answer_for(target)
        question = (
            f"For {query_for(target)}, select the highest revision. Return exactly: {expected}"
        )
    elif variant == "negative":
        # Each individual attribute is common, but this exact composite key is deliberately absent.
        absent = dict(target)
        absent["date"] = "2046-02-28"
        expected = "NOT FOUND"
        question = f"Find the unique entry with {query_for(absent)}. Return exactly: NOT FOUND"
    else:
        raise QualityError(f"unknown retrieval variant: {variant}")
    haystack = "\n".join(lines)
    messages = [
        {
            "role": "system",
            "content": (
                "You audit a shipment ledger. Match every field of the composite key exactly; "
                "nearby and near-matching entries are distractors. Return only the requested format."
            ),
        },
        {
            "role": "user",
            "content": f"<document>\n{haystack}\n</document>\n\n{question}",
        },
    ]
    return Fixture(
        name=f"niah-{nominal_tokens // 1024}k-d{depth_percent}-{variant}",
        nominal_tokens=nominal_tokens,
        depth_percent=depth_percent,
        variant=variant,
        messages=messages,
        expected=expected,
    )


def query_signature(values: dict[str, Any]) -> tuple[str, str, str]:
    return str(values["date"]), str(values["clerk"]), str(values["depot"])


def compact_answer(label: str, values: dict[str, Any]) -> str:
    return (
        f"{label}={values['shipment']}|{values['material']}|"
        f"{values['destination']}|{values['revision']}"
    )


def build_max_needle_fixture(
    needle_count: int,
    layout_seed: int,
    target_prompt_tokens: int,
    record_count: int | None = None,
) -> Fixture:
    if needle_count not in MAX_NEEDLE_COUNTS or layout_seed not in MAX_NEEDLE_LAYOUTS:
        raise QualityError("unsupported maximum-context needle fixture")
    if target_prompt_tokens <= MAX_NEEDLE_OUTPUT_TOKENS:
        raise QualityError("maximum-context needle prompt budget is too small")
    if record_count is None:
        record_count = max(4096, target_prompt_tokens // 52)
    if record_count < needle_count * 32:
        raise QualityError("maximum-context needle document is too short")

    records = [ledger_values(index) for index in range(record_count)]
    targets = [
        ledger_values(2_000_000 + layout_seed * 10_000 + index)
        for index in range(needle_count)
    ]
    signatures = [query_signature(values) for values in targets]
    if len(set(signatures)) != needle_count:
        raise QualityError("maximum-context needle keys are not unique")

    # Stratification covers the complete context while deterministic jitter prevents a regular
    # spacing pattern from becoming an accidental retrieval cue.
    rng = random.Random(0x4B564E00 + layout_seed * 257 + needle_count)
    positions: list[int] = []
    margin = 16
    usable = record_count - 2 * margin
    for index in range(needle_count):
        center = margin + int((index + 0.5) * usable / needle_count)
        jitter_limit = max(1, usable // (needle_count * 4))
        positions.append(
            max(
                margin,
                min(
                    record_count - margin - 1,
                    center + rng.randint(-jitter_limit, jitter_limit),
                ),
            )
        )
    if len(set(positions)) != needle_count:
        raise QualityError("maximum-context needle positions overlap")

    target_signatures = set(signatures)
    for index, values in enumerate(records):
        if query_signature(values) in target_signatures:
            replacement = ledger_values(4_000_000 + layout_seed * record_count + index)
            while query_signature(replacement) in target_signatures:
                replacement = ledger_values(4_000_000 + mix64(index + int(replacement["shipment"])))
            records[index] = replacement

    reserved = set(positions)
    for target_index, (position, target) in enumerate(zip(positions, targets, strict=True)):
        candidates = hard_distractors(target, 3_000_000 + layout_seed * 1000 + target_index)
        offsets = (-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12)
        for candidate_index, (offset, candidate) in enumerate(
            zip(offsets, candidates, strict=True)
        ):
            selector = candidate_index % 3
            if selector == 0:
                field = "date"
                alternatives = (
                    f"{year:04d}-{month:02d}-{day:02d}"
                    for year in range(2029, 2048)
                    for month in range(1, 13)
                    for day in range(1, 29)
                )
            elif selector == 1:
                field = "clerk"
                alternatives = (
                    f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES
                )
            else:
                field = "depot"
                alternatives = iter(DEPOTS)
            for alternative in alternatives:
                candidate[field] = alternative
                if query_signature(candidate) not in target_signatures:
                    break
            else:
                raise QualityError("failed to construct a unique near-match distractor")
            destination = position + offset
            if destination < 0 or destination >= record_count or destination in reserved:
                raise QualityError("maximum-context needle distractor placement overlaps")
            records[destination] = candidate
            reserved.add(destination)
        records[position] = target

    labels = [f"N{index:02d}" for index in range(needle_count)]
    needles = tuple(
        Needle(
            label=label,
            query=query_for(target),
            signature=query_signature(target),
            answer=compact_answer(label, target),
            depth_percent=100.0 * position / (record_count - 1),
        )
        for label, target, position in zip(labels, targets, positions, strict=True)
    )
    for needle in needles:
        matches = sum(query_signature(values) == needle.signature for values in records)
        if matches != 1:
            raise QualityError(
                f"maximum-context needle {needle.label} has {matches} exact document matches"
            )
    query_order = list(range(needle_count))
    rng.shuffle(query_order)
    queries = "\n".join(
        f"{labels[index]}: {query_for(targets[index])}" for index in query_order
    )
    expected = "\n".join(needles[index].answer for index in query_order)
    document = "\n".join(ledger_line(index, values) for index, values in enumerate(records))
    messages = [
        {
            "role": "system",
            "content": (
                "You audit a shipment ledger. Resolve every composite key exactly. Nearby records "
                "are adversarial near-matches. Return one requested label and value per line, in "
                "the requested order, with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"<document>\n{document}\n</document>\n\n"
                "Retrieve all requested records. The exact line format is "
                "LABEL=SHIPMENT|MATERIAL|DESTINATION|REVISION.\n"
                f"<queries>\n{queries}\n</queries>"
            ),
        },
    ]
    return Fixture(
        name=f"niah-max-n{needle_count}-s{layout_seed}",
        nominal_tokens=target_prompt_tokens,
        depth_percent=50,
        variant="max-needles",
        messages=messages,
        expected=expected,
        max_tokens=MAX_NEEDLE_OUTPUT_TOKENS,
        needles=needles,
        record_count=record_count,
        target_prompt_tokens=target_prompt_tokens,
        layout_seed=layout_seed,
    )


def build_loop_fixture(max_tokens: int) -> Fixture:
    path_count = max(1400, (max_tokens + 29) // 30)
    paths = "".join(
        f'<path id="spoke-{index:04d}" d="M 512 512 L {64 + index % 896} '
        f'{64 + (index * 37) % 896}"/>\n'
        for index in range(path_count)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">\n'
        '<title>Pelican riding a bicycle</title>\n'
        f"{paths}</svg>"
    )
    messages = [
        {"role": "user", "content": "Generate an SVG of a pelican riding a bicycle."},
        {"role": "assistant", "content": svg},
        {
            "role": "user",
            "content": (
                "Repeat the exact previous assistant response verbatim. Do not reason, summarize, "
                "correct, or add commentary."
            ),
        },
    ]
    return Fixture(
        name="pelican-svg-exact-recall",
        nominal_tokens=8192,
        depth_percent=0,
        variant="loop-probe",
        messages=messages,
        expected=svg,
        max_tokens=max_tokens,
    )


def build_fixtures(
    preset: str,
    include_loop: bool,
    loop_tokens: int,
    max_context: int = 98304,
    needle_counts: Sequence[int] = MAX_NEEDLE_COUNTS,
    needle_layouts: Sequence[int] = MAX_NEEDLE_LAYOUTS,
) -> list[Fixture]:
    if preset == "max-needles":
        target = max_context - MAX_NEEDLE_OUTPUT_TOKENS - MAX_NEEDLE_CONTEXT_MARGIN
        fixtures = [
            build_max_needle_fixture(count, layout, target)
            for count in needle_counts
            for layout in needle_layouts
        ]
    else:
        lengths, depths, variants = PRESETS[preset]
        fixtures = [
            build_fixture(length, depth, variant)
            for length in lengths
            for depth in depths
            for variant in variants
        ]
    if include_loop:
        fixtures.append(build_loop_fixture(loop_tokens))
    return fixtures


def fixture_identity(fixture: Fixture) -> str:
    represented = {
        "messages": fixture.messages,
        "expected": fixture.expected,
        "max_tokens": fixture.max_tokens,
    }
    encoded = json.dumps(represented, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def post_json_path(
    connection: corpus.http.client.HTTPConnection, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Connection": "keep-alive",
            },
        )
    except (OSError, corpus.http.client.HTTPException) as exc:
        raise QualityError(f"HTTP request failed: {exc}") from exc
    return corpus.receive_json(connection)


def count_prompt_tokens(
    connection: corpus.http.client.HTTPConnection, fixture: Fixture
) -> int:
    messages = fixture.messages
    system: str | None = None
    if messages and messages[0].get("role") == "system":
        system = messages[0]["content"]
        messages = messages[1:]
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "max_tokens": fixture.max_tokens,
        "messages": messages,
        "thinking": {"type": "disabled"},
    }
    if system is not None:
        payload["system"] = system
    response = post_json_path(
        connection,
        "/v1/messages/count_tokens",
        payload,
    )
    tokens = response.get("input_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        raise QualityError("token-count response lacks a positive input_tokens value")
    return tokens


def calibrate_max_needle_fixture(
    connection: corpus.http.client.HTTPConnection, fixture: Fixture
) -> tuple[Fixture, int]:
    if not fixture.needles:
        return fixture, count_prompt_tokens(connection, fixture)

    target = fixture.target_prompt_tokens
    current = fixture
    current_tokens = count_prompt_tokens(connection, current)
    low: tuple[int, Fixture, int] | None = None
    high: tuple[int, Fixture, int] | None = None
    for _ in range(16):
        represented = (current.record_count, current, current_tokens)
        if current_tokens <= target:
            low = represented
        else:
            high = represented
        if low is not None and high is not None and high[0] - low[0] <= 1:
            break
        if low is not None and high is not None:
            next_count = (low[0] + high[0]) // 2
        else:
            average = current_tokens / current.record_count
            estimate = current.record_count + int((target - current_tokens) / average)
            step = max(8, abs(estimate - current.record_count) // 4)
            next_count = estimate + step if low is not None else estimate - step
        next_count = max(len(fixture.needles) * 32, next_count)
        if next_count == current.record_count:
            next_count += 1 if current_tokens <= target else -1
        current = build_max_needle_fixture(
            len(fixture.needles), fixture.layout_seed, target, next_count
        )
        current_tokens = count_prompt_tokens(connection, current)
    if low is None:
        raise QualityError("failed to calibrate maximum-context needle fixture below token budget")
    return low[1], low[2]


def normalized_answer(text: str) -> str:
    return " ".join(text.strip().split())


def response_text(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise QualityError(f"malformed Chat Completions response: {exc}") from exc
    return str(content) if content else str(reasoning)


def common_prefix_length(first: str, second: str) -> int:
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[index] == second[index]:
        index += 1
    return index


def unwrapped_payload(text: str) -> str:
    for prefix in ("```svg\n", "```xml\n", "```\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.endswith("\n```"):
        text = text[:-4]
    return text


def repetition_stats(text: str, maximum_period: int = 512, minimum_repeats: int = 8) -> dict[str, Any]:
    if not text:
        return {"period_chars": 0, "repeated_chars": 0, "repeated_fraction": 0.0}
    best_period = 0
    best_repeated = 0
    for period in range(1, min(maximum_period, len(text) // minimum_repeats) + 1):
        repeated = 0
        index = len(text) - 1
        while index - period >= 0 and text[index] == text[index - period]:
            repeated += 1
            index -= 1
        repeated += period
        if repeated >= period * minimum_repeats and repeated > best_repeated:
            best_period = period
            best_repeated = repeated
    return {
        "period_chars": best_period,
        "repeated_chars": best_repeated,
        "repeated_fraction": best_repeated / len(text),
    }


NEEDLE_LINE = re.compile(r"(?m)^\s*(N\d{2})=([^\r\n]+?)\s*$")


def needle_recall_stats(text: str, fixture: Fixture) -> dict[str, Any] | None:
    if not fixture.needles:
        return None
    expected = {needle.label: needle.answer for needle in fixture.needles}
    observed: dict[str, list[str]] = {}
    for match in NEEDLE_LINE.finditer(text):
        label = match.group(1)
        observed.setdefault(label, []).append(f"{label}={match.group(2)}")

    recalled = [
        needle.label
        for needle in fixture.needles
        if needle.answer in observed.get(needle.label, [])
    ]
    missed = [needle.label for needle in fixture.needles if needle.label not in recalled]
    incorrect = sorted(
        label
        for label, answers in observed.items()
        if label not in expected or any(answer != expected.get(label) for answer in answers)
    )
    duplicates = sorted(label for label, answers in observed.items() if len(answers) > 1)
    depth_buckets: dict[str, dict[str, int | float]] = {}
    for name, lower, upper in (
        ("early", 0.0, 33.333),
        ("middle", 33.333, 66.667),
        ("late", 66.667, 100.001),
    ):
        labels = [
            needle.label
            for needle in fixture.needles
            if lower <= needle.depth_percent < upper
        ]
        correct = sum(label in recalled for label in labels)
        depth_buckets[name] = {
            "correct": correct,
            "total": len(labels),
            "recall": correct / len(labels) if labels else 0.0,
        }
    total = len(fixture.needles)
    return {
        "correct": len(recalled),
        "total": total,
        "recall": len(recalled) / total,
        "all_recalled": len(recalled) == total,
        "recalled_labels": recalled,
        "missed_labels": missed,
        "incorrect_labels": incorrect,
        "duplicate_labels": duplicates,
        "needles": [
            {
                "label": needle.label,
                "depth_percent": needle.depth_percent,
                "correct": needle.label in recalled,
                "observed": observed.get(needle.label, []),
            }
            for needle in fixture.needles
        ],
        "depth_buckets": depth_buckets,
    }


def resolved_context_capacity(args: argparse.Namespace, mode: str) -> tuple[int, int]:
    kv_capacity = args.kv_capacity
    if kv_capacity is None:
        kv_capacity = (
            args.max_context
            if mode == "mtp0" or getattr(args, "preset", None) == "max-needles"
            else min(args.max_context, 86016)
        )
    return min(args.max_context, kv_capacity), kv_capacity


def profile_identity(
    args: argparse.Namespace, serve: Path, artifact: Path, kv_dtype: str, mode: str
) -> str:
    max_context, kv_capacity = resolved_context_capacity(args, mode)
    represented = {
        "serve": str(serve),
        "artifact": str(artifact),
        "kv_dtype": kv_dtype,
        "mode": mode,
        "device": args.device,
        "max_context": max_context,
        "kv_capacity": kv_capacity,
        "max_concurrency": args.max_concurrency,
        "prefill_chunk": args.prefill_chunk,
    }
    encoded = json.dumps(represented, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def server_command(
    serve: Path,
    artifact: Path,
    kv_dtype: str,
    mode: str,
    request_log: Path,
    args: argparse.Namespace,
) -> list[str]:
    backend, draft_tokens = MODES[mode]
    # Normal matched campaigns keep one logical capacity across KV formats within each mode.
    # Maximum-context KVarN explicitly uses the requested model limit instead.
    max_context, kv_capacity = resolved_context_capacity(args, mode)
    command = [
        str(serve),
        str(artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model-id",
        MODEL_ID,
        "--device",
        str(args.device),
        "--max-context",
        str(max_context),
        "--kv-capacity",
        str(kv_capacity),
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-pending-requests",
        "1",
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--kv-dtype",
        kv_dtype,
        "--no-prefix-reuse",
        "--greedy",
        "--log-stats-interval-ms",
        "0",
        "--request-log-jsonl",
        str(request_log),
    ]
    if backend != "none":
        command.extend(
            ["--spec", backend, "--draft-tokens", str(draft_tokens), "--lm-head-draft"]
        )
    return command


def request_payload(fixture: Fixture) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": fixture.messages,
        "max_completion_tokens": fixture.max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "enable_thinking": False,
    }


def validate_start(event: dict[str, Any], kv_dtype: str, mode: str) -> tuple[str, str]:
    corpus.require_server_log_identity(event, "server_start")
    engine = event.get("engine", {})
    backend, draft_tokens = MODES[mode]
    expected = {
        "kv_cache": KV_LOG_NAMES[kv_dtype],
        "speculative_backend": backend,
        "speculative_draft_window": draft_tokens,
        "prefix_reuse": False,
    }
    actual = {name: engine.get(name) for name in expected}
    if actual != expected:
        raise QualityError(f"server configuration mismatch: {actual!r} != {expected!r}")
    instance = event.get("server_instance_id")
    weights_id = event.get("artifact", {}).get("weights_id")
    if not isinstance(instance, str) or not isinstance(weights_id, str):
        raise QualityError("server startup record lacks identity")
    return instance, weights_id


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["kv_dtype"]), str(record["mode"]), str(record["fixture"])


def load_records(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("artifact_type") != "ninfer_kv_cache_quality_result":
                raise QualityError("unexpected artifact type")
            key = record_key(record)
        except (json.JSONDecodeError, KeyError, QualityError) as exc:
            raise QualityError(f"invalid result at {path}:{line_number}: {exc}") from exc
        if key in records:
            raise QualityError(f"duplicate result at {path}:{line_number}: {key}")
        records[key] = record
    return records


def build_record(
    fixture: Fixture,
    kv_dtype: str,
    mode: str,
    weights_id: str,
    response: dict[str, Any],
    event: dict[str, Any],
    fixture_id: str,
    profile_id: str,
    counted_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    corpus.require_server_log_identity(event, "request_done")
    text = response_text(response)
    payload = unwrapped_payload(text)
    payload_prefix_chars = common_prefix_length(payload, fixture.expected)
    result = event.get("result", {})
    timings = event.get("timings_seconds", {})
    exact = normalized_answer(text) == normalized_answer(fixture.expected)
    needle_recall = needle_recall_stats(text, fixture)
    return {
        "artifact_type": "ninfer_kv_cache_quality_result",
        "schema_version": 1,
        "target": "qwen3_8_27b",
        "weights_id": weights_id,
        "kv_dtype": kv_dtype,
        "mode": mode,
        "fixture": fixture.name,
        "fixture_id": fixture_id,
        "profile_id": profile_id,
        "nominal_tokens": fixture.nominal_tokens,
        "depth_percent": fixture.depth_percent,
        "variant": fixture.variant,
        "expected": fixture.expected,
        "response_text": text,
        "exact": exact,
        "contains_expected": normalized_answer(fixture.expected) in normalized_answer(text),
        "payload_prefix_chars": payload_prefix_chars,
        "payload_prefix_exact": payload_prefix_chars == min(len(payload), len(fixture.expected)),
        "expected_coverage": payload_prefix_chars / len(fixture.expected),
        "counted_prompt_tokens": counted_prompt_tokens,
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
        "finish_reason": result.get("finish_reason"),
        "prefill_seconds": float(timings.get("prefill", 0.0)),
        "decode_seconds": float(timings.get("decode", 0.0)),
        "repetition": repetition_stats(text),
        "needle_recall": needle_recall,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_block(
    args: argparse.Namespace,
    serve: Path,
    artifact: Path,
    output: Path,
    kv_dtype: str,
    mode: str,
    fixtures: Sequence[Fixture],
    records: dict[tuple[str, str, str], dict[str, Any]],
    run_handle: Any,
) -> None:
    profile_id = profile_identity(args, serve, artifact, kv_dtype, mode)
    needs_calibration = any(fixture.needles for fixture in fixtures)
    if not needs_calibration:
        missing = []
        for fixture in fixtures:
            existing = records.get((kv_dtype, mode, fixture.name))
            if (
                existing is None
                or existing.get("fixture_id") != fixture_identity(fixture)
                or existing.get("profile_id") != profile_id
            ):
                missing.append((fixture, None))
        if not missing:
            return
    request_log = output / "server" / f"{kv_dtype}-{mode}.jsonl"
    command = server_command(serve, artifact, kv_dtype, mode, request_log, args)
    print(f"starting {kv_dtype}/{mode}", flush=True)
    with corpus.RunningServer(command, "127.0.0.1", args.port, request_log) as server:
        instance, weights_id = validate_start(server.wait_until_ready(), kv_dtype, mode)
        connection = corpus.http.client.HTTPConnection(
            "127.0.0.1", args.port, timeout=corpus.REQUEST_TIMEOUT_SECONDS
        )
        try:
            if needs_calibration:
                missing = []
                for fixture in fixtures:
                    calibrated, counted = calibrate_max_needle_fixture(connection, fixture)
                    existing = records.get((kv_dtype, mode, calibrated.name))
                    if (
                        existing is None
                        or existing.get("fixture_id") != fixture_identity(calibrated)
                        or existing.get("profile_id") != profile_id
                    ):
                        missing.append((calibrated, counted))
                    print(
                        f"  calibrated {calibrated.name}: prompt={counted} "
                        f"records={calibrated.record_count}",
                        flush=True,
                    )
            print(f"running {len(missing)} request(s)", flush=True)
            for fixture, counted_prompt_tokens in missing:
                response = corpus.post_json(connection, request_payload(fixture))
                event = server.wait_for_request_done(instance)
                record = build_record(
                    fixture,
                    kv_dtype,
                    mode,
                    weights_id,
                    response,
                    event,
                    fixture_identity(fixture),
                    profile_id,
                    counted_prompt_tokens,
                )
                if fixture.needles and (
                    record["prompt_tokens"] > fixture.target_prompt_tokens
                    or record["prompt_tokens"] + fixture.max_tokens > args.max_context
                ):
                    raise QualityError(
                        f"{fixture.name} exceeded its calibrated context envelope"
                    )
                append_record(run_handle, record)
                records[record_key(record)] = record
                recall = record["needle_recall"]
                recall_text = (
                    f" recall={recall['correct']}/{recall['total']} "
                    f"incorrect={recall['incorrect_labels']} "
                    f"duplicates={recall['duplicate_labels']} "
                    f"depth={recall['depth_buckets']}"
                    if recall is not None
                    else ""
                )
                print(
                    f"  {fixture.name}: exact={record['exact']} prompt={record['prompt_tokens']} "
                    f"completion={record['completion_tokens']}{recall_text}",
                    flush=True,
                )
        finally:
            connection.close()


def mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return statistics.fmean(collected) if collected else None


def build_summary(
    records: dict[tuple[str, str, str], dict[str, Any]],
    kv_dtypes: Sequence[str],
    modes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for kv_dtype in kv_dtypes:
            selected = [
                record
                for (record_dtype, record_mode, _), record in records.items()
                if record_dtype == kv_dtype
                and record_mode == mode
                and record["variant"] != "loop-probe"
            ]
            if not selected:
                continue
            exact_accuracy = mean(float(record["exact"]) for record in selected)
            contains_accuracy = mean(float(record["contains_expected"]) for record in selected)
            needle_records = [record for record in selected if record.get("needle_recall")]
            needle_correct = sum(record["needle_recall"]["correct"] for record in needle_records)
            needle_total = sum(record["needle_recall"]["total"] for record in needle_records)
            rows.append(
                {
                    "mode": mode,
                    "kv_dtype": kv_dtype,
                    "samples": len(selected),
                    "exact_accuracy": exact_accuracy,
                    "contains_accuracy": contains_accuracy,
                    "needle_correct": needle_correct,
                    "needle_total": needle_total,
                    "needle_recall": needle_correct / needle_total if needle_total else None,
                    "prompt_tokens_mean": mean(record["prompt_tokens"] for record in selected),
                    "prefill_tok_s_mean": mean(
                        record["prompt_tokens"] / record["prefill_seconds"]
                        for record in selected
                        if record["prefill_seconds"] > 0
                    ),
                    "decode_tok_s_mean": mean(
                        max(record["completion_tokens"] - 1, 0) / record["decode_seconds"]
                        for record in selected
                        if record["decode_seconds"] > 0
                    ),
                }
            )
        baseline = next(
            (row for row in rows if row["mode"] == mode and row["kv_dtype"] == "bf16"), None
        )
        if baseline is not None:
            for row in rows:
                if row["mode"] == mode:
                    row["exact_accuracy_delta_vs_bf16"] = (
                        row["exact_accuracy"] - baseline["exact_accuracy"]
                    )
    return rows


def pairwise_rows(
    records: dict[tuple[str, str, str], dict[str, Any]], modes: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        fixture_names = sorted(
            fixture
            for kv_dtype, record_mode, fixture in records
            if kv_dtype == "bf16" and record_mode == mode
        )
        for fixture in fixture_names:
            baseline = records[("bf16", mode, fixture)]
            for kv_dtype in ("int8", "kvarn"):
                compared = records.get((kv_dtype, mode, fixture))
                if compared is None:
                    continue
                first = baseline["response_text"]
                second = compared["response_text"]
                rows.append(
                    {
                        "mode": mode,
                        "fixture": fixture,
                        "kv_dtype": kv_dtype,
                        "response_equal_bf16": first == second,
                        "common_prefix_chars": common_prefix_length(first, second),
                        "bf16_chars": len(first),
                        "compared_chars": len(second),
                    }
                )
    return rows


def write_reports(
    output: Path,
    records: dict[tuple[str, str, str], dict[str, Any]],
    kv_dtypes: Sequence[str],
    modes: Sequence[str],
) -> None:
    summary = build_summary(records, kv_dtypes, modes)
    pairs = pairwise_rows(records, modes)
    needle_records = sorted(
        (record for record in records.values() if record.get("needle_recall")),
        key=lambda record: (record["mode"], record["kv_dtype"], record["fixture"]),
    )
    needle_results = [
        {
            "mode": record["mode"],
            "kv_dtype": record["kv_dtype"],
            "fixture": record["fixture"],
            "prompt_tokens": record["prompt_tokens"],
            "completion_tokens": record["completion_tokens"],
            "exact": record["exact"],
            "needle_recall": record["needle_recall"],
            "repetition": record["repetition"],
        }
        for record in needle_records
    ]
    report = {
        "artifact_type": "ninfer_kv_cache_quality_summary",
        "schema_version": 1,
        "summary": summary,
        "pairwise_vs_bf16": pairs,
        "maximum_context_needles": needle_results,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if summary:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(dict.fromkeys(name for row in summary for name in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
    lines = [
        "# KV-cache quality summary",
        "",
        "When present, BF16 is the uncompressed production KV baseline; model weights are unchanged.",
        "",
        "| Mode | KV | Samples | Exact | Contains | Needles | Delta vs BF16 | Prefill tok/s | Decode tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        delta = row.get("exact_accuracy_delta_vs_bf16")
        lines.append(
            f"| {row['mode']} | {row['kv_dtype']} | {row['samples']} | "
            f"{100 * row['exact_accuracy']:.1f}% | {100 * row['contains_accuracy']:.1f}% | "
            f"{row['needle_correct']}/{row['needle_total']} | {100 * delta:.1f} pp | "
            f"{row['prefill_tok_s_mean']:.1f} | "
            f"{row['decode_tok_s_mean']:.1f} |"
            if delta is not None
            else f"| {row['mode']} | {row['kv_dtype']} | {row['samples']} | "
            f"{100 * row['exact_accuracy']:.1f}% | {100 * row['contains_accuracy']:.1f}% | "
            f"{row['needle_correct']}/{row['needle_total']} | n/a | "
            f"{row['prefill_tok_s_mean']:.1f} | {row['decode_tok_s_mean']:.1f} |"
        )
    if needle_records:
        lines.extend(
            [
                "",
                "## Maximum-context multi-needle recall",
                "",
                "| Mode | KV | Fixture | Prompt | Recall | Incorrect | Duplicates | Early | Middle | Late | Repetition |",
                "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in needle_records:
            recall = record["needle_recall"]
            buckets = recall["depth_buckets"]
            repetition = record["repetition"]
            lines.append(
                f"| {record['mode']} | {record['kv_dtype']} | {record['fixture']} | "
                f"{record['prompt_tokens']} | {recall['correct']}/{recall['total']} | "
                f"{', '.join(recall['incorrect_labels']) or '-'} | "
                f"{', '.join(recall['duplicate_labels']) or '-'} | "
                f"{buckets['early']['correct']}/{buckets['early']['total']} | "
                f"{buckets['middle']['correct']}/{buckets['middle']['total']} | "
                f"{buckets['late']['correct']}/{buckets['late']['total']} | "
                f"{100 * repetition['repeated_fraction']:.1f}% |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.port < 1
        or args.port > 65535
        or args.device < 0
        or args.max_concurrency < 1
        or args.max_concurrency > 8
    ):
        raise QualityError("invalid port, device, or concurrency")
    if args.max_context <= 0 or args.prefill_chunk <= 0 or args.loop_output_tokens <= 0:
        raise QualityError("context, chunk, and output-token values must be positive")
    if args.kv_capacity is not None and args.kv_capacity <= 0:
        raise QualityError("KV capacity must be positive")
    artifact = args.artifact.expanduser().resolve()
    serve = args.serve.expanduser().resolve()
    if not artifact.is_file() or not serve.is_file() or not os.access(serve, os.X_OK):
        raise QualityError("artifact or executable is unavailable")
    kv_dtypes = args.kv_dtype or (["kvarn"] if args.preset == "max-needles" else list(KV_DTYPES))
    modes = args.mode or ["mtp0"]
    if len(kv_dtypes) != len(set(kv_dtypes)) or len(modes) != len(set(modes)):
        raise QualityError("duplicate KV dtype or mode")
    if args.preset == "max-needles" and args.max_concurrency != 1:
        raise QualityError("maximum-context multi-needle qualification requires concurrency 1")
    if args.preset != "max-needles" and (args.needle_count or args.needle_layout):
        raise QualityError("needle selection requires --preset max-needles")
    needle_counts = args.needle_count or list(MAX_NEEDLE_COUNTS)
    needle_layouts = args.needle_layout or list(MAX_NEEDLE_LAYOUTS)
    if len(needle_counts) != len(set(needle_counts)) or len(needle_layouts) != len(
        set(needle_layouts)
    ):
        raise QualityError("duplicate needle count or layout")
    fixtures = (
        [build_loop_fixture(args.loop_output_tokens)]
        if args.loop_only
        else build_fixtures(
            args.preset,
            args.include_loop_probe,
            args.loop_output_tokens,
            args.max_context,
            needle_counts,
            needle_layouts,
        )
    )
    if max(fixture.nominal_tokens + fixture.max_tokens for fixture in fixtures) > args.max_context:
        raise QualityError("configured max context is smaller than the nominal fixture envelope")

    output = args.output.expanduser().resolve()
    (output / "server").mkdir(parents=True, exist_ok=True)
    run_path = output / "run.jsonl"
    records = load_records(run_path)
    with run_path.open("a", encoding="utf-8") as run_handle:
        for mode in modes:
            for kv_dtype in kv_dtypes:
                run_block(
                    args, serve, artifact, output, kv_dtype, mode, fixtures, records, run_handle
                )
    write_reports(output, records, kv_dtypes, modes)
    print(f"summary: {output / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted; completed records remain resumable", file=sys.stderr)
        raise SystemExit(130) from None
