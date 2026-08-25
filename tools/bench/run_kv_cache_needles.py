#!/usr/bin/env python3
"""Compare INT8 and KVarN on deterministic maximum-context needle retrieval."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_corpus as corpus


KV_LOG_NAMES = {"int8": "int8-group64", "kvarn": "kvarn-k4v2-group64"}
SEEDS = tuple(range(5))
NEEDLE_COUNT = 64
OUTPUT_TOKENS = 2048
CONTEXT_MARGIN = 128
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
NEEDLE_LINE = re.compile(r"(?m)^\s*(N\d+)=([^\r\n]+?)\s*$")


class QualificationError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Needle:
    label: str
    signature: tuple[str, str, str]
    answer: str
    depth_percent: float


@dataclasses.dataclass(frozen=True)
class Fixture:
    seed: int
    target_prompt_tokens: int
    record_count: int
    messages: list[dict[str, str]]
    expected: str
    needles: tuple[Needle, ...]

    @property
    def name(self) -> str:
        return f"niah-max-n{len(self.needles)}-s{self.seed}"


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def ledger_values(index: int) -> dict[str, Any]:
    value = mix64(index + 0xC0FFEE)
    return {
        "date": f"{2029 + value % 19:04d}-{1 + (value >> 8) % 12:02d}-{1 + (value >> 16) % 28:02d}",
        "clerk": f"{FIRST_NAMES[(value >> 24) % len(FIRST_NAMES)]} "
        f"{LAST_NAMES[(value >> 29) % len(LAST_NAMES)]}",
        "depot": DEPOTS[(value >> 35) % len(DEPOTS)],
        "shipment": 100 + (value >> 40) % 900,
        "material": MATERIALS[(value >> 50) % len(MATERIALS)],
        "destination": DESTINATIONS[(value >> 54) % len(DESTINATIONS)],
        "revision": 1 + (value >> 61) % 3,
    }


def signature(values: dict[str, Any]) -> tuple[str, str, str]:
    return str(values["date"]), str(values["clerk"]), str(values["depot"])


def ledger_line(index: int, values: dict[str, Any]) -> str:
    return (
        f"Entry {index:06d} | date {values['date']} | clerk {values['clerk']} | "
        f"depot {values['depot']} | shipment {values['shipment']} crates | "
        f"material {values['material']} | destination {values['destination']} | "
        f"revision {values['revision']}."
    )


def query(values: dict[str, Any]) -> str:
    return f"date {values['date']}, clerk {values['clerk']}, depot {values['depot']}"


def compact_answer(label: str, values: dict[str, Any]) -> str:
    return (
        f"{label}={values['shipment']}|{values['material']}|"
        f"{values['destination']}|{values['revision']}"
    )


def hard_distractors(target: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(12):
        candidate = dict(ledger_values(seed + index))
        selector = index % 3
        if selector != 0:
            candidate["date"] = target["date"]
        if selector != 1:
            candidate["clerk"] = target["clerk"]
        if selector != 2:
            candidate["depot"] = target["depot"]
        result.append(candidate)
    return result


def build_fixture(seed: int, target_prompt_tokens: int, record_count: int | None = None) -> Fixture:
    if seed not in SEEDS or target_prompt_tokens <= OUTPUT_TOKENS:
        raise QualificationError("invalid needle fixture profile")
    if record_count is None:
        record_count = max(4096, target_prompt_tokens // 52)
    margin = 16
    spacing = (record_count - 2 * margin) // NEEDLE_COUNT
    if spacing >= 25:
        offsets = (-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12)
    elif spacing >= 13:
        offsets = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
    elif spacing >= 9:
        offsets = (-4, -3, -2, -1, 1, 2, 3, 4)
    else:
        raise QualificationError("context is too short for 64 separated needle regions")

    records = [ledger_values(index) for index in range(record_count)]
    targets = [ledger_values(2_000_000 + seed * 10_000 + index) for index in range(NEEDLE_COUNT)]
    target_signatures = {signature(values) for values in targets}
    if len(target_signatures) != NEEDLE_COUNT:
        raise QualificationError("needle signatures are not unique")
    for index, values in enumerate(records):
        if signature(values) in target_signatures:
            replacement = ledger_values(4_000_000 + seed * record_count + index)
            while signature(replacement) in target_signatures:
                replacement = ledger_values(4_000_000 + mix64(index + replacement["shipment"]))
            records[index] = replacement

    rng = random.Random(0x4B564E00 + seed * 257 + NEEDLE_COUNT)
    usable = record_count - 2 * margin
    positions: list[int] = []
    distractor_span = 2 * max(abs(offset) for offset in offsets) + 1
    for index in range(NEEDLE_COUNT):
        center = margin + int((index + 0.5) * usable / NEEDLE_COUNT)
        region = usable // NEEDLE_COUNT
        jitter = min(max(1, usable // (NEEDLE_COUNT * 4)), max(0, (region - distractor_span) // 2))
        positions.append(
            max(margin, min(record_count - margin - 1, center + rng.randint(-jitter, jitter)))
        )
    if len(set(positions)) != NEEDLE_COUNT:
        raise QualificationError("needle positions overlap")

    reserved = set(positions)
    for target_index, (position, target) in enumerate(zip(positions, targets, strict=True)):
        distractors = hard_distractors(target, 3_000_000 + seed * 1000 + target_index)
        for candidate_index, (offset, candidate) in enumerate(zip(offsets, distractors, strict=False)):
            field = ("date", "clerk", "depot")[candidate_index % 3]
            alternatives: Sequence[str]
            if field == "date":
                alternatives = tuple(
                    f"{year:04d}-{month:02d}-{day:02d}"
                    for year in range(2029, 2048)
                    for month in range(1, 13)
                    for day in range(1, 29)
                )
            elif field == "clerk":
                alternatives = tuple(
                    f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES
                )
            else:
                alternatives = DEPOTS
            for alternative in alternatives:
                candidate[field] = alternative
                if signature(candidate) not in target_signatures:
                    break
            destination = position + offset
            if destination < 0 or destination >= record_count or destination in reserved:
                raise QualificationError("needle distractor placement overlaps")
            records[destination] = candidate
            reserved.add(destination)
        records[position] = target

    labels = [f"N{index:02d}" for index in range(NEEDLE_COUNT)]
    needles = tuple(
        Needle(
            label,
            signature(target),
            compact_answer(label, target),
            100.0 * position / (record_count - 1),
        )
        for label, target, position in zip(labels, targets, positions, strict=True)
    )
    for needle in needles:
        if sum(signature(values) == needle.signature for values in records) != 1:
            raise QualificationError(f"{needle.label} is not unique in the document")
    order = list(range(NEEDLE_COUNT))
    rng.shuffle(order)
    queries = "\n".join(f"{labels[index]}: {query(targets[index])}" for index in order)
    expected = "\n".join(needles[index].answer for index in order)
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
    return Fixture(seed, target_prompt_tokens, record_count, messages, expected, needles)


def fixture_identity(fixture: Fixture) -> str:
    represented = json.dumps(
        {"messages": fixture.messages, "expected": fixture.expected, "max_tokens": OUTPUT_TOKENS},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(represented).hexdigest()


def response_text(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
        return str(message.get("content") or message.get("reasoning_content") or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise QualificationError(f"malformed response: {exc}") from exc


def recall_stats(text: str, fixture: Fixture) -> dict[str, Any]:
    observed: dict[str, list[str]] = {}
    for match in NEEDLE_LINE.finditer(text):
        label = match.group(1)
        observed.setdefault(label, []).append(f"{label}={match.group(2)}")
    correct = [
        needle.label
        for needle in fixture.needles
        if needle.answer in observed.get(needle.label, [])
    ]
    expected = {needle.label: needle.answer for needle in fixture.needles}
    incorrect = sorted(
        label
        for label, answers in observed.items()
        if label not in expected or any(answer != expected.get(label) for answer in answers)
    )
    buckets: dict[str, dict[str, int | float]] = {}
    for name, lower, upper in (
        ("early", 0.0, 33.333),
        ("middle", 33.333, 66.667),
        ("late", 66.667, 100.001),
    ):
        labels = [
            needle.label for needle in fixture.needles if lower <= needle.depth_percent < upper
        ]
        count = sum(label in correct for label in labels)
        buckets[name] = {"correct": count, "total": len(labels), "recall": count / len(labels)}
    return {
        "correct": len(correct),
        "total": len(fixture.needles),
        "recall": len(correct) / len(fixture.needles),
        "missed_labels": [needle.label for needle in fixture.needles if needle.label not in correct],
        "incorrect_labels": incorrect,
        "duplicate_labels": sorted(label for label, answers in observed.items() if len(answers) > 1),
        "depth_buckets": buckets,
    }


def count_tokens(connection: corpus.http.client.HTTPConnection, model_id: str, fixture: Fixture) -> int:
    response = post_json(
        connection,
        "/v1/messages/count_tokens",
        {
            "model": model_id,
            "system": fixture.messages[0]["content"],
            "messages": fixture.messages[1:],
            "max_tokens": OUTPUT_TOKENS,
            "thinking": {"type": "disabled"},
        },
    )
    value = response.get("input_tokens")
    if not isinstance(value, int) or value <= 0:
        raise QualificationError("token-count response has no positive input_tokens")
    return value


def calibrate_fixture(
    connection: corpus.http.client.HTTPConnection, model_id: str, seed: int, target_tokens: int
) -> tuple[Fixture, int]:
    current = build_fixture(seed, target_tokens)
    tokens = count_tokens(connection, model_id, current)
    low: tuple[Fixture, int] | None = None
    high: tuple[Fixture, int] | None = None
    for _ in range(20):
        if tokens <= target_tokens:
            low = (current, tokens)
        else:
            high = (current, tokens)
        if low and high and high[0].record_count - low[0].record_count <= 1:
            break
        if low and high:
            next_count = (low[0].record_count + high[0].record_count) // 2
        else:
            estimate = max(NEEDLE_COUNT * 9 + 32, int(current.record_count * target_tokens / tokens))
            next_count = estimate + max(8, abs(estimate - current.record_count) // 4) * (1 if low else -1)
        if next_count == current.record_count:
            next_count += 1 if low else -1
        next_count = max(NEEDLE_COUNT * 9 + 32, next_count)
        current = build_fixture(seed, target_tokens, next_count)
        tokens = count_tokens(connection, model_id, current)
    if low is None:
        raise QualificationError("failed to calibrate a needle document within context")
    return low


def post_json(
    connection: corpus.http.client.HTTPConnection, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    return corpus.receive_json(connection)


def server_command(args: argparse.Namespace, dtype: str, server_log: Path) -> list[str]:
    return [
        str(args.serve),
        str(args.artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model-id",
        args.model_id,
        "--device",
        str(args.device),
        "--max-context",
        str(args.max_context),
        "--kv-capacity",
        str(args.max_context),
        "--max-concurrency",
        "1",
        "--max-pending-requests",
        "1",
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--kv-dtype",
        dtype,
        "--no-prefix-reuse",
        "--greedy",
        "--log-stats-interval-ms",
        "0",
        "--request-log-jsonl",
        str(server_log),
    ]


def validate_start(event: dict[str, Any], args: argparse.Namespace, dtype: str) -> tuple[str, str]:
    corpus.require_server_log_identity(event, "server_start")
    if event.get("server", {}).get("public_model_id") != args.model_id:
        raise QualificationError("server model identity mismatch")
    if event.get("engine", {}).get("kv_cache") != KV_LOG_NAMES[dtype]:
        raise QualificationError("server KV format mismatch")
    instance = event.get("server_instance_id")
    weights_id = event.get("artifact", {}).get("weights_id")
    if not isinstance(instance, str) or not isinstance(weights_id, str):
        raise QualificationError("server startup record lacks identity")
    return instance, weights_id


def load_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("artifact_type") != "ninfer_kv_cache_needle_result":
                raise ValueError("wrong artifact type")
            key = str(record["kv_dtype"]), int(record["seed"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise QualificationError(f"invalid result at {path}:{line_number}: {exc}") from exc
        if key in records:
            raise QualificationError(f"duplicate result for {key}")
        records[key] = record
    return records


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_dtype(
    args: argparse.Namespace,
    dtype: str,
    records: dict[tuple[str, int], dict[str, Any]],
    handle: Any,
) -> None:
    server_log = args.output / "server" / f"{dtype}.jsonl"
    with corpus.RunningServer(
        server_command(args, dtype, server_log), "127.0.0.1", args.port, server_log
    ) as server:
        instance, weights_id = validate_start(server.wait_until_ready(), args, dtype)
        connection = corpus.http.client.HTTPConnection(
            "127.0.0.1", args.port, timeout=corpus.REQUEST_TIMEOUT_SECONDS
        )
        try:
            for seed in args.seed:
                fixture, prompt_tokens = calibrate_fixture(
                    connection, args.model_id, seed, args.max_context - OUTPUT_TOKENS - CONTEXT_MARGIN
                )
                identity = fixture_identity(fixture)
                existing = records.get((dtype, seed))
                if (
                    existing
                    and existing.get("fixture_id") == identity
                    and existing.get("max_context") == args.max_context
                    and Path(existing.get("artifact", "")).resolve() == args.artifact
                ):
                    print(f"skip {dtype}/seed-{seed}: existing matching result", flush=True)
                    continue
                if existing:
                    raise QualificationError(
                        f"existing {dtype}/seed-{seed} result has a different profile; "
                        "use a new output directory"
                    )
                response = corpus.post_json(
                    connection,
                    {
                        "model": args.model_id,
                        "messages": fixture.messages,
                        "max_completion_tokens": OUTPUT_TOKENS,
                        "temperature": 0,
                        "seed": 42,
                        "stream": False,
                        "enable_thinking": False,
                    },
                )
                event = server.wait_for_request_done(instance)
                corpus.require_server_log_identity(event, "request_done")
                text = response_text(response)
                recall = recall_stats(text, fixture)
                result = event.get("result", {})
                timings = event.get("timings_seconds", {})
                record = {
                    "artifact_type": "ninfer_kv_cache_needle_result",
                    "schema_version": 1,
                    "artifact": str(args.artifact),
                    "weights_id": weights_id,
                    "model_id": args.model_id,
                    "kv_dtype": dtype,
                    "seed": seed,
                    "max_context": args.max_context,
                    "fixture_id": identity,
                    "prompt_tokens": int(result.get("prompt_tokens", 0)),
                    "counted_prompt_tokens": prompt_tokens,
                    "completion_tokens": int(result.get("completion_tokens", 0)),
                    "finish_reason": result.get("finish_reason"),
                    "prefill_seconds": float(timings.get("prefill", 0.0)),
                    "decode_seconds": float(timings.get("decode", 0.0)),
                    "recall": recall,
                    "response_text": text,
                    "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                append_record(handle, record)
                records[(dtype, seed)] = record
                print(
                    f"{dtype}/seed-{seed}: recall={recall['correct']}/{recall['total']} "
                    f"prompt={record['prompt_tokens']} completion={record['completion_tokens']}",
                    flush=True,
                )
        finally:
            connection.close()


def write_summary(args: argparse.Namespace, records: dict[tuple[str, int], dict[str, Any]]) -> None:
    rows = []
    for dtype in args.kv_dtype:
        selected = [records[(dtype, seed)] for seed in args.seed if (dtype, seed) in records]
        correct = sum(record["recall"]["correct"] for record in selected)
        total = sum(record["recall"]["total"] for record in selected)
        per_seed = [
            {
                "seed": int(record["seed"]),
                "correct": int(record["recall"]["correct"]),
                "total": int(record["recall"]["total"]),
                "recall": float(record["recall"]["recall"]),
            }
            for record in selected
        ]
        rows.append(
            {
                "kv_dtype": dtype,
                "seeds": len(selected),
                "correct": correct,
                "total": total,
                "mean_recall": statistics.fmean(row["recall"] for row in per_seed)
                if per_seed
                else None,
                "per_seed": per_seed,
            }
        )
    summary = {
        "artifact_type": "ninfer_kv_cache_needle_summary",
        "schema_version": 1,
        "max_context": args.max_context,
        "needle_count_per_seed": NEEDLE_COUNT,
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# KV-cache needle qualification",
        "",
        f"Context limit: {args.max_context}; deterministic needles per seed: {NEEDLE_COUNT}.",
        "",
        "| KV format | Seeds | Mean recall | Aggregate | Per-seed correct |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        mean_recall = row["mean_recall"]
        mean_text = f"{100.0 * mean_recall:.2f}%" if mean_recall is not None else "n/a"
        per_seed = ", ".join(
            f"s{seed['seed']}={seed['correct']}/{seed['total']}" for seed in row["per_seed"]
        )
        lines.append(
            f"| {row['kv_dtype']} | {row['seeds']} | {mean_text} | "
            f"{row['correct']}/{row['total']} | {per_seed or '-'} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--serve", type=Path, default=REPO_ROOT / "build/apps/ninfer-serve")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-dtype", action="append", choices=tuple(KV_LOG_NAMES))
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--max-context", type=int, default=196608)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    args.artifact = args.artifact.expanduser().resolve()
    args.serve = args.serve.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.kv_dtype = args.kv_dtype or list(KV_LOG_NAMES)
    args.seed = args.seed or list(SEEDS)
    if len(args.kv_dtype) != len(set(args.kv_dtype)) or len(args.seed) != len(set(args.seed)):
        raise QualificationError("duplicate KV format or seed")
    if not args.artifact.is_file() or not args.serve.is_file() or not os.access(args.serve, os.X_OK):
        raise QualificationError("artifact or server executable is unavailable")
    if args.max_context < 65536 or args.prefill_chunk <= 0 or not 1 <= args.port <= 65535:
        raise QualificationError("invalid context, prefill chunk, or port")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    (args.output / "server").mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.jsonl"
    records = load_records(run_path)
    with run_path.open("a", encoding="utf-8") as handle:
        for dtype in args.kv_dtype:
            run_dtype(args, dtype, records, handle)
    write_summary(args, records)
    print(f"summary: {args.output / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted; completed results remain resumable", file=sys.stderr)
        raise SystemExit(130) from None
