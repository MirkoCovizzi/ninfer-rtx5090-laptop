from __future__ import annotations

import json

from tools.bench.run_ninfer_bench_matrix import BenchCase, report_rows


def test_schema_v13_report_is_flattened_for_matrix_summary(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 13,
                "artifact_type": "ninfer_bench_report",
                "tool": "ninfer_bench",
                "artifact": {"path": "model.ninfer"},
                "environment": {"gpu_name": "RTX 5090"},
                "load": {
                    "target": "qwen3_6_27b",
                    "weights_id": "nvfp4",
                    "load_seconds": 2.5,
                    "upload_seconds": 2.0,
                    "artifact_bytes_read": 17_500_000_000,
                    "host_to_device_bytes": 17_400_000_000,
                    "peak_staging_bytes": 134_217_728,
                },
                "memory": {
                    "kv_capacity": 8192,
                    "kv_payload_bytes": 123_456,
                    "weights": {"capacity_bytes": 17_400_000_000},
                    "sequence": {"capacity_bytes": 2_000_000_000},
                    "workspace": {"capacity_bytes": 100_000_000},
                    "request_transient": {"capacity_bytes": 50_000_000},
                    "cuda_graph_allowance_bytes": 150_000_000,
                },
                "config": {
                    "max_context": 4096,
                    "prefill_chunk": 1024,
                    "kv_cache": "int8-group64",
                    "mtp_draft_tokens": 5,
                    "proposal_head": "optimized",
                    "mtp_policy": "adaptive",
                    "decode_path": "cuda-graph",
                    "decode_graph_prime": {"primed": True, "output_tokens": 13},
                    "repetitions": 2,
                    "warmup": 1,
                },
                "tests": [
                    {
                        "label": "tg3",
                        "kind": "tg",
                        "n_prompt": 0,
                        "n_gen": 3,
                        "requested_output_tokens": 4,
                        "workspace_peak_bytes": 1_048_576,
                        "workspace_allocator_peak_bytes": 524_288,
                        "decode_output_tok_s_mean": 4.5,
                        "decode_engine_tok_s_mean": 6.5,
                        "total_seconds_mean": 0.875,
                        "speculative": {
                            "backend": "mtp",
                            "acceptance_rate": 0.8,
                            "acceptance_length": 5.0,
                            "rounds": 1,
                            "drafted_tokens": 5,
                            "accepted_tokens": 4,
                            "fallback_steps": 3,
                            "accepted_per_position": [1, 1, 1, 1, 0],
                            "adaptive": True,
                            "drafted_per_position": [1, 1, 1, 1, 1],
                            "rounds_per_window": [3, 0, 0, 0, 1],
                            "window_transitions": 2,
                            "window_transition_counts": [
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 1,
                                0, 0, 0, 0, 0,
                                0, 0, 1, 0, 0,
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = report_rows(
        report_path,
        BenchCase("pure_decode", "tg3_k5_graph", (), repetitions=2, warmup=1),
    )

    assert len(rows) == 1
    row = rows[0]
    assert (row["suite"], row["case"], row["label"], row["kind"]) == (
        "pure_decode",
        "tg3_k5_graph",
        "tg3",
        "tg",
    )
    assert (row["target"], row["weights_id"], row["artifact_path"], row["gpu_name"]) == (
        "qwen3_6_27b",
        "nvfp4",
        "model.ninfer",
        "RTX 5090",
    )
    assert (row["decode_path"], row["decode_graph_primed"]) == (
        "cuda-graph",
        True,
    )
    assert row["mtp_policy"] == "adaptive"
    assert row["decode_graph_prime_output_tokens"] == 13
    assert row["kv_capacity"] == 8192
    assert row["spec_backend"] == "mtp"
    assert row["spec_adaptive"] is True
    assert row["spec_drafted_per_position"] == "[1,1,1,1,1]"
    assert row["spec_rounds_per_window"] == "[3,0,0,0,1]"
    assert row["spec_window_transitions"] == 2
    assert row["spec_window_transition_counts"] == (
        "[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0]"
    )
    assert row["host_to_device_bytes"] == 17_400_000_000
    assert row["workspace_capacity_bytes"] == 100_000_000
    assert row["request_transient_capacity_bytes"] == 50_000_000
    assert row["cuda_graph_allowance_bytes"] == 150_000_000
    assert row["workspace_peak_bytes"] == 1_048_576
    assert row["workspace_allocator_peak_bytes"] == 524_288
    assert row["decode_output_tok_s_mean"] == 4.5
    assert row["decode_engine_tok_s_mean"] == 6.5
    assert row["spec_fallback_steps"] == 3
    assert row["spec_accepted_per_position"] == "[1,1,1,1,0]"
