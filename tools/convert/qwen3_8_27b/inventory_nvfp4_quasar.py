"""Persistent-object contract for the Qwen3.8-27B QUASAR NVFP4 artifact."""

from __future__ import annotations

from . import inventory_nvfp4 as base


MODEL_ID = base.MODEL_ID
WEIGHTS_ID = base.WEIGHTS_ID
TARGET_KEY = base.TARGET_KEY

BF16 = base.BF16
FP32 = base.FP32
I32 = base.I32
Q4 = base.Q4
Q5 = base.Q5
Q6 = base.Q6
W8 = base.W8
NVFP4 = base.NVFP4
FP8 = base.FP8

CONTIGUOUS_LAYOUT = base.CONTIGUOUS_LAYOUT
ROW_SPLIT_LAYOUT = base.ROW_SPLIT_LAYOUT
BLOCK_SCALE_LAYOUT = base.BLOCK_SCALE_LAYOUT
ROW_SCALE_LAYOUT = base.ROW_SCALE_LAYOUT

ResourceSpec = base.ResourceSpec
StoredObjectSpec = base.StoredObjectSpec
TensorSpec = base.TensorSpec

RESOURCE_SPECS = base.RESOURCE_SPECS
FULL_ATTENTION_LAYERS = base.FULL_ATTENTION_LAYERS
GDN_LAYERS = base.GDN_LAYERS


def tensor_spec(
    name: str, shape: tuple[int, ...], numeric_format: str
) -> TensorSpec:
    return base.tensor_spec(name, shape, numeric_format)


def _input_divisor_name(name: str) -> str:
    prefix, suffix = name.rsplit("/", 1)
    if suffix == "query_key_gate_value" and prefix.endswith("/attention"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/attention"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "query_key_value_z" and prefix.endswith("/gdn"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/gdn"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "gate_up" and prefix.endswith("/mlp"):
        return prefix + "/gate_up_projection/input_scale_divisor"
    if suffix == "down" and prefix.endswith("/mlp"):
        return prefix + "/down_projection/input_scale_divisor"
    raise ValueError(f"no QUASAR input-divisor role for {name}")


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for original in base.TEXT_CORE_TENSOR_SPECS:
        if original.format != FP8:
            specs.append(original)
            continue
        if original.name in ("text/token_embedding", "text/output_head"):
            specs.append(tensor_spec(original.name, original.shape, W8))
            continue
        replacement = tensor_spec(original.name, original.shape, NVFP4)
        specs.extend(
            (
                replacement,
                tensor_spec(_input_divisor_name(original.name), (), FP32),
            )
        )
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = base.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = base.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = base.VISION_TENSOR_SPECS

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4, FP8)
LAYOUT_NAMES = (
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    BLOCK_SCALE_LAYOUT,
    ROW_SCALE_LAYOUT,
)
FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

NVFP4_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == NVFP4
)
FP8_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == FP8)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)

LOGICAL_ROW_VIEW_SPECS = base.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = base.ALIAS_SPECS


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("QUASAR NVFP4 inventory contains duplicate object names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(FP8_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
    ) != (915, 2, 12, 333, 1262, 1268, 256, 0, 256):
        raise ValueError("registered QUASAR NVFP4 inventory is incomplete")
    if FORMAT_COUNTS != {
        BF16: 534,
        FP32: 352,
        I32: 1,
        Q4: 55,
        Q5: 54,
        Q6: 1,
        W8: 9,
        NVFP4: 256,
        FP8: 0,
    }:
        raise ValueError(f"unexpected QUASAR numeric allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 887,
        ROW_SPLIT_LAYOUT: 119,
        BLOCK_SCALE_LAYOUT: 256,
        ROW_SCALE_LAYOUT: 0,
    }:
        raise ValueError(f"unexpected QUASAR layout allocation: {LAYOUT_COUNTS}")


validate_inventory()
