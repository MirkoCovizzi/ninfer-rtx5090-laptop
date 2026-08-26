"""Closed dual-source recipe for the Qwen3.8-27B QUASAR NVFP4 artifact."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import recipe as official_recipe

from . import inventory_nvfp4_quasar as inventory
from . import recipe_nvfp4 as shared


BASE_REPOSITORY = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QUANTIZED_REPOSITORY = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
QUANTIZED_REVISION = "d8e6fbfa3e3a78899b440222b827430045a05b44"

RowRange = shared.RowRange
MatrixSource = shared.MatrixSource
MatrixPart = shared.MatrixPart
Nvfp4WeightRecipe = shared.Nvfp4WeightRecipe
InputDivisorRecipe = shared.InputDivisorRecipe


@dataclass(frozen=True, slots=True)
class ControlRecipe:
    object_name: str
    sources: tuple[MatrixSource, MatrixSource]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    begin = 256 if gate else 0
    return MatrixPart(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def _build_matrix_recipes() -> tuple[
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
    tuple[tuple[MatrixSource, ...], ...],
    tuple[ControlRecipe, ...],
]:
    weights: list[Nvfp4WeightRecipe] = []
    inputs: list[InputDivisorRecipe] = []
    divisor_groups: list[tuple[MatrixSource, ...]] = []
    controls: list[ControlRecipe] = []

    def add_weight(
        object_name: str,
        shape: tuple[int, int],
        parts: tuple[MatrixPart, ...],
        sources: tuple[MatrixSource, ...],
        input_name: str,
    ) -> None:
        weights.append(Nvfp4WeightRecipe(object_name, shape, parts, sources))
        inputs.append(InputDivisorRecipe(input_name, sources, (object_name,)))
        if len(sources) > 1:
            divisor_groups.append(sources)

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            group = (query, key, value)
            add_weight(
                object_prefix + "attention/query_key_gate_value",
                (14336, 5120),
                (
                    _q_part(query, False),
                    _all(key),
                    _q_part(query, True),
                    _all(value),
                ),
                group,
                object_prefix
                + "attention/input_projection/input_scale_divisor",
            )
            add_weight(
                object_prefix + "attention/output",
                output.shape,
                (_all(output),),
                (output,),
                object_prefix
                + "attention/output_projection/input_scale_divisor",
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(
                source_prefix + "linear_attn.out_proj", 5120, 6144
            )
            group = (query_key_value, z)
            add_weight(
                object_prefix + "gdn/query_key_value_z",
                (16384, 5120),
                (_all(query_key_value), _all(z)),
                group,
                object_prefix + "gdn/input_projection/input_scale_divisor",
            )
            add_weight(
                object_prefix + "gdn/output",
                output.shape,
                (_all(output),),
                (output,),
                object_prefix + "gdn/output_projection/input_scale_divisor",
            )
            controls.append(
                ControlRecipe(
                    object_prefix + "gdn/a_b_projection",
                    (
                        _source(
                            source_prefix + "linear_attn.in_proj_a", 48, 5120
                        ),
                        _source(
                            source_prefix + "linear_attn.in_proj_b", 48, 5120
                        ),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        gate_up_group = (gate, up)
        add_weight(
            object_prefix + "mlp/gate_up",
            (34816, 5120),
            (_all(gate), _all(up)),
            gate_up_group,
            object_prefix + "mlp/gate_up_projection/input_scale_divisor",
        )
        add_weight(
            object_prefix + "mlp/down",
            down.shape,
            (_all(down),),
            (down,),
            object_prefix + "mlp/down_projection/input_scale_divisor",
        )

    return tuple(weights), tuple(inputs), tuple(divisor_groups), tuple(controls)


(
    NVFP4_WEIGHT_RECIPES,
    INPUT_DIVISOR_RECIPES,
    WEIGHT_DIVISOR_GROUPS,
    CONTROL_RECIPES,
) = _build_matrix_recipes()
NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in NVFP4_WEIGHT_RECIPES
}
INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in INPUT_DIVISOR_RECIPES
}
CONTROLS_BY_NAME = {item.object_name: item for item in CONTROL_RECIPES}

NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source
        for recipe in NVFP4_WEIGHT_RECIPES
        for part in recipe.parts
    )
)
CONTROL_SOURCES = tuple(
    source for recipe in CONTROL_RECIPES for source in recipe.sources
)
ALL_NVFP4_SOURCES = tuple(dict.fromkeys(NVFP4_SOURCES + CONTROL_SOURCES))

QUANTIZED_DIRECT_RECIPES = tuple(
    recipe
    for recipe in shared.QUANTIZED_DIRECT_RECIPES
    if recipe.object_name not in CONTROLS_BY_NAME
)
QUANTIZED_DIRECT_BY_NAME = {
    item.object_name: item for item in QUANTIZED_DIRECT_RECIPES
}
QUANTIZED_DIRECT_SPECS = tuple(
    spec
    for spec in inventory.TEXT_CORE_TENSOR_SPECS
    if spec.name in QUANTIZED_DIRECT_BY_NAME
)
QUANTIZED_MTP_SPECS = inventory.MTP_TENSOR_SPECS
QUANTIZED_MTP_RECIPES = tuple(
    official_recipe.RECIPES_BY_NAME[spec.name]
    for spec in QUANTIZED_MTP_SPECS
)
QUANTIZED_MTP_BY_NAME = {
    item.object_name: item for item in QUANTIZED_MTP_RECIPES
}

OFFICIAL_TENSOR_SPECS = tuple(
    spec
    for spec in inventory.TENSOR_SPECS
    if spec.name in ("text/token_embedding", "text/output_head")
    or spec.name.startswith("text/draft_head")
    or spec.name.startswith("vision/")
)
OFFICIAL_RECIPES = tuple(
    official_recipe.RECIPES_BY_NAME[spec.name]
    for spec in OFFICIAL_TENSOR_SPECS
)
OFFICIAL_RECIPES_BY_NAME = {
    item.object_name: item for item in OFFICIAL_RECIPES
}


def _validate_matrix_recipe(recipe: Nvfp4WeightRecipe) -> None:
    rows = sum(part.output_rows for part in recipe.parts)
    if not recipe.parts or (rows, recipe.parts[0].source.shape[1]) != recipe.shape:
        raise ValueError(f"{recipe.object_name}: invalid fused row geometry")
    if any(part.source.shape[1] != recipe.shape[1] for part in recipe.parts):
        raise ValueError(f"{recipe.object_name}: incompatible source K")


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(
        QUANTIZED_DIRECT_RECIPES, QUANTIZED_DIRECT_SPECS
    )
    family_recipe.validate_recipe_coverage(
        QUANTIZED_MTP_RECIPES, QUANTIZED_MTP_SPECS
    )
    family_recipe.validate_recipe_coverage(
        OFFICIAL_RECIPES, OFFICIAL_TENSOR_SPECS
    )
    if (
        len(NVFP4_WEIGHT_RECIPES),
        len(INPUT_DIVISOR_RECIPES),
        len(WEIGHT_DIVISOR_GROUPS),
        len(NVFP4_SOURCES),
        len(CONTROL_RECIPES),
        len(CONTROL_SOURCES),
        len(ALL_NVFP4_SOURCES),
        len(QUANTIZED_DIRECT_RECIPES),
        len(QUANTIZED_MTP_RECIPES),
        len(OFFICIAL_RECIPES),
    ) != (256, 256, 128, 400, 48, 96, 496, 353, 12, 337):
        raise ValueError("Qwen3.8 QUASAR NVFP4 source recipe is incomplete")
    ownership = (
        set(NVFP4_WEIGHTS_BY_NAME),
        set(INPUT_DIVISORS_BY_NAME),
        set(CONTROLS_BY_NAME),
        set(QUANTIZED_DIRECT_BY_NAME),
        set(QUANTIZED_MTP_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME),
    )
    all_names: set[str] = set()
    for names in ownership:
        if all_names.intersection(names):
            raise ValueError("more than one source route owns an artifact tensor")
        all_names.update(names)
    if all_names != {spec.name for spec in inventory.TENSOR_SPECS}:
        raise ValueError("source routes do not cover the QUASAR tensor inventory")
    if tuple(NVFP4_WEIGHTS_BY_NAME) != tuple(
        spec.name for spec in inventory.NVFP4_TENSOR_SPECS
    ):
        raise ValueError("QUASAR NVFP4 recipe order does not match inventory")
    if tuple(INPUT_DIVISORS_BY_NAME) != tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    ):
        raise ValueError("QUASAR input-divisor order does not match inventory")
    for recipe in NVFP4_WEIGHT_RECIPES:
        _validate_matrix_recipe(recipe)
    bound_weights = tuple(
        name for site in INPUT_DIVISOR_RECIPES for name in site.weight_names
    )
    if len(bound_weights) != 256 or set(bound_weights) != set(
        NVFP4_WEIGHTS_BY_NAME
    ):
        raise ValueError("QUASAR input-divisor sites do not cover NVFP4 parents")


def _merge_requirement(
    result: dict[str, tuple[tuple[int, ...], str]],
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    signature = (shape, dtype)
    previous = result.setdefault(name, signature)
    if previous != signature:
        raise ValueError(f"inconsistent QUASAR source declaration for {name}")


def _source_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in ALL_NVFP4_SOURCES:
        n, k = source.shape
        _merge_requirement(
            result, source.field("weight_packed"), (n, k // 2), "U8"
        )
        _merge_requirement(
            result, source.field("weight_scale"), (n, k // 16), "F8_E4M3"
        )
        _merge_requirement(
            result, source.field("weight_global_scale"), (1,), "F32"
        )
        _merge_requirement(
            result, source.field("input_global_scale"), (1,), "F32"
        )
    for source in family_recipe.source_requirements(
        QUANTIZED_DIRECT_RECIPES + QUANTIZED_MTP_RECIPES
    ).values():
        _merge_requirement(result, source.name, source.shape, source.dtype)
    return result


SOURCE_REQUIREMENTS = _source_requirements()
EXPECTED_QUANTIZED_FIELDS = frozenset(
    source.field(suffix)
    for source in ALL_NVFP4_SOURCES
    for suffix in (
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
    )
)


def preflight_quantized_metadata(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    missing = set(SOURCE_REQUIREMENTS).difference(reader.names)
    if missing:
        raise ValueError(f"QUASAR source is missing {sorted(missing)[0]}")
    metadata = reader.metadata(reader.names)
    actual_quantized_fields = frozenset(
        name
        for name, item in metadata.items()
        if item.dtype in ("F8_E4M3", "F32", "U8")
        and any(
            name.endswith("." + suffix)
            for suffix in (
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            )
        )
    )
    if actual_quantized_fields != EXPECTED_QUANTIZED_FIELDS:
        unexpected = actual_quantized_fields.difference(EXPECTED_QUANTIZED_FIELDS)
        absent = EXPECTED_QUANTIZED_FIELDS.difference(actual_quantized_fields)
        detail = sorted(unexpected or absent)[0]
        raise ValueError(f"QUASAR source allocation is not closed: {detail}")

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shards.add(actual.shard)
    return family_recipe.SourcePreflight(
        recipe_count=(
            len(NVFP4_WEIGHT_RECIPES)
            + len(INPUT_DIVISOR_RECIPES)
            + len(CONTROL_RECIPES)
            + len(QUANTIZED_DIRECT_RECIPES)
            + len(QUANTIZED_MTP_RECIPES)
        ),
        source_tensor_count=len(SOURCE_REQUIREMENTS),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def preflight_official_sources(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    return family_recipe.preflight_source_reader(reader, OFFICIAL_RECIPES)


def _word(tensor: torch.Tensor, name: str) -> int:
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError(f"{name}: divisor must be FP32[1]")
    word = int(tensor.detach().contiguous().cpu().view(torch.int32).item())
    word &= 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return word


def _same_divisor(
    reader: ShardReader,
    sources: Iterable[MatrixSource],
    suffix: str,
) -> int:
    items = tuple(sources)
    words = tuple(
        _word(reader.get(source.field(suffix)), source.field(suffix))
        for source in items
    )
    if len(set(words)) != 1:
        raise ValueError(f"{items[0].name}: fused {suffix} words differ")
    return words[0]


def validate_nvfp4_words(reader: ShardReader) -> None:
    for source in ALL_NVFP4_SOURCES:
        scales = reader.get(source.field("weight_scale")).view(torch.uint8)
        invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
        if bool(invalid.any()):
            raise ValueError(
                f"{source.field('weight_scale')}: invalid E4M3FN scale word"
            )
        _word(
            reader.get(source.field("weight_global_scale")),
            source.field("weight_global_scale"),
        )
        _word(
            reader.get(source.field("input_global_scale")),
            source.field("input_global_scale"),
        )
    for group in WEIGHT_DIVISOR_GROUPS:
        _same_divisor(reader, group, "weight_global_scale")
    for recipe in INPUT_DIVISOR_RECIPES:
        _same_divisor(reader, recipe.sources, "input_global_scale")


def materialize_nvfp4_weight(
    recipe: Nvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    return shared.materialize_nvfp4_weight(recipe, reader)


def materialize_input_divisor(
    recipe: InputDivisorRecipe,
    reader: ShardReader,
) -> torch.Tensor:
    word = _same_divisor(reader, recipe.sources, "input_global_scale")
    return torch.frombuffer(
        bytearray(struct.pack("<I", word)), dtype=torch.float32
    ).reshape(())


def _decode_nvfp4_source(
    source: MatrixSource,
    reader: ShardReader,
) -> torch.Tensor:
    packed = reader.get(source.field("weight_packed")).to(torch.uint8)
    codes = torch.empty(source.shape, dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    lookup = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.float32,
    )
    scales = reader.get(source.field("weight_scale")).to(torch.float32)
    divisor = reader.get(source.field("weight_global_scale")).to(torch.float32)
    decoded = lookup[codes.to(torch.long)]
    decoded *= scales.repeat_interleave(16, dim=1)
    decoded /= divisor
    return decoded.to(torch.bfloat16)


def materialize_control(
    recipe: ControlRecipe,
    reader: ShardReader,
) -> torch.Tensor:
    value = torch.cat(
        tuple(_decode_nvfp4_source(source, reader) for source in recipe.sources),
        dim=0,
    )
    if tuple(value.shape) != (96, 5120):
        raise ValueError(f"{recipe.object_name}: decoded control shape mismatch")
    return value


def materialize_quantized_direct(
    object_name: str,
    reader: ShardReader,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        QUANTIZED_DIRECT_BY_NAME[object_name], reader
    )


def materialize_quantized_mtp(
    object_name: str,
    reader: ShardReader,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        QUANTIZED_MTP_BY_NAME[object_name], reader
    )


def materialize_official(
    object_name: str,
    reader: ShardReader,
    derived_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        OFFICIAL_RECIPES_BY_NAME[object_name], reader, derived_tensors
    )


validate_recipe()
