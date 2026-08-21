"""Checkpoint format and model-size metadata helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .dataset import _MODEL_CONFIGS

CHECKPOINT_FORMAT_VERSION = 1


def _normalize_model_size(value: str) -> str:
    """Normalize model-size aliases such as 'yolox_l' and 'yolox-l' to 'l'."""
    if not isinstance(value, str):
        raise TypeError(f"model_sizeは文字列で指定してください: {type(value)}")

    normalized = value.lower().removeprefix("yolox_").removeprefix("yolox-")
    if normalized not in _MODEL_CONFIGS:
        raise ValueError(
            f"未対応のモデルサイズ: '{value}'\n"
            f"使用可能: {list(_MODEL_CONFIGS.keys())} または 'yolox_{{size}}'"
        )
    return normalized


def checkpoint_model_metadata(model_size: str | None) -> dict[str, Any]:
    """Build versioned checkpoint metadata for a known model size."""
    metadata: dict[str, Any] = {"format_version": CHECKPOINT_FORMAT_VERSION}
    if model_size is None:
        return metadata

    normalized = _normalize_model_size(model_size)
    cfg = _MODEL_CONFIGS[normalized]
    metadata.update(
        {
            "model_size": normalized,
            "depth": cfg["depth"],
            "width": cfg["width"],
        }
    )
    return metadata


def resolve_checkpoint_model_size(
    checkpoint: Mapping[str, Any],
    explicit_model_size: str | None = None,
) -> str | None:
    """Resolve and cross-check checkpoint and caller-provided model sizes."""
    _validate_format_version(checkpoint)
    explicit = (
        _normalize_model_size(explicit_model_size)
        if explicit_model_size is not None
        else None
    )

    stored: str | None = None
    raw_size = checkpoint.get("model_size")
    if raw_size is not None:
        if not isinstance(raw_size, str):
            raise ValueError(
                "checkpointのmodel_sizeが文字列ではありません: "
                f"{type(raw_size)}"
            )
        stored = _normalize_model_size(raw_size)

    dimensions = _model_size_from_dimensions(checkpoint)
    if stored is not None and dimensions is not None and stored != dimensions:
        raise ValueError(
            "checkpointのモデルサイズ情報が矛盾しています: "
            f"model_size='{stored}', depth/width='{dimensions}'"
        )

    detected = stored or dimensions
    if explicit is not None and detected is not None and explicit != detected:
        raise ValueError(
            "指定されたモデルサイズがcheckpointと一致しません: "
            f"指定='{explicit}', checkpoint='{detected}'"
        )
    return detected or explicit


def _validate_format_version(checkpoint: Mapping[str, Any]) -> None:
    raw_version = checkpoint.get("format_version")
    if raw_version is None:
        return
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError(
            "checkpointのformat_versionは整数で指定する必要があります: "
            f"{raw_version!r}"
        )
    if raw_version < 1 or raw_version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"未対応のcheckpoint format_version: {raw_version} "
            f"(対応バージョン: {CHECKPOINT_FORMAT_VERSION})"
        )


def _model_size_from_dimensions(checkpoint: Mapping[str, Any]) -> str | None:
    has_depth = "depth" in checkpoint and checkpoint["depth"] is not None
    has_width = "width" in checkpoint and checkpoint["width"] is not None
    if not has_depth and not has_width:
        return None
    if has_depth != has_width:
        raise ValueError(
            "checkpointのモデルサイズ情報が不完全です: "
            "depthとwidthの両方が必要です"
        )

    try:
        depth = float(checkpoint["depth"])
        width = float(checkpoint["width"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "checkpointのdepthとwidthは数値で指定する必要があります"
        ) from exc

    for size, cfg in _MODEL_CONFIGS.items():
        if math.isclose(depth, cfg["depth"], abs_tol=1e-9) and math.isclose(
            width, cfg["width"], abs_tol=1e-9
        ):
            return size
    raise ValueError(
        "checkpointのdepth/widthに対応するモデルサイズがありません: "
        f"depth={depth}, width={width}"
    )
