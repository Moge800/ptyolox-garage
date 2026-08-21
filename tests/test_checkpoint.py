"""Tests for checkpoint model-size metadata."""

from pathlib import Path

import pytest
import torch
from yolox.models import YoloPafpn, YoloxHead, YoloxModule

from ptyolox_garage._checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    checkpoint_model_metadata,
    resolve_checkpoint_model_size,
)
from ptyolox_garage.wrapper import YOLOX


def test_checkpoint_metadata_contains_canonical_model_size() -> None:
    assert checkpoint_model_metadata("yolox-l") == {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_size": "l",
        "depth": 1.0,
        "width": 1.0,
    }


def test_checkpoint_metadata_without_known_size_is_still_versioned() -> None:
    assert checkpoint_model_metadata(None) == {
        "format_version": CHECKPOINT_FORMAT_VERSION
    }


@pytest.mark.parametrize(
    "checkpoint,expected",
    [
        ({"model_size": "tiny"}, "tiny"),
        ({"model_size": "yolox_m"}, "m"),
        ({"depth": 1.33, "width": 1.25}, "x"),
        ({}, None),
    ],
)
def test_resolve_checkpoint_model_size(
    checkpoint: dict[str, object], expected: str | None
) -> None:
    assert resolve_checkpoint_model_size(checkpoint) == expected


def test_explicit_model_size_supports_legacy_checkpoint() -> None:
    assert resolve_checkpoint_model_size({}, "yolox-s") == "s"


def test_explicit_model_size_must_match_checkpoint() -> None:
    with pytest.raises(ValueError, match="一致しません"):
        resolve_checkpoint_model_size({"model_size": "l"}, "s")


def test_stored_size_must_match_dimensions() -> None:
    with pytest.raises(ValueError, match="矛盾"):
        resolve_checkpoint_model_size(
            {"model_size": "l", "depth": 0.33, "width": 0.50}
        )


def test_partial_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="depthとwidthの両方"):
        resolve_checkpoint_model_size({"depth": 1.0})


def test_unknown_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="対応するモデルサイズがありません"):
        resolve_checkpoint_model_size({"depth": 9.0, "width": 9.0})


@pytest.mark.parametrize("format_version", [0, 2, True, "1"])
def test_unsupported_or_invalid_format_version_is_rejected(
    format_version: object,
) -> None:
    with pytest.raises(ValueError, match="format_version"):
        resolve_checkpoint_model_size({"format_version": format_version})


def _write_real_state_dict_checkpoint(
    path: Path, metadata: dict[str, object]
) -> None:
    model = YoloxModule(
        YoloPafpn(0.33, 0.25, in_channels=[256, 512, 1024]),
        YoloxHead(2, 0.25, in_channels=[256, 512, 1024]),
    )
    torch.save(
        {
            "model": model.state_dict(),
            "names": {0: "class-0", 1: "class-1"},
            "nc": 2,
            "input_size": [64, 64],
            **metadata,
        },
        path,
    )


def test_state_dict_rebuilds_from_checkpoint_metadata(tmp_path: Path) -> None:
    path = tmp_path / "renamed-model.pt"
    _write_real_state_dict_checkpoint(path, checkpoint_model_metadata("nano"))

    wrapper = YOLOX(path, verbose=False)

    assert isinstance(wrapper.model, YoloxModule)
    assert wrapper._model_size == "nano"
    assert wrapper._num_classes == 2


def test_legacy_state_dict_rebuilds_from_explicit_size(tmp_path: Path) -> None:
    path = tmp_path / "sjm.pt"
    _write_real_state_dict_checkpoint(path, {})

    wrapper = YOLOX(path, verbose=False, model_size="nano")

    assert isinstance(wrapper.model, YoloxModule)
    assert wrapper._model_size == "nano"
    assert wrapper._num_classes == 2
