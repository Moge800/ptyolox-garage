"""Tests for model serialization device helpers."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ptyolox_garage._model_io import (
    assert_all_on_cpu,
    model_device_label,
    model_devices,
    temporary_model_device,
)
from ptyolox_garage._trainer import _YOLOXTrainer


class _ModelWithBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("running_value", torch.tensor(2.0))


def test_model_device_helpers_include_parameters_and_buffers() -> None:
    model = _ModelWithBuffer()

    assert model_devices(model) == ("cpu",)
    assert model_device_label(model) == "cpu"
    assert_all_on_cpu(model)


def test_cpu_assertion_reports_non_cpu_buffer() -> None:
    model = _ModelWithBuffer()
    model.register_buffer("stray", torch.empty(1, device="meta"))

    with pytest.raises(RuntimeError, match="stray"):
        assert_all_on_cpu(model)


def test_temporary_device_rejects_mixed_device_model() -> None:
    model = _ModelWithBuffer()
    model.register_buffer("stray", torch.empty(1, device="meta"))

    with pytest.raises(RuntimeError, match="複数デバイス"):
        with temporary_model_device(model, "cpu"):
            pass


def test_package_model_honors_to_cpu_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _YOLOXTrainer(
        model_size="nano",
        num_classes=1,
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        device="cuda:1",
    )

    class TrackingModel(_ModelWithBuffer):
        def __init__(self) -> None:
            super().__init__()
            self.requested_devices: list[str] = []

        def to(self, device: object) -> "TrackingModel":
            self.requested_devices.append(str(device))
            return self

    model = TrackingModel()
    saved: dict[str, object] = {}
    monkeypatch.setattr(trainer, "_build_model", lambda *args: model)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {}})

    def capture_save(payload: object, path: object) -> None:
        saved["payload"] = payload
        saved["path"] = path

    monkeypatch.setattr(torch, "save", capture_save)
    output_path = tmp_path / "model.pt"

    result = trainer.package_model(
        {0: "part"},
        checkpoint_path=str(tmp_path / "checkpoint.pth"),
        output_model_path=str(output_path),
        to_cpu=False,
    )

    assert result == str(output_path)
    assert model.requested_devices == ["cuda:1"]
    assert saved["path"] == str(output_path)
    payload = saved["payload"]
    assert isinstance(payload, dict)
    assert payload["format_version"] == 1
    assert payload["model_size"] == "nano"
    assert payload["depth"] == pytest.approx(0.33)
    assert payload["width"] == pytest.approx(0.25)
