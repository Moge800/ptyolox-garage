"""ONNX export contract and runtime compatibility tests."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from google.protobuf.message import DecodeError
from yolox.models import YoloPafpn, YoloxHead, YoloxModule

from ptyolox_garage.wrapper import YOLOX


def _make_real_yolox_wrapper() -> YOLOX:
    torch.manual_seed(7)
    backbone = YoloPafpn(0.33, 0.25, in_channels=[256, 512, 1024])
    head = YoloxHead(2, 0.25, in_channels=[256, 512, 1024])

    wrapper = YOLOX("nano", verbose=False)
    wrapper.model = YoloxModule(backbone, head)
    wrapper._num_classes = 2
    wrapper._class_names = {0: "class-0", 1: "class-1"}
    wrapper._input_size = (64, 64)
    wrapper._current_device = "cpu"
    return wrapper


def _pytorch_output(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(input_tensor).cpu().numpy()
    finally:
        model.train(was_training)


def test_real_yolox_export_matches_onnx_runtime(tmp_path: Path) -> None:
    wrapper = _make_real_yolox_wrapper()
    assert wrapper.model is not None
    wrapper.model.train()
    input_tensor = torch.linspace(
        0.0,
        255.0,
        steps=2 * 3 * 64 * 64,
        dtype=torch.float32,
    ).reshape(2, 3, 64, 64)
    expected = _pytorch_output(wrapper.model, input_tensor)
    output_path = tmp_path / "model.onnx"

    result = wrapper.export(format="onnx", output_path=str(output_path))

    session = ort.InferenceSession(result, providers=["CPUExecutionProvider"])
    actual = session.run(["output"], {"images": input_tensor.numpy()})[0]

    assert wrapper.model.training is True
    assert wrapper._current_device == "cpu"
    assert actual.shape == expected.shape
    assert actual.shape[0] == 2
    assert actual.shape[1] > 1
    assert actual.shape[2] == 7
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    exported = onnx.load(result)
    assert exported.graph.input[0].name == "images"
    assert exported.graph.output[0].name == "output"
    assert exported.opset_import[0].version == 11

    input_dimensions = exported.graph.input[0].type.tensor_type.shape.dim
    assert input_dimensions[0].dim_param == "batch"
    assert [dimension.dim_value for dimension in input_dimensions[1:]] == [3, 64, 64]


class _FailureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("offset", torch.tensor(0.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight + self.offset


def test_export_failure_restores_state_and_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = YOLOX("nano", verbose=False)
    wrapper.model = _FailureModel()
    wrapper.model.train()
    wrapper._input_size = (64, 64)
    wrapper._current_device = "cpu"
    output_path = tmp_path / "existing.onnx"
    output_path.write_bytes(b"existing model")

    def fail_export(*args: object, **kwargs: object) -> None:
        raise RuntimeError("export failed")

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(RuntimeError, match="export failed"):
        wrapper.export(output_path=str(output_path))

    assert output_path.read_bytes() == b"existing model"
    assert wrapper.model.training is True
    assert wrapper._current_device == "cpu"
    assert not list(tmp_path.glob("*.tmp.onnx"))


def test_invalid_export_does_not_replace_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = YOLOX("nano", verbose=False)
    wrapper.model = _FailureModel()
    wrapper._input_size = (64, 64)
    output_path = tmp_path / "existing.onnx"
    output_path.write_bytes(b"existing model")

    def write_invalid_model(
        model: object,
        args: object,
        path: str,
        **kwargs: object,
    ) -> None:
        Path(path).write_bytes(b"not an ONNX graph")

    monkeypatch.setattr(torch.onnx, "export", write_invalid_model)

    with pytest.raises(DecodeError):
        wrapper.export(output_path=str(output_path))

    assert output_path.read_bytes() == b"existing model"
    assert not list(tmp_path.glob("*.tmp.onnx"))
