"""Tests for wrapper behavior that does not require the yolox package."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ptyolox_garage.wrapper import (
    YOLOX,
    YOLOXBoxes,
    YOLOXResult,
    _letterbox,
    _nms_fallback,
    _normalize_model_size,
    _YOLOXBox,
)

# ---------------------------------------------------------------------------
# _normalize_model_size
# ---------------------------------------------------------------------------


class TestNormalizeModelSize:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("l", "l"),
            ("yolox_l", "l"),
            ("yolox-l", "l"),
            ("YOLOX_S", "s"),
            ("nano", "nano"),
            ("x", "x"),
        ],
    )
    def test_valid(self, inp: str, expected: str) -> None:
        assert _normalize_model_size(inp) == expected

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            _normalize_model_size("unknown_model")


# ---------------------------------------------------------------------------
# _letterbox
# ---------------------------------------------------------------------------


class TestLetterbox:
    def test_output_shape(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result, ratio = _letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)

    def test_ratio_correct(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, ratio = _letterbox(img, (640, 640))
        assert ratio == pytest.approx(1.0)

    def test_smaller_image_upscaled(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result, ratio = _letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        assert ratio == pytest.approx(6.4)

    def test_fill_value(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # Landscape image
        result, _ = _letterbox(img, (640, 640), fill_value=114)
        # Top-left placement leaves padding in the bottom-right corner.
        assert result[-1, -1, 0] == 114


# ---------------------------------------------------------------------------
# _nms_fallback
# ---------------------------------------------------------------------------


class TestNmsFallback:
    def test_empty_input(self) -> None:
        boxes = torch.zeros((0, 4))
        scores = torch.zeros(0)
        keep = _nms_fallback(boxes, scores, 0.5)
        assert keep.numel() == 0

    def test_single_box(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9])
        keep = _nms_fallback(boxes, scores, 0.5)
        assert keep.tolist() == [0]

    def test_overlapping_boxes_suppressed(self) -> None:
        # Two nearly overlapping boxes leave only the higher-scoring box.
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.5, 0.5, 10.5, 10.5],
            ]
        )
        scores = torch.tensor([0.9, 0.8])
        keep = _nms_fallback(boxes, scores, 0.5)
        assert len(keep) == 1
        assert keep[0].item() == 0

    def test_non_overlapping_boxes_kept(self) -> None:
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8])
        keep = _nms_fallback(boxes, scores, 0.5)
        assert len(keep) == 2


# ---------------------------------------------------------------------------
# YOLOXBoxes / _YOLOXBox
# ---------------------------------------------------------------------------


class TestYOLOXBoxes:
    def _make_boxes(self, n: int) -> YOLOXBoxes:
        return YOLOXBoxes(
            xyxy=torch.rand(n, 4),
            conf=torch.rand(n),
            cls=torch.zeros(n),
        )

    def test_len(self) -> None:
        assert len(self._make_boxes(3)) == 3

    def test_iter_yields_box_objects(self) -> None:
        boxes = self._make_boxes(3)
        items = list(boxes)
        assert len(items) == 3
        assert all(isinstance(b, _YOLOXBox) for b in items)

    def test_box_shapes(self) -> None:
        boxes = self._make_boxes(2)
        for box in boxes:
            assert box.xyxy.shape == (1, 4)
            assert box.conf.shape == (1,)
            assert box.cls.shape == (1,)

    def test_empty_boxes(self) -> None:
        boxes = self._make_boxes(0)
        assert len(boxes) == 0
        assert list(boxes) == []


# ---------------------------------------------------------------------------
# YOLOXResult.plot()
# ---------------------------------------------------------------------------


class TestYOLOXResultPlot:
    def test_plot_returns_ndarray(self) -> None:
        boxes = YOLOXBoxes(
            xyxy=torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
        )
        orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = YOLOXResult(
            boxes=boxes,
            names={0: "cat"},
            orig_shape=(100, 100),
            orig_img=orig_img,
        )
        annotated = result.plot()
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (100, 100, 3)

    def test_plot_no_boxes(self) -> None:
        boxes = YOLOXBoxes(
            xyxy=torch.zeros((0, 4)),
            conf=torch.zeros(0),
            cls=torch.zeros(0),
        )
        result = YOLOXResult(
            boxes=boxes,
            names={},
            orig_shape=(100, 100),
        )
        annotated = result.plot()
        assert annotated.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# YOLOX.__init__ with model-size strings
# ---------------------------------------------------------------------------


class TestYOLOXInit:
    def test_init_with_model_size(self) -> None:
        model = YOLOX("l", verbose=False)
        assert model._model_size == "l"
        assert model.model is None

    def test_init_with_yolox_prefix(self) -> None:
        model = YOLOX("yolox_s", verbose=False)
        assert model._model_size == "s"

    def test_init_with_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            YOLOX("invalid_model", verbose=False)

    def test_init_with_nonexistent_pt_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            YOLOX("nonexistent.pt", verbose=False)

    def test_predict_without_model_raises(self) -> None:
        model = YOLOX("l", verbose=False)
        with pytest.raises(RuntimeError, match="モデルが読み込まれていません"):
            model.predict("dummy.jpg")


class _DummyModel(nn.Module):
    """Module-level dummy model that torch.save can pickle."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.zeros(b, 100, 6)  # [batch, anchors, 5+nc]


# ---------------------------------------------------------------------------
# YOLOX.save() and load() with a mock model
# ---------------------------------------------------------------------------


class TestYOLOXSaveLoad:
    def _make_mock_pt(self, tmp_path: Path) -> str:
        """Create a minimal .pt file."""
        model = _DummyModel()
        path = str(tmp_path / "dummy.pt")
        torch.save(
            {"model": model, "names": {0: "cat"}, "nc": 1, "input_size": [64, 64]},
            path,
        )
        return path

    def test_load_and_save(self, tmp_path: Path) -> None:
        pt_path = self._make_mock_pt(tmp_path)
        model = YOLOX(pt_path, verbose=False)
        assert model.model is not None
        assert model._class_names == {0: "cat"}
        assert model._num_classes == 1
        assert model._input_size == (64, 64)

        save_path = str(tmp_path / "saved.pt")
        model.save(save_path)
        assert Path(save_path).exists()

    def test_reload_saved(self, tmp_path: Path) -> None:
        pt_path = self._make_mock_pt(tmp_path)
        model = YOLOX(pt_path, verbose=False)
        save_path = str(tmp_path / "saved.pt")
        model.save(save_path)

        model2 = YOLOX(save_path, verbose=False)
        assert model2._class_names == {0: "cat"}
        assert model2._num_classes == 1


# ---------------------------------------------------------------------------
# GUI entry point
# ---------------------------------------------------------------------------


class TestGUIEntryPoint:
    def test_gui_main_is_importable(self) -> None:
        """Import the ptyolox-garage script entry point."""
        from ptyolox_garage_bootstrap import main as gui_main

        assert callable(gui_main)

    def test_bootstrap_import_is_lightweight(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import ptyolox_garage_bootstrap; "
                "assert 'ptyolox_garage' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

    def test_bootstrap_logs_before_delegating(self, capsys, monkeypatch) -> None:
        import ptyolox_garage_bootstrap as bootstrap
        from ptyolox_garage.gui import app

        delegated = []
        monkeypatch.setattr(app, "main", lambda *, started_at: delegated.append(started_at))
        monkeypatch.setattr(bootstrap.time, "perf_counter", lambda: 12.5)

        bootstrap.main()

        assert delegated == [12.5]
        assert capsys.readouterr().out.splitlines() == [
            "[PTYOLOX Garage] Starting...",
            "[PTYOLOX Garage] Loading GUI components...",
        ]

    def test_bootstrap_reports_failed_stage(self, capsys, monkeypatch) -> None:
        import ptyolox_garage_bootstrap as bootstrap
        from ptyolox_garage.gui import app

        def fail(*, started_at: float) -> None:
            bootstrap.startup_log(
                "Loading ML dependencies...", stage="loading ML dependencies"
            )
            raise RuntimeError("missing dependency")

        monkeypatch.setattr(app, "main", fail)

        with pytest.raises(RuntimeError, match="missing dependency"):
            bootstrap.main()

        captured = capsys.readouterr()
        assert "Loading ML dependencies..." in captured.out
        assert captured.err.strip() == (
            "[PTYOLOX Garage] Failed while loading ML dependencies: missing dependency"
        )

    def test_gui_main_reports_ready_time(self, capsys, monkeypatch) -> None:
        from ptyolox_garage.gui import app

        mainloop_calls = []

        class FakeApp:
            def mainloop(self) -> None:
                mainloop_calls.append(True)

        monkeypatch.setattr(app, "App", FakeApp)
        monkeypatch.setattr(app.time, "perf_counter", lambda: 14.2)

        app.main(started_at=10.0)

        assert mainloop_calls == [True]
        assert capsys.readouterr().out.strip() == "[PTYOLOX Garage] Ready (4.2s)"

    def test_root_main_delegates_to_bootstrap(self, monkeypatch) -> None:
        """Have main.py call the lightweight bootstrap entry point."""
        import importlib

        mod = importlib.import_module("main")
        import ptyolox_garage_bootstrap as bootstrap

        called = []
        monkeypatch.setattr(bootstrap, "main", lambda: called.append(True))
        mod.main()

        assert called == [True]
