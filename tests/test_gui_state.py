from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ptyolox_garage.config import AppConfig
from ptyolox_garage.gui.app import App
from ptyolox_garage.gui.export_tab import ExportTab
from ptyolox_garage.gui.train_tab import TrainTab


class _Value:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


def test_training_profile_save_preserves_language_and_inference_settings(tmp_path: Path) -> None:
    config = AppConfig(tmp_path / "config.ini")
    config.set("default", "language", "ja")
    config.set("default", "conf", "0.73")
    config.set("default", "iou", "0.61")

    tab = object.__new__(TrainTab)
    tab._config_mgr = config
    tab._profile_var = _Value("default")  # type: ignore[assignment]
    tab._device_var = _Value("cuda:0")  # type: ignore[assignment]
    tab._model_var = _Value("m")  # type: ignore[assignment]
    tab._batch_var = _Value(8)  # type: ignore[assignment]
    tab._imgsz_var = _Value(640)  # type: ignore[assignment]
    tab._workers_var = _Value(2)  # type: ignore[assignment]
    tab._val_split_var = _Value(0.2)  # type: ignore[assignment]

    tab.save_profile()

    saved = config.get("default")
    assert saved.language == "ja"
    assert saved.conf == 0.73
    assert saved.iou == 0.61
    assert saved.device == "cuda:0"
    assert saved.model_size == "m"


def test_language_change_is_rejected_while_training() -> None:
    language_var = _Value("ja")
    app = SimpleNamespace(
        _train_tab=SimpleNamespace(is_busy=lambda: True),
        _export_tab=SimpleNamespace(is_busy=lambda: False),
        _language_var=language_var,
        _configured_language="en",
    )

    with patch("tkinter.messagebox.showwarning") as showwarning:
        App._change_language(app)  # type: ignore[arg-type]

    assert language_var.get() == "en"
    showwarning.assert_called_once()


def test_background_task_state() -> None:
    train_tab = object.__new__(TrainTab)
    train_tab._running = True
    export_tab = object.__new__(ExportTab)
    export_tab._running = False

    assert train_tab.is_busy() is True
    assert export_tab.is_busy() is False
