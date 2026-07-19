import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ptyolox_garage._trainer import TrainingStopped
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


def test_profile_is_restored_after_language_change() -> None:
    profile_var = _Value("default")
    loaded_profiles: list[str] = []
    app = SimpleNamespace(
        config_mgr=SimpleNamespace(profiles=lambda: ["default", "factory_pc"]),
        _profile_var=profile_var,
        _on_profile_changed=lambda: loaded_profiles.append(str(profile_var.get())),
    )

    App._restore_profile(app, "factory_pc")  # type: ignore[arg-type]

    assert profile_var.get() == "factory_pc"
    assert loaded_profiles == ["factory_pc"]


def test_missing_profile_falls_back_to_default() -> None:
    profile_var = _Value("factory_pc")
    loaded_profiles: list[str] = []
    app = SimpleNamespace(
        config_mgr=SimpleNamespace(profiles=lambda: ["default"]),
        _profile_var=profile_var,
        _on_profile_changed=lambda: loaded_profiles.append(str(profile_var.get())),
    )

    App._restore_profile(app, "factory_pc")  # type: ignore[arg-type]

    assert profile_var.get() == "default"
    assert loaded_profiles == ["default"]


def test_background_task_state() -> None:
    train_tab = object.__new__(TrainTab)
    train_tab._running = True
    export_tab = object.__new__(ExportTab)
    export_tab._running = False

    assert train_tab.is_busy() is True
    assert export_tab.is_busy() is False


def test_training_worker_passes_stop_event_and_handles_cancellation(monkeypatch) -> None:
    tab = object.__new__(TrainTab)
    tab._model_var = _Value("nano")  # type: ignore[assignment]
    tab._batch_var = _Value(16)  # type: ignore[assignment]
    tab._device_var = _Value("cpu")  # type: ignore[assignment]
    tab._imgsz_var = _Value(640)  # type: ignore[assignment]
    tab._workers_var = _Value(2)  # type: ignore[assignment]
    tab._val_split_var = _Value(0.2)  # type: ignore[assignment]
    tab._stop_event = threading.Event()
    tab._log_queue = queue.Queue()
    tab._train_succeeded = True
    captured: dict[str, object] = {}

    class FakeYOLOX:
        def __init__(self, model: str, verbose: bool) -> None:
            pass

        def train(self, **kwargs: object) -> None:
            captured.update(kwargs)
            raise TrainingStopped("stopped")

    monkeypatch.setattr("ptyolox_garage.gui.train_tab.YOLOX", FakeYOLOX)

    tab._run_training("data.yaml", [10, 20])

    assert captured["stop_event"] is tab._stop_event
    assert tab._train_succeeded is False
    assert tab._log_queue.get_nowait() is not None
    assert tab._log_queue.get_nowait() is None


def test_stop_request_disables_button_and_explains_stage_boundary() -> None:
    tab = object.__new__(TrainTab)
    tab._stop_event = threading.Event()
    button_updates: list[dict[str, str]] = []
    logs: list[str] = []
    tab._stop_btn = SimpleNamespace(config=lambda **kwargs: button_updates.append(kwargs))
    tab._append_log = logs.append

    tab._stop()

    assert tab._stop_event.is_set()
    assert button_updates == [{"state": "disabled"}]
    assert len(logs) == 1
    assert "current stage" in logs[0] or "現在のステージ" in logs[0]
