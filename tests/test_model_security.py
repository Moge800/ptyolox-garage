from types import SimpleNamespace
from unittest.mock import patch

from ptyolox_garage.gui.camera_tab import CameraTab
from ptyolox_garage.gui.export_tab import ExportTab
from ptyolox_garage.gui.infer_tab import InferTab
from ptyolox_garage.gui.model_security import confirm_trusted_model_file


class _Value:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


def test_trusted_model_confirmation_uses_a_warning_dialog() -> None:
    with patch("ptyolox_garage.gui.model_security.messagebox.askokcancel", return_value=True) as confirm:
        assert confirm_trusted_model_file("model.pt") is True

    confirm.assert_called_once()
    assert confirm.call_args.kwargs["icon"] == "warning"
    assert confirm.call_args.kwargs["default"] == "cancel"


def test_inference_does_not_load_a_model_when_confirmation_is_cancelled() -> None:
    tab = object.__new__(InferTab)
    tab._model_var = _Value("model.pt")  # type: ignore[assignment]
    tab._source_var = _Value("image.jpg")  # type: ignore[assignment]
    tab._loaded_model_path = ""

    with patch("ptyolox_garage.gui.infer_tab.confirm_trusted_model_file", return_value=False), patch(
        "ptyolox_garage.gui.infer_tab.YOLOX"
    ) as yolox:
        tab._run()

    yolox.assert_not_called()


def test_camera_does_not_load_a_model_when_confirmation_is_cancelled() -> None:
    tab = object.__new__(CameraTab)
    tab._model_var = _Value("model.pt")  # type: ignore[assignment]

    with patch("ptyolox_garage.gui.camera_tab.confirm_trusted_model_file", return_value=False), patch(
        "ptyolox_garage.gui.camera_tab.YOLOX"
    ) as yolox:
        tab._start()

    yolox.assert_not_called()


def test_export_does_not_start_when_confirmation_is_cancelled() -> None:
    tab = object.__new__(ExportTab)
    tab._model_var = _Value("model.pt")  # type: ignore[assignment]
    tab._output_var = _Value("")  # type: ignore[assignment]
    tab._export_btn = SimpleNamespace(config=lambda **kwargs: None)
    tab._status_label = SimpleNamespace(config=lambda **kwargs: None)

    with patch("ptyolox_garage.gui.export_tab.confirm_trusted_model_file", return_value=False), patch(
        "ptyolox_garage.gui.export_tab.threading.Thread"
    ) as thread:
        tab._export()

    thread.assert_not_called()
