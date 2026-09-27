"""Tests for training data input forms."""

from pathlib import Path

import pytest

import ptyolox_garage.wrapper as wrapper_module
from ptyolox_garage._trainer import TrainingStopped
from ptyolox_garage.gui.train_tab import TrainTab
from ptyolox_garage.wrapper import YOLOX


class _Value:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def _make_export(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    annotations = root / "result.json"
    annotations.write_text("{}", encoding="utf-8")
    images = root / "images"
    images.mkdir()
    return annotations, images


def test_coco_directory_resolves_label_studio_layout(tmp_path: Path) -> None:
    export = tmp_path / "export"
    annotations, images = _make_export(export)

    cfg = YOLOX._load_data_config(str(export))

    assert cfg["coco_json"] == str(annotations)
    assert cfg["images_dir"] == str(images)


def test_coco_json_requires_explicit_image_directory(tmp_path: Path) -> None:
    annotations, images = _make_export(tmp_path / "export")

    with pytest.raises(ValueError, match="images_dir"):
        YOLOX._load_data_config(str(annotations))

    cfg = YOLOX._load_data_config(str(annotations), images_dir=str(images))
    assert cfg["coco_json"] == str(annotations)
    assert cfg["images_dir"] == str(images)


def test_yaml_remains_supported_with_relative_paths(tmp_path: Path) -> None:
    export = tmp_path / "export"
    annotations, images = _make_export(export)
    config = tmp_path / "data.yaml"
    config.write_text(
        "coco_json: export/result.json\nimages_dir: export/images\n"
        "output_dir: prepared\nval_split: 0.3\n",
        encoding="utf-8",
    )

    cfg = YOLOX._load_data_config(str(config))

    assert cfg == {
        "coco_json": str(annotations),
        "images_dir": str(images),
        "output_dir": str(tmp_path / "prepared"),
        "val_split": 0.3,
    }


@pytest.mark.parametrize("missing", ["result.json", "images"])
def test_incomplete_coco_directory_fails_early(tmp_path: Path, missing: str) -> None:
    export = tmp_path / "export"
    _make_export(export)
    if missing == "result.json":
        (export / missing).unlink()
    else:
        (export / missing).rmdir()

    with pytest.raises(FileNotFoundError, match="COCO JSON|画像ディレクトリ"):
        YOLOX._load_data_config(str(export))


def test_training_tab_passes_images_only_for_json() -> None:
    tab = object.__new__(TrainTab)
    tab._data_modes = {"yaml": "YAML", "json": "COCO JSON", "directory": "COCO directory"}
    tab._data_mode_var = _Value("COCO JSON")  # type: ignore[assignment]
    tab._data_var = _Value("export/result.json")  # type: ignore[assignment]
    tab._images_var = _Value("export/images")  # type: ignore[assignment]
    tab._output_var = _Value("output")  # type: ignore[assignment]

    assert tab._training_data_options() == ("export/result.json", "export/images", "output")

    tab._data_mode_var = _Value("COCO directory")  # type: ignore[assignment]
    tab._data_var = _Value("export")  # type: ignore[assignment]
    assert tab._training_data_options() == ("export", None, "output")

    tab._data_mode_var = _Value("YAML")  # type: ignore[assignment]
    tab._data_var = _Value("data.yaml")  # type: ignore[assignment]
    assert tab._training_data_options() == ("data.yaml", None, "output")


def test_training_tab_rejects_json_without_images() -> None:
    tab = object.__new__(TrainTab)
    tab._data_modes = {"json": "COCO JSON"}
    tab._data_mode_var = _Value("COCO JSON")  # type: ignore[assignment]
    tab._data_var = _Value("export/result.json")  # type: ignore[assignment]
    tab._images_var = _Value("")  # type: ignore[assignment]
    tab._output_var = _Value("")  # type: ignore[assignment]

    with pytest.raises(ValueError, match="画像ディレクトリ|image directory"):
        tab._training_data_options()


@pytest.mark.parametrize("form", ["json", "directory"])
def test_train_passes_coco_paths_and_output_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, form: str
) -> None:
    annotations, images = _make_export(tmp_path / "export")
    captured: dict[str, object] = {}

    class FakePreparer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def prepare(self) -> tuple[dict[int, str], int]:
            return {0: "part"}, 1

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def train_sequential(self, **kwargs: object) -> None:
            raise TrainingStopped("stop before training")

    monkeypatch.setattr(wrapper_module, "DatasetPreparer", FakePreparer)
    monkeypatch.setattr(wrapper_module, "_YOLOXTrainer", FakeTrainer)
    output = tmp_path / "custom_output"
    data = annotations if form == "json" else annotations.parent

    with pytest.raises(TrainingStopped, match="stop before training"):
        YOLOX("nano", verbose=False).train(
            data=str(data),
            images_dir=str(images) if form == "json" else None,
            output_dir=str(output),
        )

    assert captured["coco_json_path"] == str(annotations)
    assert captured["images_dir"] == str(images)
    assert captured["output_dir"] == str(output / "dataset")
    assert (output / "class_names.json").exists()
