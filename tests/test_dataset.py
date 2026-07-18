"""Tests for DatasetPreparer."""

import json
from pathlib import Path

import pytest

from ptyolox_garage.dataset import DatasetPreparer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coco_dir(tmp_path: Path) -> Path:
    """Create a minimal Label Studio COCO export in tmp_path."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create five placeholder images.
    import cv2
    import numpy as np
    for i in range(5):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img{i:03d}.jpg"), img)

    # COCO JSON
    coco = {
        "images": [
            {"id": i, "file_name": f"img{i:03d}.jpg", "width": 100, "height": 100}
            for i in range(5)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "segmentation": [],
                "iscrowd": 0,
            }
            for i in range(5)
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
    }
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDatasetPreparer:
    def test_prepare_returns_class_names(self, coco_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "dataset"
        preparer = DatasetPreparer(
            coco_json_path=str(coco_dir / "result.json"),
            images_dir=str(coco_dir / "images"),
            output_dir=str(output),
            val_split=0.2,
        )
        class_names, num_classes = preparer.prepare()

        # Normalize Label Studio category IDs 1 and 2 to zero-based IDs.
        assert num_classes == 2
        assert set(class_names.values()) == {"cat", "dog"}

    def test_prepare_creates_directory_structure(self, coco_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "dataset"
        preparer = DatasetPreparer(
            coco_json_path=str(coco_dir / "result.json"),
            images_dir=str(coco_dir / "images"),
            output_dir=str(output),
        )
        preparer.prepare()

        assert (output / "annotations" / "instances_train.json").exists()
        assert (output / "annotations" / "instances_val.json").exists()
        assert (output / "train").is_dir()
        assert (output / "val").is_dir()

    def test_split_ratio(self, coco_dir: Path, tmp_path: Path) -> None:
        """Split five images into four training and one validation image."""
        output = tmp_path / "dataset"
        preparer = DatasetPreparer(
            coco_json_path=str(coco_dir / "result.json"),
            images_dir=str(coco_dir / "images"),
            output_dir=str(output),
            val_split=0.2,
        )
        preparer.prepare()

        train_json = json.loads((output / "annotations" / "instances_train.json").read_text())
        val_json = json.loads((output / "annotations" / "instances_val.json").read_text())

        assert len(train_json["images"]) + len(val_json["images"]) == 5
        assert len(val_json["images"]) >= 1

    def test_annotation_category_id_remapped(self, coco_dir: Path, tmp_path: Path) -> None:
        """Normalize category_id values to zero-based IDs."""
        output = tmp_path / "dataset"
        preparer = DatasetPreparer(
            coco_json_path=str(coco_dir / "result.json"),
            images_dir=str(coco_dir / "images"),
            output_dir=str(output),
        )
        preparer.prepare()

        train_json = json.loads((output / "annotations" / "instances_train.json").read_text())
        cat_ids = {ann["category_id"] for ann in train_json["annotations"]}
        # Only zero and one are valid after zero-based normalization.
        assert cat_ids.issubset({0, 1})

    def test_too_few_images_raises(self, tmp_path: Path) -> None:
        """Raise ValueError when no more than one image is available."""
        import cv2
        import numpy as np

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img.jpg"), img)

        coco = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [],
            "categories": [{"id": 1, "name": "cat"}],
        }
        (tmp_path / "result.json").write_text(json.dumps(coco), encoding="utf-8")

        preparer = DatasetPreparer(
            coco_json_path=str(tmp_path / "result.json"),
            images_dir=str(images_dir),
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises(ValueError, match="最低 2 枚"):
            preparer.prepare()

    def test_missing_categories_raises(self, tmp_path: Path) -> None:
        coco = {"images": [], "annotations": [], "categories": []}
        (tmp_path / "result.json").write_text(json.dumps(coco), encoding="utf-8")

        preparer = DatasetPreparer(
            coco_json_path=str(tmp_path / "result.json"),
            images_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises(ValueError, match="categories"):
            preparer.prepare()
