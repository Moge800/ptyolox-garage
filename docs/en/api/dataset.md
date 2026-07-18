# API Reference — DatasetPreparer (dataset)

The `ptyolox_garage.dataset` module converts Label Studio COCO exports into a YOLOX training directory structure.

---

## `DatasetPreparer` Class

### Constructor

```python
DatasetPreparer(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
)
```

| Parameter | Description |
|-----------|-------------|
| `coco_json_path` | Path to COCO-format JSON file exported from Label Studio |
| `images_dir` | Directory containing image files |
| `output_dir` | Output directory |
| `val_split` | Validation data ratio (0.0–1.0, default 0.2) |
| `seed` | Random split seed (default 42) |

---

### `prepare()`

```python
def prepare(self) -> tuple[dict[int, str], int]
```

Executes dataset preparation.

**Processing steps:**

1. Load COCO JSON and remap category IDs to start from 0
2. Resolve image paths
3. Split into training and validation sets
4. Output annotation JSON and image files for each split

**Returns:**

| Element | Type | Description |
|---------|------|-------------|
| `class_names` | `dict[int, str]` | Class ID → class name mapping |
| `num_classes` | `int` | Number of classes |

**Output directory structure:**

```
output_dir/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg / *.png
└── val/
    └── *.jpg / *.png
```

---

### Internal Methods

| Method | Description |
|--------|-------------|
| `_load_coco()` | Load COCO JSON and remap category IDs to start from 0 |
| `_resolve_image_paths()` | Resolve actual paths from filenames in JSON |
| `_split()` | Split records into training and validation sets |
| `_write_split(split, records)` | Write annotation JSON and copy images |

---

### Error Cases

| Condition | Exception |
|-----------|-----------|
| Fewer than 2 images | `ValueError` |
| No categories defined | `ValueError` |

---

## Usage Example

```python
from ptyolox_garage.dataset import DatasetPreparer

preparer = DatasetPreparer(
    coco_json_path="export/result.json",
    images_dir="export/images",
    output_dir="dataset",
    val_split=0.2,
)

class_names, num_classes = preparer.prepare()
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
# Number of classes: 3
# Class names: {0: "cat", 1: "dog", 2: "bird"}
```

## Integration with Label Studio

1. Complete annotation work in Label Studio
2. **Export** → Select **COCO** format
3. Pass the exported JSON file and image directory to `DatasetPreparer`
4. Run `prepare()` to generate a YOLOX training dataset
