# API Reference — YOLOX (wrapper)

The `ptyolox_garage.wrapper` module provides the main class that integrates YOLOX model loading, inference, training, and export.

---

## Model Sizes

| Size | depth | width |
|------|-------|-------|
| `nano` | 0.33 | 0.25 |
| `tiny` | 0.33 | 0.375 |
| `s` | 0.33 | 0.50 |
| `m` | 0.67 | 0.75 |
| `l` | 1.00 | 1.00 |
| `x` | 1.33 | 1.25 |

> Size strings accept `"l"`, `"yolox_l"`, or `"yolox-l"` formats.

---

## `YOLOX` Class

### Constructor

```python
YOLOX(model: str, verbose: bool = True)
```

| Parameter | Description |
|-----------|-------------|
| `model` | Model size string (e.g., `"l"`) or `.pt` file path |
| `verbose` | Enable verbose logging |

**Behavior:**
- Size string → Builds an untrained model architecture
- `.pt` path → Restores model and settings from checkpoint

---

### `train()`

```python
def train(
    self,
    data: str,
    epochs: int | list[int] = 300,
    batch: int = 16,
    device: str = "cpu",
    imgsz: int = 640,
    workers: int = 4,
    val_split: float | None = None,
    pretrained_weights: str | None = None,
    on_log: Callable[[str], None] | None = None,
    on_stage_done: Callable[[int, int, str], None] | None = None,
    **kwargs: Any,
) -> "YOLOX"
```

Train a YOLOX model. Executes training in stages according to the epoch schedule.

| Parameter | Description |
|-----------|-------------|
| `data` | Path to `data.yaml` |
| `epochs` | Total epochs or staged schedule (e.g., `[100, 200, 300]`) |
| `batch` | Batch size |
| `device` | Device to use (`"cpu"` / `"cuda:0"`) |
| `imgsz` | Input image size |
| `workers` | DataLoader worker count |
| `val_split` | Validation data split ratio |
| `pretrained_weights` | Path to pretrained weights |
| `on_log` | Log output callback `(text: str) -> None` |
| `on_stage_done` | Stage completion callback `(stage_idx, epoch, ckpt_path) -> None` |

**Returns:** `YOLOX` (for method chaining)

---

### `predict()`

```python
def predict(
    self,
    source: str | Path | np.ndarray | list,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    verbose: bool = False,
    save: bool = False,
    **kwargs: Any,
) -> list[YOLOXResult]
```

Run inference on images.

| Parameter | Description |
|-----------|-------------|
| `source` | Image path, NumPy array, or list thereof |
| `conf` | Confidence threshold |
| `iou` | NMS IoU threshold |
| `device` | Device to use |
| `verbose` | Verbose logging |
| `save` | Save result images |

**Returns:** `list[YOLOXResult]` (one per image)

---

### `export()`

```python
def export(self, format: str = "onnx", **kwargs: Any) -> str
```

Export the model to the specified format.

| Parameter | Description |
|-----------|-------------|
| `format` | Output format (currently only `"onnx"` is supported) |

**Returns:** Output file path

---

### `fuse()`

```python
def fuse(self) -> "YOLOX"
```

Fuse BatchNorm layers into convolutions for faster inference.

**Returns:** `YOLOX` (for method chaining)

---

### `save()`

```python
def save(self, path: str) -> None
```

Save the model as a `.pt` file with metadata.

---

## `YOLOXResult` Class

Container for inference results. Provides an interface compatible with ultralytics `Results`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `boxes` | `YOLOXBoxes` | Collection of detection boxes |
| `names` | `dict[int, str]` | Class ID → class name mapping |
| `orig_shape` | `tuple[int, int]` | Original image size `(H, W)` |
| `orig_img` | `np.ndarray \| None` | Original image (BGR) |

### `plot()`

```python
def plot(self, orig_img: np.ndarray | None = None) -> np.ndarray
```

Returns an image with detection results drawn as bounding boxes.

---

## `YOLOXBoxes` Class

Collection of detection boxes.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `xyxy` | `torch.Tensor` | Coordinates `[N, 4]` (x1, y1, x2, y2) |
| `conf` | `torch.Tensor` | Confidence scores `[N]` |
| `cls` | `torch.Tensor` | Class IDs `[N]` |

### Methods

- `__len__() -> int` — Number of boxes
- `__iter__()` — Iterate over individual boxes (`_YOLOXBox`)

---

## Utility Functions

### `_letterbox()`

```python
def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    fill_value: int = 114,
) -> tuple[np.ndarray, float]
```

Letterbox resize while preserving aspect ratio.

### `_postprocess()`

```python
def _postprocess(
    outputs: torch.Tensor,
    ratio: float,
    orig_h: int,
    orig_w: int,
    conf_thre: float,
    iou_thre: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Decode YOLOX output and apply NMS. Returns `(boxes, scores, class_ids)`.
