# API Reference — _YOLOXTrainer (trainer)

The `ptyolox_garage._trainer` module is an internal module that manages YOLOX training execution.

> This module is a private API. It is typically used indirectly through `YOLOX.train()`.

---

## `_YOLOXTrainer` Class

### Constructor

```python
_YOLOXTrainer(
    model_size: str,
    num_classes: int,
    dataset_dir: str,
    output_dir: str,
    input_size: tuple[int, int] = (640, 640),
    batch_size: int = 16,
    device: str = "cpu",
    num_workers: int = 4,
    basic_lr_per_img: float = 0.01 / 64,
    pretrained_weights: str | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `model_size` | Model size (`"nano"` to `"x"`) |
| `num_classes` | Number of detection classes |
| `dataset_dir` | Dataset directory output by `DatasetPreparer` |
| `output_dir` | Output directory for checkpoints, etc. |
| `input_size` | Input image size `(H, W)` |
| `batch_size` | Training batch size |
| `device` | Device to use |
| `num_workers` | DataLoader worker count |
| `basic_lr_per_img` | Base learning rate per image |
| `pretrained_weights` | Path to pretrained weights |

---

### `train_sequential()`

```python
def train_sequential(
    self,
    epoch_schedule: list[int],
    on_log: Callable[[str], None] | None = None,
    on_stage_done: Callable[[int, int, str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> str
```

Executes staged training according to the epoch schedule.

| Parameter | Description |
|-----------|-------------|
| `epoch_schedule` | List of staged epoch counts (e.g., `[100, 200, 300]`) |
| `on_log` | Log callback `(text) -> None` |
| `on_stage_done` | Stage completion callback `(stage_idx, epoch, ckpt_path) -> None` |
| `stop_event` | Interruption signal (`threading.Event`) |

**Returns:** Path to the final checkpoint

**How staged training works:**

```
epoch_schedule = [100, 200, 300]

Stage 0: Epochs 1–100   (resume=False)
Stage 1: Epochs 101–200 (resume=True)
Stage 2: Epochs 201–300 (resume=True)
```

Checkpoints are saved between stages, and each subsequent stage resumes from the previous one.

---

### `package_model()`

```python
def package_model(
    self,
    class_names: dict[int, str],
    checkpoint_path: str | None = None,
    output_model_path: str | None = None,
) -> str
```

Packages a training checkpoint into a YOLOX-compatible `.pt` file.

| Parameter | Description |
|-----------|-------------|
| `class_names` | Class ID → class name mapping |
| `checkpoint_path` | Source checkpoint. Auto-searches for best if `None` |
| `output_model_path` | Output path. Auto-generated if `None` |

**Returns:** Output file path

---

## `_build_config()` Function

```python
def _build_config(
    name: str,
    num_classes: int,
    data_dir: str,
    input_size: tuple[int, int],
    depth: float,
    width: float,
    max_epoch: int,
    num_workers: int,
    basic_lr_per_img: float,
    train_ann: str,
    val_ann: str,
    output_dir: str,
)
```

Dynamically generates a `YoloxConfig` subclass and returns a `YoloxConfig` instance with training settings (epochs, data paths, etc.) embedded.

---

## Log Management

### `_LogRedirector`

A context manager that redirects stdout/stderr to a callback function.  
Used to forward training log output to the GUI text widget.

```python
with _LogRedirector(callback=on_log):
    # print output within this block is forwarded to on_log
    trainer.train(...)
```

### `_CallbackStream`

A `TextIOBase` subclass that forwards `write()` calls to a callback function.
