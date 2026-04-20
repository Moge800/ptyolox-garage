# YOLOX Wrapper

A Python toolkit for training, inference, and ONNX export of YOLOX object detection models, powered by [pixeltable-yolox](https://github.com/pixeltable/pixeltable-yolox).  
Provides an ultralytics YOLO-like API and a tkinter-based GUI.

## Features

- **Training** — Train directly from Label Studio COCO exports with staged epoch scheduling
- **Inference** — Run predictions on image files, NumPy arrays, or directories
- **Real-time Inference** — Live object detection via USB camera (GUI)
- **ONNX Export** — Convert `.pt` models to ONNX format
- **Config Profiles** — Manage multiple environment settings with `config.ini`
- **GUI** — Four-tab interface for training, inference, camera, and export

## Requirements

- Python **3.10** or later
- [uv](https://docs.astral.sh/uv/) (recommended)

## Installation

```bash
git clone https://github.com/Moge800/yolox_wrapper.git
cd yolox_wrapper
uv sync
```

## Quick Start

### Code API

```python
from yolox_wrapper import YOLOX

# Training
model = YOLOX("l")
model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0", batch=16)

# Inference
model = YOLOX("best_model.pt")
results = model.predict("image.jpg", conf=0.3)
for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    annotated = r.plot()

# ONNX export
model.export(format="onnx")
```

### GUI

```bash
uv run yolox-gui
```

## data.yaml Format

```yaml
coco_json: /path/to/result.json    # Label Studio COCO export JSON
images_dir: /path/to/images        # Image directory
output_dir: /path/to/output        # Output directory (optional)
val_split: 0.2                     # Validation split ratio (optional)
```

## Model Sizes

| Size | depth | width |
|------|-------|-------|
| `nano` | 0.33 | 0.25 |
| `tiny` | 0.33 | 0.375 |
| `s` | 0.33 | 0.50 |
| `m` | 0.67 | 0.75 |
| `l` | 1.00 | 1.00 |
| `x` | 1.33 | 1.25 |

## Config Profiles

Manage multiple environment settings with `config.ini`:

```ini
[default]
device = cpu
model_size = l
batch_size = 16
imgsz = 640

[factory_pc]
device = cuda:0
model_size = l
batch_size = 16
imgsz = 640
```

## Tests

```bash
uv run pytest
```

## Documentation

See [docs/](docs/en/index.md) for detailed documentation.

- [Installation](docs/en/installation.md)
- [Quick Start](docs/en/quickstart.md)
- [Configuration](docs/en/configuration.md)
- [GUI Guide](docs/en/gui.md)
- API Reference: [YOLOX](docs/en/api/wrapper.md) / [AppConfig](docs/en/api/config.md) / [DatasetPreparer](docs/en/api/dataset.md) / [Trainer](docs/en/api/trainer.md)
- [Development Guide](docs/en/development.md)

## License

[MIT License](LICENSE)
