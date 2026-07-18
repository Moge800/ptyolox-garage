# PTYOLOX Garage

[日本語](https://github.com/Moge800/ptyolox-garage/blob/main/README_JP.md)

PTYOLOX Garage is a practical desktop and Python toolkit for training, testing, and exporting object-detection models with [Pixeltable YOLOX](https://github.com/pixeltable/pixeltable-yolox). It provides an Ultralytics-style API and a bilingual tkinter GUI for repeatable local and offline workflows.

## Features

- Prepare Label Studio COCO exports for training
- Train YOLOX nano, tiny, s, m, l, and x models with staged epoch schedules
- Run inference on images, NumPy arrays, directories, and USB cameras
- Export trained models to ONNX
- Switch the GUI between English and Japanese
- Store reusable CPU/GPU configuration profiles

## Requirements

- Python 3.10–3.13
- Windows 11 is the primary supported platform
- NVIDIA CUDA GPU is recommended for training; CPU inference is supported

On Linux, tkinter may need to be installed through the operating-system package manager.

## Installation

From PyPI:

```bash
pip install ptyolox-garage
```

For development with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Moge800/ptyolox-garage.git
cd ptyolox-garage
uv sync --group dev
```

PyTorch builds are hardware-specific. Install the appropriate PyTorch build from the [official selector](https://pytorch.org/get-started/locally/) when CUDA support is required.

## GUI

```bash
ptyolox-garage
```

![PTYOLOX Garage training screen](https://raw.githubusercontent.com/Moge800/ptyolox-garage/main/docs/assets/ptyolox-garage-gui.png)

The GUI contains four work areas: training, image inference, live camera inference, and ONNX export. The initial language follows the operating-system locale and can be changed from the Language menu.

Configuration is stored in the operating system's user configuration directory. On Windows, the default location is `%APPDATA%\ptyolox-garage\config.ini`.

## Python API

```python
from ptyolox_garage import YOLOX

# Train from a Label Studio COCO export.
model = YOLOX("l")
model.train(
    data="data.yaml",
    epochs=[100, 200, 300],
    device="cuda:0",
    batch=16,
)

# Run inference.
model = YOLOX("best_model.pt")
results = model.predict("image.jpg", conf=0.3)
annotated = results[0].plot()

# Export to ONNX.
model.export(format="onnx")
```

## Dataset Configuration

`data.yaml` points to a Label Studio COCO export and its image directory:

```yaml
coco_json: C:/datasets/widgets/result.json
images_dir: C:/datasets/widgets/images
output_dir: C:/datasets/widgets/prepared
val_split: 0.2
```

PTYOLOX Garage remaps COCO category IDs, validates image paths, creates train/validation splits, and writes the directory structure expected by Pixeltable YOLOX.

## Model Sizes

| Name | Depth | Width |
|---|---:|---:|
| `nano` | 0.33 | 0.25 |
| `tiny` | 0.33 | 0.375 |
| `s` | 0.33 | 0.50 |
| `m` | 0.67 | 0.75 |
| `l` | 1.00 | 1.00 |
| `x` | 1.33 | 1.25 |

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv build
```

Detailed guides are available in the [English documentation](https://github.com/Moge800/ptyolox-garage/tree/main/docs/en) and [Japanese documentation](https://github.com/Moge800/ptyolox-garage/tree/main/docs/jp).

## Attribution

PTYOLOX Garage is built on [Pixeltable YOLOX](https://github.com/pixeltable/pixeltable-yolox), which is derived from [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). PTYOLOX Garage is an independent project and is not an official Pixeltable product.

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
