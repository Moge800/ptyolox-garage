# Installation

## Requirements

- Python **3.10** or later
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Moge800/yolox_wrapper.git
cd yolox_wrapper
```

### 2. Create Virtual Environment and Install Dependencies

```bash
uv sync
```

> `uv sync` automatically installs all dependencies defined in `pyproject.toml`, including pixeltable-yolox.

### 3. Verify Installation

```bash
uv run python -c "from yolox_wrapper import YOLOX; print('OK')"
```

---

## Main Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.5.1 | Deep learning framework |
| torchvision | >= 0.20.1 | Image processing utilities (NMS, etc.) |
| opencv-python | >= 4.8 | Image loading and drawing |
| numpy | >= 1.24, < 2.0 | Numerical computation |
| pillow | >= 10.0 | Image display in GUI |
| pyyaml | >= 6.0 | data.yaml parsing |
| pixeltable-yolox | >= 0.4.2 | YOLOX model definitions and training logic (Python 3.9–3.13) |
| onnx | >= 1.16 | ONNX export |
| ml-dtypes | >= 0.5.0 | onnx dependency |
| beep-lite | >= 0.1.3 | Training completion notification sound |

### Development Packages

```bash
uv sync --group dev
```

| Package | Purpose |
|---------|---------|
| pytest | Test execution |
| pytest-cov | Coverage measurement |
| ruff | Linter and formatter |

---

## GPU Support

To use a CUDA-capable GPU, ensure that the CUDA build of PyTorch is installed.  
Set `device = cuda:0` in `config.ini` to enable GPU-accelerated training and inference.

```python
import torch
print(torch.cuda.is_available())  # True means GPU is available
```
