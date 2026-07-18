# Quick Start

## Using the Code API

### Training

```python
from ptyolox_garage import YOLOX

model = YOLOX("l")  # Model size: nano / tiny / s / m / l / x
model.train(
    data="data.yaml",          # Path to Label Studio COCO export config
    epochs=[100, 200, 300],    # Staged epoch schedule
    device="cuda:0",           # Use GPU
    batch=16,
    imgsz=640,
)
```

### Inference

```python
from ptyolox_garage import YOLOX

model = YOLOX("best_model.pt")            # Load trained model
results = model.predict("image.jpg", conf=0.3)

for result in results:
    print(result.boxes.xyxy)   # Detection box coordinates
    print(result.boxes.conf)   # Confidence scores
    print(result.boxes.cls)    # Class IDs

    # Draw results on image
    annotated = result.plot()
```

### ONNX Export

```python
model = YOLOX("best_model.pt")
onnx_path = model.export(format="onnx")
print(f"Exported to {onnx_path}")
```

---

## Using the GUI

```bash
uv run ptyolox-garage
```

or

```bash
uv run python main.py
```

The GUI will launch with four tabs (Train, Infer, Camera, Export).  
See [GUI Guide](gui.md) for details.

---

## data.yaml Format

Training requires a `data.yaml` that specifies the COCO-format JSON exported from Label Studio and the image directory.

```yaml
coco_json: /path/to/result.json    # Label Studio COCO export JSON
images_dir: /path/to/images        # Image directory

# Optional
output_dir: /path/to/output        # Output directory (defaults to current directory)
val_split: 0.2                     # Validation split ratio (default: 0.2)
```

> Relative paths in `coco_json` and `images_dir` are resolved relative to the directory containing `data.yaml`.
