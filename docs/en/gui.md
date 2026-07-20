# GUI Guide

PTYOLOX Garage includes a tkinter-based GUI application.

## Launching

```bash
uv run ptyolox-garage
# or
uv run python main.py
```

## Layout

The application consists of the following elements:

- **Menu Bar** — File (save settings) and Profile (add/remove) operations
- **Profile Selector** — Dropdown at the top to switch configuration profiles
- **Tabs** — Four tabs: Train, Infer, Camera, Export

---

## Train Tab

Train YOLOX models.

### Input Fields

| Field | Description |
|-------|-------------|
| data.yaml | Path to Label Studio COCO export configuration file |
| Model Size | Select from `nano` / `tiny` / `s` / `m` / `l` / `x` |
| Epochs | Comma-separated staged schedule (e.g., `100,200,300`) |
| Batch Size | Training batch size |
| Image Size | Input image side length (32–1920, step 32) |
| Workers | Number of parallel data loading workers |
| Val Split | Validation data ratio (0.05–0.5) |
| Device | CPU or GPU (cuda:0) |

### Operations

1. Configure each field
2. Click **Start** to begin training (runs in a background thread)
3. Training progress is displayed in real-time in the log area
4. Progress bar shows stage-by-stage progress
5. Click **Stop After Stage** to stop before the next stage; the active stage completes first

> A beep sound notifies you when training is complete.

---

## Infer Tab

Run inference on single images using a trained model.

### Input Fields

| Field | Description |
|-------|-------------|
| Model Path | Path to `.pt` file |
| Image Path | Image file or directory for inference |
| Confidence Threshold | Detection confidence filter (0.01–1.0) |
| IoU Threshold | NMS IoU filter (0.01–1.0) |
| Device | CPU or GPU |

### Operations

1. Select model and image
2. Adjust thresholds
3. Run inference to display result image and detection list

> The model is lazily loaded and only reloaded when the path changes.
> PyTorch checkpoint files such as `.pt` can execute arbitrary code when loaded. Confirm the GUI warning only for files from a trusted source.

---

## Camera Tab

Real-time object detection from a USB camera.

### Input Fields

| Field | Description |
|-------|-------------|
| Model Path | Path to `.pt` file |
| Camera Index | Camera device index (0–9) |
| Confidence Threshold | Detection confidence filter |
| IoU Threshold | NMS IoU filter |
| Device | CPU or GPU |

### Operations

1. Set model and camera index
2. Click **Start** to begin live inference
3. FPS is displayed in real-time
4. Click **Stop** to stop

> Frames are updated targeting approximately 30fps.
> Confirm the model-file security warning only when you trust the file's source.

---

## Export Tab

Export trained models to ONNX format.

### Input Fields

| Field | Description |
|-------|-------------|
| Input Model | Path to `.pt` file |
| Output Path | Save path for `.onnx` file (auto-set if omitted) |

### Operations

1. Select input model (output path is auto-suggested)
2. Click **Export** to run export
3. Status displays green (success) or red (error)

> PyTorch checkpoint files such as `.pt` can execute arbitrary code when loaded. Confirm the GUI warning only for files from a trusted source.

---

## Profile Management

- **Add Profile**: Menu → Profile → Add Profile, enter profile name
- **Remove Profile**: Menu → Profile → Remove Profile to delete the current profile
- **Save Settings**: Menu → File → Save Config to save all tab settings to `config.ini`

> The `default` profile cannot be deleted.
