# PTYOLOX Garage Documentation

PTYOLOX Garage is a Python toolkit for training, inference, and ONNX export of YOLOX object detection models, powered by [pixeltable-yolox](https://github.com/pixeltable/pixeltable-yolox).
It provides an ultralytics YOLO-like API and a tkinter-based GUI.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Training** | Train directly from Label Studio COCO exports with staged epoch scheduling |
| **Inference** | Run predictions on image files, NumPy arrays, or directories. Built-in NMS and letterbox processing |
| **Real-time Inference** | Live object detection via USB camera from the GUI |
| **ONNX Export** | Convert trained `.pt` models to ONNX format |
| **Config Profiles** | Manage multiple environment settings (dev PC, factory PC, etc.) with `config.ini` |
| **GUI** | Four-tab interface for training, inference, camera, and export |

---

## Documentation

- [Installation](installation.md) — Environment setup and dependencies
- [Quick Start](quickstart.md) — Get up and running quickly
- [Configuration](configuration.md) — `config.ini` profile management
- [GUI Guide](gui.md) — How to use the GUI application
- API Reference
  - [YOLOX (wrapper)](api/wrapper.md) — Main wrapper class
  - [AppConfig / ProfileParams (config)](api/config.md) — Configuration management
  - [DatasetPreparer (dataset)](api/dataset.md) — Dataset preparation
  - [_YOLOXTrainer (trainer)](api/trainer.md) — Internal training engine
- [Development Guide](development.md) — Testing, coding conventions, contributing
