# Development Guide

## Development Environment Setup

```bash
git clone https://github.com/Moge800/ptyolox-garage.git
cd ptyolox-garage
uv sync --group dev
```

---

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/ptyolox_garage
```

### Test Structure

| File | Target | Content |
|------|--------|---------|
| `tests/test_config.py` | `config.py` | AppConfig read/write, profile management |
| `tests/test_dataset.py` | `dataset.py` | DatasetPreparer split and remap processing |
| `tests/test_wrapper.py` | `wrapper.py` | Model initialization, letterbox, NMS, inference results |

> `test_wrapper.py` covers testable areas without requiring the YOLOX package itself.

---

## Linter and Formatter

```bash
uv run ruff check .        # Lint check
uv run ruff format .       # Format
```

---

## Project Structure

```
ptyolox_garage/
├── main.py                     # GUI entry point
├── config.example.ini                  # Application settings
├── pyproject.toml              # Package configuration
├── src/
│   └── ptyolox_garage/
│       ├── __init__.py         # Package exports
│       ├── config.py           # Config management (AppConfig, ProfileParams)
│       ├── dataset.py          # Dataset preparation (DatasetPreparer)
│       ├── wrapper.py          # Main wrapper (YOLOX, YOLOXResult, YOLOXBoxes)
│       ├── _trainer.py         # Internal training engine (_YOLOXTrainer)
│       └── gui/
│           ├── __init__.py
│           ├── app.py          # Main window (App)
│           ├── train_tab.py    # Train tab (TrainTab)
│           ├── infer_tab.py    # Infer tab (InferTab)
│           ├── camera_tab.py   # Camera tab (CameraTab)
│           └── export_tab.py   # Export tab (ExportTab)
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   └── test_wrapper.py
└── docs/                       # Documentation
```

---

## Dependency Structure

```
wrapper.py ──→ _trainer.py ──→ dataset.py
    │
    └──→ dataset.py

config.py (independent)

gui/app.py ──→ gui/*_tab.py ──→ wrapper.py, config.py
```

---

## License

Apache License 2.0 — See [LICENSE](../../LICENSE) for details.

---

## Release to PyPI

PyPI publication uses Trusted Publishing. Configure the PyPI publisher with
these values:

| Setting | Value |
|---|---|
| Owner | `Moge800` |
| Repository | `ptyolox-garage` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Package versions are generated from Git tags. Publish a GitHub Release from a
`v<version>` tag; for example, `v0.2.1` creates package version `0.2.1`.
The workflow verifies that the generated wheel version matches the Release tag.
Manual runs build from `main` and skip files that already exist on PyPI. A
GitHub Release fails when files for its version already exist on PyPI,
preventing release assets from diverging from the published package. The
workflow runs tests and Ruff for the exact release tag or `main` commit, then
builds the wheel and sdist, publishes both to PyPI, and attaches them to a
GitHub Release when one triggered the run.
