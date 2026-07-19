"""PTYOLOX Garage package.

Example::

    from ptyolox_garage import YOLOX

    model = YOLOX("l")
    model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0")

    model = YOLOX("yolox_l.pt")
    results = model.predict("image.jpg", conf=0.3)
    model.export(format="onnx")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import AppConfig, ProfileParams
    from .wrapper import YOLOX, TrainingStopped, YOLOXBoxes, YOLOXResult

__all__ = ["TrainingStopped", "YOLOX", "YOLOXBoxes", "YOLOXResult", "AppConfig", "ProfileParams"]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load public exports only when they are first accessed."""
    if name in {"AppConfig", "ProfileParams"}:
        from . import config

        value = getattr(config, name)
    elif name in {"TrainingStopped", "YOLOX", "YOLOXBoxes", "YOLOXResult"}:
        from . import wrapper

        value = getattr(wrapper, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive attribute listings."""
    return sorted(set(globals()) | set(__all__))
