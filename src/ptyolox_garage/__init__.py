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

from .config import AppConfig, ProfileParams
from .wrapper import YOLOX, YOLOXBoxes, YOLOXResult

__all__ = ["YOLOX", "YOLOXBoxes", "YOLOXResult", "AppConfig", "ProfileParams"]

__version__ = "0.1.0"
