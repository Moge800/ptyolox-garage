"""Internal helpers for model serialization and device handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn


def model_devices(model: nn.Module) -> tuple[str, ...]:
    """Return the distinct devices used by registered parameters and buffers."""
    devices = {str(tensor.device) for tensor in model.parameters()}
    devices.update(str(tensor.device) for tensor in model.buffers())
    return tuple(sorted(devices))


def model_device_label(model: nn.Module) -> str:
    """Return a stable device label suitable for logs and checkpoint metadata."""
    devices = model_devices(model)
    return "/".join(devices) if devices else "unknown"


def assert_all_on_cpu(model: nn.Module) -> None:
    """Raise when a registered parameter or buffer is not on the CPU."""
    stray = [
        name
        for name, tensor in list(model.named_parameters())
        + list(model.named_buffers())
        if tensor.device.type != "cpu"
    ]
    if stray:
        preview = ", ".join(stray[:5])
        raise RuntimeError(f"CPU以外のテンソルが残っています: {preview}")


@contextmanager
def temporary_model_device(
    model: nn.Module,
    target_device: str | None,
) -> Iterator[nn.Module]:
    """Temporarily move a single-device model and always restore its device."""
    devices = model_devices(model)
    if len(devices) > 1:
        raise RuntimeError(
            "複数デバイスに分散したモデルは保存・エクスポートできません: "
            + ", ".join(devices)
        )

    original_device = devices[0] if devices else None
    should_move = target_device is not None and original_device != target_device
    if should_move:
        model.to(target_device)

    try:
        yield model
    finally:
        if should_move and original_device is not None:
            model.to(original_device)
