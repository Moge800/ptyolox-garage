"""Lightweight console entry point for launching PTYOLOX Garage."""

from __future__ import annotations

import sys
import time

_PREFIX = "[PTYOLOX Garage]"
_startup_stage = "starting"


def startup_log(message: str, *, stage: str | None = None) -> None:
    """Print a startup message immediately and optionally record its stage."""
    global _startup_stage
    if stage is not None:
        _startup_stage = stage
    print(f"{_PREFIX} {message}", flush=True)


def main() -> None:
    """Launch the GUI while reporting startup progress to the console."""
    started_at = time.perf_counter()
    startup_log("Starting...", stage="starting")

    try:
        startup_log("Loading GUI components...", stage="loading GUI components")
        from ptyolox_garage.gui.app import main as gui_main

        gui_main(started_at=started_at)
    except Exception as exc:
        print(
            f"{_PREFIX} Failed while {_startup_stage}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
