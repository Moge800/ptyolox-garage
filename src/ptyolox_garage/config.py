"""Configuration management.

Read and write config.ini in the operating system's standard user configuration
directory. Settings are stored in sections that can be selected in the GUI.

Example::

    cfg = AppConfig()
    params = cfg.get("factory_pc")
    print(params.device)  # "cuda:0"

    cfg.set("factory_pc", "batch_size", "32")
    cfg.save()
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, fields
from pathlib import Path

from platformdirs import user_config_path

_DEFAULT_CONFIG_PATH = (
    user_config_path("ptyolox-garage", appauthor=False, roaming=True) / "config.ini"
)

_DEFAULTS: dict[str, str] = {
    "device": "cpu",
    "model_size": "l",
    "batch_size": "16",
    "imgsz": "640",
    "workers": "4",
    "val_split": "0.2",
    "output_dir": "",
    "conf": "0.25",
    "iou": "0.45",
    "language": "auto",
}


@dataclass
class ProfileParams:
    """Parameters for one configuration profile."""

    device: str = "cpu"
    model_size: str = "l"
    batch_size: int = 16
    imgsz: int = 640
    workers: int = 4
    val_split: float = 0.2
    output_dir: str = ""
    conf: float = 0.25
    iou: float = 0.45
    language: str = "auto"


class AppConfig:
    """Manage reading and writing config.ini."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._parser = configparser.ConfigParser(defaults=_DEFAULTS)
        self.load()

    # ------------------------------------------------------------------
    # Loading and saving
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load config.ini, using defaults when the file does not exist."""
        if self._path.exists():
            self._parser.read(self._path, encoding="utf-8")
        # Ensure the default section exists.
        if not self._parser.has_section("default"):
            self._parser.add_section("default")

    def save(self) -> None:
        """Write the current configuration to config.ini."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            self._parser.write(f)

    # ------------------------------------------------------------------
    # Profile operations
    # ------------------------------------------------------------------

    def profiles(self) -> list[str]:
        """Return the available profile names."""
        return self._parser.sections()

    def get(self, profile: str = "default") -> ProfileParams:
        """Return parameters for the specified profile."""
        if not self._parser.has_section(profile):
            profile = "default"
        sec = self._parser[profile]
        return ProfileParams(
            device=sec.get("device", "cpu"),
            model_size=sec.get("model_size", "l"),
            batch_size=sec.getint("batch_size", 16),
            imgsz=sec.getint("imgsz", 640),
            workers=sec.getint("workers", 4),
            val_split=sec.getfloat("val_split", 0.2),
            output_dir=sec.get("output_dir", ""),
            conf=sec.getfloat("conf", 0.25),
            iou=sec.getfloat("iou", 0.45),
            language=sec.get("language", "auto"),
        )

    def set(self, profile: str, key: str, value: str) -> None:
        """Update a profile value without writing it to disk until save()."""
        if not self._parser.has_section(profile):
            self._parser.add_section(profile)
        self._parser.set(profile, key, value)

    def set_params(self, profile: str, params: ProfileParams) -> None:
        """Update all values in a profile from ProfileParams."""
        if not self._parser.has_section(profile):
            self._parser.add_section(profile)
        for f in fields(params):
            self._parser.set(profile, f.name, str(getattr(params, f.name)))

    def add_profile(self, profile: str) -> None:
        """Add a profile initialized with values from the default profile."""
        if self._parser.has_section(profile):
            return
        self._parser.add_section(profile)
        defaults = self.get("default")
        self.set_params(profile, defaults)

    def remove_profile(self, profile: str) -> bool:
        """Remove a profile; the default profile cannot be removed."""
        if profile == "default":
            return False
        return self._parser.remove_section(profile)
