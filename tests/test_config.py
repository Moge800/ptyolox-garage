"""Tests for AppConfig."""

from pathlib import Path

import pytest

from ptyolox_garage.config import AppConfig, ProfileParams


class TestAppConfig:
    def test_load_defaults_when_no_file(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        p = cfg.get("default")
        assert p.device == "cpu"
        assert p.model_size == "l"
        assert p.batch_size == 16
        assert p.language == "auto"

    def test_profiles_returns_sections(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        cfg.add_profile("factory_pc")
        assert "default" in cfg.profiles()
        assert "factory_pc" in cfg.profiles()

    def test_set_and_get(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        cfg.set("default", "device", "cuda:0")
        assert cfg.get("default").device == "cuda:0"

    def test_set_params_roundtrip(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        params = ProfileParams(
            device="cuda:0",
            model_size="m",
            batch_size=32,
            imgsz=416,
            workers=2,
            val_split=0.15,
            output_dir="D:/work",
            conf=0.3,
            iou=0.5,
        )
        cfg.set_params("default", params)
        loaded = cfg.get("default")
        assert loaded.device == "cuda:0"
        assert loaded.model_size == "m"
        assert loaded.batch_size == 32
        assert loaded.imgsz == 416
        assert loaded.val_split == pytest.approx(0.15)

    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "config.ini"
        cfg = AppConfig(path)
        cfg.set("default", "batch_size", "64")
        cfg.save()

        cfg2 = AppConfig(path)
        assert cfg2.get("default").batch_size == 64

    def test_remove_profile(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        cfg.add_profile("test_profile")
        assert cfg.remove_profile("test_profile") is True
        assert "test_profile" not in cfg.profiles()

    def test_cannot_remove_default(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        assert cfg.remove_profile("default") is False
        assert "default" in cfg.profiles()

    def test_get_unknown_profile_falls_back_to_default(self, tmp_path: Path) -> None:
        cfg = AppConfig(tmp_path / "config.ini")
        p = cfg.get("nonexistent")
        assert p.device == "cpu"  # Value inherited from the default profile.
