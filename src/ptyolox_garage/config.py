"""設定管理モジュール

config.ini を OS 標準のユーザー設定ディレクトリから読み書きします。
セクションごとに設定を保存し、GUI で切り替えられます。

使用例::

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

_DEFAULT_CONFIG_PATH = user_config_path("ptyolox-garage", appauthor=False) / "config.ini"

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
    """1プロファイル分のパラメータ"""

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
    """config.ini の読み書きを管理するクラス"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._parser = configparser.ConfigParser(defaults=_DEFAULTS)
        self.load()

    # ------------------------------------------------------------------
    # 読み込み / 保存
    # ------------------------------------------------------------------

    def load(self) -> None:
        """config.ini を読み込む (ファイルがなければデフォルト値を使用)"""
        if self._path.exists():
            self._parser.read(self._path, encoding="utf-8")
        # default セクションがなければ作成
        if not self._parser.has_section("default"):
            self._parser.add_section("default")

    def save(self) -> None:
        """config.ini に書き込む"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            self._parser.write(f)

    # ------------------------------------------------------------------
    # プロファイル操作
    # ------------------------------------------------------------------

    def profiles(self) -> list[str]:
        """利用可能なプロファイル名の一覧を返す"""
        return self._parser.sections()

    def get(self, profile: str = "default") -> ProfileParams:
        """指定プロファイルのパラメータを返す"""
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
        """指定プロファイルの値を更新する (save() するまでファイルには書かれない)"""
        if not self._parser.has_section(profile):
            self._parser.add_section(profile)
        self._parser.set(profile, key, value)

    def set_params(self, profile: str, params: ProfileParams) -> None:
        """ProfileParams を一括で書き込む"""
        if not self._parser.has_section(profile):
            self._parser.add_section(profile)
        for f in fields(params):
            self._parser.set(profile, f.name, str(getattr(params, f.name)))

    def add_profile(self, profile: str) -> None:
        """新規プロファイルを追加する (default の値をコピー)"""
        if self._parser.has_section(profile):
            return
        self._parser.add_section(profile)
        defaults = self.get("default")
        self.set_params(profile, defaults)

    def remove_profile(self, profile: str) -> bool:
        """プロファイルを削除する。default は削除不可。"""
        if profile == "default":
            return False
        return self._parser.remove_section(profile)
