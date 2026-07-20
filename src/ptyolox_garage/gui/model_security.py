"""Safety confirmation for loading PyTorch model files from the GUI."""

from __future__ import annotations

from tkinter import messagebox

from ..i18n import tr


def confirm_trusted_model_file(model_path: str) -> bool:
    """Ask the user to confirm that a model file is from a trusted source."""
    return bool(
        messagebox.askokcancel(
            tr("セキュリティ警告", "Security Warning"),
            tr(
                "PyTorch の .pt ファイルは読み込み時に任意のコードを実行する可能性があります。\n"
                "信頼できる作成元のモデルだけを開いてください。\n\n"
                f"このファイルを読み込みますか？\n{model_path}",
                "PyTorch .pt files can execute arbitrary code when loaded.\n"
                "Open model files only when you trust their source.\n\n"
                f"Load this file?\n{model_path}",
            ),
            icon="warning",
            default="cancel",
        )
    )
