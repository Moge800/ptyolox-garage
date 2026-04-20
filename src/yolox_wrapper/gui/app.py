"""メインウィンドウ"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..config import AppConfig
from .camera_tab import CameraTab
from .export_tab import ExportTab
from .infer_tab import InferTab
from .train_tab import TrainTab


class App(tk.Tk):
    """YOLOX Wrapper メインウィンドウ"""

    def __init__(self) -> None:
        super().__init__()
        self.title("YOLOX Wrapper")
        self.geometry("960x720")
        self.minsize(800, 600)

        self.config_mgr = AppConfig()

        self._build_menu()
        self._build_profile_bar()
        self._build_notebook()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="設定を保存", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.destroy)

        profile_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="プロファイル", menu=profile_menu)
        profile_menu.add_command(
            label="プロファイルを追加...", command=self._add_profile
        )
        profile_menu.add_command(
            label="プロファイルを削除", command=self._remove_profile
        )

    def _build_profile_bar(self) -> None:
        """プロファイル選択バー"""
        bar = ttk.Frame(self, padding=(4, 2))
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="設定プロファイル:").pack(side="left")

        self._profile_var = tk.StringVar(value="default")
        self._profile_cb = ttk.Combobox(
            bar,
            textvariable=self._profile_var,
            values=self.config_mgr.profiles(),
            state="readonly",
            width=20,
        )
        self._profile_cb.pack(side="left", padx=(4, 0))
        self._profile_cb.bind("<<ComboboxSelected>>", self._on_profile_changed)

        ttk.Separator(self, orient="horizontal").pack(fill="x")

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=4, pady=4)

        self._train_tab = TrainTab(self._nb, self.config_mgr, self._profile_var)
        self._infer_tab = InferTab(self._nb, self.config_mgr, self._profile_var)
        self._camera_tab = CameraTab(self._nb, self.config_mgr, self._profile_var)
        self._export_tab = ExportTab(self._nb)

        self._nb.add(self._train_tab, text="  学習  ")
        self._nb.add(self._infer_tab, text="  推論テスト  ")
        self._nb.add(self._camera_tab, text="  カメラテスト  ")
        self._nb.add(self._export_tab, text="  ONNX エクスポート  ")

    def _on_profile_changed(self, _event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        profile = self._profile_var.get()
        self._train_tab.load_profile(profile)
        self._infer_tab.load_profile(profile)
        self._camera_tab.load_profile(profile)

    def _save_config(self) -> None:
        self._train_tab.save_profile()
        self._infer_tab.save_profile()
        self._camera_tab.save_profile()
        self.config_mgr.save()
        self._show_status("設定を保存しました")

    def _add_profile(self) -> None:
        from tkinter.simpledialog import askstring

        name = askstring("プロファイル追加", "新しいプロファイル名:", parent=self)
        if name and name.strip():
            self.config_mgr.add_profile(name.strip())
            self._refresh_profile_list()

    def _remove_profile(self) -> None:
        profile = self._profile_var.get()
        if profile == "default":
            from tkinter.messagebox import showwarning

            showwarning(
                "削除不可", "'default' プロファイルは削除できません。", parent=self
            )
            return
        from tkinter.messagebox import askyesno

        if askyesno("確認", f"プロファイル '{profile}' を削除しますか？", parent=self):
            self.config_mgr.remove_profile(profile)
            self._profile_var.set("default")
            self._refresh_profile_list()
            self._on_profile_changed(None)

    def _refresh_profile_list(self) -> None:
        self._profile_cb["values"] = self.config_mgr.profiles()

    def _show_status(self, msg: str) -> None:
        # ウィンドウタイトルに一時的に表示
        original = self.title()
        self.title(f"{original} — {msg}")
        self.after(2000, lambda: self.title(original))

    def destroy(self) -> None:
        self._camera_tab.stop()
        super().destroy()


def main() -> None:
    app = App()
    app.mainloop()
