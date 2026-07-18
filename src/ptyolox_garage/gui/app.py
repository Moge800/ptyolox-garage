"""Main application window."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from ptyolox_garage_bootstrap import startup_log

from ..config import AppConfig
from ..i18n import set_language, tr


class App(tk.Tk):
    """PTYOLOX Garage main window."""

    def __init__(self) -> None:
        super().__init__()
        self.geometry("960x720")
        self.minsize(800, 600)

        startup_log("Loading configuration...", stage="loading configuration")
        self.config_mgr = AppConfig()
        startup_log("Resolving language...", stage="resolving language")
        self._configured_language = self.config_mgr.get("default").language
        set_language(self._configured_language)
        self.title("PTYOLOX Garage")

        startup_log("Building interface...", stage="building interface")
        self._starting = True
        self._build_ui()
        self._starting = False

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_profile_bar()
        self._build_notebook()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label=tr("ファイル", "File"), menu=file_menu)
        file_menu.add_command(label=tr("設定を保存", "Save Config"), command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label=tr("終了", "Exit"), command=self.destroy)

        profile_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label=tr("プロファイル", "Profile"), menu=profile_menu)
        profile_menu.add_command(
            label=tr("プロファイルを追加...", "Add Profile..."), command=self._add_profile
        )
        profile_menu.add_command(
            label=tr("プロファイルを削除", "Remove Profile"), command=self._remove_profile
        )

        language_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label=tr("言語", "Language"), menu=language_menu)
        self._language_var = tk.StringVar(value=self._configured_language)
        for value, label in (("auto", tr("自動", "Auto")), ("ja", "日本語"), ("en", "English")):
            language_menu.add_radiobutton(
                label=label,
                value=value,
                variable=self._language_var,
                command=self._change_language,
            )

    def _build_profile_bar(self) -> None:
        """Build the profile-selection bar."""
        bar = ttk.Frame(self, padding=(4, 2))
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text=tr("設定プロファイル:", "Config profile:")).pack(side="left")

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
        if self._starting:
            startup_log("Loading ML dependencies...", stage="loading ML dependencies")
        from .camera_tab import CameraTab
        from .export_tab import ExportTab
        from .infer_tab import InferTab
        from .train_tab import TrainTab

        if self._starting:
            startup_log("Building tabs...", stage="building tabs")
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=4, pady=4)

        self._train_tab = TrainTab(self._nb, self.config_mgr, self._profile_var)
        self._infer_tab = InferTab(self._nb, self.config_mgr, self._profile_var)
        self._camera_tab = CameraTab(self._nb, self.config_mgr, self._profile_var)
        self._export_tab = ExportTab(self._nb)

        self._nb.add(self._train_tab, text=tr("  学習  ", "  Train  "))
        self._nb.add(self._infer_tab, text=tr("  推論テスト  ", "  Inference  "))
        self._nb.add(self._camera_tab, text=tr("  カメラテスト  ", "  Camera  "))
        self._nb.add(self._export_tab, text=tr("  ONNX エクスポート  ", "  ONNX Export  "))

    def _on_profile_changed(self, _event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        profile = self._profile_var.get()
        self._train_tab.load_profile(profile)
        self._infer_tab.load_profile(profile)
        self._camera_tab.load_profile(profile)

    def _restore_profile(self, profile: str) -> None:
        """Restore a profile selection after rebuilding the translated UI."""
        if profile not in self.config_mgr.profiles():
            profile = "default"
        self._profile_var.set(profile)
        self._on_profile_changed()

    def _save_config(self) -> None:
        self._train_tab.save_profile()
        self._infer_tab.save_profile()
        self._camera_tab.save_profile()
        self.config_mgr.save()
        self._show_status(tr("設定を保存しました", "Configuration saved"))

    def _add_profile(self) -> None:
        from tkinter.simpledialog import askstring

        name = askstring(
            tr("プロファイル追加", "Add Profile"),
            tr("新しいプロファイル名:", "New profile name:"),
            parent=self,
        )
        if name and name.strip():
            self.config_mgr.add_profile(name.strip())
            self._refresh_profile_list()

    def _remove_profile(self) -> None:
        profile = self._profile_var.get()
        if profile == "default":
            from tkinter.messagebox import showwarning

            showwarning(
                tr("削除不可", "Cannot Remove"),
                tr("'default' プロファイルは削除できません。", "The 'default' profile cannot be removed."),
                parent=self,
            )
            return
        from tkinter.messagebox import askyesno

        if askyesno(
            tr("確認", "Confirm"),
            tr(f"プロファイル '{profile}' を削除しますか？", f"Remove profile '{profile}'?"),
            parent=self,
        ):
            self.config_mgr.remove_profile(profile)
            self._profile_var.set("default")
            self._refresh_profile_list()
            self._on_profile_changed(None)

    def _refresh_profile_list(self) -> None:
        self._profile_cb["values"] = self.config_mgr.profiles()

    def _change_language(self) -> None:
        if self._train_tab.is_busy() or self._export_tab.is_busy():
            from tkinter.messagebox import showwarning

            self._language_var.set(self._configured_language)
            showwarning(
                tr("処理中", "Task in Progress"),
                tr(
                    "学習またはエクスポート中は言語を変更できません。",
                    "The language cannot be changed during training or export.",
                ),
                parent=self,
            )
            return

        selected_profile = self._profile_var.get()
        self._train_tab.save_profile()
        self._infer_tab.save_profile()
        self._camera_tab.save_profile()
        self._camera_tab.stop()
        self._configured_language = self._language_var.get()
        self.config_mgr.set("default", "language", self._configured_language)
        self.config_mgr.save()
        set_language(self._configured_language)
        for child in self.winfo_children():
            child.destroy()
        self.config(menu="")
        self.title("PTYOLOX Garage")
        self._build_ui()
        self._restore_profile(selected_profile)

    def _show_status(self, msg: str) -> None:
        # Show the status temporarily in the window title.
        original = self.title()
        self.title(f"{original} — {msg}")
        self.after(2000, lambda: self.title(original))

    def destroy(self) -> None:
        self._camera_tab.stop()
        super().destroy()


def main(started_at: float | None = None) -> None:
    if started_at is None:
        started_at = time.perf_counter()
    app = App()
    elapsed = time.perf_counter() - started_at
    startup_log(f"Ready ({elapsed:.1f}s)", stage="running GUI")
    app.mainloop()
