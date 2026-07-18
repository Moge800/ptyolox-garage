"""ONNX エクスポートタブ"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..i18n import tr
from ..wrapper import YOLOX


class ExportTab(ttk.Frame):
    """ONNX エクスポートタブ"""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        frame = ttk.LabelFrame(self, text=tr("ONNX エクスポート設定", "ONNX Export Settings"), padding=12)
        frame.pack(padx=16, pady=16, fill="x")

        # モデルパス
        ttk.Label(frame, text=tr("モデルパス (.pt):", "Model path (.pt):")).grid(
            row=0, column=0, sticky="w", pady=4
        )
        self._model_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._model_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(frame, text="...", width=3, command=self._browse_model).grid(
            row=0, column=2, padx=(2, 0)
        )

        # 出力パス
        ttk.Label(frame, text=tr("出力パス (.onnx):", "Output path (.onnx):")).grid(
            row=1, column=0, sticky="w", pady=4
        )
        self._output_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._output_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(frame, text="...", width=3, command=self._browse_output).grid(
            row=1, column=2, padx=(2, 0)
        )
        ttk.Label(
            frame,
            text=tr("(空欄 = モデルと同じディレクトリに保存)", "(Blank = save next to the model)"),
            foreground="gray",
        ).grid(row=2, column=1, sticky="w")

        frame.columnconfigure(1, weight=1)

        # ボタン + ステータス
        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=16, pady=8, fill="x")
        self._export_btn = ttk.Button(
            btn_frame, text=tr("エクスポート実行", "Export"), command=self._export
        )
        self._export_btn.pack(side="left")

        self._status_label = ttk.Label(self, text="", foreground="gray")
        self._status_label.pack(padx=16, anchor="w")

    def _export(self) -> None:
        model_path = self._model_var.get().strip()
        if not model_path:
            messagebox.showwarning(tr("入力エラー", "Input Error"), tr("モデルパスを指定してください。", "Specify a model path."))
            return

        output_path = self._output_var.get().strip() or None
        self._export_btn.config(state="disabled")
        self._status_label.config(text=tr("エクスポート中...", "Exporting..."), foreground="gray")

        threading.Thread(
            target=self._run_export,
            args=(model_path, output_path),
            daemon=True,
        ).start()

    def _run_export(self, model_path: str, output_path: str | None) -> None:
        try:
            model = YOLOX(model_path)
            result = model.export(format="onnx", output_path=output_path)
            self.after(0, lambda: self._on_done(result))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self._on_error(msg))

    def _on_done(self, output_path: str) -> None:
        self._export_btn.config(state="normal")
        self._status_label.config(text=tr(f"完了: {output_path}", f"Done: {output_path}"), foreground="green")
        messagebox.showinfo(
            tr("エクスポート完了", "Export Complete"),
            tr(f"ONNX ファイルを保存しました:\n{output_path}", f"Saved ONNX file:\n{output_path}"),
        )

    def _on_error(self, msg: str) -> None:
        self._export_btn.config(state="normal")
        self._status_label.config(text=tr(f"エラー: {msg}", f"Error: {msg}"), foreground="red")
        messagebox.showerror(tr("エクスポートエラー", "Export Error"), msg)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[(tr("PyTorch モデル", "PyTorch Model"), "*.pt"), (tr("全ファイル", "All Files"), "*.*")]
        )
        if path:
            self._model_var.set(path)
            # 出力パスを自動補完
            if not self._output_var.get():
                self._output_var.set(str(Path(path).with_suffix(".onnx")))

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".onnx",
            filetypes=[("ONNX", "*.onnx"), (tr("全ファイル", "All Files"), "*.*")],
        )
        if path:
            self._output_var.set(path)
