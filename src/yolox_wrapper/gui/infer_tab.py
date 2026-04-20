"""推論テストタブ"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from ..config import AppConfig
from ..wrapper import YOLOX


class InferTab(ttk.Frame):
    """推論テストタブ"""

    def __init__(
        self,
        parent: tk.Widget,
        config_mgr: AppConfig,
        profile_var: tk.StringVar,
    ) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._profile_var = profile_var
        self._model: YOLOX | None = None
        self._loaded_model_path = ""

        self._build()
        self.load_profile(profile_var.get())

    def _build(self) -> None:
        # 左: 設定
        left = ttk.LabelFrame(self, text="推論設定", padding=8)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)

        # モデルパス
        self._model_var = tk.StringVar()
        self._add_path_row(
            left,
            "モデル (.pt):",
            self._model_var,
            is_file=True,
            filetypes=[("PyTorch モデル", "*.pt"), ("全ファイル", "*.*")],
        )

        # 画像パス
        self._source_var = tk.StringVar()
        ttk.Label(left, text="画像 / ディレクトリ:").pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self._source_var).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row, text="...", width=3, command=self._browse_source).pack(
            side="left"
        )

        # conf / iou
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text="conf 閾値:").pack(anchor="w")
        self._conf_var = tk.DoubleVar(value=0.25)
        self._conf_label = ttk.Label(left, text="0.25")
        ttk.Scale(
            left,
            from_=0.01,
            to=1.0,
            variable=self._conf_var,
            orient="horizontal",
            length=160,
            command=lambda v: self._conf_label.config(text=f"{float(v):.2f}"),
        ).pack(anchor="w")
        self._conf_label.pack(anchor="e")

        ttk.Label(left, text="IOU 閾値:").pack(anchor="w", pady=(6, 0))
        self._iou_var = tk.DoubleVar(value=0.45)
        self._iou_label = ttk.Label(left, text="0.45")
        ttk.Scale(
            left,
            from_=0.01,
            to=1.0,
            variable=self._iou_var,
            orient="horizontal",
            length=160,
            command=lambda v: self._iou_label.config(text=f"{float(v):.2f}"),
        ).pack(anchor="w")
        self._iou_label.pack(anchor="e")

        # デバイス
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text="デバイス:").pack(anchor="w")
        self._device_var = tk.StringVar(value="cpu")
        dev_frame = ttk.Frame(left)
        dev_frame.pack(anchor="w")
        ttk.Radiobutton(
            dev_frame, text="CPU", variable=self._device_var, value="cpu"
        ).pack(side="left")
        self._gpu_radio = ttk.Radiobutton(
            dev_frame, text="GPU", variable=self._device_var, value="cuda:0"
        )
        self._gpu_radio.pack(side="left")
        self._check_gpu()

        # ボタン
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(left, text="推論実行", command=self._run).pack(fill="x")

        # 右: 結果
        right = ttk.Frame(self, padding=(4, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        # 画像表示
        self._img_label = ttk.Label(
            right, text="(画像がここに表示されます)", anchor="center", relief="sunken"
        )
        self._img_label.pack(fill="both", expand=True)

        # 検出結果テキスト
        ttk.Label(right, text="検出結果:").pack(anchor="w", pady=(4, 0))
        self._result_text = scrolledtext.ScrolledText(
            right, height=6, state="disabled", wrap="word", font=("Consolas", 9)
        )
        self._result_text.pack(fill="x")

    # ------------------------------------------------------------------
    # プロファイル連携
    # ------------------------------------------------------------------

    def load_profile(self, profile: str) -> None:
        p = self._config_mgr.get(profile)
        self._device_var.set(p.device)
        self._conf_var.set(p.conf)
        self._conf_label.config(text=f"{p.conf:.2f}")
        self._iou_var.set(p.iou)
        self._iou_label.config(text=f"{p.iou:.2f}")

    def save_profile(self) -> None:
        profile = self._profile_var.get()
        self._config_mgr.set(profile, "device", self._device_var.get())
        self._config_mgr.set(profile, "conf", str(round(self._conf_var.get(), 2)))
        self._config_mgr.set(profile, "iou", str(round(self._iou_var.get(), 2)))

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def _run(self) -> None:
        model_path = self._model_var.get().strip()
        source = self._source_var.get().strip()
        if not model_path:
            messagebox.showwarning("入力エラー", "モデルパスを指定してください。")
            return
        if not source:
            messagebox.showwarning("入力エラー", "画像パスを指定してください。")
            return

        try:
            if model_path != self._loaded_model_path:
                self._model = YOLOX(model_path)
                self._loaded_model_path = model_path

            results = self._model.predict(  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
                source,
                conf=self._conf_var.get(),
                iou=self._iou_var.get(),
                device=self._device_var.get(),
            )
        except Exception as e:
            messagebox.showerror("推論エラー", str(e))
            return

        if results:
            self._display_result(results[0])

    def _display_result(self, result: Any) -> None:
        # 画像表示
        annotated = result.plot()
        self._show_image(annotated)

        # テキスト
        lines = [
            f"検出数: {len(result.boxes)}",
        ]
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = result.names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].cpu().numpy()
            lines.append(
                f"  [{i}] {name} conf={conf:.3f} "
                f"xyxy=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]"
            )

        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("end", "\n".join(lines))
        self._result_text.config(state="disabled")

    def _show_image(self, bgr: np.ndarray) -> None:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        w = self._img_label.winfo_width() or 640
        h = self._img_label.winfo_height() or 480
        pil.thumbnail((w, h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        self._img_label.config(image=photo, text="")
        self._img_label.image = photo  # type: ignore[attr-defined]  # ty: ignore[invalid-assignment]

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _browse_source(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("画像", "*.jpg *.jpeg *.png *.bmp"), ("全ファイル", "*.*")]
        )
        if path:
            self._source_var.set(path)

    def _check_gpu(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                self._gpu_radio.config(state="disabled")
                self._device_var.set("cpu")
        except ImportError:
            self._gpu_radio.config(state="disabled")

    def _add_path_row(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.StringVar,
        is_file: bool = True,
        filetypes: list[Any] | None = None,
    ) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        def cmd(v=var, ft=filetypes):
            v.set(filedialog.askopenfilename(filetypes=ft or [("全ファイル", "*.*")]))

        ttk.Button(row, text="...", width=3, command=cmd).pack(side="left")
