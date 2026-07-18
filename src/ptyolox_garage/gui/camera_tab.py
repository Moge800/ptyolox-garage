"""Camera testing tab."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..config import AppConfig
from ..i18n import tr
from ..wrapper import YOLOX


class CameraTab(ttk.Frame):
    """Live camera testing tab."""

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
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)

        self._build()
        self.load_profile(profile_var.get())

    def _build(self) -> None:
        # Left: settings
        left = ttk.LabelFrame(self, text=tr("カメラ設定", "Camera Settings"), padding=8)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)

        # Model path
        self._model_var = tk.StringVar()
        ttk.Label(left, text=tr("モデル (.pt):", "Model (.pt):")).pack(anchor="w")
        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self._model_var).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row, text="...", width=3, command=self._browse_model).pack(
            side="left"
        )

        # Camera index
        ttk.Label(left, text=tr("カメラ番号:", "Camera index:")).pack(anchor="w", pady=(6, 0))
        self._cam_var = tk.IntVar(value=0)
        ttk.Spinbox(left, from_=0, to=9, textvariable=self._cam_var, width=5).pack(
            anchor="w"
        )

        # conf / iou
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text=tr("conf 閾値:", "Confidence threshold:")).pack(anchor="w")
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

        ttk.Label(left, text=tr("IOU 閾値:", "IoU threshold:")).pack(anchor="w", pady=(6, 0))
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

        # Device
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text=tr("デバイス:", "Device:")).pack(anchor="w")
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

        # Controls
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x")
        self._start_btn = ttk.Button(btn_frame, text=tr("開始", "Start"), command=self._start)
        self._start_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self._stop_btn = ttk.Button(
            btn_frame, text=tr("停止", "Stop"), command=self._stop, state="disabled"
        )
        self._stop_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))

        # FPS
        self._fps_label = ttk.Label(left, text="FPS: --")
        self._fps_label.pack(anchor="w", pady=(8, 0))

        # Right: video
        right = ttk.Frame(self, padding=(4, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        self._video_label = ttk.Label(
            right,
            text=tr("(カメラ映像がここに表示されます)", "(Camera preview)"),
            anchor="center",
            relief="sunken",
        )
        self._video_label.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Profile integration
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
    # Camera controls
    # ------------------------------------------------------------------

    def _start(self) -> None:
        model_path = self._model_var.get().strip()
        if not model_path:
            messagebox.showwarning(tr("入力エラー", "Input Error"), tr("モデルパスを指定してください。", "Specify a model path."))
            return

        try:
            self._model = YOLOX(model_path)
        except Exception as e:
            messagebox.showerror(tr("モデル読み込みエラー", "Model Loading Error"), str(e))
            return

        self._stop_event.clear()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self._thread.start()
        self.after(33, self._show_frame)

    def _stop(self) -> None:
        self._stop_event.set()

    def stop(self) -> None:
        """Release camera resources when the application closes."""
        self._stop_event.set()

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._cam_var.get())
        if not cap.isOpened():
            self._frame_queue.put(None)
            return

        device = self._device_var.get()
        fps_times: list[float] = []

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                results = self._model.predict(  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
                    frame,
                    conf=self._conf_var.get(),
                    iou=self._iou_var.get(),
                    device=device,
                )
                annotated = results[0].plot() if results else frame

                fps_times.append(time.time() - t0)
                if len(fps_times) > 10:
                    fps_times.pop(0)
                fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                try:
                    self._frame_queue.put_nowait(annotated)
                except queue.Full:
                    pass
        finally:
            cap.release()
            self._frame_queue.put(None)

    def _show_frame(self) -> None:
        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            self.after(33, self._show_frame)
            return

        if frame is None:
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._fps_label.config(text="FPS: --")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w = self._video_label.winfo_width() or 640
        h = self._video_label.winfo_height() or 480
        pil.thumbnail((w, h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._video_label.config(image=photo, text="")
        self._video_label.image = photo  # type: ignore[attr-defined]  # ty: ignore[invalid-assignment]

        self.after(33, self._show_frame)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[(tr("PyTorch モデル", "PyTorch Model"), "*.pt"), (tr("全ファイル", "All Files"), "*.*")]
        )
        if path:
            self._model_var.set(path)

    def _check_gpu(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                self._gpu_radio.config(state="disabled")
                self._device_var.set("cpu")
        except ImportError:
            self._gpu_radio.config(state="disabled")
