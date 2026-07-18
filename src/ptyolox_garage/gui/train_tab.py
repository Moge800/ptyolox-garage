"""学習タブ"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import beep_lite

from ..config import AppConfig, ProfileParams
from ..i18n import tr
from ..wrapper import YOLOX


class TrainTab(ttk.Frame):
    """学習タブ"""

    def __init__(
        self,
        parent: tk.Widget,
        config_mgr: AppConfig,
        profile_var: tk.StringVar,
    ) -> None:
        super().__init__(parent)
        self._config_mgr = config_mgr
        self._profile_var = profile_var
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._train_succeeded = False

        self._build()
        self.load_profile(profile_var.get())

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # 左ペイン: 設定
        left = ttk.LabelFrame(self, text=tr("学習設定", "Training Settings"), padding=8)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)

        # data.yaml
        self._data_var = tk.StringVar()
        self._add_path_row(
            left,
            "data.yaml:",
            self._data_var,
            is_file=True,
            filetypes=[("YAML", "*.yaml *.yml"), (tr("全ファイル", "All Files"), "*.*")],
        )

        # モデルサイズ
        ttk.Label(left, text=tr("モデルサイズ:", "Model size:")).pack(anchor="w", pady=(6, 0))
        self._model_var = tk.StringVar(value="l")
        ttk.Combobox(
            left,
            textvariable=self._model_var,
            values=["nano", "tiny", "s", "m", "l", "x"],
            state="readonly",
            width=10,
        ).pack(anchor="w")

        # エポックスケジュール
        ttk.Label(left, text=tr("エポック (例: 100,200,300):", "Epochs (e.g. 100,200,300):")).pack(
            anchor="w", pady=(6, 0)
        )
        self._epochs_var = tk.StringVar(value="100,200,300")
        ttk.Entry(left, textvariable=self._epochs_var, width=20).pack(anchor="w")

        # バッチサイズ
        ttk.Label(left, text=tr("バッチサイズ:", "Batch size:")).pack(anchor="w", pady=(6, 0))
        self._batch_var = tk.IntVar(value=16)
        ttk.Spinbox(left, from_=1, to=256, textvariable=self._batch_var, width=8).pack(
            anchor="w"
        )

        # 入力サイズ
        ttk.Label(left, text=tr("入力サイズ (imgsz):", "Input size (imgsz):")).pack(anchor="w", pady=(6, 0))
        self._imgsz_var = tk.IntVar(value=640)
        ttk.Spinbox(
            left, from_=32, to=1920, increment=32, textvariable=self._imgsz_var, width=8
        ).pack(anchor="w")

        # ワーカー数
        ttk.Label(left, text=tr("ワーカー数:", "Workers:")).pack(anchor="w", pady=(6, 0))
        self._workers_var = tk.IntVar(value=4)
        ttk.Spinbox(left, from_=0, to=16, textvariable=self._workers_var, width=8).pack(
            anchor="w"
        )

        # val 分割比
        ttk.Label(left, text=tr("val 分割比:", "Validation split:")).pack(anchor="w", pady=(6, 0))
        self._val_split_var = tk.DoubleVar(value=0.2)
        val_frame = ttk.Frame(left)
        val_frame.pack(anchor="w")
        ttk.Scale(
            val_frame,
            from_=0.05,
            to=0.5,
            variable=self._val_split_var,
            orient="horizontal",
            length=120,
            command=lambda v: self._val_label.config(text=f"{float(v):.0%}"),
        ).pack(side="left")
        self._val_label = ttk.Label(val_frame, text="20%", width=5)
        self._val_label.pack(side="left")

        # デバイス
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text=tr("デバイス:", "Device:")).pack(anchor="w")
        self._device_var = tk.StringVar(value="cpu")
        device_frame = ttk.Frame(left)
        device_frame.pack(anchor="w")
        ttk.Radiobutton(
            device_frame, text="CPU", variable=self._device_var, value="cpu"
        ).pack(side="left")
        self._gpu_radio = ttk.Radiobutton(
            device_frame, text="GPU (cuda:0)", variable=self._device_var, value="cuda:0"
        )
        self._gpu_radio.pack(side="left")

        self._check_gpu()

        # ボタン
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x")
        self._start_btn = ttk.Button(btn_frame, text=tr("学習開始", "Start Training"), command=self._start)
        self._start_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self._stop_btn = ttk.Button(
            btn_frame, text=tr("停止", "Stop"), command=self._stop, state="disabled"
        )
        self._stop_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))

        # 右ペイン: 進捗 + ログ
        right = ttk.Frame(self, padding=(4, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        # ステージ進捗
        prog_frame = ttk.Frame(right)
        prog_frame.pack(fill="x")
        self._stage_label = ttk.Label(prog_frame, text=tr("待機中", "Idle"))
        self._stage_label.pack(side="left")
        self._progress = ttk.Progressbar(prog_frame, length=200, mode="determinate")
        self._progress.pack(side="right", fill="x", expand=True, padx=(8, 0))

        # ログ
        ttk.Label(right, text=tr("ログ:", "Log:")).pack(anchor="w", pady=(8, 0))
        self._log = scrolledtext.ScrolledText(
            right, state="disabled", wrap="word", font=("Consolas", 9), height=30
        )
        self._log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # プロファイル連携
    # ------------------------------------------------------------------

    def load_profile(self, profile: str) -> None:
        p = self._config_mgr.get(profile)
        self._model_var.set(p.model_size)
        self._batch_var.set(p.batch_size)
        self._imgsz_var.set(p.imgsz)
        self._workers_var.set(p.workers)
        self._val_split_var.set(p.val_split)
        self._val_label.config(text=f"{p.val_split:.0%}")
        self._device_var.set(p.device)

    def save_profile(self) -> None:
        profile = self._profile_var.get()
        p = ProfileParams(
            device=self._device_var.get(),
            model_size=self._model_var.get(),
            batch_size=self._batch_var.get(),
            imgsz=self._imgsz_var.get(),
            workers=self._workers_var.get(),
            val_split=self._val_split_var.get(),
            output_dir="",
            conf=0.25,
            iou=0.45,
        )
        self._config_mgr.set_params(profile, p)

    # ------------------------------------------------------------------
    # 学習制御
    # ------------------------------------------------------------------

    def _start(self) -> None:
        data_path = self._data_var.get().strip()
        if not data_path:
            messagebox.showwarning(
                tr("入力エラー", "Input Error"),
                tr("data.yaml パスを指定してください。", "Specify a data.yaml path."),
            )
            return

        try:
            epoch_schedule = self._parse_epochs(self._epochs_var.get())
        except ValueError as e:
            messagebox.showerror(tr("入力エラー", "Input Error"), str(e))
            return

        self._stop_event.clear()
        self._set_running(True)
        self._progress["maximum"] = len(epoch_schedule)
        self._progress["value"] = 0
        self._stage_label.config(text=f"Stage 0/{len(epoch_schedule)}")
        self._log_clear()

        self._thread = threading.Thread(
            target=self._run_training,
            args=(data_path, epoch_schedule),
            daemon=True,
        )
        self._thread.start()
        self.after(100, self._poll_log)

    def _stop(self) -> None:
        self._stop_event.set()
        self._append_log(tr("--- 停止リクエスト送信 ---", "--- Stop requested ---"))

    def _run_training(self, data_path: str, epoch_schedule: list[int]) -> None:
        try:
            model = YOLOX(self._model_var.get(), verbose=False)
            model.train(
                data=data_path,
                epochs=epoch_schedule,
                batch=self._batch_var.get(),
                device=self._device_var.get(),
                imgsz=self._imgsz_var.get(),
                workers=self._workers_var.get(),
                val_split=self._val_split_var.get(),
                on_log=self._log_queue.put,
                on_stage_done=self._on_stage_done,
            )
            self._log_queue.put(tr("[完了] 全ステージの学習が終了しました。", "[Done] All training stages completed."))
            self._train_succeeded = True
        except Exception as e:
            self._log_queue.put(tr(f"[エラー] {e}", f"[Error] {e}"))
            self._train_succeeded = False
        finally:
            self._log_queue.put(None)  # 終了シグナル

    def _on_stage_done(self, stage_idx: int, epoch: int, ckpt_path: str) -> None:
        total = self._progress["maximum"]
        self._log_queue.put(
            tr(
                f"[Stage {stage_idx + 1}/{int(total)}] epoch {epoch} 完了: {ckpt_path}",
                f"[Stage {stage_idx + 1}/{int(total)}] epoch {epoch} done: {ckpt_path}",
            )
        )

    def _poll_log(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line is None:
                    self._set_running(False)
                    # ステージ進捗を最終状態に更新
                    self._progress["value"] = self._progress["maximum"]
                    self._stage_label.config(text=tr("完了", "Done"))
                    if self._train_succeeded:
                        beep_lite.ok()
                    else:
                        beep_lite.ng()
                    return
                self._append_log(line)
                # ステージ完了行からプログレスを進める
                if line.startswith("[Stage "):
                    try:
                        done = int(line.split("/")[0].split()[-1])
                        self._progress["value"] = done
                        total = int(self._progress["maximum"])
                        self._stage_label.config(text=f"Stage {done}/{total}")
                    except (ValueError, IndexError):
                        pass
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _parse_epochs(self, text: str) -> list[int]:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError(tr("エポック数を入力してください。", "Enter at least one epoch value."))
        try:
            values = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(
                tr(f"エポック数は整数で指定してください: {text}", f"Epochs must be integers: {text}")
            ) from e
        if any(v <= 0 for v in values):
            raise ValueError(tr("エポック数は 1 以上の整数で指定してください。", "Epochs must be integers greater than zero."))
        return sorted(set(values))

    def _check_gpu(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                self._gpu_radio.config(state="disabled")
                self._device_var.set("cpu")
        except ImportError:
            self._gpu_radio.config(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        stop_state = "normal" if running else "disabled"
        self._start_btn.config(state=state)
        self._stop_btn.config(state=stop_state)

    def _append_log(self, text: str) -> None:
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _log_clear(self) -> None:
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

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
        cmd = (
            (
                lambda v=var, ft=filetypes: v.set(
                    filedialog.askopenfilename(filetypes=ft or [(tr("全ファイル", "All Files"), "*.*")])
                )
            )
            if is_file
            else (lambda v=var: v.set(filedialog.askdirectory()))
        )
        ttk.Button(row, text="...", width=3, command=cmd).pack(side="left")
