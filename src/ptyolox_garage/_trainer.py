"""Internal YOLOX training implementation.

This module is used internally by ptyolox_garage.wrapper.YOLOX.train().

Sequential epoch training with epoch_schedule=[100, 200, 300]:
        Stage 0: max_epoch=100, resume=False → stage_0_epoch100_ckpt.pth
        Stage 1: max_epoch=200, resume=True  → resume from epoch 101
        Stage 2: max_epoch=300, resume=True  → resume from epoch 201
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dataset import _MODEL_CONFIGS


class TrainingStopped(RuntimeError):
    """Raised when a staged training run is stopped between stages."""

# ---------------------------------------------------------------------------
# Log redirection
# ---------------------------------------------------------------------------


class _LogRedirector:
    """Temporarily redirect stdout and stderr to a callback."""

    def __init__(self, callback: Callable[[str], None] | None) -> None:
        self._callback = callback
        self._original_stdout = None
        self._original_stderr = None

    def __enter__(self) -> _LogRedirector:
        if self._callback is None:
            return self
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _CallbackStream(self._callback)
        sys.stderr = _CallbackStream(self._callback)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr


class _CallbackStream(io.TextIOBase):
    """Stream that forwards write() calls to a callback."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback

    def write(self, text: str) -> int:
        if text and text.strip():
            self._callback(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _build_config(
    name: str,
    num_classes: int,
    data_dir: str,
    input_size: tuple[int, int],
    depth: float,
    width: float,
    max_epoch: int,
    num_workers: int,
    basic_lr_per_img: float,
    train_ann: str,
    val_ann: str,
    output_dir: str,
):
    """Create and return a configured YoloxConfig subclass."""
    try:
        from yolox.config import YoloxConfig
    except ImportError as e:
        raise ImportError(
            "pixeltable-yolox パッケージが必要です。\n" "  pip install pixeltable-yolox"
        ) from e

    class _Exp(YoloxConfig):
        def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
            from yolox.data import CocoDataset, TrainTransform

            return CocoDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache,
                cache_type=cache_type,
                name="train",
            )

        def get_eval_dataset(self, **kwargs):
            from yolox.data import CocoDataset, ValTransform

            return CocoDataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                img_size=self.test_size,
                preproc=ValTransform(legacy=False),
                name="val",
            )

        def get_evaluator(
            self, batch_size, is_distributed, testdev=False, legacy=False
        ):
            from yolox.evaluators import COCOEvaluator

            return COCOEvaluator(
                dataloader=self.get_eval_loader(batch_size, is_distributed),
                img_size=self.test_size[0],  # type: ignore[index]
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
                per_class_AP=True,
                per_class_AR=True,
            )

    cfg = _Exp(name=name)
    cfg.num_classes = num_classes
    cfg.depth = depth
    cfg.width = width
    cfg.input_size = input_size
    cfg.test_size = input_size
    cfg.max_epoch = max_epoch
    cfg.warmup_epochs = max(1, min(5, max_epoch // 10))
    cfg.no_aug_epochs = max(1, min(15, max_epoch // 5))
    cfg.basic_lr_per_img = basic_lr_per_img
    cfg.data_dir = data_dir
    cfg.train_ann = train_ann
    cfg.val_ann = val_ann
    cfg.data_num_workers = num_workers
    cfg.eval_interval = 5
    cfg.test_conf = 0.01
    cfg.nmsthre = 0.65
    cfg.output_dir = output_dir
    cfg.save_history_ckpt = True
    return cfg


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class _YOLOXTrainer:
    """Manage YOLOX training.

    This class is normally used through YOLOX.train().
    """

    def __init__(
        self,
        model_size: str,
        num_classes: int,
        dataset_dir: str,
        output_dir: str,
        input_size: tuple[int, int] = (640, 640),
        batch_size: int = 16,
        device: str = "cpu",
        num_workers: int = 4,
        basic_lr_per_img: float = 0.01 / 64,
        pretrained_weights: str | None = None,
    ) -> None:
        if model_size not in _MODEL_CONFIGS:
            raise ValueError(
                f"未対応の model_size: '{model_size}'\n"
                f"使用可能: {list(_MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.basic_lr_per_img = basic_lr_per_img
        self.pretrained_weights = pretrained_weights

    def train_sequential(
        self,
        epoch_schedule: list[int],
        on_log: Callable[[str], None] | None = None,
        on_stage_done: Callable[[int, int, str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        """Run sequential epoch training.

        Args:
            epoch_schedule: Epoch targets, for example ``[100, 200, 300]``.
            on_log: Callback that receives log lines.
            on_stage_done: Stage completion callback receiving the stage index,
                epoch, and checkpoint path.
            stop_event: Stop before the next stage when this event is set.

        Returns:
            Path to the final checkpoint.
        """
        self._raise_if_stopped(stop_event, 0, len(epoch_schedule), on_log)

        try:
            import torch
            from yolox.core.trainer import Trainer
        except ImportError as e:
            raise ImportError(
                "pixeltable-yolox パッケージが必要です。\n"
                "  pip install pixeltable-yolox"
            ) from e

        cfg = _MODEL_CONFIGS[self.model_size]
        num_gpu, device_id = self._parse_device(self.device)

        if device_id >= 0:
            torch.cuda.set_device(device_id)

        train_output_dir = str(self.output_dir / "train_output")
        exp_name = f"yolox_{self.model_size}"
        last_ckpt_path = ""

        for i, target_epoch in enumerate(epoch_schedule):
            self._raise_if_stopped(stop_event, i, len(epoch_schedule), on_log)

            if on_log:
                on_log(
                    f"[Trainer] ===== Stage {i + 1}/{len(epoch_schedule)} | target epoch: {target_epoch} ====="
                )

            config = _build_config(
                name=exp_name,
                num_classes=self.num_classes,
                data_dir=self.dataset_dir,
                input_size=self.input_size,
                depth=cfg["depth"],
                width=cfg["width"],
                max_epoch=target_epoch,
                num_workers=self.num_workers,
                basic_lr_per_img=self.basic_lr_per_img,
                train_ann="instances_train.json",
                val_ann="instances_val.json",
                output_dir=train_output_dir,
            )

            resume = i > 0
            ckpt_path = self.pretrained_weights if i == 0 else None
            args = argparse.Namespace(
                name=exp_name,
                resume=resume,
                ckpt=ckpt_path,
                start_epoch=None,
                fp16=False,
                batch_size=self.batch_size,
                cache=None,
            )

            with _LogRedirector(on_log):
                if num_gpu > 1:
                    from yolox.core.launch import launch

                    launch(
                        lambda local_rank, _cfg=config, _args=args: Trainer(
                            _cfg, _args
                        ).train(),
                        num_gpus_per_machine=num_gpu,
                        num_machines=1,
                        machine_rank=0,
                        backend="nccl",
                        dist_url="auto",
                    )
                else:
                    trainer = Trainer(config, args)
                    trainer.train()

            # Save a checkpoint snapshot for this stage.
            last_ckpt = Path(train_output_dir) / exp_name / "latest_ckpt.pth"
            snap = (
                Path(train_output_dir)
                / exp_name
                / f"stage_{i}_epoch{target_epoch}_ckpt.pth"
            )
            if last_ckpt.exists():
                shutil.copy2(last_ckpt, snap)
                last_ckpt_path = str(snap)

            if on_stage_done:
                on_stage_done(i, target_epoch, last_ckpt_path)

        return last_ckpt_path

    @staticmethod
    def _raise_if_stopped(
        stop_event: threading.Event | None,
        stage_index: int,
        total_stages: int,
        on_log: Callable[[str], None] | None,
    ) -> None:
        """Raise a cancellation signal before starting the next stage."""
        if stop_event is None or not stop_event.is_set():
            return

        msg = f"[Trainer] 中断 (Stage {stage_index}/{total_stages})"
        if on_log:
            on_log(msg)
        else:
            print(msg)
        raise TrainingStopped(msg)

    def package_model(
        self,
        class_names: dict[int, str],
        checkpoint_path: str | None = None,
        output_model_path: str | None = None,
    ) -> str:
        """Convert a checkpoint into a .pt file readable by YOLOX.

        Args:
            class_names: Class-name mapping such as ``{0: "cat", 1: "dog"}``.
            checkpoint_path: Source checkpoint; discovered automatically if omitted.
            output_model_path: Destination path; defaults to
                ``output_dir/yolox_{model_size}.pt``.

        Returns:
            Path to the saved .pt file.
        """
        import torch

        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()

        print(f"[Trainer] チェックポイント読み込み: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg = _MODEL_CONFIGS[self.model_size]
        model = self._build_model(cfg["depth"], cfg["width"], len(class_names))

        state_dict = ckpt.get("model", ckpt)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        else:
            model = state_dict

        model.eval()

        if output_model_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_model_path = str(self.output_dir / f"yolox_{self.model_size}.pt")

        torch.save(
            {
                "model": model,
                "names": class_names,
                "nc": len(class_names),
                "input_size": list(self.input_size),
                "depth": cfg["depth"],
                "width": cfg["width"],
            },
            output_model_path,
        )
        print(f"[Trainer] モデルを保存しました: {output_model_path}")
        return output_model_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _parse_device(self, device: str) -> tuple[int, int]:
        """Return the GPU count and device ID for a device string."""
        if device == "cpu":
            return 0, -1
        if device == "cuda":
            return 1, 0
        if device.startswith("cuda:"):
            return 1, int(device.split(":")[1])
        return 0, -1

    def _find_best_checkpoint(self) -> str:
        train_dir = self.output_dir / "train_output" / f"yolox_{self.model_size}"
        for name in ["best_ckpt.pth", "latest_ckpt.pth"]:
            p = train_dir / name
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {train_dir}\n"
            "先に train_sequential() を実行してください。"
        )

    def _build_model(self, depth: float, width: float, num_classes: int):
        try:
            from yolox.models import YoloPafpn, YoloxHead, YoloxModule
        except ImportError as e:
            raise ImportError(
                "pixeltable-yolox パッケージが必要です。\n  pip install pixeltable-yolox"
            ) from e

        in_channels = [256, 512, 1024]
        backbone = YoloPafpn(depth, width, in_channels=in_channels)
        head = YoloxHead(num_classes, width, in_channels=in_channels)
        return YoloxModule(backbone, head)
