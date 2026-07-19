"""PTYOLOX Garage Module

This module provides a YOLOX wrapper with an Ultralytics-style API.

Example:
    >>> from ptyolox_garage import YOLOX

    # Train a new model.
    >>> model = YOLOX("l")
    >>> model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0", batch=16)

    # Load a trained model.
    >>> model = YOLOX("yolox_l.pt")
    >>> model.fuse()
    >>> results = model.predict("image.jpg", conf=0.3, device="cuda:0")
    >>> for box in results[0].boxes:
    ...     cls_id = int(box.cls[0].item())
    ...     conf   = float(box.conf[0].item())
    ...     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    # Export to ONNX.
    >>> model.export(format="onnx")

Model storage format (torch.save):
    torch.save(
        {"model": model, "names": {0: "cat", 1: "dog"}, "nc": 2, "input_size": [640, 640]},
        "my_model.pt",
    )
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from ._trainer import TrainingStopped, _YOLOXTrainer
from .dataset import _MODEL_CONFIGS, DatasetPreparer

__all__ = ["TrainingStopped", "YOLOX", "YOLOXBoxes", "YOLOXResult"]

# ---------------------------------------------------------------------------
# Model-size normalization
# ---------------------------------------------------------------------------


def _normalize_model_size(s: str) -> str:
    """Normalize model-size aliases such as 'yolox_l' and 'yolox-l' to 'l'."""
    normalized = s.lower().removeprefix("yolox_").removeprefix("yolox-")
    if normalized not in _MODEL_CONFIGS:
        raise ValueError(
            f"未対応のモデルサイズ: '{s}'\n"
            f"使用可能: {list(_MODEL_CONFIGS.keys())} または 'yolox_{{size}}'"
        )
    return normalized


# ---------------------------------------------------------------------------
# Result objects inspired by Ultralytics Boxes and Results
# ---------------------------------------------------------------------------


class _YOLOXBox:
    """Single bounding box with an Ultralytics-style interface.

    Attributes:
        xyxy: Tensor with shape [1, 4]; ``box.xyxy[0]`` has shape [4].
        conf: Tensor with shape [1]; ``box.conf[0].item()`` returns a float.
        cls: Tensor with shape [1]; ``box.cls[0].item()`` returns a float.
    """

    __slots__ = ("xyxy", "conf", "cls")

    def __init__(
        self,
        xyxy: torch.Tensor,
        conf: torch.Tensor,
        cls: torch.Tensor,
    ) -> None:
        self.xyxy = xyxy  # [1, 4]
        self.conf = conf  # [1]
        self.cls = cls  # [1]


class YOLOXBoxes:
    """Bounding-box collection with an Ultralytics-style interface.

    Attributes:
        xyxy: Tensor with shape [N, 4].
        conf: Tensor with shape [N].
        cls: Tensor with shape [N].
    """

    def __init__(
        self,
        xyxy: torch.Tensor,
        conf: torch.Tensor,
        cls: torch.Tensor,
    ) -> None:
        self.xyxy = xyxy  # [N, 4]
        self.conf = conf  # [N]
        self.cls = cls  # [N]

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield _YOLOXBox(
                self.xyxy[i : i + 1],
                self.conf[i : i + 1],
                self.cls[i : i + 1],
            )


class YOLOXResult:
    """Inference result with an Ultralytics-style interface.

    Attributes:
        boxes: YOLOXBoxes instance.
        names: Class-name mapping ``{int: str}``.
        orig_shape: Original image size as ``(height, width)``.
        orig_img: Original image array used by ``plot()``.
    """

    def __init__(
        self,
        boxes: YOLOXBoxes,
        names: dict[int, str],
        orig_shape: tuple[int, int],
        orig_img: np.ndarray | None = None,
    ) -> None:
        self.boxes = boxes
        self.names = names
        self.orig_shape = orig_shape
        self.orig_img = orig_img
        self.results_dict: dict[str, float] = {}

    def plot(self, orig_img: np.ndarray | None = None) -> np.ndarray:
        """Draw detections and return an image, matching ``Result.plot()``."""
        img = orig_img if orig_img is not None else self.orig_img
        if img is None:
            h, w = self.orig_shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
        result = img.copy()

        for box in self.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = f"{self.names.get(cls_id, str(cls_id))} {conf:.2f}"

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                result,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                result, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return result


# ---------------------------------------------------------------------------
# Preprocessing and postprocessing utilities
# ---------------------------------------------------------------------------


def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    fill_value: int = 114,
) -> tuple[np.ndarray, float]:
    """Resize with letterboxing while preserving the aspect ratio."""
    h, w = image.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    rh, rw = int(h * r), int(w * r)

    padded = np.full((nh, nw, 3), fill_value, dtype=np.uint8)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded[:rh, :rw] = resized
    return padded, r


def _nms_fallback(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Implement NMS without depending on torchvision."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        idx = int(order[0].item())
        keep.append(idx)
        if order.numel() == 1:
            break
        order = order[1:]
        ix1 = x1[order].clamp(min=float(x1[idx]))
        iy1 = y1[order].clamp(min=float(y1[idx]))
        ix2 = x2[order].clamp(max=float(x2[idx]))
        iy2 = y2[order].clamp(max=float(y2[idx]))
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (areas[idx] + areas[order] - inter + 1e-6)
        order = order[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _apply_class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Run class-aware NMS with torchvision or a per-class fallback."""
    try:
        from torchvision.ops import batched_nms

        return batched_nms(boxes, scores, class_ids, iou_threshold)
    except (ImportError, RuntimeError):
        return _class_aware_nms_fallback(boxes, scores, class_ids, iou_threshold)


def _class_aware_nms_fallback(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Apply fallback NMS independently to each detection class."""
    kept_by_class: list[torch.Tensor] = []
    for class_id in class_ids.unique(sorted=True):
        class_indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        class_keep = _nms_fallback(boxes[class_indices], scores[class_indices], iou_threshold)
        kept_by_class.append(class_indices[class_keep])

    if not kept_by_class:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    keep = torch.cat(kept_by_class)
    return keep[scores[keep].argsort(descending=True)]


def _postprocess(
    outputs: torch.Tensor,
    ratio: float,
    orig_h: int,
    orig_w: int,
    conf_thre: float,
    iou_thre: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode YOLOX inference output and apply NMS."""
    pred = outputs[0]  # [N, C+5]

    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes_xyxy = torch.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1
    )

    obj_conf = pred[:, 4]
    cls_confs = pred[:, 5:]
    cls_scores, cls_ids = cls_confs.max(dim=1)
    scores = obj_conf * cls_scores

    mask = scores >= conf_thre
    if not mask.any():
        empty4 = torch.zeros((0, 4), dtype=torch.float32)
        return empty4, torch.zeros(0), torch.zeros(0)

    boxes_f = boxes_xyxy[mask]
    scores_f = scores[mask]
    cls_f = cls_ids[mask]

    boxes_f = boxes_f / ratio
    boxes_f[:, 0::2].clamp_(0.0, float(orig_w))
    boxes_f[:, 1::2].clamp_(0.0, float(orig_h))

    keep = _apply_class_aware_nms(boxes_f, scores_f, cls_f, iou_thre)
    return boxes_f[keep].cpu(), scores_f[keep].cpu(), cls_f[keep].float().cpu()


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------


class YOLOX:
    """YOLOX model wrapper with an Ultralytics-style API.

    Example::

        # Train a new model.
        model = YOLOX("l")
        model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0")

        # Load a trained model.
        model = YOLOX("yolox_l.pt")
        results = model.predict("image.jpg", conf=0.3)
        model.export(format="onnx")
    """

    def __init__(self, model: str, verbose: bool = True) -> None:
        """Initialize the wrapper.

        Args:
            model: Model-size string such as ``"nano"``, ``"l"``, or
                ``"yolox_l"``, or a path to a trained .pt model.
            verbose: Whether to emit detailed logs.
        """
        self._verbose = verbose
        self.model: nn.Module | None = None
        self._num_classes: int = 80
        self._input_size: tuple[int, int] = (640, 640)
        self._class_names: dict[int, str] = {}
        self._current_device: str = "cpu"
        self._model_path: str | None = None
        self._model_size: str | None = None

        p = Path(model)
        if p.suffix == ".pt" or p.exists():
            self._load_checkpoint(str(p), verbose)
            self._model_path = str(p)
        else:
            self._model_size = _normalize_model_size(model)
            if verbose:
                print(f"[YOLOX] モデルサイズ: {self._model_size} (学習前)")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data: str,
        epochs: int | list[int] = 300,
        batch: int = 16,
        device: str = "cpu",
        imgsz: int = 640,
        workers: int = 4,
        val_split: float | None = None,
        pretrained_weights: str | None = None,
        on_log: Callable[[str], None] | None = None,
        on_stage_done: Callable[[int, int, str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> YOLOX:
        """Train the model.

        Args:
            data: Path to data.yaml for a Label Studio COCO export.
            epochs: Epoch count or sequential targets such as ``[100, 200, 300]``.
            batch: Batch size.
            device: Device string: ``'cpu'``, ``'cuda'``, or ``'cuda:0'``.
            imgsz: Input image size.
            workers: Number of data-loader workers.
            val_split: Validation fraction; defaults to the data.yaml value.
            pretrained_weights: .pt or .pth weights for fine-tuning. When omitted,
                use the model previously loaded with ``YOLOX("model.pt")``.
            on_log: Log callback used by the GUI.
            on_stage_done: Stage-completion callback used by the GUI.
            stop_event: Signal that stops before the next training stage.

        Returns:
            This instance, allowing method chaining.
        """
        # Infer model_size when fine-tuning from a loaded .pt file.
        if self._model_size is None and self._model_path is not None:
            self._model_size = self._infer_model_size_from_path(self._model_path)

        if self._model_size is None:
            raise RuntimeError("モデルサイズが設定されていません。")

        # Reuse the loaded model for fine-tuning when weights were not specified.
        if pretrained_weights is None and self._model_path is not None:
            pretrained_weights = self._model_path

        # Load data.yaml.
        data_cfg = self._load_data_config(data)
        output_dir = data_cfg.get("output_dir", "./yolox_work")
        effective_val_split = (
            val_split
            if val_split is not None
            else float(data_cfg.get("val_split", 0.2))
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Normalize the epoch schedule.
        if isinstance(epochs, int):
            epoch_schedule = [epochs]
        else:
            epoch_schedule = sorted(set(epochs))

        if on_log:
            on_log(
                f"[YOLOX] 学習開始: model={self._model_size}, epochs={epoch_schedule}, device={device}"
            )

        # 1. Prepare the dataset.
        if on_log:
            on_log("[YOLOX] データセット整備中...")
        preparer = DatasetPreparer(
            coco_json_path=data_cfg["coco_json"],
            images_dir=data_cfg["images_dir"],
            output_dir=str(output_path / "dataset"),
            val_split=effective_val_split,
        )
        class_names, num_classes = preparer.prepare()

        # Save class names for package_model().
        names_path = output_path / "class_names.json"
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        # 2. Train the model.
        trainer = _YOLOXTrainer(
            model_size=self._model_size,
            num_classes=num_classes,
            dataset_dir=str(output_path / "dataset"),
            output_dir=str(output_path),
            input_size=(imgsz, imgsz),
            batch_size=batch,
            device=device,
            num_workers=workers,
            pretrained_weights=pretrained_weights,
        )

        trainer.train_sequential(
            epoch_schedule=epoch_schedule,
            on_log=on_log,
            on_stage_done=on_stage_done,
            stop_event=stop_event,
        )

        # 3. Package the model and assign it to self.model.
        if on_log:
            on_log("[YOLOX] モデルをパッケージ化中...")
        model_path = trainer.package_model(
            class_names=class_names,
            output_model_path=str(output_path / f"yolox_{self._model_size}.pt"),
        )

        self._load_checkpoint(model_path, verbose=True)
        self._model_path = model_path
        return self

    # ------------------------------------------------------------------
    # Model-size inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_model_size_from_path(model_path: str) -> str:
        """Infer model_size from a trained .pt file.

        Use saved depth and width values when available, otherwise inspect the
        model filename for a size token.
        """
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            depth = ckpt.get("depth")
            width = ckpt.get("width")
            if depth is not None and width is not None:
                depth, width = float(depth), float(width)
                for size, cfg in _MODEL_CONFIGS.items():
                    if cfg["depth"] == depth and cfg["width"] == width:
                        return size

        # Infer from the filename.
        stem = Path(model_path).stem.lower()
        for size in _MODEL_CONFIGS:
            if size in stem:
                return size

        raise ValueError(
            f"モデルサイズを自動判定できません: {model_path}\n"
            "YOLOX('l', ...) のように明示的にモデルサイズを指定してください。"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, model_path: str, verbose: bool) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

        if isinstance(ckpt, nn.Module):
            self.model = ckpt
        elif isinstance(ckpt, dict):
            model_obj = ckpt.get("model")
            if isinstance(model_obj, nn.Module):
                self.model = model_obj
            elif isinstance(model_obj, dict):
                self.model = self._build_from_state_dict(model_obj, ckpt)
            else:
                raise ValueError(
                    f"未対応のチェックポイント形式です。\n"
                    f"キー: {list(ckpt.keys())}\n"
                    "YOLOX モデルは以下の形式で保存してください:\n"
                    "  torch.save({'model': model, 'names': {0:'cls0'}, 'nc': 1}, 'model.pt')"
                )

            if "names" in ckpt:
                raw = ckpt["names"]
                if isinstance(raw, dict):
                    self._class_names = {int(k): str(v) for k, v in raw.items()}
                else:
                    self._class_names = {i: str(v) for i, v in enumerate(raw)}
            if "nc" in ckpt:
                self._num_classes = int(ckpt["nc"])
            if "input_size" in ckpt:
                raw_size = ckpt["input_size"]
                self._input_size = (int(raw_size[0]), int(raw_size[1]))
        else:
            raise ValueError(f"未対応のチェックポイント形式: {type(ckpt)}")

        self.model.eval()
        self._infer_class_info()

        if verbose:
            print(
                f"[YOLOX] モデル読み込み完了: {model_path} (クラス数: {self._num_classes})"
            )

    def _build_from_state_dict(self, state_dict: dict, ckpt: dict) -> nn.Module:
        try:
            from yolox.models import YoloPafpn, YoloxHead, YoloxModule

            nc = int(ckpt.get("nc", 80))
            depth = float(ckpt.get("depth", 0.33))
            width = float(ckpt.get("width", 0.50))
            in_ch = [256, 512, 1024]

            backbone = YoloPafpn(depth, width, in_channels=in_ch)
            head = YoloxHead(nc, width, in_channels=in_ch)
            model = YoloxModule(backbone, head)
            model.load_state_dict(state_dict)
            return model

        except ImportError as e:
            raise ImportError(
                "state_dict 形式のチェックポイントを読み込むには pixeltable-yolox パッケージが必要です。\n"
                "  pip install pixeltable-yolox"
            ) from e

    def _infer_class_info(self) -> None:
        if self.model is None:
            return
        try:
            head = getattr(self.model, "head", None)
            if head is not None and hasattr(head, "num_classes"):
                self._num_classes = int(head.num_classes)
            elif hasattr(self.model, "num_classes"):
                self._num_classes = int(cast(int, self.model.num_classes))
        except Exception:
            pass

        if not self._class_names:
            self._class_names = {i: str(i) for i in range(self._num_classes)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def fuse(self) -> YOLOX:
        """Optimize the model, including BatchNorm fusion."""
        if self.model is not None:
            try:
                for m in self.model.modules():
                    if hasattr(m, "switch_to_deploy"):
                        m.switch_to_deploy()  # ty: ignore[call-non-callable]
            except Exception:
                pass
        return self

    def predict(
        self,
        source: str | Path | np.ndarray | list,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        verbose: bool = False,
    ) -> list[YOLOXResult]:
        """Run inference.

        Args:
            source: Image path, NumPy array, or a list of either.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            device: Device such as ``'cpu'``, ``'cuda'``, or ``'cuda:0'``.
            verbose: Whether to report processing time.

        Returns:
            One YOLOXResult per input image.
        """
        if self.model is None:
            raise RuntimeError(
                "モデルが読み込まれていません。先に train() を実行するか、"
                "学習済みモデルを YOLOX('path/to/model.pt') で読み込んでください。"
            )

        if device != self._current_device:
            self.model = self.model.to(device)
            self._current_device = device

        images = self._collect_images(source)
        results: list[YOLOXResult] = []

        with torch.no_grad():
            for image in images:
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                orig_h, orig_w = image.shape[:2]

                padded, ratio = _letterbox(image, self._input_size)
                tensor = (
                    torch.from_numpy(padded)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )

                t0 = time.time()
                outputs = self.model(tensor)
                if verbose:
                    print(f"[YOLOX] 推論時間: {(time.time() - t0) * 1000:.1f}ms")

                boxes_t, scores_t, cls_t = _postprocess(
                    outputs, ratio, orig_h, orig_w, conf, iou
                )

                if len(boxes_t) > 0:
                    yolox_boxes = YOLOXBoxes(boxes_t, scores_t, cls_t)
                else:
                    yolox_boxes = YOLOXBoxes(
                        torch.zeros((0, 4), dtype=torch.float32),
                        torch.zeros(0, dtype=torch.float32),
                        torch.zeros(0, dtype=torch.float32),
                    )

                results.append(
                    YOLOXResult(
                        boxes=yolox_boxes,
                        names=self._class_names,
                        orig_shape=(orig_h, orig_w),
                        orig_img=image,
                    )
                )

        return results

    @staticmethod
    def _collect_images(source: str | Path | np.ndarray | list) -> list[np.ndarray]:
        """Load an image, a non-recursive image directory, or a list of inputs."""
        if isinstance(source, np.ndarray):
            return [source]
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                image_paths = sorted(
                    (
                        candidate
                        for candidate in path.iterdir()
                        if candidate.is_file()
                        and candidate.suffix.lower()
                        in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
                    ),
                    key=lambda candidate: candidate.name.lower(),
                )
                if not image_paths:
                    raise ValueError(f"画像ファイルが見つかりません: {source}")
                return [YOLOX._read_image(image_path) for image_path in image_paths]
            return [YOLOX._read_image(path)]
        if isinstance(source, list):
            images = []
            for s in source:
                if isinstance(s, np.ndarray):
                    images.append(s)
                else:
                    images.append(YOLOX._read_image(Path(s)))
            return images
        raise ValueError(f"未対応の入力タイプ: {type(source)}")

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"画像を読み込めません: {path}")
        return image

    # ------------------------------------------------------------------
    # Saving and export
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model."""
        if self.model is None:
            raise RuntimeError("モデルが読み込まれていません")
        torch.save(
            {
                "model": self.model,
                "names": self._class_names,
                "nc": self._num_classes,
                "input_size": list(self._input_size),
            },
            path,
        )
        if self._verbose:
            print(f"[YOLOX] モデルを保存しました: {path}")

    def export(self, format: str = "onnx", output_path: str | None = None) -> str:
        """Export the model.

        Args:
            format: Export format; currently only ``'onnx'`` is supported.
            output_path: Destination path for the exported model.

        Returns:
            Path to the exported file.
        """
        if self.model is None:
            raise RuntimeError("モデルが読み込まれていません")
        if format.lower() != "onnx":
            raise ValueError(
                f"未対応のエクスポート形式: {format}\n"
                "現在は 'onnx' のみ対応しています。"
            )
        return self._export_onnx(output_path)

    def _export_onnx(self, output_path: str | None = None) -> str:
        if output_path is None:
            base = self._model_path or "yolox_model"
            output_path = str(Path(base).with_suffix(".onnx"))

        dummy = torch.zeros(1, 3, *self._input_size)
        prev_device = self._current_device
        model_cpu = self.model.cpu().eval()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]

        torch.onnx.export(
            model_cpu,
            (dummy,),
            output_path,
            input_names=["images"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        )

        if prev_device != "cpu":
            self.model = self.model.to(prev_device)  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
            self._current_device = prev_device

        print(f"[YOLOX] ONNX エクスポート完了: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # data.yaml loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data_config(data: str) -> dict[str, Any]:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"data.yaml が見つかりません: {data}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for key in ("coco_json", "images_dir"):
            if key not in cfg:
                raise ValueError(
                    f"data.yaml に '{key}' が必要です。\n"
                    "必須キー: coco_json, images_dir\n"
                    "オプション: output_dir, val_split"
                )

        # Resolve relative paths against the location of data.yaml.
        base = path.parent
        for key in ("coco_json", "images_dir", "output_dir"):
            if key in cfg:
                p = Path(cfg[key])
                if not p.is_absolute():
                    cfg[key] = str(base / p)

        return cfg
