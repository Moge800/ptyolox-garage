"""データセット準備モジュール

Label Studio の COCO エクスポートを YOLOX 学習構造へ変換します。

Label Studio からのエクスポート手順:
    1. Label Studio プロジェクト → Export
    2. 形式: COCO JSON を選択
    3. ZIP を展開すると以下の構造になる:
        export_dir/
        ├── result.json          ← annotations (COCO 形式)
        └── images/              ← 画像ファイル群

YOLOX が期待する出力構造:
    output_dir/
    ├── annotations/
    │   ├── instances_train.json
    │   └── instances_val.json
    ├── train/
    │   └── *.jpg / *.png ...
    └── val/
        └── *.jpg / *.png ...
"""

from __future__ import annotations

import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Any

_MODEL_CONFIGS: dict[str, dict[str, float]] = {
    "nano": {"depth": 0.33, "width": 0.25},
    "tiny": {"depth": 0.33, "width": 0.375},
    "s": {"depth": 0.33, "width": 0.50},
    "m": {"depth": 0.67, "width": 0.75},
    "l": {"depth": 1.00, "width": 1.00},
    "x": {"depth": 1.33, "width": 1.25},
}


class DatasetPreparer:
    """Label Studio COCO エクスポートを YOLOX 学習構造へ変換するクラス"""

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        output_dir: str,
        val_split: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.seed = seed

        self._coco: dict[str, Any] = {}
        self._class_names: dict[int, str] = {}
        self._num_classes: int = 0
        self._id_remap: dict[int, int] = {}

    def prepare(self) -> tuple[dict[int, str], int]:
        """データセットを整備し (クラス名辞書, クラス数) を返す"""
        print("[Dataset] COCO JSON を読み込み中...")
        self._load_coco()

        print("[Dataset] 画像ファイルを確認中...")
        image_records = self._resolve_image_paths()

        print(f"[Dataset] 有効な画像: {len(image_records)} 枚")
        print(f"[Dataset] クラス: {list(self._class_names.values())}")

        train_records, val_records = self._split(image_records)
        print(f"[Dataset] 分割: train={len(train_records)}, val={len(val_records)}")

        self._write_split("train", train_records)
        self._write_split("val", val_records)

        print("[Dataset] データセット準備完了")
        return self._class_names, self._num_classes

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------

    def _load_coco(self) -> None:
        with open(self.coco_json_path, encoding="utf-8") as f:
            self._coco = json.load(f)

        raw_cats = self._coco.get("categories", [])
        if not raw_cats:
            raise ValueError(
                "COCO JSON に categories が含まれていません。"
                "Label Studio から COCO 形式でエクスポートしてください。"
            )

        # category_id → name マッピング (0-indexed に正規化)
        # Label Studio は 1-indexed の category_id を使う場合がある
        cat_ids_sorted = sorted(c["id"] for c in raw_cats)
        self._id_remap = {orig: new for new, orig in enumerate(cat_ids_sorted)}

        self._class_names = {self._id_remap[c["id"]]: c["name"] for c in raw_cats}
        self._num_classes = len(self._class_names)

    def _resolve_image_paths(self) -> list[dict[str, Any]]:
        """JSON の file_name から実際のファイルパスを解決する"""
        ann_by_image: dict[int, list[dict]] = {}
        for ann in self._coco.get("annotations", []):
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        records: list[dict[str, Any]] = []
        missing: list[str] = []

        for img_info in self._coco.get("images", []):
            file_name = img_info["file_name"]

            candidates = [
                self.images_dir / file_name,
                self.images_dir / Path(file_name).name,
                Path(file_name),
            ]
            found = next((p for p in candidates if p.exists()), None)

            if found is None:
                missing.append(file_name)
                continue

            records.append(
                {
                    "id": img_info["id"],
                    "path": found,
                    "file_name": Path(file_name).name,
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "anns": ann_by_image.get(img_info["id"], []),
                }
            )

        if missing:
            print(
                f"[Dataset] 警告: {len(missing)} 枚の画像が見つかりませんでした "
                f"(最初の3件: {missing[:3]})"
            )

        return records

    def _split(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict], list[dict]]:
        if len(records) < 2:
            raise ValueError(
                f"データセットの画像が {len(records)} 枚しかありません。"
                "train/val 分割には最低 2 枚必要です。"
            )

        val_split = self.val_split
        if not (0.0 < val_split < 1.0):
            warnings.warn(
                f"val_split={val_split} は範囲外 (0.0 < val_split < 1.0)。"
                "デフォルト値 0.2 を使用します。",
                UserWarning,
                stacklevel=3,
            )
            val_split = 0.2

        rng = random.Random(self.seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_split))
        n_val = max(1, min(n_val, len(shuffled) - 1))
        return shuffled[n_val:], shuffled[:n_val]

    def _write_split(self, split: str, records: list[dict[str, Any]]) -> None:
        img_dir = self.output_dir / split
        ann_dir = self.output_dir / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        new_images: list[dict] = []
        new_anns: list[dict] = []
        ann_id = 0

        for new_img_id, rec in enumerate(records):
            dst = img_dir / rec["file_name"]
            if not dst.exists():
                shutil.copy2(rec["path"], dst)

            new_images.append(
                {
                    "id": new_img_id,
                    "file_name": rec["file_name"],
                    "width": rec["width"],
                    "height": rec["height"],
                }
            )

            for ann in rec["anns"]:
                new_cat_id = self._id_remap.get(ann["category_id"])
                if new_cat_id is None:
                    continue

                x, y, w, h = ann["bbox"]
                new_anns.append(
                    {
                        "id": ann_id,
                        "image_id": new_img_id,
                        "category_id": new_cat_id,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "segmentation": [],
                        "iscrowd": 0,
                        "ignore": 0,
                    }
                )
                ann_id += 1

        coco_out = {
            "images": new_images,
            "annotations": new_anns,
            "categories": [
                {"id": k, "name": v} for k, v in sorted(self._class_names.items())
            ],
            "info": {"contributor": "yolox_wrapper"},
        }

        json_path = ann_dir / f"instances_{split}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_out, f, ensure_ascii=False, indent=2)

        print(
            f"[Dataset]  {split}: {len(new_images)} 画像, {len(new_anns)} アノテーション → {json_path}"
        )
