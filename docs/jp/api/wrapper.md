# API リファレンス — YOLOX (wrapper)

`ptyolox_garage.wrapper` モジュールは、YOLOX モデルの読み込み・推論・学習・エクスポートを統合するメインクラスを提供します。

---

## モデルサイズ

| サイズ名 | depth | width |
|---------|-------|-------|
| `nano` | 0.33 | 0.25 |
| `tiny` | 0.33 | 0.375 |
| `s` | 0.33 | 0.50 |
| `m` | 0.67 | 0.75 |
| `l` | 1.00 | 1.00 |
| `x` | 1.33 | 1.25 |

> サイズ文字列は `"l"`, `"yolox_l"`, `"yolox-l"` のいずれでも受け付けます。

---

## `YOLOX` クラス

### コンストラクタ

```python
YOLOX(model: str, verbose: bool = True)
```

| 引数 | 説明 |
|------|------|
| `model` | モデルサイズ文字列（`"l"` 等）または `.pt` ファイルパス |
| `verbose` | 詳細ログ出力の有効化 |

**動作:**
- サイズ文字列の場合 → 未学習のモデルアーキテクチャを構築
- `.pt` パスの場合 → チェックポイントからモデルと設定を復元

---

### `train()`

```python
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
    **kwargs: Any,
) -> "YOLOX"
```

YOLOX モデルを学習します。エポックスケジュールに従って段階的に学習を実行します。

| 引数 | 説明 |
|------|------|
| `data` | `data.yaml` のパス |
| `epochs` | 総エポック数または段階的スケジュール（例: `[100, 200, 300]`） |
| `batch` | バッチサイズ |
| `device` | 使用デバイス（`"cpu"` / `"cuda:0"`） |
| `imgsz` | 入力画像サイズ |
| `workers` | DataLoader ワーカー数 |
| `val_split` | 検証データの分割比率 |
| `pretrained_weights` | 事前学習済みウェイトのパス |
| `on_log` | ログ出力コールバック `(text: str) -> None` |
| `on_stage_done` | ステージ完了コールバック `(stage_idx, epoch, ckpt_path) -> None` |

**戻り値:** `YOLOX` (メソッドチェーン用)

---

### `predict()`

```python
def predict(
    self,
    source: str | Path | np.ndarray | list,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    verbose: bool = False,
    save: bool = False,
    **kwargs: Any,
) -> list[YOLOXResult]
```

画像に対して推論を実行します。

| 引数 | 説明 |
|------|------|
| `source` | 画像パス、NumPy 配列、またはそれらのリスト |
| `conf` | 信頼度しきい値 |
| `iou` | NMS の IoU しきい値 |
| `device` | 使用デバイス |
| `verbose` | 詳細ログ |
| `save` | 結果画像の保存 |

**戻り値:** `list[YOLOXResult]` (画像ごとに 1 つ)

---

### `export()`

```python
def export(self, format: str = "onnx", **kwargs: Any) -> str
```

モデルを指定形式にエクスポートします。

| 引数 | 説明 |
|------|------|
| `format` | 出力形式（現在は `"onnx"` のみ対応） |

**戻り値:** 出力ファイルのパス

---

### `fuse()`

```python
def fuse(self) -> "YOLOX"
```

BatchNorm レイヤーを畳み込みに融合し、推論を高速化します。

**戻り値:** `YOLOX` (メソッドチェーン用)

---

### `save()`

```python
def save(self, path: str) -> None
```

モデルをメタデータ付き `.pt` ファイルとして保存します。

---

## `YOLOXResult` クラス

推論結果のコンテナです。ultralytics の `Results` と互換性のあるインターフェースを提供します。

### 属性

| 属性 | 型 | 説明 |
|------|-----|------|
| `boxes` | `YOLOXBoxes` | 検出ボックスのコレクション |
| `names` | `dict[int, str]` | クラス ID → クラス名のマッピング |
| `orig_shape` | `tuple[int, int]` | 元画像のサイズ `(H, W)` |
| `orig_img` | `np.ndarray \| None` | 元画像（BGR） |

### `plot()`

```python
def plot(self, orig_img: np.ndarray | None = None) -> np.ndarray
```

検出結果をバウンディングボックスとして描画した画像を返します。

---

## `YOLOXBoxes` クラス

検出ボックスのコレクションです。

### 属性

| 属性 | 型 | 説明 |
|------|-----|------|
| `xyxy` | `torch.Tensor` | 座標 `[N, 4]`（x1, y1, x2, y2） |
| `conf` | `torch.Tensor` | 信頼度スコア `[N]` |
| `cls` | `torch.Tensor` | クラス ID `[N]` |

### メソッド

- `__len__() -> int` — ボックス数
- `__iter__()` — 個別ボックス (`_YOLOXBox`) のイテレーション

---

## ユーティリティ関数

### `_letterbox()`

```python
def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    fill_value: int = 114,
) -> tuple[np.ndarray, float]
```

アスペクト比を保ったままレターボックスリサイズを行います。

### `_postprocess()`

```python
def _postprocess(
    outputs: torch.Tensor,
    ratio: float,
    orig_h: int,
    orig_w: int,
    conf_thre: float,
    iou_thre: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

YOLOX の出力をデコードし NMS を適用します。戻り値は `(boxes, scores, class_ids)` です。
