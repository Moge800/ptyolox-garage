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
YOLOX(model: str | os.PathLike[str], verbose: bool = True)
```

| 引数 | 説明 |
|------|------|
| `model` | モデルサイズ文字列（`"l"` 等）またはcheckpointファイルパス |
| `verbose` | 詳細ログ出力の有効化 |

**動作:**
- サイズ文字列の場合 → 未学習のモデルアーキテクチャを構築
- `.pt`や`.pth`などのcheckpointパスの場合 → checkpointからモデルと設定を復元

既知のモデルサイズは同名のファイルより優先されます。サイズ名と衝突する拡張子なしの
checkpointを読み込む場合は、`Path`オブジェクトまたは`./l`のような明示的な相対パスを
指定してください。存在しないパス形式の値は`FileNotFoundError`、パスではない未知の値は
モデルサイズの`ValueError`になります。

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
    stop_event: threading.Event | None = None,
    package_to_cpu: bool = True,
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
| `stop_event` | 次のステージ開始前に停止するシグナル。実行中のステージは完了します |
| `package_to_cpu` | 配布しやすいよう最終 `.pt` をCPU tensorで保存 |

**戻り値:** `YOLOX` (メソッドチェーン用)

**例外:** ステージ開始前に `stop_event` が設定されている場合は `TrainingStopped`。

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
) -> list[YOLOXResult]
```

画像に対して推論を実行します。

| 引数 | 説明 |
|------|------|
| `source` | 画像パス、非再帰の画像ディレクトリ、NumPy 配列、またはそれらのリスト |
| `conf` | 信頼度しきい値 |
| `iou` | NMS の IoU しきい値 |
| `device` | 使用デバイス |
| `verbose` | 詳細ログ |

**戻り値:** `list[YOLOXResult]` (画像ごとに 1 つ)

---

### `export()`

```python
def export(self, format: str = "onnx", output_path: str | None = None) -> str
```

モデルを指定形式にエクスポートします。

| 引数 | 説明 |
|------|------|
| `format` | 出力形式（現在は `"onnx"` のみ対応） |
| `output_path` | 出力先。省略時は読み込んだモデルパスから生成 |

**戻り値:** 出力ファイルのパス

ONNXグラフのインターフェースは次の仕様で固定します。

- `images`: `float32[batch, 3, height, width]`。BGR、値域`0..255`、
  アスペクト比を維持したletterbox処理済みで余白値は`114`
- letterboxではリサイズ画像を左上へ配置し、余白は右側と下側だけに追加。座標復元は
  x/y方向のpadding offsetを引かず、リサイズ比率による除算だけを実行
- `output`: `float32[batch, anchors, 5 + num_classes]`。center-x、center-y、
  width、height、object confidence、各class confidenceを格納
- Opset 11、batch軸のみ可変、入力解像度はcheckpointの設定で固定
- confidence filtering、元画像座標への復元、class-aware NMSはONNXの外で実行

各候補のclass IDには最大class confidenceのindexを使用します。最終的な検出scoreは
`objectness * max(class confidence)`です。この合成後のscoreを設定されたconfidence
閾値と比較し、class-aware NMSでも同じscoreを候補の順位付けに使用します。

エクスポートはCPUで実行します。成功・失敗にかかわらず、メモリ上のモデルは元の
deviceとtrain/eval状態へ戻ります。ONNX検証に成功した完成ファイルだけが指定した
出力先を置き換えます。

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
def save(self, path: str, *, to_cpu: bool = True) -> None
```

モデルをメタデータ付き `.pt` ファイルとして保存します。既定では登録済みの
parametersとbuffersをすべてCPUに配置して保存するため、CPU専用の実機でも
読み込めます。保存後、メモリ上のモデルは元のdeviceとtrain/eval状態へ戻ります。

| 引数 | 説明 |
|------|------|
| `path` | 出力する `.pt` のパス |
| `to_cpu` | `True`ならCPU tensorで保存、`False`なら現在のdeviceを維持 |

---

## `YOLOXResult` クラス

推論結果のコンテナです。ultralytics の `Results` 風インターフェースを提供します。

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
