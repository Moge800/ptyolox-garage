# クイックスタート

## コード API で使う

### 学習

```python
from ptyolox_garage import YOLOX

model = YOLOX("l")  # モデルサイズ: nano / tiny / s / m / l / x
model.train(
    data="data.yaml",          # Label Studio COCO エクスポートへのパス設定
    epochs=[100, 200, 300],    # 段階的エポックスケジュール
    device="cuda:0",           # GPU を使用
    batch=16,
    imgsz=640,
)
```

### 推論

```python
from ptyolox_garage import YOLOX

model = YOLOX("best_model.pt")            # 学習済みモデルを読み込み
results = model.predict("image.jpg", conf=0.3)

for result in results:
    print(result.boxes.xyxy)   # 検出ボックス座標
    print(result.boxes.conf)   # 信頼度スコア
    print(result.boxes.cls)    # クラス ID

    # 結果を画像に描画
    annotated = result.plot()
```

モデルサイズ情報のない旧checkpointも、上の例のように推論に使用できます。
追加学習する場合だけ、既知のアーキテクチャを指定します。

```python
legacy = YOLOX("legacy-model.pt", model_size="l")
legacy.train(data="data.yaml")
```

checkpointのファイル名からモデルサイズを推測することはありません。

### ONNX エクスポート

```python
model = YOLOX("best_model.pt")
onnx_path = model.export(format="onnx")
print(f"Exported to {onnx_path}")
```

グラフにはletterbox処理済みのBGR `float32` tensorを値域`0..255`で入力します。
出力はconfidence filteringとNMSを適用する前の候補predictionです。

---

## GUI で使う

```bash
uv run ptyolox-garage
```

または

```bash
uv run python main.py
```

GUI が起動し、4 つのタブ（学習・推論・カメラ・エクスポート）が表示されます。  
詳細は [GUI ガイド](gui.md) を参照してください。

---

## data.yaml の書き方

学習には、`result.json`と`images/`が並ぶLabel Studioのエクスポートディレクトリ、またはCOCO JSONと画像ディレクトリを指定できます。

```python
model.train(data="export")
model.train(data="export/result.json", images_dir="export/images")
```

従来の`data.yaml`形式も利用できます。

```yaml
coco_json: /path/to/result.json    # Label Studio COCO エクスポート JSON
images_dir: /path/to/images        # 画像ディレクトリ

# オプション
output_dir: /path/to/output        # 作業ディレクトリ（省略時は ./yolox_work）
val_split: 0.2                     # 検証データの割合（デフォルト: 0.2）
```

> `coco_json` と `images_dir` は相対パスの場合、`data.yaml` のあるディレクトリを基準に解決されます。

JSON・ディレクトリ入力時は`train(output_dir=...)`で作業ディレクトリを変更できます。
