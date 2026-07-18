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

### ONNX エクスポート

```python
model = YOLOX("best_model.pt")
onnx_path = model.export(format="onnx")
print(f"Exported to {onnx_path}")
```

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

学習には、Label Studio からエクスポートした COCO 形式の JSON と画像ディレクトリを指定する `data.yaml` が必要です。

```yaml
coco_json: /path/to/result.json    # Label Studio COCO エクスポート JSON
images_dir: /path/to/images        # 画像ディレクトリ

# オプション
output_dir: /path/to/output        # 出力先（省略時はカレントディレクトリ）
val_split: 0.2                     # 検証データの割合（デフォルト: 0.2）
```

> `coco_json` と `images_dir` は相対パスの場合、`data.yaml` のあるディレクトリを基準に解決されます。
