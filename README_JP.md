# YOLOX Wrapper

[pixeltable-yolox](https://github.com/pixeltable/pixeltable-yolox) をバックエンドとした YOLOX 物体検出モデルの学習・推論・ONNX エクスポートを簡単に扱える Python ツールキットです。  
ultralytics YOLO ライクな API と tkinter ベースの GUI を提供します。

## 主な機能

- **学習** — Label Studio の COCO エクスポートから直接学習。段階的エポックスケジュール対応
- **推論** — 画像ファイル・NumPy 配列・ディレクトリを指定して推論
- **リアルタイム推論** — USB カメラでのライブ物体検出（GUI）
- **ONNX エクスポート** — `.pt` モデルを ONNX 形式に変換
- **設定プロファイル** — `config.ini` で複数環境の設定を管理
- **GUI** — 学習・推論・カメラ・エクスポートの 4 タブ構成

## 必要環境

- Python **3.10** 以上
- [uv](https://docs.astral.sh/uv/) (推奨)

## インストール

```bash
git clone https://github.com/Moge800/yolox_wrapper.git
cd yolox_wrapper
uv sync
```

## クイックスタート

### コード API

```python
from yolox_wrapper import YOLOX

# 学習
model = YOLOX("l")
model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0", batch=16)

# 推論
model = YOLOX("best_model.pt")
results = model.predict("image.jpg", conf=0.3)
for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    annotated = r.plot()

# ONNX エクスポート
model.export(format="onnx")
```

### GUI

```bash
uv run yolox-gui
```

## data.yaml の書き方

```yaml
coco_json: /path/to/result.json    # Label Studio COCO エクスポート JSON
images_dir: /path/to/images        # 画像ディレクトリ
output_dir: /path/to/output        # 出力先（オプション）
val_split: 0.2                     # 検証データの割合（オプション）
```

## モデルサイズ

| サイズ | depth | width |
|--------|-------|-------|
| `nano` | 0.33 | 0.25 |
| `tiny` | 0.33 | 0.375 |
| `s` | 0.33 | 0.50 |
| `m` | 0.67 | 0.75 |
| `l` | 1.00 | 1.00 |
| `x` | 1.33 | 1.25 |

## 設定プロファイル

`config.ini` で複数の環境設定を管理できます。

```ini
[default]
device = cpu
model_size = l
batch_size = 16
imgsz = 640

[factory_pc]
device = cuda:0
model_size = l
batch_size = 16
imgsz = 640
```

## テスト

```bash
uv run pytest
```

## ドキュメント

詳細なドキュメントは [docs/](docs/jp/index.md) を参照してください。

- [インストール](docs/jp/installation.md)
- [クイックスタート](docs/jp/quickstart.md)
- [設定ファイル](docs/jp/configuration.md)
- [GUI ガイド](docs/jp/gui.md)
- API リファレンス: [YOLOX](docs/jp/api/wrapper.md) / [AppConfig](docs/jp/api/config.md) / [DatasetPreparer](docs/jp/api/dataset.md) / [Trainer](docs/jp/api/trainer.md)
- [開発ガイド](docs/jp/development.md)

## ライセンス

[MIT License](LICENSE)
