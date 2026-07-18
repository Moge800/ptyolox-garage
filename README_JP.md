# PTYOLOX Garage

[English](https://github.com/Moge800/ptyolox-garage/blob/main/README.md)

PTYOLOX Garageは、[Pixeltable YOLOX](https://github.com/pixeltable/pixeltable-yolox)を利用した物体検出モデルの学習・動作確認・エクスポートをまとめて扱うデスクトップ/Pythonツールキットです。Ultralytics風のAPIと、日本語・英語対応のtkinter GUIを提供します。

## 主な機能

- Label StudioのCOCOエクスポートを学習データへ変換
- YOLOX nano、tiny、s、m、l、xの段階的エポック学習
- 画像、NumPy配列、ディレクトリ、USBカメラからの推論
- 学習済みモデルのONNXエクスポート
- GUIの日本語・英語切り替え
- CPU/GPU環境ごとの設定プロファイル

## 必要環境

- Python 3.10–3.13
- 主なサポート対象はWindows 11
- 学習にはNVIDIA CUDA GPUを推奨。CPU推論にも対応

Linuxでは、OSのパッケージマネージャーからtkinterの追加インストールが必要な場合があります。

## インストール

PyPIからインストール:

```bash
pip install ptyolox-garage
```

[uv](https://docs.astral.sh/uv/)を使用した開発環境:

```bash
git clone https://github.com/Moge800/ptyolox-garage.git
cd ptyolox-garage
uv sync --group dev
```

PyTorchはハードウェアごとに配布物が異なります。CUDAを使用する場合は、[PyTorch公式インストールページ](https://pytorch.org/get-started/locally/)から環境に合うビルドを導入してください。

## GUI

```bash
ptyolox-garage
```

GUIには、学習、画像推論、ライブカメラ推論、ONNXエクスポートの4画面があります。初期言語はOSのロケールから選択され、Languageメニューから変更できます。

設定はOSのユーザー設定ディレクトリへ保存されます。Windowsでは標準で`%APPDATA%\ptyolox-garage\config.ini`を使用します。

## Python API

```python
from ptyolox_garage import YOLOX

# Label Studio COCOエクスポートから学習
model = YOLOX("l")
model.train(
    data="data.yaml",
    epochs=[100, 200, 300],
    device="cuda:0",
    batch=16,
)

# 推論
model = YOLOX("best_model.pt")
results = model.predict("image.jpg", conf=0.3)
annotated = results[0].plot()

# ONNXエクスポート
model.export(format="onnx")
```

## データセット設定

`data.yaml`にLabel StudioのCOCO JSONと画像ディレクトリを指定します。

```yaml
coco_json: C:/datasets/widgets/result.json
images_dir: C:/datasets/widgets/images
output_dir: C:/datasets/widgets/prepared
val_split: 0.2
```

PTYOLOX GarageはCOCOカテゴリIDの再割り当て、画像パスの検証、学習/検証データの分割、Pixeltable YOLOX用ディレクトリの生成を行います。

## モデルサイズ

| 名前 | Depth | Width |
|---|---:|---:|
| `nano` | 0.33 | 0.25 |
| `tiny` | 0.33 | 0.375 |
| `s` | 0.33 | 0.50 |
| `m` | 0.67 | 0.75 |
| `l` | 1.00 | 1.00 |
| `x` | 1.33 | 1.25 |

## 開発

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv build
```

詳しい説明は[日本語ドキュメント](https://github.com/Moge800/ptyolox-garage/tree/main/docs/jp)を参照してください。英語版は[English documentation](https://github.com/Moge800/ptyolox-garage/tree/main/docs/en)にあります。

## クレジット

PTYOLOX Garageは[Pixeltable YOLOX](https://github.com/pixeltable/pixeltable-yolox)をバックエンドとして使用し、その上流には[Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)があります。本プロジェクトは独立したプロジェクトであり、Pixeltableの公式製品ではありません。

## ライセンス

[Apache License 2.0](LICENSE)で提供します。クレジットは[NOTICE](NOTICE)を参照してください。
