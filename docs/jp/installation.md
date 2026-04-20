# インストール

## 必要環境

- Python **3.10** 以上
- [uv](https://docs.astral.sh/uv/) (推奨パッケージマネージャー)

## 手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/Moge800/yolox_wrapper.git
cd yolox_wrapper
```

### 2. 仮想環境の作成と依存パッケージのインストール

```bash
uv sync
```

> `uv sync` により `pyproject.toml` に定義されたすべての依存パッケージ（pixeltable-yolox を含む）が自動的にインストールされます。

### 3. 動作確認

```bash
uv run python -c "from yolox_wrapper import YOLOX; print('OK')"
```

---

## 主な依存パッケージ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| torch | >= 2.5.1 | ディープラーニングフレームワーク |
| torchvision | >= 0.20.1 | NMS 等の画像処理ユーティリティ |
| opencv-python | >= 4.8 | 画像の読み込み・描画 |
| numpy | >= 1.24, < 2.0 | 数値計算 |
| pillow | >= 10.0 | GUI での画像表示 |
| pyyaml | >= 6.0 | data.yaml のパース |
| pixeltable-yolox | >= 0.4.2 | YOLOX モデル定義・学習ロジック (Python 3.9–3.13 対応) |
| onnx | >= 1.16 | ONNX エクスポート |
| ml-dtypes | >= 0.5.0 | onnx の依存ライブラリ |
| beep-lite | >= 0.1.3 | 学習完了通知音 |

### 開発用パッケージ

```bash
uv sync --group dev
```

| パッケージ | 用途 |
|-----------|------|
| pytest | テスト実行 |
| pytest-cov | カバレッジ計測 |
| ruff | リンター・フォーマッター |

---

## GPU サポート

CUDA 対応 GPU を使用する場合は、PyTorch の CUDA ビルドがインストールされていることを確認してください。  
`config.ini` で `device = cuda:0` を設定すると、GPU を利用した学習・推論が可能になります。

```python
import torch
print(torch.cuda.is_available())  # True であればGPU利用可能
```
