# 開発ガイド

## 開発環境のセットアップ

```bash
git clone https://github.com/Moge800/ptyolox-garage.git
cd ptyolox-garage
uv sync --group dev
```

---

## テストの実行

```bash
uv run pytest
```

カバレッジ付きで実行:

```bash
uv run pytest --cov=src/ptyolox_garage
```

### テスト構成

| ファイル | 対象 | 内容 |
|---------|------|------|
| `tests/test_config.py` | `config.py` | AppConfig の読み書き、プロファイル管理 |
| `tests/test_dataset.py` | `dataset.py` | DatasetPreparer の分割・リマップ処理 |
| `tests/test_wrapper.py` | `wrapper.py` | モデル初期化、レターボックス、NMS、推論結果 |

> `test_wrapper.py` は YOLOX パッケージ本体がなくてもテスト可能な範囲をカバーしています。

---

## リンター・フォーマッター

```bash
uv run ruff check .        # リントチェック
uv run ruff format .       # フォーマット
```

---

## プロジェクト構造

```
ptyolox_garage/
├── main.py                     # GUI エントリポイント
├── config.example.ini                  # アプリケーション設定
├── pyproject.toml              # パッケージ設定
├── src/
│   └── ptyolox_garage/
│       ├── __init__.py         # パッケージエクスポート
│       ├── config.py           # 設定管理 (AppConfig, ProfileParams)
│       ├── dataset.py          # データセット準備 (DatasetPreparer)
│       ├── wrapper.py          # メインラッパー (YOLOX, YOLOXResult, YOLOXBoxes)
│       ├── _trainer.py         # 内部学習エンジン (_YOLOXTrainer)
│       └── gui/
│           ├── __init__.py
│           ├── app.py          # メインウィンドウ (App)
│           ├── train_tab.py    # 学習タブ (TrainTab)
│           ├── infer_tab.py    # 推論タブ (InferTab)
│           ├── camera_tab.py   # カメラタブ (CameraTab)
│           └── export_tab.py   # エクスポートタブ (ExportTab)
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   └── test_wrapper.py
└── docs/                       # ドキュメント
```

---

## 依存関係の構造

```
wrapper.py ──→ _trainer.py ──→ dataset.py
    │
    └──→ dataset.py

config.py （独立）

gui/app.py ──→ gui/*_tab.py ──→ wrapper.py, config.py
```

---

## ライセンス

Apache License 2.0 — 詳細は [LICENSE](../../LICENSE) を参照してください。

---

## PyPIリリース

PyPI公開にはTrusted Publishingを使用します。PyPI側のPublisherには以下を設定します。

| 設定 | 値 |
|---|---|
| Owner | `Moge800` |
| Repository | `ptyolox-garage` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

パッケージversionはGitタグから自動生成されます。`v<version>`形式のタグで
GitHub Releaseを公開してください。たとえば、`v0.2.1`からはパッケージversion
`0.2.1`が生成されます。workflowは生成したwheelのversionとReleaseタグが一致する
ことを検証します。手動実行では`main`からビルドし、既存のPyPIファイルはスキップ
されます。GitHub Releaseでは同一versionのファイルがPyPIに既に存在すると失敗し、
Release成果物との不一致を防ぎます。Actionsは対象のReleaseタグまたは`main`
コミットに対してテストとRuffを実行し、wheel/sdist生成、PyPI公開、Release起点の
場合はReleaseへの成果物添付を実行します。
