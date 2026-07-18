# PTYOLOX Garage ドキュメント

PTYOLOX Garage は、[pixeltable-yolox](https://github.com/pixeltable/pixeltable-yolox) をバックエンドとした YOLOX 物体検出モデルの学習・推論・ONNX エクスポートを簡単に扱える Python ツールキットです。
ultralytics YOLO ライクな API と、tkinter ベースの GUI を提供します。

---

## 主な機能

| 機能 | 説明 |
|------|------|
| **学習** | Label Studio の COCO エクスポートからそのまま学習可能。段階的エポックスケジュール対応 |
| **推論** | 画像ファイル・NumPy 配列・ディレクトリ指定で推論。NMS とレターボックス処理を内蔵 |
| **リアルタイム推論** | USB カメラでのライブ検出を GUI からワンクリックで実行 |
| **ONNX エクスポート** | 学習済み `.pt` モデルを ONNX 形式に変換 |
| **設定プロファイル** | `config.ini` で複数環境（開発 PC・工場 PC 等）の設定を管理 |
| **GUI** | 学習・推論・カメラ・エクスポートの 4 タブ構成 |

---

## ドキュメント構成

- [インストール](installation.md) — 環境構築と依存パッケージの導入
- [クイックスタート](quickstart.md) — 最短で動かすためのガイド
- [設定ファイル](configuration.md) — `config.ini` のプロファイル管理
- [GUI ガイド](gui.md) — GUI アプリケーションの使い方
- API リファレンス
  - [YOLOX (wrapper)](api/wrapper.md) — メインラッパークラス
  - [AppConfig / ProfileParams (config)](api/config.md) — 設定管理
  - [DatasetPreparer (dataset)](api/dataset.md) — データセット準備
  - [_YOLOXTrainer (trainer)](api/trainer.md) — 内部学習エンジン
- [開発ガイド](development.md) — テスト・コーディング規約・コントリビューション
