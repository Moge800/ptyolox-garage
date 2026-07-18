# 設定ファイル (`config.ini`)

PTYOLOX Garage は `config.ini` ファイルでアプリケーション設定を管理します。
複数のプロファイルを定義し、GUI のドロップダウンで切り替えて使用できます。

設定ファイルはOS標準のユーザー設定ディレクトリへ保存されます。Windowsでの
標準パスは`%APPDATA%\ptyolox-garage\config.ini`です。リポジトリ内の
`config.example.ini`は参照用であり、実行時には直接使用しません。

## ファイル形式

```ini
; PTYOLOX Garage 設定ファイル

[default]
device = cpu
model_size = l
batch_size = 16
imgsz = 640
workers = 4
val_split = 0.2
output_dir =
conf = 0.25
iou = 0.45
language = auto

[factory_pc]
device = cuda:0
model_size = l
batch_size = 16
imgsz = 640
workers = 4
val_split = 0.2
output_dir =
conf = 0.25
iou = 0.45
```

## パラメータ一覧

| キー | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `device` | str | `cpu` | 使用デバイス。`cpu` または `cuda:0` 等 |
| `model_size` | str | `l` | モデルサイズ。`nano` / `tiny` / `s` / `m` / `l` / `x` |
| `batch_size` | int | `16` | 学習時のバッチサイズ |
| `imgsz` | int | `640` | 入力画像サイズ（正方形） |
| `workers` | int | `4` | DataLoader のワーカー数 |
| `val_split` | float | `0.2` | 検証データの割合（0.0〜1.0） |
| `output_dir` | str | `""` | 出力先ディレクトリ（空欄の場合はデフォルト） |
| `conf` | float | `0.25` | 推論時の信頼度しきい値 |
| `iou` | float | `0.45` | NMS の IoU しきい値 |
| `language` | str | `auto` | GUI言語。`auto` / `ja` / `en` |

## プロファイル

- `[default]` セクションは必須です。削除できません。
- 任意のセクション名でプロファイルを追加できます（例: `[factory_pc]`, `[dev_pc]`）。
- GUI のプロファイルドロップダウンで切り替えると、各タブの設定が自動的に反映されます。
- 存在しないプロファイルを指定した場合は `[default]` にフォールバックします。

## コードからの操作

```python
from ptyolox_garage import AppConfig, ProfileParams

config = AppConfig()          # config.ini を読み込み
config.load()

# プロファイル一覧
print(config.profiles())     # ["default", "factory_pc", "dev_pc"]

# 値の取得
params = config.get("factory_pc")
print(params.device)          # "cuda:0"

# 値の更新
config.set("factory_pc", "batch_size", "32")
config.save()

# プロファイルの追加・削除
config.add_profile("new_env")
config.remove_profile("dev_pc")
```
