# API リファレンス — AppConfig / ProfileParams (config)

`ptyolox_garage.config` モジュールは、`config.ini` を介したアプリケーション設定の読み書きを管理します。

---

## `ProfileParams` データクラス

1 つのプロファイルのパラメータを保持するデータクラスです。

```python
@dataclass
class ProfileParams:
    device: str = "cpu"
    model_size: str = "l"
    batch_size: int = 16
    imgsz: int = 640
    workers: int = 4
    val_split: float = 0.2
    output_dir: str = ""
    conf: float = 0.25
    iou: float = 0.45
```

### フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `device` | `str` | `"cpu"` | 使用デバイス（`"cpu"` / `"cuda:0"` 等） |
| `model_size` | `str` | `"l"` | モデルサイズ（`nano`/`tiny`/`s`/`m`/`l`/`x`） |
| `batch_size` | `int` | `16` | 学習バッチサイズ |
| `imgsz` | `int` | `640` | 入力画像サイズ（正方形） |
| `workers` | `int` | `4` | DataLoader ワーカー数 |
| `val_split` | `float` | `0.2` | 検証データの分割比率 |
| `output_dir` | `str` | `""` | 出力先ディレクトリ |
| `conf` | `float` | `0.25` | 推論時の信頼度しきい値 |
| `iou` | `float` | `0.45` | NMS の IoU しきい値 |

---

## `AppConfig` クラス

`config.ini` の読み書きとプロファイル管理を行います。

### コンストラクタ

```python
AppConfig(config_path: str | Path | None = None)
```

| 引数 | 説明 |
|------|------|
| `config_path` | `config.ini` のパス。`None` の場合はデフォルトパスを使用 |

---

### メソッド

#### `load()`

```python
def load(self) -> None
```

`config.ini` を読み込みます。ファイルが存在しない場合はデフォルト値で新規作成します。

#### `save()`

```python
def save(self) -> None
```

現在の設定を `config.ini` に書き出します。

#### `profiles()`

```python
def profiles(self) -> list[str]
```

定義済みの全プロファイル名をリストで返します。

#### `get()`

```python
def get(self, profile: str = "default") -> ProfileParams
```

指定プロファイルのパラメータを `ProfileParams` として返します。  
存在しないプロファイルを指定した場合は `"default"` にフォールバックします。

#### `set()`

```python
def set(self, profile: str, key: str, value: str) -> None
```

指定プロファイルの単一キーを更新します。

#### `set_params()`

```python
def set_params(self, profile: str, params: ProfileParams) -> None
```

`ProfileParams` の全フィールドを一括で書き込みます。

#### `add_profile()`

```python
def add_profile(self, profile: str) -> None
```

デフォルト値で新しいプロファイルを追加します。

#### `remove_profile()`

```python
def remove_profile(self, profile: str) -> bool
```

指定プロファイルを削除します。`"default"` は削除できず `False` を返します。

---

## 使用例

```python
from ptyolox_garage import AppConfig, ProfileParams

# 読み込み
config = AppConfig()
config.load()

# プロファイル一覧
print(config.profiles())  # ["default", "factory_pc", ...]

# パラメータ取得
params = config.get("factory_pc")
print(params.device)       # "cuda:0"
print(params.batch_size)   # 16

# 個別値の更新
config.set("factory_pc", "batch_size", "32")

# 一括更新
new_params = ProfileParams(device="cuda:1", model_size="x", batch_size=64)
config.set_params("factory_pc", new_params)

# プロファイル操作
config.add_profile("test_env")
config.remove_profile("test_env")  # True

# 保存
config.save()
```
