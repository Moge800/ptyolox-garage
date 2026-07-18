# API リファレンス — DatasetPreparer (dataset)

`ptyolox_garage.dataset` モジュールは、Label Studio の COCO エクスポートを YOLOX の学習用ディレクトリ構造に変換します。

---

## `DatasetPreparer` クラス

### コンストラクタ

```python
DatasetPreparer(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
)
```

| 引数 | 説明 |
|------|------|
| `coco_json_path` | Label Studio がエクスポートした COCO 形式 JSON ファイルのパス |
| `images_dir` | 画像ファイルが格納されたディレクトリ |
| `output_dir` | 出力先ディレクトリ |
| `val_split` | 検証データの割合（0.0〜1.0、デフォルト 0.2） |
| `seed` | ランダム分割のシード値（デフォルト 42） |

---

### `prepare()`

```python
def prepare(self) -> tuple[dict[int, str], int]
```

データセットの準備を実行します。

**処理内容:**

1. COCO JSON を読み込み、カテゴリ ID を 0 始まりにリマップ
2. 画像パスを解決
3. 学習用・検証用に分割
4. 各スプリットのアノテーション JSON と画像ファイルを出力

**戻り値:**

| 要素 | 型 | 説明 |
|------|-----|------|
| `class_names` | `dict[int, str]` | クラス ID → クラス名のマッピング |
| `num_classes` | `int` | クラス数 |

**出力ディレクトリ構造:**

```
output_dir/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg / *.png
└── val/
    └── *.jpg / *.png
```

---

### 内部メソッド

| メソッド | 説明 |
|---------|------|
| `_load_coco()` | COCO JSON を読み込み、カテゴリ ID を 0 始まりにリマップ |
| `_resolve_image_paths()` | JSON 内のファイル名から実際のパスを解決 |
| `_split()` | 学習用と検証用にレコードを分割 |
| `_write_split(split, records)` | アノテーション JSON の書き出しと画像のコピー |

---

### エラーケース

| 条件 | 例外 |
|------|------|
| 画像が 2 枚未満 | `ValueError` |
| カテゴリが定義されていない | `ValueError` |

---

## 使用例

```python
from ptyolox_garage.dataset import DatasetPreparer

preparer = DatasetPreparer(
    coco_json_path="export/result.json",
    images_dir="export/images",
    output_dir="dataset",
    val_split=0.2,
)

class_names, num_classes = preparer.prepare()
print(f"クラス数: {num_classes}")
print(f"クラス名: {class_names}")
# クラス数: 3
# クラス名: {0: "cat", 1: "dog", 2: "bird"}
```

## Label Studio との連携

1. Label Studio でアノテーション作業を完了
2. **Export** → **COCO** 形式でエクスポート
3. エクスポートされた JSON ファイルと画像ディレクトリを `DatasetPreparer` に渡す
4. `prepare()` を実行すると YOLOX 学習用のデータセットが生成される
