# API リファレンス — _YOLOXTrainer (trainer)

`ptyolox_garage._trainer` モジュールは、YOLOX の学習実行を管理する内部モジュールです。

> このモジュールはプライベート API です。通常は `YOLOX.train()` を通じて間接的に利用されます。

---

## `_YOLOXTrainer` クラス

### コンストラクタ

```python
_YOLOXTrainer(
    model_size: str,
    num_classes: int,
    dataset_dir: str,
    output_dir: str,
    input_size: tuple[int, int] = (640, 640),
    batch_size: int = 16,
    device: str = "cpu",
    num_workers: int = 4,
    basic_lr_per_img: float = 0.01 / 64,
    pretrained_weights: str | None = None,
)
```

| 引数 | 説明 |
|------|------|
| `model_size` | モデルサイズ（`"nano"` ～ `"x"`） |
| `num_classes` | 検出クラス数 |
| `dataset_dir` | `DatasetPreparer` が出力したデータセットディレクトリ |
| `output_dir` | チェックポイント等の出力先 |
| `input_size` | 入力画像サイズ `(H, W)` |
| `batch_size` | 学習バッチサイズ |
| `device` | 使用デバイス |
| `num_workers` | DataLoader ワーカー数 |
| `basic_lr_per_img` | 画像あたりの基本学習率 |
| `pretrained_weights` | 事前学習済みウェイトのパス |

---

### `train_sequential()`

```python
def train_sequential(
    self,
    epoch_schedule: list[int],
    on_log: Callable[[str], None] | None = None,
    on_stage_done: Callable[[int, int, str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> str
```

エポックスケジュールに従って段階的に学習を実行します。

| 引数 | 説明 |
|------|------|
| `epoch_schedule` | 段階的エポック数のリスト（例: `[100, 200, 300]`） |
| `on_log` | ログコールバック `(text) -> None` |
| `on_stage_done` | ステージ完了コールバック `(stage_idx, epoch, ckpt_path) -> None` |
| `stop_event` | 中断シグナル（`threading.Event`） |

**戻り値:** 最終チェックポイントのパス

**段階的学習の仕組み:**

```
epoch_schedule = [100, 200, 300]

ステージ 0: エポック 1〜100   (resume=False)
ステージ 1: エポック 101〜200 (resume=True)
ステージ 2: エポック 201〜300 (resume=True)
```

各ステージ間でチェックポイントが保存され、次のステージは前のステージから再開します。

---

### `package_model()`

```python
def package_model(
    self,
    class_names: dict[int, str],
    checkpoint_path: str | None = None,
    output_model_path: str | None = None,
) -> str
```

学習チェックポイントを YOLOX 互換の `.pt` ファイルにパッケージングします。

| 引数 | 説明 |
|------|------|
| `class_names` | クラス ID → クラス名のマッピング |
| `checkpoint_path` | 変換元のチェックポイント。`None` の場合はベストを自動検索 |
| `output_model_path` | 出力先パス。`None` の場合は自動生成 |

**戻り値:** 出力ファイルのパス

---

## `_build_config()` 関数

```python
def _build_config(
    name: str,
    num_classes: int,
    data_dir: str,
    input_size: tuple[int, int],
    depth: float,
    width: float,
    max_epoch: int,
    num_workers: int,
    basic_lr_per_img: float,
    train_ann: str,
    val_ann: str,
    output_dir: str,
)
```

`YoloxConfig` サブクラスを動的に生成し、学習設定（エポック数、データパス等）を組み込んだ `YoloxConfig` インスタンスを返します。

---

## ログ管理

### `_LogRedirector`

stdout/stderr をコールバック関数にリダイレクトするコンテキストマネージャーです。  
学習中のログ出力を GUI のテキストウィジェットに転送するために使用されます。

```python
with _LogRedirector(callback=on_log):
    # この中の print 出力が on_log に転送される
    trainer.train(...)
```

### `_CallbackStream`

`TextIOBase` のサブクラスで、`write()` 呼び出しをコールバック関数に転送します。
