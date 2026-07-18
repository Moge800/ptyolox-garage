# API Reference — AppConfig / ProfileParams (config)

The `ptyolox_garage.config` module manages application settings through `config.ini`.

---

## `ProfileParams` Dataclass

A dataclass that holds parameters for a single profile.

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

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `str` | `"cpu"` | Device to use (`"cpu"` / `"cuda:0"`, etc.) |
| `model_size` | `str` | `"l"` | Model size (`nano`/`tiny`/`s`/`m`/`l`/`x`) |
| `batch_size` | `int` | `16` | Training batch size |
| `imgsz` | `int` | `640` | Input image size (square) |
| `workers` | `int` | `4` | DataLoader worker count |
| `val_split` | `float` | `0.2` | Validation data split ratio |
| `output_dir` | `str` | `""` | Output directory |
| `conf` | `float` | `0.25` | Inference confidence threshold |
| `iou` | `float` | `0.45` | NMS IoU threshold |

---

## `AppConfig` Class

Manages reading/writing `config.ini` and profile management.

### Constructor

```python
AppConfig(config_path: str | Path | None = None)
```

| Parameter | Description |
|-----------|-------------|
| `config_path` | Path to `config.ini`. Uses default path if `None` |

---

### Methods

#### `load()`

```python
def load(self) -> None
```

Loads `config.ini`. Creates a new file with defaults if it doesn't exist.

#### `save()`

```python
def save(self) -> None
```

Writes the current settings to `config.ini`.

#### `profiles()`

```python
def profiles(self) -> list[str]
```

Returns a list of all defined profile names.

#### `get()`

```python
def get(self, profile: str = "default") -> ProfileParams
```

Returns the parameters for the specified profile as `ProfileParams`.  
Falls back to `"default"` if the specified profile doesn't exist.

#### `set()`

```python
def set(self, profile: str, key: str, value: str) -> None
```

Updates a single key in the specified profile.

#### `set_params()`

```python
def set_params(self, profile: str, params: ProfileParams) -> None
```

Writes all fields of `ProfileParams` at once.

#### `add_profile()`

```python
def add_profile(self, profile: str) -> None
```

Adds a new profile with default values.

#### `remove_profile()`

```python
def remove_profile(self, profile: str) -> bool
```

Removes the specified profile. Cannot remove `"default"`, returns `False`.

---

## Usage Example

```python
from ptyolox_garage import AppConfig, ProfileParams

# Load
config = AppConfig()
config.load()

# List profiles
print(config.profiles())  # ["default", "factory_pc", ...]

# Get parameters
params = config.get("factory_pc")
print(params.device)       # "cuda:0"
print(params.batch_size)   # 16

# Update individual value
config.set("factory_pc", "batch_size", "32")

# Bulk update
new_params = ProfileParams(device="cuda:1", model_size="x", batch_size=64)
config.set_params("factory_pc", new_params)

# Profile operations
config.add_profile("test_env")
config.remove_profile("test_env")  # True

# Save
config.save()
```
