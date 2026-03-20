# How to Run Experiments

All configuration is managed with **Hydra** — override any parameter on the command line.

---

# Preprocessing

```bash
# Single file
uv run preprocess input_path=data/nii_test/file.nii preprocessing=light

# Full PPMR dataset (recursive)
uv run preprocess input_path=data/PPMR preprocessing=light recursive=true
# → data/PPMR_processed/PMGcases/<subject_id>/
# → data/PPMR_processed/controlcases/<subject_id>/
```

## Presets

| Preset | Bilateral | CLAHE | Canny blend |
|--------|-----------|-------|-------------|
| `default` | d=9, σ=75/75 | 2.0 | 0.20 |
| `light` | d=5, σ=50/50 | 2.0 | 0.15 |
| `minimal` | d=3, σ=25/25 | 1.5 | 0.05 |
| `no_filter` | d=1, σ=1/1 | 2.0 | 0.00 |
| `no_bilateral` | skipped | 1.5 | 0.05 |

`edge_first=true` swaps bilateral and canny order (default: bilateral → canny).

## Key Parameters

```
normalization.method       min_max | zscore
clahe.clip_limit           float
bilateral.diameter         int
canny.blend_alpha          float  (0.0 = no edge blending)
```

## Multirun

```bash
uv run preprocess -m input_path=data/nii_test/file.nii bilateral.diameter=3,5,7,9
uv run preprocess -m input_path=data/nii_test/file.nii preprocessing=default,light,minimal
```

---

# Training

```bash
uv run train                                             # defaults: ResNet101, 20 epochs, lr=1e-4
uv run train model.name=densenet201
uv run train train.num_epochs=30 train.learning_rate=1e-3
uv run train data_loader.train_raw=true                  # skip preprocessing
uv run train data_loader.augment=false                   # no training-time augmentation
uv run train data_loader.data_dir=data/PPMR_light        # different preprocessing preset
```

## Parameters

### `model.*`
```
model.name             resnet101 | densenet201
model.dropout_p        float
model.freeze_backbone  true | false
```

### `train.*`
```
train.batch_size       int     (default: 32)
train.num_epochs       int     (default: 20)
train.learning_rate    float   (default: 1e-4)
train.weight_decay     float   (default: 1e-5)
train.device           cuda | mps | cpu
train.val_frac         float   (default: 0.2)
train.test_frac        float   (default: 0.2)
train.seed             int     (default: 42)
```

### `data_loader.*`
```
data_loader.data_dir           str    (preprocessed root, default: data/PPMR_default)
data_loader.raw_data_dir       str    (raw PPMR root,    default: data/PPMR)
data_loader.train_raw          true | false   (default: false)
data_loader.augment            true | false   (default: true)
data_loader.pmg_negative_mode  correct | paper
```

`pmg_negative_mode`:
- `correct` — label=2 → HC, label=3 excluded
- `paper` — all PMG-folder slices → positive (replicates Guha & Bhandage 2025)

## Multirun

```bash
uv run train -m model.name=resnet101,densenet201
uv run train -m train.learning_rate=1e-3,1e-4,1e-5
uv run train -m model.name=resnet101,densenet201 train.learning_rate=1e-3,1e-4
```

## Outputs

**Checkpoints** — saved to `results/checkpoints/`:
```
<model>_<raw|preprocessed>_best.pt    ← best val loss
<model>_<raw|preprocessed>_final.pt   ← last epoch
```

**Metrics CSV** — saved to `results/metrics/<model>_<raw|preprocessed>_metrics.csv`:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `{split}_loss` | BCE loss |
| `{split}_acc` | Accuracy |
| `{split}_precision` | Precision |
| `{split}_recall` | Recall |
| `{split}_f1` | F1 score |
| `{split}_kappa` | Cohen's Kappa |

`{split}` = `train`, `val`, `test`. Written row-by-row so partial runs are recoverable.

Override output directory:
```bash
uv run train checkpoint_dir=results/my_run
```

## Post-hoc Evaluation

```python
from src.func.evaluation import evaluate_model
metrics = evaluate_model(model, test_loader, device, split="test")
# prints accuracy, precision, recall, F1, Cohen's Kappa + confusion matrix counts
```

---

## Key Files

| File | Purpose |
|------|---------|
| `hydra/config.yaml` | Main Hydra config |
| `hydra/preprocessing/*.yaml` | Preprocessing presets |
| `hydra/model/model.yaml` | Model defaults |
| `hydra/model/train.yaml` | Training defaults |
| `hydra/model/data_loader.yaml` | Data pipeline defaults |
| `src/cli/train.py` | Training entry point |
| `src/func/models/get_train.py` | Training logic + CSV logging |
| `src/func/data/get_loader.py` | Dataset & dataloader |
| `src/func/evaluation/classification_metrics.py` | Acc, precision, recall, F1, Cohen's Kappa |
