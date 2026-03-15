# How to Run Experiments

Configuration is managed with **Hydra**. All results are saved to `results/` automatically.

---

## Quick Start

`input_path` accepts **a single file or a directory**.  When a directory is
given, every `.nii`, `.nii.gz`, `.jpg`, and `.jpeg` file inside is processed.

### Single NIfTI file
```bash
uv run preprocess input_path=data/nii_test/BraTS20_Training_022_t1.nii preprocessing=light

# Specify an axial slice (default: middle slice)
uv run preprocess input_path=data/nii_test/BraTS20_Training_022_t1.nii slice_idx=80
```

### Single JPEG file
```bash
# slice_idx is ignored for JPEG inputs
uv run preprocess \
    input_path=data/PPMR/PMGstudycaseslabelled/2/2cor_1/2cor_1_100_1.jpg \
    preprocessing=light
```

### All NIfTI files in a directory (flat)
```bash
uv run preprocess input_path=data/nii_test preprocessing=light
# → outputs written to results/nii_test_light/
```

### All JPEGs in a patient folder (flat)
```bash
uv run preprocess \
    input_path=data/PPMR/PMGstudycaseslabelled/2/2cor_1 \
    preprocessing=light
# → outputs written to data/PPMR_processed/PMGcases/2cor_1/
```

### All JPEGs across the full PMG tree (recursive)
```bash
uv run preprocess \
    input_path=data/PPMR/PMGstudycaseslabelled \
    preprocessing=light \
    recursive=true
# → outputs written to data/PPMR_processed/PMGcases/<subject_id>/
```

### Full dataset — PMG cases + controls (recursive)
```bash
uvr preprocess \
    input_path=data/PPMR \
    preprocessing=light \
    recursive=true
# → data/PPMR_processed/PMGcases/<subject_id>/
# → data/PPMR_processed/controlcases/<subject_id>/
```

---

## Presets

Defined in `hydra/preprocessing/`:

| Preset | Bilateral filter | CLAHE | Canny blend | Use when |
|--------|-----------------|-------|-------------|----------|
| `default` | d=9, σ=75/75 | 2.0 | 0.20 | Replicate the paper |
| `light` | d=5, σ=50/50 | 2.0 | 0.15 | Less blur, sharper edges |
| `minimal` | d=3, σ=25/25 | 1.5 | 0.05 | Preserve original detail |
| `no_filter` | d=1, σ=1/1 | 2.0 | 0.00 | Maximum preservation |

---

## Override Parameters

Any parameter can be overridden on top of a preset:

```bash
uv run preprocess \
    input_path=data/nii_test/file.nii \
    preprocessing=light \
    bilateral.diameter=7 \
    clahe.clip_limit=1.5 \
    canny.blend_alpha=0.10
```

Full parameter list:
```
normalization.method          min_max | zscore
normalization.output_range    [0.0,1.0]
clahe.clip_limit              float  (higher = more contrast)
clahe.tile_grid_size          [8,8]
bilateral.diameter            int    (neighbourhood size)
bilateral.sigma_color         float
bilateral.sigma_space         float
canny.low_threshold           int
canny.high_threshold          int
canny.aperture_size           3 | 5 | 7
canny.blend_alpha             float  (0.0 = no edge blending)
```

---

## Parameter Sweeps (multirun)

Add `-m` and comma-separate values to run multiple experiments at once:

```bash
# Sweep bilateral filter size
uv run preprocess -m \
    input_path=data/nii_test/file.nii \
    bilateral.diameter=3,5,7,9

# Compare all presets on a JPEG
uv run preprocess -m \
    input_path=data/PPMR/PMGstudycaseslabelled/2/2cor_1/2cor_1_100_1.jpg \
    preprocessing=default,light,minimal,no_filter

# Grid search (3×2 = 6 runs)
uv run preprocess -m \
    input_path=data/nii_test/file.nii \
    bilateral.diameter=3,5,7 \
    clahe.clip_limit=1.5,2.0
```

---

## Output

### PPMR input → `data/PPMR_processed/`
```
data/PPMR_processed/
├── PMGcases/
│   └── <subject_id>/
│       └── <slice>_preprocessed_<preset>.jpg
└── controlcases/
    └── <subject_id>/
        └── <slice>_preprocessed_<preset>.jpg
```

### Non-PPMR single file → saved directly to `data/`
```
data/
└── BraTS20_Training_022_t1_preprocessed_light.jpg
```

### Multirun → timestamped subdirectories
```
results/
└── 2026-02-16/14-35-20/
    ├── 0/   ← first combination
    ├── 1/
    └── 2/
```

Each run prints quality metrics:
```
PSNR: 25.02 dB       (higher = less distortion)
SSIM: 0.8178         (closer to 1.0 = more similar)
Entropy change: ...
```

---

## Common Workflows

```bash
# Replicate the paper
uv run preprocess input_path=data/nii_test/file.nii preprocessing=default

# Ablation: disable one step at a time
uv run preprocess input_path=data/nii_test/file.nii bilateral.diameter=1   # no bilateral
uv run preprocess input_path=data/nii_test/file.nii canny.blend_alpha=0.0  # no edges
uv run preprocess input_path=data/nii_test/file.nii clahe.clip_limit=1.0   # weak CLAHE

# Reproduce a past result
cat results/BraTS20_Training_022_t1_config_light.yaml   # check saved config
uv run preprocess input_path=data/nii_test/file.nii preprocessing=light
```

---

## Key Files (Preprocessing)

| File | Purpose |
|------|---------|
| `hydra/config.yaml` | Main Hydra config |
| `hydra/preprocessing/*.yaml` | Preset definitions |
| `src/cli/preprocess.py` | CLI entry point |
| `src/func/utils/loader.py` | JPEG + NIfTI loader |
| `src/main/configurable_pipeline.py` | Pipeline logic |

---

# Training

Configuration is managed with **Hydra**. All parameters have sensible defaults — override only what you need.

---

## Quick Start

```bash
# Run with all defaults (ResNet101, 20 epochs, lr=1e-4)
uv run train

# Change model
uv run train model.name=densenet201

# Change training hyperparameters
uv run train train.num_epochs=30 train.learning_rate=1e-3

# Change data directory (e.g. different preprocessing preset)
uv run train data_loader.data_dir=data/PPMR_light
```

---

## Override Parameters

### `model.*` — Model architecture

```
model.name             resnet101 | densenet201
model.pretrained       true | false        (use ImageNet weights)
model.dropout_p        float               (dropout before classifier head)
model.freeze_backbone  true | false        (freeze conv layers, train head only)
```

### `train.*` — Training loop

```
train.batch_size       int                 (samples per batch, default: 32)
train.num_epochs       int                 (default: 20)
train.learning_rate    float               (default: 1e-4)
train.weight_decay     float               (L2 regularisation, default: 1e-5)
train.num_workers      int                 (dataloader workers, default: 4)
train.device           cuda | mps | cpu    (default: cuda; falls back to cpu if unavailable)
train.val_frac         float               (fraction of patients for val, default: 0.15)
train.test_frac        float               (fraction of patients for test, default: 0.15)
train.seed             int                 (random seed, default: 42)
```

### `data_loader.*` — Data pipeline

```
data_loader.data_dir           str         (root dir with PMGcases/ and controlcases/)
data_loader.crop_size          int         (spatial size after crop, default: 224)
data_loader.scale              [float,float]  (random crop area range, default: [0.8,1.0])
data_loader.mean               [float,float,float]  (ImageNet: [0.485,0.456,0.406])
data_loader.std                [float,float,float]  (ImageNet: [0.229,0.224,0.225])
data_loader.pmg_negative_mode  correct | paper
```

`pmg_negative_mode`:
- `correct` — label=2 (PMG patient, slice looks normal) → HC; label=3 excluded
- `paper` — replicates Guha & Bhandage 2025: all PMG-folder slices → positive

---

## Parameter Sweeps (multirun)

Add `-m` and comma-separate values to sweep multiple runs:

```bash
# Compare models
uv run train -m model.name=resnet101,densenet201

# Sweep learning rate
uv run train -m train.learning_rate=1e-3,1e-4,1e-5

# Grid search — model × learning rate (2×3 = 6 runs)
uv run train -m \
    model.name=resnet101,densenet201 \
    train.learning_rate=1e-3,1e-4,1e-5

# Compare label modes
uv run train -m data_loader.pmg_negative_mode=correct,paper
```

---

## Common Workflows

```bash
# Replicate the paper (DenseNet201, paper label mode)
uv run train model.name=densenet201 data_loader.pmg_negative_mode=paper

# Fine-tune full network (unfreeze backbone)
uv run train model.freeze_backbone=false train.learning_rate=1e-5

# Quick smoke test (few epochs, small batch)
uv run train train.num_epochs=2 train.batch_size=8

# Run locally on Apple Silicon (default is cuda for HPC)
uv run train train.device=mps

# Train on a different preprocessing preset
uv run train data_loader.data_dir=data/PPMR_light
uv run train data_loader.data_dir=data/PPMR_minimal
```

---

## Key Files (Training)

| File | Purpose |
|------|---------|
| `hydra/model/model.yaml` | Model defaults |
| `hydra/model/train.yaml` | Training loop defaults |
| `hydra/model/data_loader.yaml` | Data pipeline defaults |
| `src/cli/train.py` | CLI entry point |
| `src/func/models/get_train.py` | Training logic |
| `src/func/data/get_loader.py` | Dataset & dataloader |
