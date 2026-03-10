# How to Run Preprocessing Experiments

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

## Key Files

| File | Purpose |
|------|---------|
| `hydra/config.yaml` | Main Hydra config |
| `hydra/preprocessing/*.yaml` | Preset definitions |
| `src/cli/preprocess.py` | CLI entry point |
| `src/func/utils/loader.py` | JPEG + NIfTI loader |
| `src/main/configurable_pipeline.py` | Pipeline logic |
