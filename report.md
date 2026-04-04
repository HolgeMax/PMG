# Replication Report: Guha & Bhandage (2025)
## "Automated Detection of Polymicrogyria in Pediatric Patients Using Deep Learning"

**Author of Replication:** Holger Max Fløe Lyng  
**Course:** Special Course — Investigating Methods for Polymicrogyria Classification  
**Date:** April 2026  
**Sessions Covered:** 1–15 (January–April 2026)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Phase 1 — Preprocessing Pipeline](#3-phase-1--preprocessing-pipeline)
4. [Phase 2 — Model Training](#4-phase-2--model-training)
5. [Phase 3 — Ablation Study](#5-phase-3--ablation-study)
6. [Phase 4 — 5-Fold Cross-Validation](#6-phase-4--5-fold-cross-validation)
7. [Critical Analysis](#7-critical-analysis)
8. [Summary of What Worked and What Did Not](#8-summary-of-what-worked-and-what-did-not)

---

## 1. Overview

Guha & Bhandage (2025) investigate the impact of image preprocessing on the detection of Polymicrogyria (PMG) from pediatric T1-weighted MRI scans using Convolutional Neural Networks (CNNs). Their core claim is that a four-step preprocessing pipeline — Min-Max normalization → CLAHE → Bilateral filtering → Canny edge detection — consistently improves classification accuracy across five pretrained CNN architectures (ResNet-50, ResNet-101, VGG-16, MobileNetV2, DenseNet-201). The paper reports remarkable results: DenseNet-201 achieves 100% test accuracy on the preprocessed dataset, and VGG-16 achieves 100% accuracy via 5-fold cross-validation on both original and preprocessed data.

This report documents the effort to reproduce these results starting from the publicly available PPMR dataset, covering 15 work sessions from January to April 2026. The replication is structured across four phases: preprocessing pipeline, model training, ablation study, and 5-fold cross-validation. Each phase documents what was attempted, what failed, what was changed, and the reasoning behind each decision.

---

## 2. Dataset

### What the Paper Uses

The paper uses the publicly available Pediatric Polymicrogyria MRI (PPMR) dataset introduced by Zhang et al. (2024), available on Kaggle. It consists of:

- **15,056 total JPEG MRI slices** from coronal 3D gradient echo T1-weighted sequences
- **23 PMG patients** → 4,517 PMG slices
- **Control patients** at a 3:1 ratio → 10,539 control slices
- **Class imbalance handling:** the paper downsamples controls to 4,517 images per class, resulting in a balanced 9,034-image dataset
- **Split:** 60% / 20% / 20% (train / val / test), with all images rescaled to 224×224 pixels and converted to RGB

### What Was Discovered During Replication

The PPMR dataset has a nuanced label structure that the paper glosses over. Each JPEG filename encodes a raw label at field index 3 (zero-indexed, split by `_`):

| Raw Label | Folder | Meaning |
|-----------|--------|---------|
| 0 | `controlcases/` | Healthy control — always HC |
| 1 | `PMGcases/` | PMG visible in this slice — positive |
| 2 | `PMGcases/` | PMG patient, but no PMG visible in this slice |
| 3 | `PMGcases/` | Uncertain / ambiguous label |

This creates a critical methodological choice:

- **"Paper" mode (`pmg_negative_mode="paper"`):** All slices from the `PMGcases/` folder — including label 2 (no PMG visible) and label 3 (uncertain) — are treated as positive. This is what Guha & Bhandage (2025) appear to have done, but it is methodologically incorrect: it trains the model to predict "this patient has PMG" rather than "this slice shows PMG features."
- **"Correct" mode (`pmg_negative_mode="correct"`):** Only label-1 slices are positive. Label-2 slices are treated as HC (the patient has PMG but this slice does not show it). Label-3 slices are excluded entirely.

A dedicated `pmg_negative_mode` configuration parameter was implemented to allow switching between modes for direct comparison.

**Additional data quality issues found:**
- One file (`10cor_1_19__1.jpg`) contains a double underscore in its filename. The label parser uses underscore-splitting; the double underscore shifts field positions and causes this slice to be silently skipped in all loaders. One PMG slice is effectively invisible.
- One preprocessed file (`2control2cor_0_019_preprocessed_minimal.jpg`) was missing from the preprocessed dataset, causing silent sample drops in preprocessed-data loaders.

**Dataset structure understanding evolved across sessions:**  
It was confirmed (via Zhang et al., Artikel 1) that each PMG patient has three *different* healthy control people matched by age and gender. The filename prefix (e.g., `10control1`, `10control2`, `10control3`, `10cor`) therefore corresponds to a unique individual. This means grouping by the first underscore-delimited field is the correct patient-level split strategy — each prefix is one patient.

---

## 3. Phase 1 — Preprocessing Pipeline

### What the Paper Does

Guha & Bhandage (2025) apply a sequential four-step pipeline to every image before model training:

1. **Grayscale conversion** — Reduces color complexity for histogram/edge operations
2. **Min-Max normalization** — Scales pixel intensities to [0, 1]
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization with clip limit 2.0 and tile grid 8×8; applied to normalized image to improve visibility of subtle structural differences
4. **Bilateral filter** — Edge-preserving noise reduction; kernel diameter 9, σ-color = 75, σ-space = 75
5. **Canny edge detection** — Hysteresis thresholding (low=50, high=200), aperture 3; edge map blended into image at α=0.20

The paper also evaluated alternatives to bilateral filtering (Non-Local Means, Anisotropic Diffusion, Wavelet Denoising) and alternatives to Canny (Sobel, Scharr, Laplacian of Gaussian), concluding that bilateral + Canny was the best combination for PMG.

### Implementation Journey

#### Session 1 - Initial Pipeline

The first implementation created individual Python modules for each step:
- `grayscale_conversion.py`, `min_max_normalization.py`, `clahe_enhancement.py`, `bilateral_filtering.py`, `canny_edge_detection.py`
- A `preprocess_pipeline.py` integrating all steps

**Issues:** The implementation was tested only on a BraTS NIfTI file, not on PPMR JPEG data. Several type-conversion bugs were present:
- `canny_edge_detection.py` had dtype issues with `cv2.addWeighted` (mixed float64/uint8 inputs)
- `min_max_normalization.py` had float64 → float32 conversion issues

These were fixed in the same session by switching to NumPy-based implementations.

#### Session 2 - Architecture Refactoring

The preprocessing code was completely refactored from flat scripts into a modular package:

```
src/config/           → PreprocessingConfig dataclasses (frozen, type-annotated)
src/func/data/normalization/  → min_max.py, zscore.py
src/func/data/edge_detection/ → canny.py, dog.py
src/func/evaluation/  → preprocessing_metrics.py (PSNR, SSIM, entropy)
src/experiments/      → ablation_study.py (early framework)
src/main/configurable_pipeline.py → unified entry point with logging
```

The DoG (Difference of Gaussians) edge detector was added as a Canny alternative for future ablation.

**Why this refactor?** The flat-script structure made it impossible to systematically vary parameters across presets or measure quantitative differences. The dataclass-based config approach allows frozen, type-checked configurations to be passed to a single `preprocess_image()` function.

#### Session 3 — Hydra Configuration System

A Hydra-based configuration system was introduced, creating four preprocessing presets:

| Preset | Description |
|--------|-------------|
| `default` | Full pipeline matching Guha & Bhandage (2025) parameters |
| `light` | Reduced bilateral (d=5, σ=50), less edge blending (15%) |
| `minimal` | Light CLAHE (clip=1.5), minimal bilateral (d=2, σ=25) |
| `no_filter` | Near-zero filtering (d=1) |

A unified CLI was created: `uv run preprocess input_path=... preprocessing=default`. The output was initially PNG; later changed to JPEG (quality 95) in Session 6 because the input data is JPEG.

**Why Hydra?** Hydra allows sweeping preprocessing parameters from the command line without changing code, which is essential for systematic ablation experiments.

#### Session 6 — JPEG Support & Batch Processing

The CLI was extended to support directory-level batch processing:
- `collect_input_files()` — accepts single file or directory (flat or recursive)
- `process_one()` — encapsulates load-preprocess-evaluate-save for one file
- `tqdm` progress bar for batch runs
- Aggregate PSNR/SSIM summary statistics after batch runs

The output format switched from PNG to JPEG to match the source dataset format.

**Deleted:** BraTS NIfTI test files (`BraTS20_Training_001_seg.nii`, etc.) that had been used for initial testing but were irrelevant to the PPMR dataset.

#### Session 8 — Output Structure Refactored

Processed images are now saved to `data/PPMR_processed/PMGcases/<subject>/<scan>/` and `data/PPMR_processed/controlcases/<subject>/<scan>/`, exactly mirroring the source directory tree. This is necessary because the data loader identifies patient IDs from directory structure, and any mismatch would break the patient-level splitting.

**Deleted in this session:** All model training, evaluation, ablation, and XAI code that had been written up to this point — `src/cli/train.py`, `src/config/training_config.py`, `src/data/`, `src/evaluation/`, `src/experiments/`, `src/models/`, `src/training/`, `src/xai/`, `src/func/data/dataloader/`, `hydra/training/`. The entire training subsystem was scrapped and redesigned from scratch.

**Why delete?** The initial model training code had been written before the data loading strategy was fully understood. The label parsing was wrong, the split was image-level (causing data leakage), and the architecture didn't match the paper. A clean redesign was the right call.

#### Session 12 — Ablation Presets Expanded

Two additional presets were added:
- `no_bilateral` — CLAHE + Canny only, skipping bilateral entirely
- `no_clahe` — Bilateral + Canny only, skipping CLAHE

An `edge_first: bool` flag was added to `preprocess_image()` so that Canny can optionally run *before* the bilateral filter. This tests whether applying edge detection to a noisier image captures different features.

**Bug fixed:** `bilateral: None` in YAML was being misread — the `Optional` import was missing from `preprocessing_config.py`, causing a runtime error when loading the `no_bilateral` preset. Fixed by adding `Optional` to imports and changing the default `bilateral` field from `BilateralFilterConfig()` (always-on) to `None`.

**Config utility bug fixed:** `config_to_preprocessing_config` crashed with `KeyError` when `bilateral` was absent from the YAML dict. Fixed by using `.get()` instead of direct key access.

#### Session 14 — Non-Deterministic Split Bug Fixed

A subtle but critical reproducibility bug was identified in `split_dataset()`:

```python
# BEFORE (non-deterministic)
patients = list(patient_map.keys())

# AFTER (deterministic)
patients = sorted(patient_map.keys())
```

Python dictionaries do not guarantee insertion order across runs when filesystem traversal order varies (e.g., macOS vs Linux, or after inode changes). Without sorting, the same `seed=42` would produce different patient assignments depending on the order files were discovered on disk — making cross-run and cross-machine comparisons meaningless.

**Why this matters:** For fair comparison between raw and preprocessed data runs, the same patients must be in the same split. With the non-deterministic bug, the two loaders (pointing to different directories) could assign different patients to train/val/test, making any performance difference partly an artifact of different test sets.

#### Preprocessing Parameters (Final Implementation)

The final `default` preset matches the paper exactly:

```
Grayscale conversion
Min-Max normalization → [0, 1]
CLAHE: clip_limit=2.0, tile_grid_size=(8, 8)
Bilateral filter: d=9, sigma_color=75, sigma_space=75
Canny edge detection: low_threshold=50, high_threshold=200, aperture_size=3
Blend: 0.20 × edges + 0.80 × filtered image
```

---

## 4. Phase 2 — Model Training

### What the Paper Does

The paper uses five pretrained CNN architectures, all loaded with ImageNet weights:
- ResNet-50, ResNet-101, VGG-16, MobileNetV2, DenseNet-201
- Each backbone has its base layers frozen; only a custom head is trained
- Custom head: Dense(256, ReLU, L2=0.001) → Dropout(0.5) → Dense(1, sigmoid)
- Input: 224×224×3 (RGB)
- Optimizer: Adam, lr=0.0005, weight_decay=0.001
- Batch size: 32, up to 10 epochs
- Loss: Binary cross-entropy (implicit from sigmoid output)
- Data split: 60% / 20% / 20% (train / val / test)

### Implementation Journey

#### Session 9 — Architecture Education

Before implementing training, the architecture was analyzed in depth:

- ResNet-101 outputs a 2048-dim feature vector → the paper's `Dense(256)` head replaces the ImageNet `Linear(2048, 1000)` classifier
- DenseNet-201 outputs a 1920-dim feature vector → same pattern
- The paper uses `sigmoid` output with binary cross-entropy; PyTorch's `BCEWithLogitsLoss` (numerically stable, combines sigmoid + BCE) was chosen instead

A guided 12-TODO exercise (`src/main/model.py`) was created to build:
- `PMGHead`: Dropout + Linear
- `build_resnet101()`: load pretrained backbone, replace `model.fc`, optional freeze
- `build_densenet201()`: load pretrained backbone, replace `model.classifier`, optional freeze

#### Session 10 — get_models.py Implementation

The exercise was completed, producing `src/func/models/get_models.py` with:
- `PMGHead(in_features)`: `nn.Dropout(p) → nn.Linear(in_features, 256) → nn.ReLU() → nn.Dropout(p) → nn.Linear(256, 1)`
- `build_resnet101(freeze_backbone=True, dropout_p=0.5)`
- `build_densenet201(freeze_backbone=True, dropout_p=0.5)`

**Note:** The replication uses ResNet-101 and DenseNet-201 as the two primary architectures. The paper tests five models, but the two strongest performers were prioritized. This is a deliberate scope reduction.

#### Session 13 — Two Critical Bugs Fixed in get_models.py

**Bug 1 — `nn.Dropout2d` on a 1D vector:**

After global-average-pool, the feature tensor is shape `(N, C)`. `nn.Dropout2d` expects `(N, C, H, W)` — applying it to a 2D tensor caused every feature element to be zeroed out during training, completely preventing the head from learning.

```python
# WRONG (was in PMGHead)
self.dropout = nn.Dropout2d(p=dropout_p)

# FIXED
self.dropout = nn.Dropout(p=dropout_p)
```

**Bug 2 — `freeze_backbone` set `requires_grad` on the module, not its parameters:**

```python
# WRONG — sets attribute on nn.Module, has no effect on gradients; also inside loop
for name, param in model.named_parameters():
    param.requires_grad = False
model.fc.requires_grad = True  # This does nothing

# FIXED — iterate over parameters of head, outside the freeze loop
for param in model.parameters():
    param.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
```

Without this fix, the backbone would have been fully trainable (not frozen), defeating the purpose of transfer learning and dramatically increasing training time and overfitting risk.

#### Session 13 — Hydra Model Configs & Double-Nesting Bug

Three Hydra config files were created: `data_loader.yaml`, `model.yaml`, `train.yaml`. An initial version had all keys wrapped in a redundant top-level key (e.g., `data_loader: { data_dir: ... }` inside a file that already packages into the `data_loader` namespace), causing double-nesting that broke config access.

Also, `mean: null` and `std: null` caused a runtime crash in `torchvision.transforms.Normalize`, which does not accept `None`. Fixed by replacing with explicit ImageNet stats: `mean: [0.485, 0.456, 0.406]`, `std: [0.229, 0.224, 0.225]`.

#### Session 12 — DataLoader & Label Mapping (Patient-Level Split)

The data loader `src/func/data/get_loader.py` was fully rewritten:

**`PMGDataset`:** Lazy-loads images from disk. Label is extracted from filename at field index 3 of underscore-split stem (e.g., `10cor_1_42_1_preprocessed.jpg` → field 3 = `1` = PMG visible).

**`split_dataset()`:** Groups all slices by patient ID (first field of filename stem), shuffles patients with seeded RNG, then assigns whole patient groups to test, val, train — ensuring no patient appears in more than one split. This prevents data leakage.

**Label dtype mismatch documented:** `__getitem__` returns `torch.long` labels; `BCEWithLogitsLoss` requires `torch.float`. The cast must happen in the training loop:
```python
loss = criterion(logit.squeeze(1), labels.float())
```

**`pmg_negative_mode` parameter:**
- `"paper"` — all `PMGcases/` slices are positive (replicates Guha & Bhandage 2025 — methodologically incorrect)
- `"correct"` — only label=1 is positive; label=2 → HC; label=3 → excluded

This allows direct comparison between the paper's (flawed) mode and a methodologically sound mode.

#### Session 14 (Earlier Numbering) — Full Training Loop

`src/func/models/get_train.py` was implemented:

- `train_one_epoch()` — Adam optimizer, BCEWithLogitsLoss, returns mean epoch loss
- `validate_one_epoch()` / `test_one_epoch()` — eval mode, no-grad
- `train(cfg)` — top-level function: resolves device, builds model, calls `split_dataset()`, instantiates datasets, runs training loop

`src/cli/train.py` was implemented as the Hydra-decorated entry point:
- `uv run train` with any override
- Registered in `pyproject.toml` as `train = "src.cli.train:train_cli"`

**Old scaffold deleted:** `src/main/configurable_train.py` was cleared — it had been an early placeholder that was fully superseded.

#### Session 13 (Later Numbering) — Training Pipeline Overhaul

The training system was substantially upgraded:

**Augmentation toggle:**  
`data_loader.augment: bool` added to Hydra config. When `false`, training uses deterministic transforms identical to validation/test. This is essential for clean ablation between augmented and non-augmented runs.

**Model checkpointing:**  
Training saves two checkpoint files per run to `results/checkpoints/`:
- `best_<run_name>.pt` — weights at epoch with lowest validation loss
- `final_<run_name>.pt` — weights at end of last epoch

**Unified epoch function:**  
Replaced separate `train_epoch()` and `eval_epoch()` with a single `_run_epoch()`. This ensures train/val/test metrics are computed identically (same code path), and eliminates a second forward pass over data.

**Per-epoch CSV metrics logging:**  
After each epoch, a row is appended to `results/metrics/<run_name>.csv` containing:
`epoch | split | loss | accuracy | precision | recall | f1 | cohen_kappa`

**New evaluation module (`src/func/evaluation/classification_metrics.py`):**
- `compute_metrics(y_true, y_pred)` — dict of accuracy, precision, recall, F1, Cohen's Kappa
- `collect_predictions(model, loader, device)` — stacked true/pred tensors
- `evaluate_model(model, loader, device)` — combined wrapper
- `print_metrics(metrics_dict)` — formatted console output

#### Session 7 (Earlier) — First Working Training Pipeline (Later Deleted)

Before the clean redesign from Session 8, a first attempt at an end-to-end training pipeline was completed. `src/data/dataset.py` had been corrupted (all functions stripped, replaced with a broken import), and was fully restored from scratch:

- `_label_from_stem()` — extracts integer label from JPEG stems
- `_collect_control_samples()` — collects HC samples
- `_collect_study_samples()` — collects PMG/HC samples, skips label-3
- `collect_all_samples()` — combines controls and study cases
- `downsample_to_balance()` — random undersampling for class balance
- `image_level_split()` — train/val/test split

End-to-end loading was confirmed: `data/PPMR` → 2706 train / 902 val / 904 test.

**This entire pipeline was deleted in Session 8.** The reason: the split was image-level, not patient-level, meaning slices from the same patient could appear in both train and test. This is data leakage. The redesigned `split_dataset()` in `get_loader.py` corrects this.

#### Balance Mode

The paper downsamples controls before splitting (pre-split downsampling). The replication implements this as a configurable parameter:

- `balance_mode: null` — no balancing (all images used, class-imbalanced)
- `balance_mode: "pre_split"` — undersample majority class before splitting (replicates paper)
- `balance_mode: "post_split"` — undersample train set only after splitting (methodologically more sound, avoids leaking information about held-out class distribution)

---

## 5. Phase 3 — Ablation Study

### What the Paper Does

Guha & Bhandage (2025) perform a **preprocessing step ablation** using their best model (DenseNet-201). They progressively add preprocessing steps and measure the effect on test performance:

| Pipeline | Test Accuracy | Test F1 |
|----------|--------------|---------|
| Original images (no preprocessing) | 80.80% | 83.78% |
| Grayscale conversion | 80.80% | 83.78% |
| Grayscale + Normalization | 81.68% | 84.41% |
| Grayscale + Normalization + CLAHE | 89.32% | 90.26% |
| Grayscale + Normalization + CLAHE + Bilateral | 96.96% | 96.96% |
| Full pipeline (+ Canny blend) | 96.96% | 97.00% |

The paper concludes that each step adds incremental value, with CLAHE providing the largest single-step gain (+7.6% accuracy) and Canny providing a marginal improvement in loss and F1 while maintaining accuracy.

### Implementation in This Replication

The replication implements a different style of ablation: **black-box (occlusion) ablation** rather than the paper's preprocessing-step ablation.

#### What Was Implemented

`src/func/evaluation/ablation_study.py` contains:

- `make_black_box(image_tensor)` — applies a random square occlusion of size `0.20 × min(H, W)` to each test image, set to zero
- `run_all_ckpts_ablation_study(data_dir, checkpoint_dir, device, cfg)` — evaluates all `.pt` / `.pth` checkpoints in `results/checkpoints/` on the occluded test set

The architecture is inferred from the checkpoint filename prefix (`resnet101_*` or `densenet201_*`), and the full metrics dict (accuracy, precision, recall, F1) is recorded per checkpoint.

Output: `results/ablation_study/ablation_results.csv` and `results/ablation_study/black_box_example.jpg` (visualization of one occluded image).

#### Bug Fixed — Checkpoint Loading

A bug was fixed where the checkpoint's epoch counter was not being correctly restored:

```python
# BEFORE
model.load_state_dict(torch.load(ckpt_path))

# AFTER
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint.get('epoch', 0)
```

This was needed because checkpoints were saved with metadata (`epoch`, `val_loss`) alongside the model state dict.

#### Difference from Paper

The paper's ablation tests the value of each preprocessing *step* by training separate models with partial pipelines. The replication's black-box ablation tests model *robustness* by occluding regions of test images. These answer different questions:

- **Paper ablation:** "Does each preprocessing step improve classification?"
- **Replication ablation:** "Where is the model looking? Does it break if a region is masked?"

The preprocessing-step ablation (matching the paper exactly) was planned but not completed in the sessions covered by this report — it would require training separate models for each of the six pipeline variants, which is computationally expensive and was deprioritized in favor of establishing the end-to-end training and cross-validation infrastructure first.

#### Visualization in Notebook

`notebooks/Metrics_exploration.ipynb` includes an ablation study visualization:

- **Session 14 (chart fix):** The ablation chart was converted from a vertical grouped bar chart to a horizontal grouped bar chart, matching the test metrics chart in the same notebook. `ax.bar()` replaced with `ax.barh()`, metric labels moved to y-axis, `ax.invert_yaxis()` added so runs read top-to-bottom, figure height set to `max(5, n * 0.6)` to scale with number of runs.

---

## 6. Phase 4 — 5-Fold Cross-Validation

### What the Paper Does

In addition to the 60/20/20 holdout split, Guha & Bhandage (2025) perform 5-fold cross-validation on the original dataset. Results (Table 3, original images):

| Architecture | Test Accuracy | Test Precision | Test Recall | Mean Val Loss | Std Dev |
|---|---|---|---|---|---|
| VGG-16 | 100% | 100% | 100% | 0.0020 | 0.0009 |
| MobileNetV2 | 98.12% | 98.01% | 98.23% | 0.0882 | 0.0109 |
| DenseNet-201 | 97.68% | 99.09% | 96.23% | 0.0793 | 0.0078 |
| ResNet-50 | 96.02% | 96.32% | 95.68% | 0.1361 | 0.0203 |
| ResNet-101 | 95.68% | 95.68% | 95.68% | 0.1519 | 0.0380 |

And on preprocessed images (Table 5):

| Architecture | Test Accuracy | Test Precision | Test Recall | Mean Val Loss | Std Dev |
|---|---|---|---|---|---|
| VGG-16 | 100% | 100% | 100% | 0.0010 | 0.0004 |
| DenseNet-201 | 99.94% | 99.89% | 100% | 0.0050 | 0.0011 |
| ResNet-101 | 99.67% | 99.78% | 99.56% | 0.0157 | 0.0018 |
| ResNet-50 | 99.61% | 100% | 99.22% | 0.0183 | 0.0037 |
| MobileNetV2 | 99.50% | 99.23% | 99.78% | 0.0110 | 0.0024 |

The paper reports that 5-fold CV significantly improves performance over the 60/20/20 split, attributing this to the increased training data per fold (80% vs 60%).

### Implementation in This Replication

#### Data Split for Cross-Validation

`src/func/data/crossval_split.py` implements `kfold_split_patients()`:

- Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on patient IDs
- Preserves HC:PMG ratio across folds
- All slices from one patient land in the same fold (prevents leakage)
- Returns a generator yielding `(train_samples, val_samples, test_samples)` per fold
- 15% of the training patients in each fold are held out as a validation set for early stopping; the test fold is the held-out fold

This matches the paper's use of 5 folds, but adds an inner validation split for checkpointing (lowest-val-loss checkpoint is used for test evaluation).

#### Training Orchestration

`src/func/models/get_crossval.py` implements `run_crossval(cfg)`:

1. Calls `kfold_split_patients()` to generate fold splits
2. For each fold `i`, calls `train_one_fold(cfg, fold_tag=f"fold{i}")` from `get_train.py`
3. Each fold saves:
   - `results/checkpoints/best_<model>_fold{i}.pt`
   - `results/checkpoints/final_<model>_fold{i}.pt`
   - `results/metrics/<model>_fold{i}.csv`
4. After all folds, saves two summary CSVs:
   - `<model>_crossval_per_fold.csv` — metrics for each fold
   - `<model>_crossval_summary.csv` — mean ± std across folds

`src/cli/crossval.py` provides the Hydra-decorated entry point: `uv run crossval`.

`hydra/crossval_config.yaml` is a separate Hydra config (distinct from the single-run `config.yaml`) to allow independent parameter control for CV experiments.

#### Session Milestone — Ablation Bug Fix (Checkpoint Counter)

From the commit history: *"ablation study checkpoint bug fixed"* and *"added ablation study of trained model and 5-fold crossvalidation"* appear in the same development push. This indicates that the checkpoint loading bug and the 5-fold CV implementation were developed together and fixed simultaneously.

---

## 7. Critical Analysis

### 7.1 Potential Overfitting in Guha & Bhandage (2025)

The paper's training curves (Figs. 4–8) show training accuracy reaching 97–100% while validation accuracy is substantially lower for some models (ResNet-50: train 81.5% vs val 76.5%). For MobileNetV2 and DenseNet-201, both train and validation metrics are near-perfect, which is suspicious given the limited dataset size (~9,000 images, all from 23 patients).

The extremely high 5-fold CV scores (VGG-16: 100% on both original and preprocessed) are implausible for a genuinely challenging medical imaging task and strongly suggest that the model is learning trivial features rather than true PMG pathology.

### 7.2 The FOV Confound

A major concern identified early in this replication is that PMG brain images and control images may differ systematically in Field of View (FOV) and voxel spacing. PMG images are typically larger (~1508×1727 pixels) while controls tend to be smaller (~512×512 pixels). When both are resized to 224×224, the apparent zoom level, texture (due to different upsampling vs downsampling), and brain-to-background ratio are all preserved as learnable signals.

Resizing to 224×224 does **not** eliminate the FOV confound. A CNN can learn to classify based on:
1. **Apparent zoom level/scale of anatomy** — how "zoomed in" the brain appears
2. **Spatial frequency of texture** — upsampled images have smoother textures than downsampled ones
3. **Brain-to-background ratio** — the fraction of the image occupied by brain tissue vs background

Proper mitigation requires skull-stripping, brain bounding-box cropping, or voxel-spacing normalization before the 224×224 resize. None of these were implemented by Guha & Bhandage (2025), and they are documented as open items in this replication.

A naive CNN baseline experiment — training a deliberately simple model to test if the FOV confound alone achieves high accuracy — was planned but not yet completed.

### 7.3 Data Leakage in the Paper

The paper does not describe its train/val/test split at the patient level. Given that the dataset contains 23 patients contributing ~150 slices each, an image-level split (randomly assigning individual slices) would likely result in slices from the same patient appearing in both train and test. This constitutes data leakage: the model sees the same patient's anatomy in training, making test performance optimistic.

This replication enforces strict patient-level splitting throughout.

### 7.4 Label Semantics

The paper treats all slices from the `PMGcases/` folder as positive, including slices where no PMG is visible in that particular frame (label=2). This means the model is trained to predict "this patient has PMG" based on the patient's folder membership, not "this slice shows PMG features." A model can trivially exploit this by learning global image statistics (e.g., FOV, overall brain morphology) without ever detecting the actual lesion.

---

## 8. Summary of What Worked and What Did Not

### Preprocessing

| Component | Status | Notes |
|-----------|--------|-------|
| Grayscale → Min-Max → CLAHE → Bilateral → Canny pipeline | Working | Exact paper parameters replicated |
| Hydra config system with presets | Working | 6 presets: default, light, minimal, no_bilateral, no_clahe, no_filter |
| Batch JPEG processing | Working | Full PPMR dataset processed |
| `edge_first` ordering flag | Working | Enables Canny before bilateral for ablation |
| Non-deterministic split bug | Fixed | `sorted()` before RNG shuffle |
| `no_bilateral` preset crash | Fixed | Missing `Optional` import + `.get()` for dict access |
| Double underscore filename | Known bug | `10cor_1_19__1.jpg` silently skipped |
| Missing preprocessed file | Known bug | `2control2cor_0_019_preprocessed_minimal.jpg` absent |

### Training

| Component | Status | Notes |
|-----------|--------|-------|
| ResNet-101 pretrained backbone | Working | ImageNet weights, frozen by default |
| DenseNet-201 pretrained backbone | Working | ImageNet weights, frozen by default |
| Custom PMGHead (2-layer) | Working | Matches paper architecture |
| `nn.Dropout2d` on 2D tensor | Fixed | Replaced with `nn.Dropout` |
| `freeze_backbone` on module vs params | Fixed | Corrected to iterate `model.fc.parameters()` |
| ImageNet mean/std `null` crash | Fixed | Replaced with explicit values |
| Hydra double-nesting config | Fixed | Removed redundant top-level wrapping |
| Patient-level split | Working | Groups by first `_`-delimited field |
| `pmg_negative_mode` toggle | Working | "paper" vs "correct" modes |
| Per-epoch CSV logging | Working | All metrics saved to `results/metrics/` |
| Model checkpointing | Working | `best_*.pt` and `final_*.pt` |
| Augmentation toggle | Working | Configurable via `data_loader.augment` |
| First training pipeline (image-level split) | Deleted | Replaced with patient-level split |
| Old training scaffold (`configurable_train.py`) | Deleted | Superseded by `get_train.py` + `train.py` |

### Ablation Study

| Component | Status | Notes |
|-----------|--------|-------|
| Black-box (occlusion) ablation | Working | 20% occlusion on test set |
| Checkpoint loading bug | Fixed | Load full dict, extract `model_state_dict` |
| Preprocessing-step ablation (matching paper) | Not yet done | Requires 6 separate training runs |
| Chart conversion (vertical → horizontal) | Working | `Metrics_exploration.ipynb` updated |

### 5-Fold Cross-Validation

| Component | Status | Notes |
|-----------|--------|-------|
| `kfold_split_patients()` generator | Working | `StratifiedKFold`, patient-level |
| `run_crossval()` orchestration | Working | Calls `train_one_fold()` per fold |
| Per-fold checkpoints and CSVs | Working | `fold{i}` suffix on all outputs |
| Summary CSV (mean ± std) | Working | `_crossval_summary.csv` |
| Separate Hydra config for CV | Working | `hydra/crossval_config.yaml` |

### Open Items (as of Session 15)

- Fix `10cor_1_19__1.jpg` double-underscore filename
- Investigate and restore missing `2control2cor_0_019_preprocessed_minimal.jpg`
- Run end-to-end training with deterministic-split fix and verify split assignments match between raw and preprocessed loaders
- Validate FOV confound via naive CNN baseline experiment
- Implement skull-stripping or brain bounding-box crop as first preprocessing step
- Run the preprocessing-step ablation (6 models with partial pipelines) to replicate Table 8 of the paper
- Address `pretrained=True` deprecation (use `weights=` API in torchvision)

---

## References

1. [Guha & Bhandage 2025 - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-25572-6)
2. [MRI Intensity Normalization Impact - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/)
3. [Image Normalization in Medical Imaging - Medium](https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1)
4. [CLAHE Parameter Optimization - ResearchGate](https://www.researchgate.net/publication/312673432)
5. [Bilateral Filter Parameter Selection - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494616300953)
6. [Automatic Bilateral Filter Parameters - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6382639/)
7. [3D Edge Detection in MRI - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11060928/)
8. [Canny Edge Detection for Parkinson's - Nature](https://www.nature.com/articles/s41598-025-98356-7)

---

*This report was compiled from SESSION.md (Sessions 1–15) and direct inspection of the codebase as of April 4, 2026.*
