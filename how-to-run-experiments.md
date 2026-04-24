# How to Run Experiments

Configuration is managed with **Hydra** — override any parameter on the command line.

---

## Volume Slicing (New Dataset)

Slices 3D NIfTI volumes into 2D JPEG slices ready for the existing training pipeline.
Run this on the server where the data lives. Paths in the CSV are resolved as local filesystem paths.

```bash
uv run slice-volumes                                                          # defaults
uv run slice-volumes volume_slicing.metadata_file=data/pmg_labels.csv
uv run slice-volumes volume_slicing.output_dir=/tmp/new_dataset
uv run slice-volumes volume_slicing.slice_selection=central                   # middle 60%
uv run slice-volumes volume_slicing.slice_selection=random volume_slicing.n_random_slices=80
uv run slice-volumes volume_slicing.slice_selection=all
```

**Quick pipeline test (local BraTS volumes):**
```bash
uv run slice-volumes volume_slicing.metadata_file=data/nii_test_labels.csv volume_slicing.output_dir=data/nii_test_sliced
```
Uses `data/nii_test_labels.csv` (3 volumes: 2 PMG, 1 HC). No server access needed.

**Metadata CSV format** (`data/pmg_labels.csv`):
```
subject,session,label,path
sub-01,ses-001,PMG,/proc_bd5/.../sub-01_ses-001_T1w.nii.gz
sub-0101,ses-001,HC,/proc_bd5/.../sub-0101_ses-001_T1w.nii.gz
```
`label` must be `PMG` (→ `PMGcases/`) or anything else, e.g. `HC` (→ `controlcases/`).

| Parameter | Default | Options / type |
|-----------|---------|----------------|
| `volume_slicing.metadata_file` | `data/pmg_labels.csv` | path to CSV |
| `volume_slicing.output_dir` | `data/new_dataset` | path to output directory |
| `volume_slicing.axis` | `2` | `0`=sagittal · `1`=coronal · `2`=axial |
| `volume_slicing.slice_selection` | `central` | `central \| all \| random` |
| `volume_slicing.central_fraction` | `0.6` | float — fraction of slices kept from centre |
| `volume_slicing.n_random_slices` | `80` | int — slices per volume when `random` |
| `volume_slicing.jpeg_quality` | `95` | int |
| `volume_slicing.seed` | `42` | int — RNG seed for `random` mode |

**Output structure** (mirrors PPMR so the existing data loader works unchanged):
```
<output_dir>/
├── PMGcases/<subject>/<session>/sub01-ses001_042_0_1.jpg
└── controlcases/<subject>/<session>/sub0101-ses001_042_0_0.jpg
```

**Full workflow for new dataset:**
```bash
# 1. Slice volumes
uv run slice-volumes volume_slicing.output_dir=/tmp/new_dataset

# 2. Preprocess (reuses existing pipeline with any preset)
uv run preprocess input_path=/tmp/new_dataset output_path=/tmp/new_dataset_default preprocessing=default recursive=true

# 3. Train
uv run train data_loader.data_dir=/tmp/new_dataset_default
```

---

## Preprocessing

```bash
uv run preprocess input_path=data/PPMR preprocessing=default recursive=true
uv run preprocess input_path=data/nii_test/file.nii preprocessing=light
uv run preprocess -m input_path=data/PPMR preprocessing=default,light,minimal   # multirun
```

**Presets:** `default` · `light` · `minimal` · `no_filter` · `no_bilateral`

| Parameter | Default | Options / type |
|-----------|---------|----------------|
| `input_path` | `data/PPMR` | path to file or directory |
| `output_path` | `data/PPMR_<preset>` | str |
| `recursive` | `false` | `true \| false` |
| `slice_idx` | `null` (middle) | int — NIfTI only |
| `edge_first` | `false` | `true \| false` — apply Canny before bilateral |
| `preprocessing.convert_to_grayscale` | `true` | `true \| false` |
| `preprocessing.normalization.method` | `min_max` | `min_max \| zscore` |
| `preprocessing.normalization.output_range` | `[0.0, 1.0]` | list[float] |
| `preprocessing.clahe.clip_limit` | `2.0` | float |
| `preprocessing.clahe.tile_grid_size` | `[8, 8]` | list[int] |
| `preprocessing.bilateral.diameter` | `9` | int |
| `preprocessing.bilateral.sigma_color` | `75.0` | float |
| `preprocessing.bilateral.sigma_space` | `75.0` | float |
| `preprocessing.canny.low_threshold` | `50` | int |
| `preprocessing.canny.high_threshold` | `200` | int |
| `preprocessing.canny.aperture_size` | `3` | int |
| `preprocessing.canny.blend_alpha` | `0.20` | float — 0.0 disables edge blending |
| `preprocessing.save` | `false` | `true \| false` — save intermediate image after each step |
| `preprocessing.save_dir` | `results/preprocessing_debug` | str — root directory for per-image step folders |

**Pipeline debug images** (`preprocessing.save=true`):

```bash
uv run preprocess input_path=data/PPMR/PMGstudycaseslabelled/34/34cor_1/34cor_1_001.jpg preprocessing.save=true
uv run preprocess input_path=data/PPMR preprocessing=default recursive=true preprocessing.save=true preprocessing.save_dir=results/my_debug
```

Saves one numbered PNG per active step under `<save_dir>/<image_stem>/`:

```
results/preprocessing_debug/
└── 34cor_1_001/
    ├── 00_input.png
    ├── 02_normalization_min_max.png
    ├── 03_clahe.png
    ├── 04_bilateral.png
    └── 05_canny.png
```

---

## Training

```bash
uv run train                                                     # defaults
uv run train model.name=densenet201
uv run train train.num_epochs=30 train.learning_rate=1e-3
uv run train data_loader.data_dir=data/PPMR_light
uv run train data_loader.balance_mode=post_split
uv run train data_loader.train_raw=true                          # skip preprocessing
uv run train -m model.name=resnet101,densenet201 train.learning_rate=1e-3,1e-4   # multirun
```

| Parameter | Default | Options / type |
|-----------|---------|----------------|
| `model.name` | `resnet101` | `resnet101 \| densenet201` |
| `model.pretrained` | `true` | `true \| false` |
| `model.dropout_p` | `0.5` | float |
| `model.freeze_backbone` | `true` | `true \| false` |
| `train.batch_size` | `32` | int |
| `train.num_epochs` | `20` | int |
| `train.learning_rate` | `5e-4` | float |
| `train.weight_decay` | `1e-3` | float |
| `train.num_workers` | `4` | int |
| `train.device` | `cuda` | `cuda \| mps \| cpu` |
| `train.val_frac` | `0.2` | float |
| `train.test_frac` | `0.2` | float |
| `train.seed` | `42` | int |
| `data_loader.data_dir` | `data/PPMR_default` | str — preprocessed data root |
| `data_loader.raw_data_dir` | `data/PPMR` | str — raw data root |
| `data_loader.train_raw` | `false` | `true \| false` — train on raw NIfTI slices |
| `data_loader.crop_size` | `224` | int |
| `data_loader.scale` | `[0.8, 1.0]` | list[float] — random crop scale range |
| `data_loader.mean` | `[0.485, 0.456, 0.406]` | list[float] — ImageNet stats |
| `data_loader.std` | `[0.229, 0.224, 0.225]` | list[float] — ImageNet stats |
| `data_loader.augment` | `true` | `true \| false` — training-time augmentation |
| `data_loader.pmg_negative_mode` | `correct` | `correct \| paper` — see below |
| `data_loader.balance_mode` | `null` | `null \| pre_split \| post_split` — see below |

**`pmg_negative_mode`**
- `correct` — label=2 → HC, label=3 excluded
- `paper` — all PMG-folder slices → positive (replicates Guha & Bhandage 2025)

**`balance_mode`**
- `null` — no balancing
- `post_split` — undersample HC in train split only, after patient-level split *(correct)*
- `pre_split` — undersample HC before split (replicates Guha & Bhandage 2025) *(methodologically incorrect)*

**Outputs:**
- `results/checkpoints/<model>_best.pt` — best val loss checkpoint
- `results/checkpoints/<model>_final.pt` — last epoch checkpoint
- `results/metrics/<model>_metrics.csv` — loss, acc, precision, recall, F1, kappa per epoch (train/val/test)

---

## Cross-Validation

Runs 5-fold (or n-fold) patient-level stratified cross-validation. Each fold trains a fresh model — one fold is test, the rest are train + val. Results are aggregated across folds (mean ± std).

```bash
uv run crossval                                                  # defaults: 5 folds, ResNet101
uv run crossval model.name=densenet201
uv run crossval crossval.n_folds=5 crossval.val_frac_of_train=0.15
uv run crossval data_loader.balance_mode=post_split
uv run crossval -m model.name=resnet101,densenet201              # multirun
```

Accepts all `model.*`, `train.*`, and `data_loader.*` overrides from Training above.

| Parameter | Default | Options / type |
|-----------|---------|----------------|
| `crossval.n_folds` | `5` | int |
| `crossval.val_frac_of_train` | `0.15` | float — fraction of non-test patients held out as val |

**Outputs** in `results/metrics/` and `results/checkpoints/`:
- `{model}_crossval_per_fold.csv` — accuracy, precision, recall, F1, kappa per fold
- `{model}_crossval_summary.csv` — mean and std across all folds
- `{model}_fold{k}_best.pt` × n\_folds — best-val-loss checkpoint per fold
- `{model}_fold{k}_final.pt` × n\_folds — last-epoch checkpoint per fold
- `{model}_fold{k}_metrics.csv` × n\_folds — per-epoch training log per fold

---

## Ablation Study

Evaluates all checkpoints on the test set with a random black-box occlusion applied to each image. Tests whether the model learned meaningful brain features or trivial differences like FOV / resolution.

```bash
uv run ablation                                                  # defaults
uv run ablation ablation.device=cuda
uv run ablation ablation.checkpoint_dir=results/checkpoints
uv run ablation ablation.box_size_frac=0.3
```

Accepts all `model.*` and `data_loader.*` overrides from Training above.

| Parameter | Default | Options / type |
|-----------|---------|----------------|
| `ablation.checkpoint_dir` | `results/checkpoints` | str — directory with `.pt` / `.pth` files |
| `ablation.output_dir` | `results/ablation_study` | str |
| `ablation.device` | `cpu` | `cuda \| cpu` |
| `ablation.box_size_frac` | `0.2` | float — occlusion side as fraction of min(h, w) |

**Outputs** in `ablation.output_dir`:
- `ablation_results.csv` — accuracy, precision, recall, F1 per checkpoint
- `black_box_example.jpg` — example occluded image
- `ablation_config.yaml` — config snapshot
