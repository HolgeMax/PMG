## Daily Summary - 09.03.26 - 1 session registered

**Key Accomplishments:**
- Fixed silent label_records bug in JPEG_exploration.ipynb (filename filter `.endswith('light.jpg')` → `.endswith('.jpg')`) that caused empty/wrong downstream plots
- Improved bubble chart: merged control sub-groups, added median centroid dimensions to legend, repositioned legend to avoid overlap
- Added Section 5 to notebook: 8×5 preprocessing comparison grid (original vs. no_filter/minimal/light/default) for 4 PMG and 4 control patients
- Completed model architecture exercise: `src/func/models/get_models.py` (PMGHead, build_resnet101, build_densenet201), `src/func/utils/get_optimizer.py`, and `src/main/train.py` (BCEWithLogitsLoss training loop)
- Appended FOV confound analysis to report.md explaining why resize to 224×224 does not eliminate the confound and documenting mitigation strategies

**Open Items:**
- Run preprocessing comparison grid on full PPMR dataset
- Implement skull-stripping or brain bounding-box crop as first preprocessing step
- Run end-to-end training with `src/main/train.py` and verify loss curves
- Validate FOV confound via naive CNN baseline experiment

---

### Session 10 - 09.03.26

#### JPEG Exploration Notebook — Bug Fix & Visual Improvements

**Bug fix — label_records empty (cell `23daf04e`):**
- Filename filter in the label-record collection loop was `.endswith('light.jpg')`, which matched no files in the current processed directory naming scheme.
- Changed filter to `.endswith('.jpg')` so all JPEG files are collected and `label_records` is populated correctly.
- This was a silent failure: downstream cells ran without error but produced empty/wrong plots.

**Bubble chart improvements — "Image-size distribution per group" (cell `8d722294`):**
- Merged the three control sub-groups (`control1`, `control2`, `control3`) into a single `Control` group so PMG vs. Control comparison is cleaner.
- Added median centroid dimensions (width × height in pixels) to each group's legend label.
- Moved legend from default position to upper-left to avoid overlap with data points.

**New section 5 — "Preprocessing comparison: Before vs After" added to notebook:**
- Added an 8-row × 5-column grid figure comparing 5 preprocessing variants for each of 4 PMG patients and 4 control patients (8 patients total).
- Columns: original, no_filter, minimal, light, default.
- Rows alternate PMG and control patients to make visual comparison easy.
- Purpose: qualitatively assess what each preprocessing preset retains vs. removes, and whether the FOV confound is visible across groups.

**Model architecture files — new implementation (completed exercise):**
- `src/func/models/get_models.py` added: contains `PMGHead` (Dropout + Linear), `build_resnet101`, and `build_densenet201` with pretrained backbone loading and optional freezing.
- `src/func/utils/get_optimizer.py` added: optimizer factory (likely per-layer lr setup for backbone vs. head).
- `src/main/train.py` added: training loop using `BCEWithLogitsLoss`, iterates over dataloader, backward pass, optimizer step.
- `src/main/model.py` changed name to train.py and moved all model into `src/func/models/get_model.py`

**Hydra config changes:**
- `hydra/config.yaml`: removed `- training: training` from defaults list (training config no longer loaded at startup).
- `hydra/preprocessing/minimal.yaml`: bilateral filter diameter reduced from 3 to 2 for truly minimal smoothing.

**report.md additions:**
- Appended a detailed prose section explaining why resizing to 224×224 does NOT eliminate the FOV confound, covering three signals that survive resizing: apparent zoom level/scale of anatomy, texture/spatial frequency differences from up- vs. down-sampling, and brain-to-background ratio.
- Added practical guidance on fixes: skull-stripping, brain bounding-box crop, voxel-spacing normalisation.

**how-to-run-experiments.md:**
- Minor typo correction: `uv run preprocess` command had been accidentally changed to `uvr preprocess`; restored to correct form.

#### Key Files Modified
- `notebooks/JPEG_exploration.ipynb` — filename filter bug fix, bubble chart improvements, new section 5 preprocessing comparison grid
- `src/func/models/get_models.py` — new file: PMGHead, build_resnet101, build_densenet201
- `src/func/utils/get_optimizer.py` — new file: optimizer factory
- `src/main/train.py` — new file: training loop
- `src/main/model.py` — deleted (superseded by src/func/models/)
- `hydra/config.yaml` — removed training defaults entry
- `hydra/preprocessing/minimal.yaml` — bilateral diameter 3 → 2
- `report.md` — FOV confound analysis section appended
- `how-to-run-experiments.md` — typo fix in preprocess command

#### Open Items
- Run preprocessing comparison grid on full PPMR dataset (currently only sample patients used in notebook)
- Implement skull-stripping or brain bounding-box crop as first preprocessing step
- Run end-to-end training with new `src/main/train.py` and verify loss curves
- Validate whether FOV confound is detectable via naive CNN baseline experiment

---

## Daily Summary - 06.03.26 - 2 sessions registered

**Key Accomplishments:**
- Designed model training architecture: clarified ResNet-101 and DenseNet-201 structures, raw output shapes (2048-dim and 1920-dim feature vectors), and why custom heads are needed to replace ImageNet classifiers
- Created `src/main/model.py` as a guided exercise covering: building a `PMGHead` (Dropout + Linear), attaching it to pretrained backbones via `model.fc` / `model.classifier`, backbone freezing, BCEWithLogitsLoss rationale, and per-layer learning rate setup
- Preprocessing pipeline output structure refactored (session 8): PPMR processed outputs now mirror source tree under `data/PPMR_processed/`; all model training code removed pending fresh redesign

**Open Items:**
- Complete model.py exercise (TODOs 1-12)
- Design and implement new model training pipeline with proper FOV/confound controls
- Implement skull-stripping or brain-crop preprocessing step
- Run `uv run preprocess` end-to-end on full PPMR dataset

---

### Session 9 - 06.03.26

#### Model Architecture — Education & Exercise Scaffold

**ResNet-101 / DenseNet-201 conceptual review:**
- Explained raw model outputs: ResNet-101 outputs 2048-dim feature vector → Linear(2048, 1000); DenseNet-201 outputs 1920-dim → Linear(1920, 1000)
- Clarified why extra layers are needed: pretrained heads target 1000 ImageNet classes; PMG is binary (1 logit + BCEWithLogitsLoss)
- Explained DenseNet-201 channel trace: 64 → 256 → 128 → 512 → 256 → 1792 → 896 → 1920

**`src/main/model.py` — Exercise created:**
- 12-TODO guided exercise covering only the adaptation layer (not the backbone from scratch)
- `PMGHead`: Dropout + Linear(in_features → 1)
- `build_resnet101`: load pretrained, replace `model.fc`, optional backbone freeze
- `build_densenet201`: load pretrained, replace `model.classifier`, optional backbone freeze
- Loss function definition (BCEWithLogitsLoss) and per-layer lr optimizer scaffold
- Self-tests verify output shapes and that frozen backbone leaves only head trainable

#### Key Files Modified
- `src/main/model.py` — Created exercise scaffold (was empty)

---

### Session 8 - 06.03.26

#### Preprocessing Pipeline — Output Structure & Cleanup

**Output directory restructured:**
- Processed images now saved to `data/PPMR_processed/PMGcases/<subject>/<scan>/` and `data/PPMR_processed/controlcases/<subject>/<scan>/`, mirroring the original source tree exactly (subject number folder + scan subfolder preserved).
- `_resolve_ppmr_output_dir` helper added to `loader.py` to detect PPMR source folders by name and map them to canonical destinations. Non-PPMR inputs fall back to `data/<dir_name>_<preset>/`.
- `preprocess.py` routes any input path containing `PPMR` to `data/PPMR_processed/` as the output root.
- `_resolve_output_dir` refactored out of `process_one` to satisfy ≤20 LOC rule (code-critic pass).

**Model training & ablation deleted:**
- Removed all model training, evaluation, ablation, and XAI code: `src/cli/train.py`, `src/config/training_config.py`, `src/data/`, `src/evaluation/`, `src/experiments/`, `src/models/`, `src/training/`, `src/xai/`, `src/func/data/dataloader/`, `hydra/training/`.
- `pyproject.toml` now registers only the `preprocess` script entry point.
- `how-to-run-experiments.md` trimmed to preprocessing only.

**FOV confound discussion:**
- Confirmed that resizing to 224×224 does NOT eliminate the FOV confound: PMG images (~1508×1727) are downsampled ×8 while controls (~512×512) are nearly unchanged, producing systematic texture, zoom-level, and brain-to-background ratio differences that survive resize and are trivially learnable by a CNN.
- Proper mitigation requires skull-stripping, brain bounding-box crop, or voxel-spacing normalisation before resize.

**Notebook fix:**
- `JPEG_exploration.ipynb` cell 2 (per-patient label distribution): `COL_COLORS` key `'No PMG (label=2)'` did not match `df_pivot` column `'No PMG visible (label=2)'`, causing blue bars to be silently skipped. Key corrected; debug `print` statements removed.

**Open Items:**
- Design and implement a new model training pipeline (fresh start) with proper FOV/confound controls.
- Implement skull-stripping or brain-crop preprocessing step.
- Run `uv run preprocess` end-to-end on full PPMR dataset to verify new output structure.

---

## Daily Summary - 04.03.26 - 2 sessions registered

**Key Accomplishments:**
- Reviewed full model structure: ResNet-101 sourced from `torchvision.models` with ImageNet weights, frozen backbone, custom 3-layer classification head (Linear 2048→256, ReLU, Dropout 0.5, Linear 256→1, Sigmoid)
- Fixed 3 bugs in `src/data/dataset.py` to enable training on preprocessed data (`results/PPMR_processed`): wrong `ppmr_root` path, `_light` directory name mismatch, label parsing breaking on `_preprocessed_light` filename suffix
- Diagnosed and fixed corrupted `src/data/dataset.py`: all core functions had been stripped and replaced with a broken import; restored all functions from scratch
- Created and implemented `src/func/data/dataloader/build_dataloaders.py` (was an empty stub) and added missing `__init__.py`
- Refactored `src/cli/train.py` to use `build_dataloaders()` utility
- Verified end-to-end: both `data/PPMR` and `results/PPMR_processed` load correctly (2706 train / 902 val / 904 test)

**Open Items:**
- Run `uv run train` end-to-end to confirm training pipeline executes without errors
- Check training curves for overfitting signatures (as noted in Guha & Bhandage paper critique)
- Begin confound ablation: train naive CNN baseline on low-level image statistics
- Investigate Llucia's reconstruction methodology

---

### Session 7 - 04.03.26

#### Dataloader Infrastructure — Bug Fix & Restoration

**Problem Diagnosed:**
`src/data/dataset.py` was found to be corrupted: all core functions had been stripped out and replaced with a single broken import pointing to a non-existent module (`src.func.utils.dataloader.build_dataloader`). The file was effectively unusable.

**Fixes Applied:**

1. **Restored `src/data/dataset.py`** — All missing functions were rewritten from scratch:
   - `_label_from_stem` — Extracts integer label from JPEG filename stems, handling both original and `_preprocessed*` suffixed filenames.
   - `_collect_control_samples` — Collects all HC samples from `PMGControlsEditedDec2021`.
   - `_collect_study_samples` — Collects PMG/HC samples from `PMGstudycaseslabelled`, skipping label-3 uncertain files.
   - `_resolve_subdir` — Auto-resolves `_light` preprocessed directory variants when the exact subdirectory name is not found.
   - `collect_all_samples` — Top-level collector combining controls and study cases.
   - `downsample_to_balance` — Random undersampling to balance class counts.
   - `image_level_split` — Reproducible train/val/test split with stratification and patient-level integrity preserved.
   - Restored missing imports: `import random` and `import numpy as np`.

2. **Created `src/func/data/dataloader/__init__.py`** — The package was missing its `__init__.py`, making it unimportable as a Python package. An empty init file was added to register it as a package.

3. **Filled in `src/func/data/dataloader/build_dataloader.py`** — The file existed but was empty. Implemented `build_dataloaders(project_root, cfg, num_workers)` which:
   - Resolves `ppmr_root` relative to `project_root`.
   - Calls `build_splits` from `src.data.dataset`.
   - Wraps the three datasets into `DataLoader` objects with configurable batch size and workers.
   - Returns `(train_loader, val_loader, test_loader)`.

4. **Updated `src/cli/train.py`** — Removed inline `DataLoader` construction and the now-redundant `build_splits` import. Replaced with a single call to `build_dataloaders()` from the new utility module.

#### Verification

End-to-end load confirmed for both data roots:
- `data/PPMR` (original JPEG layout): 2706 train / 902 val / 904 test
- `results/PPMR_processed` (preprocessed `_light` layout): 2706 train / 902 val / 904 test

#### Key Files Modified
- `src/data/dataset.py` — Fully restored from corrupted state; all core dataset functions reinstated.
- `src/func/data/dataloader/__init__.py` — Created (was missing); registers the package.
- `src/func/data/dataloader/build_dataloader.py` — Implemented `build_dataloaders()` utility function.
- `src/cli/train.py` — Refactored to use `build_dataloaders()` instead of inline DataLoader construction.

---

## Daily Summary - 02.03.26 - 1 session registered

**Key Accomplishments:**
- Switched preprocessing pipeline output format from PNG to JPEG (quality 95)
- Added directory batch mode (flat + recursive) to the CLI with tqdm progress and aggregate PSNR/SSIM summary
- Fixed bubble plot bug in `JPEG_exploration.ipynb` — trailing space in `'PMG '` key prevented PMG group from rendering in red
- Download PPMR dataset from Kaggle and run CLI with `recursive=true`
- Begin Phase 1 Week 1: EDA on PPMR dataset

**Open Items:**
- Investigate Llucia's reconstruction methodology
- Confirm DTU submission deadline

---

## Daily Summary - 16.02.26 - 2 sessions registered

**Key Accomplishments:**
- Implemented Hydra configuration system with 4 preprocessing presets (default, light, minimal, no_filter)
- Created unified CLI command (`uv run preprocess`) with full Hydra integration
- Automated output management with descriptive filenames and config saving
- Analyzed Zhang et al. (2024) cDCM loss methodology and created comprehensive documentation
- Located and verified PPMR dataset on Kaggle (743 MB, 15,056 slices, 23 patients)
- Reviewed project timeline documents (15-week plan, March 2 - June 12, 2026)

**Open Items:**
- Download PPMR dataset from Kaggle
- Begin Phase 1 implementation (Week 1: environment setup, EDA)
- Confirm DTU submission deadline
- Run preprocessing experiments with different presets on downloaded data

---

## Daily Summary - 30.01.26 - 2 sessions registered

**Key Accomplishments:**
- Researched & documented Guha & Bhandage (2025) preprocessing methodology
- Refactored `src/` with modular, configurable architecture (config, normalization, edge detection, evaluation, experiments)
- Created `report.md`, `plan.md`, `how-to.md`
- Added DoG edge detector as Canny alternative

**Open Items:**
- Run ablation study | Implement Nyul normalization | Train CNN validation

---

### Session 2 - 30.01.26 (Claude Code)

#### Research & Documentation
- Created `CLAUDE.md` context file for Claude Code integration
- Researched Guha & Bhandage (2025) paper preprocessing methodology
- Created `report.md` with detailed analysis of 4 preprocessing steps:
  - Min-Max Normalization, CLAHE, Bilateral Filter, Canny Edge Detection
  - Documented exact parameters, purposes, and research evidence
  - Added critical analysis of reproducibility issues and data splitting concerns

#### Code Architecture Refactoring
- Completely refactored `src/` following code-writer agent principles:
  - Created `src/config/` with `PreprocessingConfig` dataclasses (frozen, type-annotated)
  - Created `src/func/data/normalization/` with `min_max.py`, `zscore.py`
  - Created `src/func/data/edge_detection/` with `canny.py`, `dog.py` (DoG alternative)
  - Created `src/func/evaluation/` with `preprocessing_metrics.py` (PSNR, SSIM, entropy)
  - Created `src/experiments/` with `ablation_study.py` framework
  - Created `src/main/configurable_pipeline.py` with logging support

#### Documentation & Tooling
- Created `plan.md` with implementation roadmap and code structure
- Created `how-to.md` for NIfTI file preprocessing
- Created `examples/run_preprocessing.py` and `examples/run_nifti_preprocessing.py`
- Added "End of Day Procedure" to CLAUDE.md and GEMINI.md
- Copied project agents to `~/.claude/agents/` for global access

#### Dependencies
- Added `opencv-python`, `scikit-image` via uv

---

### Session 1 - Previous (Gemini)

### New Features & Implementations
- Created six project agents (Planner, Researcher, Code-Writer, Executor, Synthesizer, Critic) in the `agents/` directory, inspired by `kevinschawinski/claude-agents`.
- Implemented core image preprocessing functions in `src/func/data/`:
    - `grayscale_conversion.py`: Converts images to grayscale.
    - `min_max_normalization.py`: Normalizes pixel intensities to [0, 1].
    - `clahe_enhancement.py`: Applies Contrast Limited Adaptive Histogram Equalization.
    - `bilateral_filtering.py`: Applies bilateral filter for noise reduction.
    - `canny_edge_detection.py`: Performs Canny edge detection and blends with the original image.
- Developed the main preprocessing pipeline (`preprocess_pipeline.py`) in `src/main/`, integrating all individual steps.

### Refinements & Bug Fixes
- **Refactored `grayscale_conversion.py`:** Added explicit checks for 2D images for improved robustness.
- **Refactored `clahe_enhancement.py`:** Incorporated assertions to validate input image data type and range, preventing errors from invalid inputs.
- **Refactored `canny_edge_detection.py`:** Enhanced data type conversions for `cv2.addWeighted` to ensure compatibility and prevent `cv2.error`.
- **Refactored `min_max_normalization.py`:** Switched to a NumPy-based implementation to ensure precise [0, 1] scaling and fixed `float64` to `float32` conversion issues.
- **Refactored `preprocess_pipeline.py`:** Moved example usage code to a dedicated Jupyter notebook (`notebooks/preprocessing_example.ipynb`) to separate core logic from demonstration.
- **Refactored `test_preprocessing.py`:** Improved test robustness by creating dummy images in-memory and adding specific assertions for grayscale conversion and min-max normalization outputs.
- **Added type annotations:** Ensured all relevant functions in `src/` (including `src/test/test_dummy_data.py`) are properly type-annotated to adhere to best practices.

### Tooling & Environment Updates
- Installed necessary libraries: `opencv-python`, `pytest`, and `nibabel`.
- Updated `pyproject.toml` and `uv.lock` to reflect new dependencies.

### Executed Tasks
- Successfully executed the preprocessing pipeline on a NIfTI file (`data/BraTS20_Training_001_seg.nii`), generating a `preprocessed_nifti_slice.png` output and displaying the result.

This session focused on building a robust and well-tested preprocessing pipeline, following clean code principles and clear agent-based development workflows.

### Session 3 - 16.02.26

#### Hydra Configuration System
- Implemented Hydra-based configuration management for preprocessing pipeline
- Created `hydra/config.yaml` as main configuration entry point
- Created 4 preprocessing presets in `hydra/preprocessing/`:
  - `default.yaml` - Matching Guha & Bhandage (2025) paper parameters
  - `light.yaml` - Reduced filtering (diameter=5, blend_alpha=0.15)
  - `minimal.yaml` - Minimal preprocessing (diameter=3, clip_limit=1.5)
  - `no_filter.yaml` - Near-zero filtering (diameter=1)
- Implemented `examples/run_preprocessing_hydra.py` with full Hydra integration

#### Automated Output Management
- Modified preprocessing to auto-save to `results/` directory
- Implemented automatic filename generation with preset names
  - Format: `{filename}_preprocessed_{preset}.png`
  - Example: `BraTS20_Training_022_t1_preprocessed_light.png`
- Auto-save configuration YAML for reproducibility
- Auto-save original slice for comparison

#### CLI Development
- Created `src/cli/preprocess.py` as unified CLI entry point
- Configured `pyproject.toml` with `[project.scripts]` entry
- Added `[build-system]` and package discovery configuration
- Installed command: `uv run preprocess input_path=... preprocessing=...`
- Supports all Hydra features: presets, overrides, parameter sweeps

#### Repository Cleanup
- Removed duplicate scripts:
  - `examples/run_preprocessing_hydra.py` (moved to CLI)
  - `examples/run_preprocessing.py` (old simple version)
  - `examples/run_nifti_preprocessing.py` (old simple version)
- Enhanced `.gitignore` with comprehensive Python/project ignores
- Ignored `results/`, `*.egg-info/`, `__pycache__/`, generated images
- Removed orphan `config.yaml` from root

#### Documentation & Learning
- Explained complete execution flow for `run_preprocessing_hydra.py`
- Documented Hydra configuration composition (defaults → presets → CLI)
- Explained `src/experiments/ablation_study.py` purpose and usage
- Clarified relationship between different preprocessing scripts

#### Key Files Modified
- `pyproject.toml` - Added build system, CLI entry point, package discovery
- `.gitignore` - Comprehensive Python/project ignores
- Created `src/cli/preprocess.py` - Main CLI command
- Added `examples/__init__.py` - Package marker

#### Technical Details
- Hydra config path resolution for installed commands
- DictConfig → PreprocessingConfig conversion
- Runtime preset detection via `HydraConfig.get().runtime.choices`
- Automatic results organization by preset type

---

### Session 6 - 02.03.26

#### JPEG Output Format & Multi-File CLI Refactor

**Overview:**
This session focused on two improvements: switching the preprocessing pipeline output format from PNG to JPEG, and fixing a rendering bug in the JPEG exploration notebook.

**src/cli/preprocess.py — Major Refactor**

The CLI was substantially refactored to support both single-file and directory-as-input modes, and to switch output format from PNG to JPEG:

- Added `collect_input_files(input_path, recursive)` helper that accepts either a single file or a directory (flat or recursive via `recursive=true` flag) and returns all supported files (`.jpg`, `.jpeg`, `.nii`, `.nii.gz`).
- Extracted `process_one(file_path, ...)` to encapsulate the load-preprocess-evaluate-save logic for a single file.
- Output format changed from `.png` to `.jpg` (quality 95 via `cv2.IMWRITE_JPEG_QUALITY`).
- Output directory logic updated: single-file runs save to `results/`; directory runs save to `results/<dir_name>_<preset>/`.
- Added `tqdm` progress bar for batch processing.
- Added aggregate summary statistics (mean ± std of PSNR and SSIM) printed after batch runs.
- Removed old `load_nifti_slice` local definition in favor of the unified `load_image` from `src/func/utils/loader.py`.
- Hydra config (`hydra/config.yaml`) updated: removed `output_path`, added `recursive: false` flag, updated default `input_path` to `data/nii_test`.

**src/func/utils/loader.py — New `load_image` Dispatcher**

- Added `load_jpeg(jpeg_path)` function: loads a JPEG as a uint8 grayscale array via OpenCV; raises `FileNotFoundError` on failure.
- Added `load_image(path, slice_idx)` dispatcher: routes to `load_nifti_slice` or `load_jpeg` based on file extension (handles `.nii.gz` compound suffix correctly); returns `(array, metadata_dict)` with `source_type` and `slice_idx` keys.
- Callers no longer need format-specific logic; the unified interface handles both NIfTI and JPEG sources transparently.

**notebooks/JPEG_exploration.ipynb — Bubble Plot Bug Fix**

- Build the .ipynb notebook, to investigate and plot data features and composition. 
- - Bar plot of PMG patients with labelled PMG slices and normal abnormal slices(PMG patient with no PMG features observed)
- - Bar plot of PMG patients+controls, with PMG slices from PMG patients indicated
- - Bar plot of PMG patients+controls, everything colourcoded
- - Pie + bar plot of data composition
- - Scatter plot of image-size distribution
- Fixed a bug where PMG data points were not rendering in red in the bubble plot.
- Root cause: a trailing space in the key `'PMG '` in the `GROUP_COLORS` dict caused a key lookup miss, so PMG points fell through to the default color.
- Fix: corrected the key to `'PMG'` (no trailing space), restoring correct red coloring for the PMG group.

#### Key Files Modified
- `src/cli/preprocess.py` — PNG-to-JPEG switch, directory batch mode, tqdm progress, aggregate metrics summary
- `src/func/utils/loader.py` — Added `load_jpeg` and unified `load_image` dispatcher
- `hydra/config.yaml` — Removed `output_path`, added `recursive` flag, updated default `input_path`
- `notebooks/JPEG_exploration.ipynb` — Fixed trailing-space key bug in `GROUP_COLORS` dict

#### Data / Cleanup
- Deleted stale BraTS test NIfTI files from `data/` (`BraTS20_Training_001_seg.nii`, `BraTS20_Training_006_seg.nii`, `BraTS20_Training_022_t1.nii`) — replaced by `data/nii_test/` directory.
- Deleted stale `results/test_output.png` and `results/test_output_original.png`.
- Added `notebooks/` directory to repository tracking.

---

### Session 5 - 23.02.26

#### Supervisor Meeting — Project & Data Discussion

**PMG Clinical Background (for report introduction):**

Polymicrogyria (PMG) is a cortical malformation that primarily occurs in the pediatric brain and can manifest clinically as seizures, developmental delays, and other neurological impairments. Diagnosis is challenging [ref needed] and is typically performed by radiologists via MRI. The morphological differences between PMG-affected and healthy control (HC) brains are often subtle, making the condition prone to misdiagnosis [ref needed]. Furthermore, the true distribution of PMG imaging features across a patient population remains unknown, which complicates the development of generalizable detection methods. Machine learning approaches have therefore been proposed to assist with this task [ref, ref, ref].

**Key Discussion Points:**

1. **Article 2 Preprocessing Reproduction (in progress):** Reproduction of the Article 2 preprocessing pipeline is partially complete. Notably, the paper does not specify how differences in Field of View (FOV) or pixel/voxel spacing are handled across scans. A first step will be to replicate their network architectures (ResNets and others) to verify whether we can reproduce the overfitting behavior reported.

2. **Demonstrating Confounding Factors:** A key concern is that models may learn to distinguish HC from PMG cases based on spurious low-level correlations (e.g., differences in brain/image size or FOV) rather than true pathological features. To investigate this:
   - Visually demonstrate differences between HC and PMG cases (e.g., using intensity histograms — confirm approach with supervisor).
   - Train a deliberately simple ("naive") CNN as a baseline. If even this model achieves high classification accuracy, it is likely exploiting image-level statistics (e.g., FOV/size differences) rather than PMG-specific abnormalities. This also serves as an independent check for whether the Article 2 model overfits to such confounds.

3. **Open Data Question:** Further investigation of the dataset is needed. In particular, review Llucia's reconstruction methodology.

---

### Session 4 - 16.02.26

#### Paper Analysis & Documentation
- Analyzed code for Artikel 2/3 (Zhang et al. 2024) on cDCM loss
- Read `papers/Artikel_3_code/` containing Deep Contrastive Metric Learning implementation
- Created comprehensive `papers/Artikel_3_code/cDCM_Loss_Explained.md` documentation
  - Explained center-based Deep Contrastive Metric (cDCM) learning innovation
  - Documented mathematical formulation (normal attraction + abnormal repulsion)
  - Detailed loss components: L_normal + θ × (L_hinge + L_smooth)
  - Added implementation details (margin=5, θ=5, latent_dim=128)
  - Included code references, advantages, results (92% recall), and comparison table

#### Dataset Research
- Located PPMR dataset on Kaggle: https://www.kaggle.com/datasets/lingfengzhang/pediatric-polymicrogyria-mri-dataset
- Verified dataset availability (743 MB, free access, 159 downloads)
- Documented dataset contents: PMG study cases + control cases, coronal slices with slice-level annotations

#### Project Planning Review
- Reviewed newly added `timeline_a_context.md` - comprehensive 15-week project timeline
  - Phase 1: Reproduction & 2D Baseline (Weeks 1-5, 2 Mar - 5 Apr)
  - Phase 2: Method Extension & Experiments (Weeks 6-10, 6 Apr - 10 May)
  - Phase 3: Writing & Buffer (Weeks 11-15, 11 May - 12 Jun)
- Reviewed `papers/my_paper/project_timeline.tex` - LaTeX formatted timeline with two versions:
  - Version A: Kaggle PPMR reproduction + extension only
  - Version B: Full pipeline including clinical epilepsy cohort (183 scans, 162 patients)
- Key project details confirmed:
  - Submission deadline (KU): 12 June 2026
  - Effective start: 2 March 2026
  - Dataset: PPMR (15,056 JPEG slices, 23 patients)

#### New Files Added to Repository
- `papers/artikel 3.pdf` - Zhang et al. (2024) paper on cDCM loss
- `papers/Artikel_3_code/` - Full codebase for center-based deep contrastive metric learning
- `papers/Artikel_3_code/cDCM_Loss_Explained.md` - Created documentation
- `papers/my_paper/project_timeline.tex` - LaTeX timeline document
- `papers/my_paper/Specialer_og_rapporter___UCPH_MATH.pdf` - Timeline PDF
- `timeline_a_context.md` - Markdown version of project timeline

#### Minor Updates
- Fixed minor formatting in SESSION.md (removed "(Claude Code)" suffix from Session 3 header)
- Updated `how-to-run-experiments.md` with enhanced documentation
- Updated `hydra/config.yaml` with default input path
- Added utility functions to `src/func/utils/cfg.py` and `src/func/utils/loader.py`

---

