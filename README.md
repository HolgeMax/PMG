# Investigating Methods for Polymicrogyria Classification

This repository contains the code and written report for the special course
*Investigating Methods for Polymicrogyria Classification* — an independent
replication and methodological critique of Guha et al. (2025) on deep-learning
detection of polymicrogyria (PMG) from paediatric brain MRI. The full pipeline
— preprocessing, training, cross-validation, evaluation, and occlusion ablation —
is exposed as Hydra-configured CLI commands; see
[`how-to-run-experiments.md`](how-to-run-experiments.md) for usage and the complete
list of overrides. The structure below maps the main components.

## Project Structure

```
PMG/
├── CLAUDE.md                          # AI agent instructions & project context
├── README.md                          # This file
├── SESSION.md                         # Session log
├── how-to-run-experiments.md          # CLI command & override reference
├── pyproject.toml                     # Package config + CLI entry points
├── uv.lock                            # Locked dependencies
│
├── hydra/                             # Hydra configuration
│   ├── config.yaml                    # Preprocessing root config
│   ├── crossval_config.yaml
│   ├── evaluate_config.yaml
│   ├── ablation_config.yaml
│   ├── volume_slicing_config.yaml
│   ├── model/                         # Shared model / train / data-loader configs
│   │   ├── data_loader.yaml
│   │   ├── model.yaml
│   │   └── train.yaml
│   └── preprocessing/                 # Preprocessing presets
│       ├── default.yaml
│       ├── no_clahe.yaml
│       ├── no_bilateral.yaml
│       └── no_filter.yaml
│
├── src/
│   ├── cli/                           # CLI entry points (see pyproject.toml [project.scripts])
│   │   ├── preprocess.py              # `uv run preprocess`
│   │   ├── train.py                   # `uv run train`
│   │   ├── crossval.py                # `uv run crossval`
│   │   ├── evaluate.py                # `uv run evaluate`
│   │   ├── ablation.py                # `uv run ablation`
│   │   └── slice_volumes.py           # `uv run slice-volumes`
│   ├── config/
│   │   └── preprocessing_config.py    # Frozen dataclasses (single source of truth)
│   ├── func/
│   │   ├── data/
│   │   │   ├── grayscale.py           # convert_to_grayscale()
│   │   │   ├── bilateral.py           # apply_bilateral_filter()
│   │   │   ├── clahe.py               # apply_clahe()
│   │   │   ├── normalization/         # min_max, zscore, dispatcher
│   │   │   ├── edge_detection/
│   │   │   │   └── canny.py           # detect_edges_canny()
│   │   │   ├── get_loader.py          # Dataset / DataLoader + patient-level split
│   │   │   ├── crossval_split.py      # kfold_split_patients()
│   │   │   └── volume_slicing.py      # NIfTI → 2D JPEG slicing
│   │   ├── evaluation/
│   │   │   ├── classification_metrics.py  # accuracy, precision, recall, F1, kappa
│   │   │   ├── preprocessing_metrics.py   # PSNR, SSIM, entropy
│   │   │   ├── ablation_study.py          # black-box occlusion
│   │   │   └── run_ablation.py            # ablation orchestration
│   │   ├── models/
│   │   │   ├── get_models.py          # ResNet-101 / DenseNet-201 factory
│   │   │   ├── get_train.py           # Training loop
│   │   │   └── get_crossval.py        # Cross-validation loop
│   │   └── utils/
│   │       ├── cfg.py                 # Hydra → dataclass conversion
│   │       └── loader.py              # File loading / output routing
│   └── main/
│       └── configurable_pipeline.py   # preprocess_image() — full preprocessing chain
│
├── notebooks/                         # EDA & results notebooks
│   ├── JPEG_exploration.ipynb
│   ├── Volume_exploration.ipynb
│   ├── intensity_distribution.ipynb
│   └── Metrics_exploration.ipynb      # Training / CV / ablation / NRU tables & figures
│
├── papers/                            # Literature + the written report
│   ├── PPMR.bib                       # Bibliography
│   ├── *.pdf                          # Reference papers
│   └── my_paper/                      # Report sources (paper_draft.md, tables, figures)
│
├── results/                           # Generated outputs
│   ├── checkpoints/                   # Trained model checkpoints (.pt)
│   ├── metrics/                       # Per-epoch / cross-validation CSVs
│   ├── ablation_study/               # Occlusion-ablation results
│   └── plots/                         # Figures used in the report
│
├── data/                              # MRI data (not tracked by git)
├── bash/                              # SLURM job scripts (run_train.sh, run_ablation.sh)
├── agents/                            # AI agent role definitions
└── PMG/                               # Obsidian knowledge-graph vault (graphify)
```
