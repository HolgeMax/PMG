# Project Structure

```
PMG/
├── CLAUDE.md                          # AI agent instructions & project context
├── GEMINI.md                          # Legacy AI agent context file
├── README.md                          # This file
├── SESSION.md                         # Session log
├── plan.md                            # Current project plan
├── how-to-run-experiments.md          # Guide for running experiments
├── pyproject.toml                     # Package configuration
│
├── agents/                            # AI agent role definitions
│   ├── code-critic.md
│   ├── code-writer.md
│   ├── documentation-writer.md
│   ├── executor.md
│   ├── planner.md
│   ├── researcher.md
│   └── synthesizer.md
│
├── hydra/                             # Hydra configuration files
│   ├── config.yaml                    # Root config
│   ├── model/
│   │   ├── data_loader.yaml
│   │   ├── model.yaml
│   │   └── train.yaml
│   └── preprocessing/
│       ├── default.yaml
│       ├── light.yaml
│       ├── minimal.yaml
│       └── no_filter.yaml
│
├── data/                              # MRI data (not tracked by git)
│
├── notebooks/
│   └── JPEG_exploration.ipynb         # EDA notebook
│
├── papers/
│   ├── Artikel 1.pdf
│   ├── Artikel 2.pdf
│   ├── Artikel 3.pdf
│   ├── Artikel_3_code/                # Source code from paper 3
│   │   ├── cDCM_Loss_Explained.md
│   │   └── Deep-Contrastive-Metric-Learning-Method-to-Detect-Polymicrogyria-in-Pediatric-Brain-MRI-main/
│   └── my_paper/                      # Working draft materials
│       ├── project_timeline.tex
│       └── Specialer_og_rapporter___UCPH_MATH.pdf
│
├── results/
│   ├── checkpoints/
│   │   └── best_resnet101.pt          # Best model checkpoint
│   ├── plots/                         # Generated figures
│   │   ├── preprocessing_comparison.png/svg
│   │   └── ...
│   ├── nii_test_light/                # Preprocessing sanity-check images
│   ├── evaluation_resnet101_*.json    # Evaluation results
│   └── YYYY-MM-DD/HH-MM-SS/          # Timestamped run logs (preprocess/train)
│
└── src/
    ├── cli/
    │   ├── preprocess.py              # CLI entry point: preprocessing pipeline
    │   └── train.py                   # CLI entry point: training pipeline
    ├── config/
    │   └── preprocessing_config.py    # PAPER_CONFIG, dataclasses
    ├── func/
    │   ├── data/
    │   │   ├── bilateral.py           # apply_bilateral_filter()
    │   │   ├── clahe.py               # apply_clahe()
    │   │   ├── grayscale.py           # convert_to_grayscale()
    │   │   ├── get_loader.py          # Dataset/DataLoader construction
    │   │   ├── normalization/
    │   │   │   ├── apply_norm.py      # Normalization dispatcher
    │   │   │   ├── min_max.py         # normalize_min_max()
    │   │   │   └── zscore.py          # normalize_zscore()
    │   │   └── edge_detection/
    │   │       ├── canny.py           # detect_edges_canny()
    │   │       └── dog.py             # detect_edges_dog()
    │   ├── evaluation/
    │   │   └── preprocessing_metrics.py  # PSNR, SSIM, entropy
    │   ├── models/
    │   │   ├── get_models.py          # Model factory
    │   │   └── get_train.py           # Training loop
    │   └── utils/
    │       ├── cfg.py                 # Config utilities
    │       └── loader.py              # Data loading helpers
    └── main/
        └── configurable_pipeline.py   # preprocess_image()
```
