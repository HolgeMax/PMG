# Plan: PMG Preprocessing Improvement & Code Structure

Following the **Planner** agent principles:
- Hard-to-vary plans (every step explains *why*)
- Popper-Deutsch falsifiability (steps can obviously succeed/fail)
- KISS (shortest path covering edge-cases)

---

## Part A: Additional Research Needed

| step_id | Task | Agent | Goal |
|---------|------|-------|------|
| R1 | Research MRI-specific normalization methods (Nyul, WhiteStripe, Z-score) | researcher | Determine if alternatives to min-max are more robust for relative MRI intensities |
| R2 | Find optimal CLAHE parameter tuning studies for brain MRI | researcher | Validate or challenge clip=2.0, tiles=8x8 |
| R3 | Research adaptive bilateral filter methods | researcher | Determine if automatic parameter selection improves PMG feature preservation |
| R4 | Investigate Difference of Gaussian (DoG) vs Canny for cortical features | researcher | DoG may be more suitable for neuroimaging per literature |
| R5 | Research preprocessing order optimization | researcher | Determine if current order (norm→CLAHE→bilateral→Canny) is optimal |
| R6 | Find ablation study methods for preprocessing pipelines | researcher | Need methodology to measure contribution of each step |

---

## Part B: Code Structure Plan

### Current Structure (Existing)
```
src/
├── func/
│   └── data/
│       ├── grayscale_conversion.py
│       ├── min_max_normalization.py
│       ├── clahe_enhancement.py
│       ├── bilateral_filtering.py
│       └── canny_edge_detection.py
└── main/
    └── preprocess_pipeline.py
```

### Proposed Structure (Improved)

| step_id | Task | Agent | Status | Goal |
|---------|------|-------|--------|------|
| C1 | Create `src/config/preprocessing_config.py` | code-writer | DONE | Centralize all preprocessing parameters in one place for easy experimentation |
| C2 | Add parameter arguments to each preprocessing function | code-writer | DONE | Enable parameter tuning without code changes |
| C3 | Create `src/func/data/normalization/` submodule | code-writer | DONE | Support multiple normalization methods (min_max, zscore, nyul) |
| C4 | Create `src/func/evaluation/preprocessing_metrics.py` | code-writer | DONE | Implement PSNR, SSIM, entropy metrics to evaluate preprocessing quality |
| C5 | Create `src/experiments/ablation_study.py` | code-writer | DONE | Systematically test contribution of each preprocessing step |
| C6 | Add logging to pipeline for reproducibility | code-writer | DONE | Record exact parameters used for each run |

### Implemented Directory Structure
```
src/
├── config/                            # IMPLEMENTED
│   ├── __init__.py
│   └── preprocessing_config.py        # Dataclasses: PreprocessingConfig, CLAHEConfig, etc.
├── func/
│   ├── data/
│   │   ├── normalization/             # IMPLEMENTED
│   │   │   ├── __init__.py
│   │   │   ├── min_max.py             # normalize_min_max(image, output_range)
│   │   │   └── zscore.py              # normalize_zscore(image, mask)
│   │   ├── edge_detection/            # IMPLEMENTED
│   │   │   ├── __init__.py
│   │   │   ├── canny.py               # detect_edges_canny(image, thresholds, blend_alpha)
│   │   │   └── dog.py                 # detect_edges_dog(image, sigma1, sigma2, blend_alpha)
│   │   ├── grayscale_conversion.py    # (existing)
│   │   ├── min_max_normalization.py   # (existing - legacy)
│   │   ├── clahe_enhancement.py       # (existing)
│   │   ├── bilateral_filtering.py     # (existing)
│   │   └── canny_edge_detection.py    # (existing - legacy)
│   └── evaluation/                    # IMPLEMENTED
│       ├── __init__.py
│       └── preprocessing_metrics.py   # compute_psnr, compute_ssim, compute_entropy
├── main/
│   ├── preprocess_pipeline.py         # (existing - legacy)
│   └── configurable_pipeline.py       # IMPLEMENTED: preprocess_image(image, config)
├── experiments/                       # IMPLEMENTED
│   ├── __init__.py
│   └── ablation_study.py              # AblationStudy class, run_ablation()
└── test/
    └── test_preprocessing.py          # (existing)
```

### Key Modules Implemented

**`src/config/preprocessing_config.py`**
- `PreprocessingConfig` - Master config dataclass (frozen for immutability)
- `NormalizationConfig` - method, output_range
- `CLAHEConfig` - clip_limit=2.0, tile_grid_size=(8,8)
- `BilateralFilterConfig` - diameter=9, sigma_color=75, sigma_space=75
- `CannyConfig` - low=50, high=200, aperture=3, blend_alpha=0.20
- `PAPER_CONFIG` - Pre-built config matching Guha & Bhandage (2025)

**`src/func/data/normalization/`**
- `normalize_min_max(image, output_range)` - Scales to [0,1] or custom range
- `normalize_zscore(image, mask)` - Zero mean, unit variance (optional mask for brain ROI)

**`src/func/data/edge_detection/`**
- `detect_edges_canny(image, low, high, aperture, blend_alpha)` - Canny + blending
- `detect_edges_dog(image, sigma1, sigma2, blend_alpha)` - DoG alternative

**`src/func/evaluation/preprocessing_metrics.py`**
- `compute_psnr(original, processed)` - Peak Signal-to-Noise Ratio
- `compute_ssim(original, processed)` - Structural Similarity Index
- `compute_entropy(image)` - Shannon entropy (information content)
- `evaluate_preprocessing(original, processed)` - Returns all metrics as TypedDict

**`src/experiments/ablation_study.py`**
- `AblationStudy` class - Generates step combinations for systematic testing
- `generate_combinations()` - All 2^n combinations
- `generate_leave_one_out()` - Remove one step at a time
- `run_ablation(image, step_functions, active_steps)` - Execute subset of steps

**`src/main/configurable_pipeline.py`**
- `preprocess_image(image, config)` - Full pipeline with logging
- Returns `(processed_image, PipelineLog)` for reproducibility

---

## Part C: Workflow Plan

| step_id | Task | Agent | Goal |
|---------|------|-------|------|
| W1 | Implement baseline replication with paper's exact parameters | executor | Establish baseline performance to compare against |
| W2 | Run preprocessing on sample images, save intermediate outputs | executor | Visually verify each step works correctly |
| W3 | Implement evaluation metrics (PSNR, SSIM, entropy) | code-writer | Quantify preprocessing quality objectively |
| W4 | Conduct ablation study: test each step's contribution | executor | Falsifiable test of which steps matter |
| W5 | Implement alternative normalization (Z-score) | code-writer | Test if min-max is optimal |
| W6 | Implement parameter grid search for CLAHE | code-writer | Find optimal clip_limit and tile_size |
| W7 | Implement DoG edge detector as alternative | code-writer | Compare against Canny for cortical features |
| W8 | Train CNN on preprocessed vs original data | executor | Final validation of preprocessing value |
| W9 | Document findings in reproducible notebook | synthesizer | Ensure transparency and reproducibility |

---

## Part D: Priority Execution Order

### Phase 1: Baseline Replication (Critical Path)
1. **C1** → Create config file (enables all other parameterization)
2. **C2** → Add parameters to functions (enables experimentation)
3. **W1** → Run baseline with paper's exact parameters
4. **W2** → Verify preprocessing visually

### Phase 2: Evaluation Framework
5. **C4** → Implement evaluation metrics
6. **W3** → Test metrics on sample images
7. **C5** → Create ablation study framework
8. **W4** → Run ablation study

### Phase 3: Improvements (Based on Research)
9. **R1-R6** → Complete additional research
10. **C3** → Implement alternative normalizations
11. **C6** → Add logging
12. **W5-W7** → Test improvements

### Phase 4: Validation
13. **W8** → Train and compare models
14. **W9** → Document everything

---

## Key Decision Points

| Decision | Options | How to Decide |
|----------|---------|---------------|
| Normalization method | min-max, Z-score, Nyul | Compare metrics + model accuracy |
| CLAHE parameters | Default vs optimized | Grid search with metrics |
| Edge detector | Canny vs DoG | Visual inspection + model accuracy |
| Preprocessing order | Current vs alternatives | Ablation with permutations |

---

## Success Criteria (Falsifiable)

1. **Replication**: Achieve similar accuracy to paper (within 2%)
2. **Improvement**: Beat paper's best result OR identify why their pipeline is already optimal
3. **Transparency**: All code, parameters, and results documented
4. **Reproducibility**: Another researcher can replicate our results from documentation alone

---

## Open Questions for User

1. Should we prioritize speed (simple improvements) or thoroughness (full ablation)?
2. Is there access to the original PMG dataset, or using BraTS as proxy?
3. Should alternative methods be implemented even if paper's method proves optimal?
