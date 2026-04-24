---
type: community
cohesion: 0.06
members: 45
---

# cDCM Loss Implementation

**Cohesion:** 0.06 - loosely connected
**Members:** 45 nodes

## Members
- [[Ablation Study CLI (black-box occlusion test)]] - document - how-to-run-experiments.md
- [[Ablation Study Framework (AblationStudy class)]] - document - plan.md
- [[Artikel 3 Code Repository (cDCM implementation)]] - document - papers/Artikel_3_code/Deep-Contrastive-Metric-Learning-Method-to-Detect-Polymicrogyria-in-Pediatric-Brain-MRI-main/README.md
- [[Configurable Preprocessing Pipeline (preprocess_image)]] - document - plan.md
- [[Corrected Data Handling Patient-Level Split Before Downsampling]] - document - papers/my_paper/methods_data_section.md
- [[Data Quality Issues Double-Underscore Filename & Missing Preprocessed File]] - document - SESSION.md
- [[Dataset Structure & Patient-Level Split Rationale]] - document - CLAUDE.md
- [[FOV  Resolution Confound Between PMG and HC]] - document - timeline_a_context.md
- [[FOV Confound Analysis Resize Does Not Eliminate Confound]] - document - SESSION.md
- [[GradCAM++ Visualisation for Model Attention]] - document - timeline_a_context.md
- [[Guha et al. (2025) — Automated PMG Detection Paper]] - document - timeline_a_context.md
- [[Guha et al. Preprocessing Pipeline]] - document - timeline_a_context.md
- [[Introduction Replication Framing of Guha et al. Critique]] - document - papers/my_paper/introduction.md
- [[Issue 1 Mislabelled Positive Class (label contamination)]] - document - papers/my_paper/methods_data_section.md
- [[Issue 2 Systematic Resolution Confound (1508x1727 vs 512x512 px)]] - document - papers/my_paper/methods_data_section.md
- [[Issue 3 Pre-Split Downsampling Introduces Data Leakage]] - document - papers/my_paper/methods_data_section.md
- [[Key Finding F1 Drops from 0.97 to 0.74 After Correcting Methodology]] - document - papers/my_paper/introduction.md
- [[Model Architecture PMGHead, ResNet-101, DenseNet-201]] - document - SESSION.md
- [[Open Items FOV Naive CNN, Skull Stripping, Deterministic Split]] - document - SESSION.md
- [[PMG Deep Learning Project Overview]] - document - timeline_a_context.md
- [[PMG Project AI Agent Context (CLAUDE.md)]] - document - CLAUDE.md
- [[PPMR Dataset (Pediatric Polymicrogyria MRI)]] - document - timeline_a_context.md
- [[PPMR Dataset Description (23 patients, 3 matched controls)]] - document - papers/my_paper/methods_data_section.md
- [[Phase 1 Reproduction & 2D Baseline (Weeks 1-5)]] - document - timeline_a_context.md
- [[Phase 2 Method Extension & Experiments (Weeks 6-10)]] - document - timeline_a_context.md
- [[Phase 3 Writing & Buffer (Weeks 11-15)]] - document - timeline_a_context.md
- [[Plan Part B Code Structure Plan]] - document - plan.md
- [[Preprocessing Evaluation Metrics (PSNR, SSIM, entropy)]] - document - plan.md
- [[PreprocessingConfig Dataclass (PAPER_CONFIG)]] - document - plan.md
- [[Rationale Subject-Level Split Prevents Data Leakage]] - document - timeline_a_context.md
- [[Rationale cDCM over Cross-Entropy for Small Imbalanced Datasets]] - document - papers/Artikel_3_code/cDCM_Loss_Explained.md
- [[SE-Dilated CNN Architecture (multi-scale, squeeze-and-excitation)]] - document - papers/Artikel_3_code/cDCM_Loss_Explained.md
- [[Session 14 (03.04.26) Intensity Distribution Analysis & Deterministic Split Fix]] - document - SESSION.md
- [[Shortcut Learning Hypothesis for High PMG Classification Accuracy]] - document - papers/my_paper/introduction.md
- [[Skull Stripping Tool References (deepbrain TF1TF2)]] - document - papers/Artikel_3_code/Deep-Contrastive-Metric-Learning-Method-to-Detect-Polymicrogyria-in-Pediatric-Brain-MRI-main/PPMR/skull_stripping/README.md
- [[Subject-Level TrainValTest Split]] - document - timeline_a_context.md
- [[Three-Phase Project Plan (Replication, Improvement, Transparency)]] - document - CLAUDE.md
- [[Zhang et al. (2024) — cDCM Loss Paper]] - document - timeline_a_context.md
- [[balance_mode Parameter (pre_split vs post_split vs null)]] - document - how-to-run-experiments.md
- [[cDCM Hyperparameters (margin=5, theta=5, latent_dim=128)]] - document - papers/Artikel_3_code/cDCM_Loss_Explained.md
- [[cDCM Loss Implementation]] - document - timeline_a_context.md
- [[cDCM Loss Center-based Deep Contrastive Metric Learning]] - document - papers/Artikel_3_code/cDCM_Loss_Explained.md
- [[cDCM Mathematical Formulation (L_normal + theta(L_hinge + L_smooth))]] - document - papers/Artikel_3_code/cDCM_Loss_Explained.md
- [[cDCM Results on PPMR (Recall 92.01%, Precision 55.04%)]] - document - papers/Artikel_3_code/cDCM_Loss_Explained.md
- [[split_dataset() Deterministic Sort Fix (sorted vs list)]] - document - SESSION.md

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/cDCM_Loss_Implementation
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_CLI and Hydra Config]]

## Top bridge nodes
- [[balance_mode Parameter (pre_split vs post_split vs null)]] - degree 3, connects to 1 community