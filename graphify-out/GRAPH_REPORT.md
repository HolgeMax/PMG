# Graph Report - /Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/specialcourses/PMG  (2026-04-23)

## Corpus Check
- 79 files · ~55,659 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 498 nodes · 730 edges · 31 communities detected
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 85 edges (avg confidence: 0.73)
- Token cost: 12,500 input · 4,200 output

## Community Hubs (Navigation)
- [[_COMMUNITY_PMG Research Literature|PMG Research Literature]]
- [[_COMMUNITY_Preprocessing Pipeline Config|Preprocessing Pipeline Config]]
- [[_COMMUNITY_cDCM Loss Implementation|cDCM Loss Implementation]]
- [[_COMMUNITY_Training Loop Core|Training Loop Core]]
- [[_COMMUNITY_Cross-Validation Pipeline|Cross-Validation Pipeline]]
- [[_COMMUNITY_Agent Framework|Agent Framework]]
- [[_COMMUNITY_Ablation Study CLI|Ablation Study CLI]]
- [[_COMMUNITY_SE-Dilated CNN Training|SE-Dilated CNN Training]]
- [[_COMMUNITY_Model Training Variant A|Model Training Variant A]]
- [[_COMMUNITY_Model Training Variant B|Model Training Variant B]]
- [[_COMMUNITY_Evaluation Metrics|Evaluation Metrics]]
- [[_COMMUNITY_SE-Dilated Metric Learning|SE-Dilated Metric Learning]]
- [[_COMMUNITY_Preprocessing Quality Metrics|Preprocessing Quality Metrics]]
- [[_COMMUNITY_Model Training Variant C|Model Training Variant C]]
- [[_COMMUNITY_Volume Slicing CLI|Volume Slicing CLI]]
- [[_COMMUNITY_Model Training Variant D|Model Training Variant D]]
- [[_COMMUNITY_ResNet50 Metric Learning|ResNet50 Metric Learning]]
- [[_COMMUNITY_Model Training Variant E|Model Training Variant E]]
- [[_COMMUNITY_Model Training Variant F|Model Training Variant F]]
- [[_COMMUNITY_CLI and Hydra Config|CLI and Hydra Config]]
- [[_COMMUNITY_Model Training Variant G|Model Training Variant G]]
- [[_COMMUNITY_Test Prediction Drawing|Test Prediction Drawing]]
- [[_COMMUNITY_Train Prediction Drawing|Train Prediction Drawing]]
- [[_COMMUNITY_Validation Prediction Drawing|Validation Prediction Drawing]]
- [[_COMMUNITY_Test Prediction Copy|Test Prediction Copy]]
- [[_COMMUNITY_Alternative Preprocessing Research|Alternative Preprocessing Research]]
- [[_COMMUNITY_CLI Entry Points|CLI Entry Points]]
- [[_COMMUNITY_PMG Clinical Background|PMG Clinical Background]]
- [[_COMMUNITY_Cross-Val CLI Node|Cross-Val CLI Node]]
- [[_COMMUNITY_Obsidian Welcome|Obsidian Welcome]]
- [[_COMMUNITY_Downsampling Control|Downsampling Control]]

## God Nodes (most connected - your core abstractions)
1. `PreprocessingConfig` - 16 edges
2. `Artikel 1: A Novel Center-Based Deep Contrastive Metric Learning Method for PMG Detection` - 16 edges
3. `Artikel 2: Automated Detection of Polymicrogyria in Pediatric Patients Using Deep Learning (Guha & Bhandage, 2025)` - 14 edges
4. `Artikel 3: Deep Contrastive Metric Learning to Detect Polymicrogyria in Pediatric Brain MRI (Lingfeng Zhang Thesis, 2022)` - 12 edges
5. `main()` - 9 edges
6. `Project Overview: Investigating Methods for PMG Classification` - 9 edges
7. `config_to_preprocessing_config()` - 8 edges
8. `PMGHead` - 8 edges
9. `evaluate_preprocessing()` - 8 edges
10. `Guha et al. (2025) — Automated PMG Detection Paper` - 8 edges

## Surprising Connections (you probably didn't know these)
- `cDCM Loss Implementation` --semantically_similar_to--> `cDCM Loss: Center-based Deep Contrastive Metric Learning`  [INFERRED] [semantically similar]
  timeline_a_context.md → papers/Artikel_3_code/cDCM_Loss_Explained.md
- `Ablation Study Framework (AblationStudy class)` --semantically_similar_to--> `Ablation Study CLI (black-box occlusion test)`  [INFERRED] [semantically similar]
  plan.md → how-to-run-experiments.md
- `Corrected Data Handling: Patient-Level Split Before Downsampling` --semantically_similar_to--> `balance_mode Parameter (pre_split vs post_split vs null)`  [INFERRED] [semantically similar]
  papers/my_paper/methods_data_section.md → how-to-run-experiments.md
- `Model Architecture: PMGHead, ResNet-101, DenseNet-201` --references--> `Guha et al. (2025) — Automated PMG Detection Paper`  [INFERRED]
  SESSION.md → timeline_a_context.md
- `split_dataset() Deterministic Sort Fix (sorted vs list)` --conceptually_related_to--> `Subject-Level Train/Val/Test Split`  [INFERRED]
  SESSION.md → timeline_a_context.md

## Hyperedges (group relationships)
- **Three Methodological Failures in Guha et al. (2025)** — methods_data_label_contamination, methods_data_resolution_confound, methods_data_presplit_downsampling [EXTRACTED 1.00]
- **Patient-Level Split Pipeline (split_dataset, balance_mode, subject-level rationale)** — timeline_a_context_subject_level_split, session_split_dataset_fix, howto_balance_mode, methods_data_corrected_procedure [INFERRED 0.85]
- **FOV Confound Investigation (detection, analysis, mitigation)** — timeline_a_context_fov_confound, session_fov_confound_analysis, methods_data_resolution_confound, introduction_shortcut_learning, howto_ablation_cli [INFERRED 0.80]
- **Core Agent Pipeline: Planner delegates to Researcher/Executor, Synthesizer produces final output reviewed by Code-Critic** — planner_agent, researcher_agent, executor_agent, synthesizer_agent, code_critic_agent [EXTRACTED 0.95]
- **PPMR Dataset Used Across Three Studies (Artikel 1 Created, Artikel 2 Applied, Artikel 3 Extended)** — artikel1_ppmr_dataset, artikel1_paper, artikel2_ppmr_dataset_used, artikel3_thesis [EXTRACTED 1.00]
- **cDCM Model Architecture: ResNet50/Custom Backbone + MLP Head + cDCM Loss Function** — artikel1_cdcm_loss, artikel1_resnet50_backbone, artikel1_mlp_head, artikel3_cdcm_loss_extended [EXTRACTED 0.95]

## Communities

### Community 0 - "PMG Research Literature"
Cohesion: 0.05
Nodes (54): 5-Fold Double Cross-Validation Strategy, Anomaly Detection Framing for PMG (Normal=HC, Anomaly=PMG), AUCROC Metric for Model Selection (Not F2 Due to Distribution Differences), cDCM Loss Function (Center-Based Deep Contrastive Metric Learning), Children's Hospital of Eastern Ontario (CHEO) Data Source, Coronal 3D Gradient Echo T1 Weighted Sequence (JPEG Export), Deep SAD Loss Function (Comparison Baseline), Euclidean Distance in Latent Space (Decision Boundary) (+46 more)

### Community 1 - "Preprocessing Pipeline Config"
Cohesion: 0.08
Nodes (40): _config_to_dict(), config_to_preprocessing_config(), Convert Hydra DictConfig to PreprocessingConfig dataclass., Convert config to serializable dict for logging., PipelineLog, preprocess_image(), Log of preprocessing parameters and steps applied., Save an intermediate pipeline image to disk as PNG.      Args:         image: Cu (+32 more)

### Community 2 - "cDCM Loss Implementation"
Cohesion: 0.06
Nodes (45): Artikel 3 Code Repository (cDCM implementation), cDCM Mathematical Formulation (L_normal + theta*(L_hinge + L_smooth)), cDCM Hyperparameters (margin=5, theta=5, latent_dim=128), cDCM Loss: Center-based Deep Contrastive Metric Learning, Rationale: cDCM over Cross-Entropy for Small Imbalanced Datasets, cDCM Results on PPMR (Recall 92.01%, Precision 55.04%), SE-Dilated CNN Architecture (multi-scale, squeeze-and-excitation), Dataset Structure & Patient-Level Split Rationale (+37 more)

### Community 3 - "Training Loop Core"
Cohesion: 0.1
Nodes (24): evaluation(), my_metrics(), train(), train_step(), valid_step(), evaluation(), my_metrics(), train() (+16 more)

### Community 4 - "Cross-Validation Pipeline"
Cohesion: 0.11
Nodes (21): crossval_cli(), main(), Run 5-fold (or n-fold) cross-validation and save per-fold + summary CSVs.      P, run_crossval(), _save_results(), build_densenet201(), build_resnet101(), PMGHead (+13 more)

### Community 5 - "Agent Framework"
Cohesion: 0.09
Nodes (28): Code-Critic Agent, Code Audit Rules (20 LOC max, PEP 8, Google Style, Type Annotations), Explanation Integrity Check (Hard-to-Vary Criterion), Maker-Checker Loop (APPROVED/REJECTED Diff-Style Feedback), Vectorisation Over Loops Rule, Code-Writer Agent, Hard-to-Vary Code Principle, Popperian Falsifiability for APIs (+20 more)

### Community 6 - "Ablation Study CLI"
Cohesion: 0.14
Nodes (19): ablation_cli(), main(), run_ablation(), _apply_occlusion(), _calculate_metrics(), _evaluate_on_modified(), _infer_model(), _load_model_params() (+11 more)

### Community 7 - "SE-Dilated CNN Training"
Cohesion: 0.22
Nodes (13): compute_loss(), dataset_collection_func(), evaluation(), get_image(), get_label(), model_class, my_metrics(), one_patient_dataset_collection_func() (+5 more)

### Community 8 - "Model Training Variant A"
Cohesion: 0.24
Nodes (12): compute_loss(), dataset_collection_func(), evaluation(), evaluation_roc(), get_image(), get_label(), model_class, my_metrics() (+4 more)

### Community 9 - "Model Training Variant B"
Cohesion: 0.23
Nodes (12): compute_loss(), dataset_collection_func(), evaluation(), evaluation_roc(), get_image(), get_label(), model_class, my_metrics() (+4 more)

### Community 10 - "Evaluation Metrics"
Cohesion: 0.18
Nodes (14): collect_predictions(), compute_metrics(), evaluate_model(), print_metrics(), Print a metrics dict in a readable table., Collect predictions, compute metrics, and print results.      Parameters     ---, Run the model over *dataloader* and collect ground-truth / predicted labels., Compute binary classification metrics from label lists.      Parameters     ---- (+6 more)

### Community 11 - "SE-Dilated Metric Learning"
Cohesion: 0.25
Nodes (11): compute_loss(), dataset_collection_func(), evaluation(), get_image(), get_label(), model_class, my_metrics(), test_usable_label() (+3 more)

### Community 12 - "Preprocessing Quality Metrics"
Cohesion: 0.2
Nodes (14): compute_entropy(), compute_psnr(), compute_ssim(), evaluate_preprocessing(), _infer_data_range(), _normalize_for_comparison(), PreprocessingMetrics, Evaluate preprocessing quality with multiple metrics.      Args:         origina (+6 more)

### Community 13 - "Model Training Variant C"
Cohesion: 0.25
Nodes (10): dataset_collection_func(), evaluation(), get_image(), get_label(), model_class, my_metrics(), test_usable_label(), train() (+2 more)

### Community 14 - "Volume Slicing CLI"
Cohesion: 0.23
Nodes (13): _extract_slice(), _label_int(), _load_volume(), main(), _output_subfolder(), Slice 3D NIfTI volumes into 2D JPEG slices.  Paths in the metadata CSV are resol, sub-01 + ses-001 -> sub01-ses001  (no underscores, safe for field[0])., Load NIfTI, orient to RAS canonical, return float64 array (H, W, D). (+5 more)

### Community 15 - "Model Training Variant D"
Cohesion: 0.35
Nodes (11): compute_loss(), dataset_collection_func(), evaluation(), evaluation_roc(), get_image(), get_label(), my_metrics(), test_usable_label() (+3 more)

### Community 16 - "ResNet50 Metric Learning"
Cohesion: 0.36
Nodes (10): compute_loss(), dataset_collection_func(), evaluation(), get_image(), get_label(), my_metrics(), test_usable_label(), train() (+2 more)

### Community 17 - "Model Training Variant E"
Cohesion: 0.35
Nodes (9): compute_loss(), evaluation(), evaluation_percentile(), get_R(), get_R_based_on_c(), my_metrics(), train(), train_step() (+1 more)

### Community 18 - "Model Training Variant F"
Cohesion: 0.4
Nodes (8): compute_loss(), evaluation(), get_R(), get_R_based_on_c(), my_metrics(), train(), train_step(), valid_step()

### Community 19 - "CLI and Hydra Config"
Cohesion: 0.2
Nodes (10): Preprocessing CLI (preprocess command with Hydra presets), Training CLI (train command), Volume Slicing CLI (slice-volumes command), Hydra Configuration System, Project Directory Structure, Results Directory (checkpoints, plots, metrics), src/func/data — Preprocessing Modules, pmg_negative_mode Parameter (correct vs paper) (+2 more)

### Community 20 - "Model Training Variant G"
Cohesion: 0.46
Nodes (6): compute_loss(), evaluation(), my_metrics(), train(), train_step(), valid_step()

### Community 21 - "Test Prediction Drawing"
Cohesion: 0.6
Nodes (3): dataset_collection_func(), evaluation(), my_metrics()

### Community 22 - "Train Prediction Drawing"
Cohesion: 0.6
Nodes (3): dataset_collection_func(), evaluation(), my_metrics()

### Community 23 - "Validation Prediction Drawing"
Cohesion: 0.6
Nodes (3): dataset_collection_func(), evaluation(), my_metrics()

### Community 24 - "Test Prediction Copy"
Cohesion: 0.6
Nodes (3): dataset_collection_func(), evaluation(), my_metrics()

### Community 28 - "Alternative Preprocessing Research"
Cohesion: 0.67
Nodes (3): DoG Edge Detector Alternative to Canny, Alternative Normalization Methods (Z-score, Nyul), Plan Part A: Additional Preprocessing Research Tasks

### Community 30 - "CLI Entry Points"
Cohesion: 1.0
Nodes (1): Command-line interface entry points.

### Community 31 - "PMG Clinical Background"
Cohesion: 1.0
Nodes (2): PMG Clinical Context (cortical malformation, epilepsy), Introduction P2: PMG Clinical Background (cortical folding, epilepsy)

### Community 43 - "Cross-Val CLI Node"
Cohesion: 1.0
Nodes (1): Cross-Validation CLI (crossval command, 5-fold patient-level)

### Community 44 - "Obsidian Welcome"
Cohesion: 1.0
Nodes (1): Obsidian Vault Welcome Note

### Community 45 - "Downsampling Control"
Cohesion: 1.0
Nodes (1): Downsampling Control Class (4517 images per class) to Avoid Augmenting PMG

## Knowledge Gaps
- **97 isolated node(s):** `Configuration for normalization step.      Attributes:         method: Normaliza`, `Configuration for CLAHE enhancement.      Attributes:         clip_limit: Thresh`, `Configuration for bilateral filtering.      Attributes:         diameter: Diamet`, `Configuration for Canny edge detection.      Attributes:         low_threshold:`, `Master configuration for the preprocessing pipeline.      Example:         >>> c` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `CLI Entry Points`** (2 nodes): `Command-line interface entry points.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `PMG Clinical Background`** (2 nodes): `PMG Clinical Context (cortical malformation, epilepsy)`, `Introduction P2: PMG Clinical Background (cortical folding, epilepsy)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cross-Val CLI Node`** (1 nodes): `Cross-Validation CLI (crossval command, 5-fold patient-level)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Obsidian Welcome`** (1 nodes): `Obsidian Vault Welcome Note`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Downsampling Control`** (1 nodes): `Downsampling Control Class (4517 images per class) to Avoid Augmenting PMG`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_run_epoch()` connect `Cross-Validation Pipeline` to `Evaluation Metrics`, `Training Loop Core`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `_evaluate_on_modified()` connect `Ablation Study CLI` to `Training Loop Core`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `model_class` connect `Training Loop Core` to `Model Training Variant D`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `PreprocessingConfig` (e.g. with `PipelineLog` and `Log of preprocessing parameters and steps applied.`) actually correct?**
  _`PreprocessingConfig` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Configuration for normalization step.      Attributes:         method: Normaliza`, `Configuration for CLAHE enhancement.      Attributes:         clip_limit: Thresh`, `Configuration for bilateral filtering.      Attributes:         diameter: Diamet` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `PMG Research Literature` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Preprocessing Pipeline Config` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._