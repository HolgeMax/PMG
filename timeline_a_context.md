# Project Timeline — Special Course: Automated Detection of Polymicrogyria Using Deep Learning

## Project Overview

This project investigates deep-learning-based detection of polymicrogyria (PMG) from brain MRI. The work proceeds in stages: (1) reproduce the 2D preprocessing and classification pipeline of Guha et al. (2025) on the public PPMR dataset; (2) improve the methodology, potentially incorporating the cDCM loss from Zhang et al. (2024) and subject-level data splitting; and (3) write the report.

## Key Information

- **Submission deadline (KU):** 12 June 2026
- **Submission deadline (DTU):** TBD — needs confirmation
- **Project type:** Special course
- **Dataset:** PPMR (Kaggle), 15,056 JPEG MRI slices from 23 patients (10,539 control, 4,517 PMG)
- **Dataset URL:** https://www.kaggle.com/datasets/lingfengzhang/pediatric-polymicrogyria-mri-dataset
- **Effective start date:** 2 March 2026
- **Total working time:** ~15 weeks (2 Mar – 12 Jun 2026)

## Reference Articles

1. **Guha et al. (2025)** — "Automated detection of polymicrogyria in pediatric patients using deep learning" (Scientific Reports). Preprocessing pipeline: grayscale → min-max normalisation → CLAHE → bilateral filtering → Canny edge detection. Models: ResNet-50, ResNet-101, VGG-16, MobileNetV2, DenseNet-201. Best result: DenseNet-201 at 100% test accuracy on preprocessed images (random 60/20/20 split). Key weakness: random split causes data leakage between patients.

2. **Zhang et al. (2024)** — "A novel center-based deep contrastive metric learning method for the detection of polymicrogyria in pediatric brain MRI" (Computerized Medical Imaging and Graphics). Proposes cDCM loss for anomaly detection on the same PPMR dataset. Uses ResNet-50 backbone with MLP head (512→128). Subject-level 5-fold double cross-validation. Achieves 88.07% recall at 71.86% precision. Key strength: handles imbalanced data and out-of-distribution samples.

---

## Phase 1: Reproduction & 2D Baseline — Weeks 1–5 (2 Mar – 5 Apr)

**Goal:** Reproduce the Guha et al. preprocessing pipeline and classification results on the PPMR dataset.

### Week 1 (2–8 Mar)
- Set up development environment (Python, PyTorch/TensorFlow, GPU access)
- Download the PPMR dataset from Kaggle
- Exploratory data analysis: class distribution, per-patient slice counts, image dimensions
- Understand the dataset structure: 23 patients, 3 controls per PMG patient, coronal T1w gradient echo JPEG exports

### Week 2 (9–15 Mar)
- Implement the preprocessing pipeline:
  - Grayscale conversion
  - Min-max normalisation (scale to 0–1)
  - CLAHE (clip limit 2.0, tile grid 8×8)
  - Bilateral filtering (kernel diameter 9, σ_colour = σ_space = 75)
  - Canny edge detection (thresholds 50/200, aperture 3, blend α = 0.20)
- Visually verify each step against figures in Guha et al.

### Week 3 (16–22 Mar)
- Reproduce the original 60/20/20 random train/validation/test split
- Balance the dataset by downsampling controls to 4,517 per class
- Train the first baseline model (DenseNet-201) on both original and preprocessed images
- Use transfer learning from ImageNet: freeze backbone, add GAP → Dense(256, ReLU, L2=0.001) → Dropout(0.5) → Sigmoid
- Optimiser: Adam (lr=0.0005, weight_decay=0.001), batch size 32, up to 10 epochs with early stopping
- Compare against reported metrics: accuracy, precision, recall, F1, Cohen's κ

### Week 4 (23–29 Mar)
- Train remaining architectures: ResNet-50, ResNet-101, VGG-16, MobileNetV2
- Run five-fold cross-validation as described in the paper
- Compile results tables comparable to Tables 2–6 in Guha et al.
- Run ablation study on preprocessing steps using best model

### Week 5 (30 Mar – 5 Apr)
- Implement subject-level train/validation/test split (no slices from the same patient in multiple sets)
- This follows Zhang et al.'s approach and eliminates data leakage
- Re-run the best-performing models under the new split
- Compare subject-split results with random-split results
- Document discrepancies and analyse impact of data leakage on reported performance

**Milestone:** Baseline results reproduced with both split strategies; discrepancies documented.

---

## Phase 2: Method Extension & Experiments — Weeks 6–10 (6 Apr – 10 May)

**Goal:** Improve the classification approach using the cDCM loss and run comparative experiments.

### Week 6 (6–12 Apr)
- Study the cDCM loss formulation (Eq. 1 in Zhang et al.) in detail
- Loss function: combines distance-to-center for normal samples + hinge-based margin loss for anomalies + sigmoid smoothing near boundary
- Implement the loss function and MLP projection head (512→128 units) on top of ResNet-50
- Centre c can be chosen randomly (proven in Appendix B of Zhang et al.); use all-ones vector
- Key hyperparameters: margin=5, α for class imbalance, latent dim=128

### Week 7 (13–19 Apr)
- Train the cDCM-based model on the PPMR dataset using subject-level split
- Implement five-fold double cross-validation (15:4:4 patient ratio for train:inner-val:outer-val)
- Use AUCROC for model selection on validation set
- Distance clipping at max threshold of 20 to prevent NaN loss
- Tune hyperparameters: margin, α, learning rate schedule (decay 0.5 on plateau)

### Week 8 (20–26 Apr)
- Run full comparative experiments across all combinations:
  - Images: {original, preprocessed}
  - Loss: {cross-entropy, cDCM}
  - Split: {random, subject-level}
- Evaluate with: accuracy, precision, recall, F1, F₂, AUC-ROC
- F₂ measure is the primary metric (recall more important than precision for clinical screening)

### Week 9 (27 Apr – 3 May)
- Implement GradCAM++ visualisations for best-performing models from each approach
- Compare attention maps: do cDCM-trained models focus on cortically relevant regions (irregular gyri, shallow sulci)?
- Analyse prediction distribution histograms (similar to Fig. 6 in Zhang et al.)

### Week 10 (4–10 May)
- Consolidate all experimental results
- Produce final tables, figures, and comparison plots
- Identify any gaps, failed experiments, or unexpected results that need follow-up
- Begin outlining the report structure

**Milestone:** All experiments completed; results tables and figures ready for the report.

---

## Phase 3: Writing & Buffer — Weeks 11–15 (11 May – 12 Jun)

**Goal:** Write the report and address remaining gaps. This phase doubles as a buffer for overrunning experiments.

### Weeks 11–12 (11–24 May)
- Write core report sections:
  - Introduction: PMG background, clinical relevance, motivation
  - Related work: Guha et al., Zhang et al., other PMG/brain MRI classification literature
  - Methods: preprocessing pipeline, model architectures, cDCM loss, evaluation protocol
  - Experimental setup: dataset description, split strategies, hyperparameters, compute details
- Draft the results section using figures and tables from Phase 2

### Week 13 (25–31 May)
- Write the discussion:
  - Impact of data leakage (random vs. subject split)
  - Comparison of cross-entropy vs. cDCM approaches
  - Limitations: single dataset, small patient count, 2D slices only
- Write the conclusion
- Re-run any experiments that need fixing or additional analysis

### Week 14 (1–7 Jun)
- Full revision pass: proofread, check references, verify figure/table consistency
- **Internal deadline: first complete draft by 7 June**

### Week 15 (8–12 Jun)
- Final polish, formatting, and submission
- **Submit by 12 June**

**Milestone:** Report submitted.

---

## Risks and Contingencies

- If reproducing Guha et al. takes longer than expected, reduce the cDCM extension to fewer model×split combinations.
- If cDCM training is unstable (NaN loss issues as noted by Zhang et al.), focus on the preprocessing + standard classification comparison.
- Phase 3 is a built-in buffer; writing can be compressed to weeks 13–15 if code work overruns.
- The DTU submission deadline must be confirmed in week 1.

## Technical Notes

- **Dataset imbalance:** PPMR has ~5:1 normal-to-PMG ratio at slice level. Guha et al. downsample controls; Zhang et al. use the imbalance directly with cDCM.
- **Subject-level splitting is critical:** neighbouring slices from the same patient are nearly identical; random splitting inflates performance.
- **Transfer learning caveat:** Zhang et al. found that ImageNet-pretrained ResNet50 with cDCM eventually predicts all samples as anomaly. They train from scratch instead. Guha et al. freeze ImageNet weights and only train the classification head.
- **Input sizes:** Guha et al. use 224×224×3 (RGB); Zhang et al. use 256×256×3 normalised to [0,1].
