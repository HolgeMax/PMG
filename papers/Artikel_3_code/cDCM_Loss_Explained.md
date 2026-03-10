# cDCM Loss: Center-based Deep Contrastive Metric Learning

## Overview

The **cDCM (center-based Deep Contrastive Metric learning) Loss** is a novel loss function proposed for detecting Polymicrogyria (PMG) in pediatric brain MRI scans. This innovation addresses the challenge of subtle differences between PMG and control MRI scans in small, imbalanced datasets.

**Paper**: "Deep Contrastive Metric Learning to Detect Polymicrogyria in Pediatric Brain MRI" (Zhang, 2022)
**ArXiv**: https://arxiv.org/abs/2211.12565

---

## The Problem

Traditional cross-entropy-based loss functions struggle with:
- **Subtle disease patterns**: PMG MRI differences from controls are not visually obvious
- **Small datasets**: Limited medical imaging data
- **Class imbalance**: Fewer PMG cases than controls
- **Unknown feature distributions**: True distribution of PMG features is not well-characterized

---

## The Innovation: How cDCM Loss Works

### Core Concept

The cDCM loss learns a **latent representation space** where:
- **Normal (control) samples** cluster tightly around a **learned center point** `c`
- **Abnormal (PMG) samples** are pushed away from the center beyond a **margin** `m`

This creates a clear separation boundary in the embedding space based on distance from center.

### Mathematical Formulation

The loss has **two components**:

#### 1. Normal Sample Loss (Attraction)
```
L_normal = mean(||z_normal - c||₂)
```
- Pulls control samples **toward** the center
- Minimizes Euclidean distance to center for normal cases

#### 2. Abnormal Sample Loss (Repulsion)
```
L_abnormal = θ × [L_hinge + L_smooth]

where:
  L_hinge = mean(max(0, m - ||z_abnormal - c||₂))
  L_smooth = mean(1 / (1 + exp(||z_abnormal - c||₂ - m)))
```
- Pushes PMG samples **away** from center beyond margin `m`
- **L_hinge**: Standard hinge loss (hard margin)
- **L_smooth**: Smooth exponential term for better gradients
- **θ**: Weight balancing normal vs abnormal loss (θ=5 in paper)

#### Combined Loss
```
L_total = L_normal + θ × (L_hinge + L_smooth)
```

---

## Implementation Details

### Key Parameters
- **Margin (m)**: 5.0 - minimum distance for abnormal samples
- **Theta (θ)**: 5.0 - weight for abnormal loss component
- **Latent dimension**: 128 - dimensionality of embedding space
- **Center (c)**: Initialized to ones, not trainable during optimization

### Prediction at Inference
```python
distance = ||z - c||₂
prediction = 1 if distance > margin else 0
```

Samples are classified as PMG if their distance from center exceeds the margin.

---

## Code Reference

The cDCM loss is implemented in:
- **File**: `all_metric_learning_classification.py`
- **Function**: `compute_loss()` (lines 288-331)
- **Model**: SE-Dilated CNN with multi-scale feature fusion
- **Training**: Adam optimizer with learning rate decay and early stopping

---

## Advantages Over Cross-Entropy

1. **Better generalization** on small datasets by learning meaningful distance metrics
2. **Robust to class imbalance** through separate loss components
3. **Interpretable predictions** based on distance from normality center
4. **Smoother optimization** via dual abnormal loss (hinge + exponential)
5. **No assumption** about underlying feature distribution

---

## Results

On the PPMR dataset (pediatric PMG MRI):
- **Recall**: 92.01%
- **Precision**: 55.04%
- **Model**: SE-Dilated CNN with dilated convolution, squeeze-and-excitation blocks, and feature fusion

The high recall makes this suitable as a **screening tool** to assist radiologists in identifying potential PMG cases.

---

## Comparison to Standard Approaches

| Method | Loss Type | Key Limitation |
|--------|-----------|----------------|
| Cross-Entropy | Classification | Poor on imbalanced, small datasets |
| Deep SAD | One-class | Assumes normal data distribution |
| **cDCM (This work)** | **Metric Learning** | **Designed for subtle, rare diseases** |

---

## Architecture Integration

The loss works with a custom **SE-Dilated CNN** architecture:
- **Dilated blocks**: Multi-scale receptive fields (dilation rates 1, 2, 3)
- **SE blocks**: Channel attention mechanism
- **Multi-scale fusion**: Global Average Pooling from multiple layers
- **Output**: 128-dimensional embedding fed to cDCM loss

---

## Citation

```
@article{zhang2022deep,
  title={Deep Contrastive Metric Learning to Detect Polymicrogyria in Pediatric Brain MRI},
  author={Zhang, Lingfeng},
  year={2022},
  institution={University of Ottawa}
}
```
