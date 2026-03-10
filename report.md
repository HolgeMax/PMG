# Research Report: Preprocessing Steps in Guha & Bhandage (2025)

**Paper:** "Automated detection of polymicrogyria in pediatric patients using deep learning"
**Authors:** Guha, S., Bhandage, V. & Agarwal, A.
**Published:** Scientific Reports 15, 41662 (2025)

---

## Executive Summary

The paper employs a 4-step preprocessing pipeline to enhance MRI brain scans for PMG classification:
1. Min-Max Normalization
2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
3. Bilateral Filtering
4. Canny Edge Detection with blending

This sequence achieved up to **10.3% accuracy improvement** (ResNet-101) by preserving critical anatomical details while reducing noise and enhancing relevant structural features.

---

## Preprocessing Step 1: Min-Max Normalization

### Purpose
Scales pixel intensities to a standardized [0, 1] range, ensuring:
- Consistent input distribution for neural networks
- Faster and more stable training convergence
- Comparability across different patients and equipment

### Method
```
normalized = (pixel - min) / (max - min)
```

### Parameters from Paper
- Input range: Original MRI intensity values
- Output range: [0, 1]

### Research Evidence

```json
{
  "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/",
  "snippet": "Experiments demonstrate that intensity normalization as a preprocessing step improves the synthesis results across all investigated synthesis algorithms.",
  "why_relevant": "Validates that normalization is essential for deep learning on MRI"
}
```

```json
{
  "source": "https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1",
  "snippet": "Intensity normalization is essential. It helps the network to converge more efficiently. Fluctuations in voxel values are reduced which leads to a smoother manifold of the loss function.",
  "why_relevant": "Explains the theoretical basis for why normalization improves training"
}
```

### Open Questions
- The paper uses simple min-max; alternatives like Z-score normalization or Nyul normalization may be more robust for MRI's relative intensity nature
- Patient-wise vs. dataset-wise normalization strategy is not specified
  - Could look at patient-wise normalization and z-score.

---

## Preprocessing Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Purpose
Enhances local contrast while pre03venting over-amplification of noise:
- Improves visibility of subtle cortical folding abnormalities
- Works on local regions (tiles) rather than global histogram
- Clip limit prevents noise amplification in homogeneous regions

### Method
1. Divide image into tiles (8x8 grid)
2. Compute histogram for each tile
3. Clip histogram at specified limit (redistributes excess)
4. Apply histogram equalization to each tile
5. Interpolate boundaries to prevent artifacts

### Parameters from Paper
| Parameter | Value |
|-----------|-------|
| Clip Limit | 2.0 |
| Tile Grid Size | 8 x 8 |

### Research Evidence

```json
{
  "source": "https://www.nature.com/articles/s41598-025-25572-6",
  "snippet": "Image pre-processing involved applying min-max normalization to scale pixel intensities, followed by contrast enhancement using CLAHE with a clip limit of 2.0 and a tile grid size of 8 by 8.",
  "why_relevant": "Direct source of exact parameters used in the target paper"
}
```

```json
{
  "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9795279/",
  "snippet": "CLAHE as preprocessing to U-Net architectures significantly improved Dice Similarity Coefficient (~0.993), Intersection over Union (~0.986) in brain tumor segmentation.",
  "why_relevant": "Demonstrates CLAHE effectiveness in brain MRI deep learning tasks"
}
```

```json
{
  "source": "https://www.researchgate.net/publication/312673432",
  "snippet": "The selection of tile size, clip-limit and the distribution which specify the desired shape of the histogram of image tiles is paramount, as it critically influences the quality of the enhanced image.",
  "why_relevant": "Highlights importance of parameter tuning for optimal results"
}
```

### Open Questions
- Optimal clip limit is domain-specific; 2.0 is a common default but may not be optimal for PMG features
- No justification provided for 8x8 tile size
- Adaptive clip limit methods (e.g., Sailfish Optimization) could improve results

---

## Preprocessing Step 3: Bilateral Filtering

### Purpose
Reduces noise while preserving edges:
- Smooths homogeneous regions (noise reduction)
- Maintains sharp boundaries (edge preservation)
- Critical for MRI which suffers from Rician noise

### Method
Weighted average where weights depend on:
1. **Spatial distance** (Gaussian on pixel distance)
2. **Intensity difference** (Gaussian on intensity similarity)

Pixels that are both close AND similar contribute more to the filtered output.

### Parameters from Paper
| Parameter | Value |
|-----------|-------|
| Kernel Diameter (d) | 9 |
| Sigma Color | 75 |
| Sigma Space | 75 |

### Research Evidence

```json
{
  "source": "https://www.nature.com/articles/s41598-025-25572-6",
  "snippet": "Noise reduction was performed using a bilateral filter with a kernel diameter of 9 and sigma values of 75 for both color and space.",
  "why_relevant": "Direct source of exact parameters from the target paper"
}
```

```json
{
  "source": "https://www.sciencedirect.com/science/article/abs/pii/S1568494616300953",
  "snippet": "The performance of bilateral filter for denoising is highly dependent on optimal parameter selection. There is not much theoretical or empirical study for optimal selection of the BF parameters.",
  "why_relevant": "Identifies that parameter selection is under-studied and could be improved"
}
```

```json
{
  "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6382639/",
  "snippet": "Automatic parameter decision system achieved MAPE of 6% and dramatically removed noise in brain MR images, outperforming several state-of-the-art methods.",
  "why_relevant": "Suggests automatic parameter tuning could improve bilateral filtering"
}
```

### Open Questions
- Paper uses fixed sigma values; adaptive methods exist
- Sigma=75 is relatively high; may over-smooth subtle PMG features
- Order of bilateral filter and CLAHE could affect results (paper applies after CLAHE)

---

## Preprocessing Step 4: Canny Edge Detection with Blending

### Purpose
Highlights structural boundaries relevant for PMG detection:
- Emphasizes cortical boundaries and gyral patterns
- Irregular gyri and shallow sulci become more visible
- Blending preserves original intensity information while adding edge information

### Method
1. Apply Gaussian smoothing (implicit in aperture)
2. Compute intensity gradients
3. Non-maximum suppression (thin edges)
4. Double threshold to classify strong/weak edges
5. Edge tracking by hysteresis
6. **Blend** edge map with original image (alpha compositing)

### Parameters from Paper
| Parameter | Value |
|-----------|-------|
| Low Threshold | 50 |
| High Threshold | 200 |
| Aperture Size | 3 |
| Alpha (edge blend) | 0.20 |
| Beta (original) | 0.80 |

### Research Evidence

```json
{
  "source": "https://www.nature.com/articles/s41598-025-25572-6",
  "snippet": "Edge enhancement utilized the Canny edge detector with thresholds of 50 and 200, an aperture size of 3, and the edge map was blended at an alpha value of 0.20.",
  "why_relevant": "Direct source of exact parameters from the target paper"
}
```

```json
{
  "source": "https://www.nature.com/articles/s41598-025-98356-7",
  "snippet": "Canny edge detection preprocessing impacts performance of machine learning models, highlighting structural features relevant for classification.",
  "why_relevant": "Validates edge detection as preprocessing for classification tasks"
}
```

```json
{
  "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11060928/",
  "snippet": "The crucial tuning parameter relates to image intensity, but image intensity is relative for most neuroimaging modalities, making performance unreliable.",
  "why_relevant": "Identifies a fundamental limitation of Canny for MRI - relative intensity"
}
```

### Open Questions
- Canny thresholds are intensity-dependent but MRI intensity is relative
- Blending alpha of 0.20 is arbitrary; optimal value unknown
- Alternative edge detectors (Difference of Gaussian) may be more suitable for neuroimaging

---

## Summary Table: Complete Preprocessing Pipeline

| Step | Method | Key Parameters | Purpose |
|------|--------|----------------|---------|
| 1 | Min-Max Normalization | Range: [0, 1] | Standardize input |
| 2 | CLAHE | clip=2.0, tiles=8x8 | Enhance local contrast |
| 3 | Bilateral Filter | d=9, sigmaColor=75, sigmaSpace=75 | Reduce noise, preserve edges |
| 4 | Canny + Blend | thresholds=50/200, alpha=0.20 | Highlight cortical boundaries |

---

## Critical Analysis

### Strengths
1. Reproducible parameters are clearly stated
2. Logical progression from normalization → enhancement → denoising → edge emphasis
3. Achieved measurable improvement (up to 10.3% accuracy gain)

### Weaknesses & Concerns
1. **No parameter optimization justification** - Default/common values used without validation
2. **No ablation study** - Contribution of each step not isolated
3. **Fixed parameters** - No adaptation to image-specific characteristics
4. **Potential information loss** - Each transformation is lossy
5. **Order sensitivity** - Preprocessing order may not be optimal

### Reproducibility Issues
- Data split strategy unclear (potential leakage)
- No code available
- Training curves suggest possible overfitting

---

## Sources

1. [Guha & Bhandage 2025 - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-25572-6)
2. [MRI Intensity Normalization Impact - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/)
3. [Image Normalization in Medical Imaging - Medium](https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1)
4. [CLAHE Parameter Optimization - ResearchGate](https://www.researchgate.net/publication/312673432)
5. [Bilateral Filter Parameter Selection - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494616300953)
6. [Automatic Bilateral Filter Parameters - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6382639/)
7. [3D Edge Detection in MRI - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11060928/)
8. [Canny Edge Detection for Parkinson's - Nature](https://www.nature.com/articles/s41598-025-98356-7)


  ┌───────────┬───────────────┬──────────────────┬─────────────────────────────────────────────────────┐
  │  Source   │ Original size │ Resize operation │                       Effect                        │
  ├───────────┼───────────────┼──────────────────┼─────────────────────────────────────────────────────┤
  │ PMG cases │ ~1800×1400    │ downsample ×8    │ content is compressed, fine detail averaged/blurred │
  ├───────────┼───────────────┼──────────────────┼─────────────────────────────────────────────────────┤
  │ Controls  │ ~250×250      │ upsample ×0.9    │ almost unchanged, detail preserved/amplified        │
  └───────────┴───────────────┴──────────────────┴─────────────────────────────────────────────────────┘

  The images now have the same pixel count, but they look systematically different in ways the model can exploit.

  ---
  The three signals that survive resizing

  1. Apparent zoom level / scale of anatomy
  A PMG brain photographed at wide FOV and squished to 224×224 shows lots of surrounding tissue and a smaller-looking brain. A control at narrow FOV looks zoomed in, with the brain filling the frame. The CNN sees
  different apparent brain sizes even though both are 224×224.

  2. Texture and spatial frequency
  Downsampling from 1800→224 blurs and averages high-frequency detail. Upsampling from 250→224 preserves or even sharpens it. The two classes end up with systematically different image textures — a low-level
  feature that early conv layers pick up immediately.

  3. Brain-to-background ratio
  Wide FOV images typically contain more skull, scalp, and air around the brain. Narrow FOV images are tighter crops with less background. After resize to 224×224, this ratio difference is preserved and visible.

  ---
  Why this matters for a CNN

  Convolutional networks trained on these resized images will find the texture and zoom-level differences far easier to learn than actual PMG cortical morphology. A model can reach high accuracy purely by learning
  "blurry, lots of background = PMG; sharp, brain fills frame = control" — without ever learning anything clinically meaningful.

  ---
  How to actually fix it

  The right solution is to standardise the images before resizing:

  1. Skull strip — remove everything outside the brain so both classes contain only brain tissue
  2. Crop to brain bounding box — ensures both classes cover the same anatomical region at the same scale before resize
  3. Voxel/pixel spacing normalisation — resample all images to the same physical resolution (mm/pixel) so that brain structures appear at the same scale

  Without at least one of these steps, any model you train on this data is at serious risk of learning the FOV difference rather than PMG pathology — and that is one of the core criticisms of the Guha & Bhandage
  (2025) paper this project is investigating.
