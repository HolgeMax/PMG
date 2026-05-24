# 3 Methods

**Supervisor comments addressed:**
- Comment M1: Renumbered to Section 3 (after Introduction and Theory). Unnecessary material removed.
- Comment M2: PPMR and clinical dataset subsections merged into a single "Experimental Setup" subsection that describes both.
- Comment M3: Methods content (what was done) cleanly separated from results/discussion content (what was found).
- Data comment 1: Basic data description moved here as the first methods subsection. Only known facts from the original papers — no analysis findings.
- Data comment 2: Resolution difference finding (1508×1727 px vs controls) → **moved to Results** (this is Llucia's finding from inspecting the data, not original dataset documentation).
- Data comment 3: Confound interpretation → split: the experimental design choice (two conditions) stays here; the shortcut learning explanation → **moved to Discussion**.
- Data comment 4: Metric inflation argument → **moved to Discussion**.
- Data comment 5: Pipeline reimplementation outputs → **moved to Results**.

---

## 3.1 Data

This study uses two datasets. The first is the publicly available Pediatric
Polymicrogyria MRI (PPMR) dataset introduced by Zhang et al. (2024) and subsequently
used by Guha et al. (2025). It consists of coronal 3D gradient-echo T1-weighted MRI
slices from 23 paediatric epilepsy patients with confirmed PMG and three age- and
gender-matched healthy controls per patient, yielding a 3:1 control-to-case ratio.
Images were exported from the Picture Archiving and Communication System (PACS) at the
Children's Hospital of Eastern Ontario and released as individual JPEG files on Kaggle.
The dataset contains 15,056 slices in total, of which 4,517 are attributed to PMG
patients and 10,539 to healthy controls. Importantly, each PMG-patient slice carries
one of three slice-level annotations provided by a paediatric neuroradiologist:
PMG-positive (label=1), PMG-negative (label=2), or uncertain (label=3). The
acquisition parameters for the two scanner types used in the PPMR dataset are
summarised in Table 2 in the Theory section.

The second dataset consists of 183 T1-weighted MRI scans from 162 unique patients
evaluated for epilepsy (77 females; mean ± SD age at acquisition: 25.8 ± 17.4 years),
retrospectively collected from the Eastern Region of Denmark. Of these, 110 scans (49
females; 24.7 ± 15.9 years) were reported to contain radiological evidence of PMG
based on clinical radiology reports. The non-PMG group (37 females; 27.5 ± 19.6
years) comprised scans in which PMG-related terminology appeared in the report but a
subsequent double-rater confirmed the absence of PMG; of these, 56 cases had other
abnormalities and 17 were reported as having a normal MRI. Scans were acquired across
multiple scanner protocols, two magnetic field strengths, and three vendors. The
acquisition parameters for the primary scanner (Hvidovre Hospital, 3 T Siemens Verio)
are summarised in Table 1 in the Theory section.

> **Note — Results:** The distribution of slice labels within the PPMR dataset (exact
> counts per label per patient), the class-correlated resolution difference between PMG
> and control images, and the output of the preprocessing pipeline as applied to both
> groups are reported in the Results section, as these are findings from inspecting and
> re-implementing the data rather than properties documented in the original papers.

---

## 3.2 Experimental Setup

To evaluate the methodological validity of Guha et al. (2025) and isolate the
contribution of each identified error, experiments were organised into seven conditions
spanning three axes: preprocessing (raw vs. preprocessed), label definition (paper
vs. corrected), and downsampling strategy (none, pre-split, post-split). Both
ResNet-101 and DenseNet-201 were evaluated under all conditions using 5-fold
cross-validation at the patient level. Table 1 provides an overview.

| Condition | Preprocessing | Labels | Downsampling | Primary comparison |
|---|---|---|---|---|
| 1 | Raw | Paper | None | Baseline |
| 2 | Raw | Corrected | None | 1 vs. 2: label correction on raw |
| 3 | Raw | Corrected | Post-split | Fully corrected; 2 vs. 6: preprocessing effect (corrected) |
| 4 | Preprocessed | Paper | None | 1 vs. 4: preprocessing effect (paper labels) |
| 5 | Preprocessed | Paper | Pre-split | 4 vs. 5: downsampling inflation; Guha et al.'s exact setup |
| 6 | Preprocessed | Corrected | None | 4 vs. 6: label correction on preprocessed |
| 7 | Preprocessed | Corrected | Pre-split | 5 vs. 7: label correction, downsampling held fixed |

**Label definitions.** The paper label definition follows Guha et al. (2025): all
4,517 slices attributed to PMG patients are treated as the positive class regardless
of slice-level annotations, and all 10,539 healthy control slices form the negative
class. The corrected label definition restricts the positive class to slices with
label=1 (PMG-positive, n=2,256). Label=2 slices (PMG-negative) are reclassified as
negative examples and pooled with the original healthy controls, yielding
10,539 + 1,386 = 11,925 negative slices. Label=3 (uncertain) slices are excluded from
all corrected conditions. All splits were performed at the patient level across 92
unique subjects (23 PMG + 69 controls), with a fixed random seed (seed=42) for
reproducibility.

The raw branch (Conditions 1–3) and preprocessed branch (Conditions 4–7) each vary
label definition and downsampling strategy in parallel. Within each branch, comparing
paper versus corrected label conditions isolates the effect of label noise. Comparing
Conditions 1 and 4 (and Conditions 2 and 6) isolates the effect of preprocessing on
performance, holding labels and downsampling fixed. The downsampling axis is tested by
comparing Conditions 4 and 5, which differ only in whether controls are downsampled
before or after splitting — the latter matching Guha et al.'s exact setup. Condition 7
extends this by applying the same pre-split downsampling under corrected labels,
allowing label correction to be isolated with downsampling strategy held fixed
(5 vs. 7). Condition 3 — raw images, corrected labels, post-split downsampling — is
the fully corrected condition: it corrects all three errors and avoids preprocessing,
eliminating the resolution-confound amplification introduced by the pipeline.

**5-fold cross-validation.** All conditions used 5-fold cross-validation at the
patient level. Patients were stratified by class (PMG vs. HC) and partitioned into 5
folds, ensuring approximately equal class representation per fold. Within each fold,
15% of the non-test patients were held out as a validation set; the model checkpoint
with the lowest validation loss was retained and evaluated on the test fold.
Performance is reported as mean ± SD across the 5 folds.

> **Note — Discussion:** The interpretation of why performance differs between
> conditions — specifically the shortcut learning hypothesis, the metric inflation
> argument, and the position-biased downsampling concern — is addressed in the
> Discussion section.

---

## 3.3 Preprocessing

The preprocessing pipeline replicates, step-for-step, the procedure reported by Guha
et al. (2025) and was applied identically across all experimental conditions and both
datasets. Implementation used Python with OpenCV and scikit-image.

**Grayscale conversion.** Each RGB JPEG image was converted to single-channel
grayscale using the perceptual luminance-weighting formula Y = 0.299R + 0.587G +
0.114B, as defined by the ITU-R BT.601 standard. This step reduces computational
complexity and is a prerequisite for the grayscale-specific filters that follow.

**Min-max normalisation.** Pixel intensities were linearly rescaled to [0, 1] on a
per-image basis, standardising the dynamic range across scans acquired under
heterogeneous imaging conditions.

**CLAHE.** Local contrast was enhanced using Contrast Limited Adaptive Histogram
Equalisation (Zuiderveld, 1994), with clip_limit=2.0 and tile_grid_size=8×8. Unlike
global histogram equalisation, CLAHE applies localised equalisation within small image
tiles and bounds bin amplification via the clip limit, preventing noise
over-amplification while improving visibility of fine cortical structures such as gyral
folding patterns and the gray-white matter boundary.

**Bilateral filtering.** Noise reduction used the bilateral filter (Tomasi &
Manduchi, 1998) with kernel diameter=9 and sigma_color=sigma_space=75. The
dual-weighting mechanism (spatial proximity + intensity similarity) suppresses noise in
homogeneous regions while preserving sharp anatomical boundaries. Following Guha et
al. (2025), this was chosen over Non-Local Means, Anisotropic Diffusion, and Wavelet
Denoising after evaluation.

**Canny edge detection.** An edge map was computed using the Canny detector (Canny,
1986) with low_threshold=50, high_threshold=200, aperture_size=3. The multi-stage
pipeline (Gaussian smoothing → gradient estimation → non-maximum suppression →
hysteresis tracking) yields precise, thin edge contours with strong noise rejection,
selected over Sobel, Scharr, and Laplacian of Gaussian after evaluation on cortical
MRI. The binary edge map was blended with the filtered image at alpha=0.20.

> **Note — Results:** Visual examples of the preprocessing output for PMG and healthy
> control images, and a quantitative characterisation of how the pipeline affects images
> from each class, are presented in the Results section.

---

## 3.4 Models and Training

Two pretrained deep convolutional neural network architectures were evaluated,
following Guha et al. (2025): ResNet-101 (He et al., 2016) and DenseNet-201 (Huang et
al., 2017), both initialised with ImageNet weights. The final fully connected layer of
each model was replaced by a two-layer classification head: a linear projection to 256
units, ReLU activation, dropout (p=0.5), and a final linear layer to a single logit
(PMG vs. non-PMG), following the head design of Guha et al. (2025). The backbone was
frozen and only the classification head was trained. Input images were resized to
224×224 pixels and replicated across three channels to match the ImageNet input format.
Standard data augmentation was applied during training: random resized crop (80–100%
of the image area, resized to 224×224 pixels) and random horizontal flip.

Models were trained using the Adam optimiser with a learning rate of 5×10⁻⁴, weight
decay of 1×10⁻³, and a batch size of 32, for 20 epochs. Binary cross-entropy with
logits (BCEWithLogitsLoss) was used as the loss function across all conditions.
Performance was evaluated using accuracy, precision, recall, F1 score, and Cohen's κ,
reported as mean ± SD across the 5 cross-validation folds. F1 score was the primary
metric in all comparisons, given its sensitivity to class imbalance.

---

## 3.5 Ablation Study

To probe whether trained models rely on spatially localised image features or on
global low-level statistics, a black-box occlusion ablation was applied to all retained
checkpoints. For each test image, a random square patch with a side length equal to
20% of the shorter image dimension was zeroed out prior to inference. The occluded
test set was constructed once per split and then passed through every available
checkpoint (ResNet-101 and DenseNet-201, across all saved folds and conditions).
Classification metrics were computed on the occluded images and compared to the
corresponding unoccluded test performance. A substantial drop under occlusion is
consistent with the model attending to specific spatial regions; a negligible drop
suggests reliance on global texture or low-level statistics distributed across the
image, which would further support the shortcut-learning hypothesis.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K.Q. (2017). Densely connected convolutional networks. *CVPR 2017*.
- Guha, S., Bhandage, V., & Agarwal, A. (2025). Automated detection of polymicrogyria in pediatric patients using deep learning. *Scientific Reports*, 15(1):41662.
- Zhang, L., Abdeen, N., & Lang, J. (2024). A novel center-based deep contrastive metric learning method for the detection of polymicrogyria in pediatric brain MRI. *Computerized Medical Imaging and Graphics*, 114:102373.
- Ganz, M., Lyng, H., & Coll, L. (2026). Comment on "Automated detection of polymicrogyria in pediatric patients using deep learning". [Letter to the editor, May 2026.]
