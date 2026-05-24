# Introduction and Abstract — Drafts

**What changed (supervisor comments addressed):**
- Comment 11: Current intro was too result-heavy ("reads like an abstract"). Results moved to a separate Abstract draft below. Introduction rewritten as a proper motivational funnel.
- Comment 1: Introduction now starts broad (ML in medical imaging) and narrows to PMG → reproducibility study.
- Comment 3: PMG clinical background introduced first; the open dataset and two published solutions introduced as the response to the problem.
- Comment 4: "Both attempts lacked domain knowledge" softened — one more so than the other.
- Comment 5: PMG clinical paragraph now precedes the ML-gap paragraph.
- Comments 2, 6, 9, 10: Language and spelling corrected throughout.
- Comment 8: Unclear "global downsampling of dataset size" phrasing clarified in the abstract.

---

## Abstract

```latex
Polymicrogyria (PMG) is a malformation of cortical development associated with
epilepsy, neurodevelopmental delay, and motor impairment. Automated detection from MRI
has been proposed as a means to support specialist neuroradiologists, but requires
methodologically sound evaluation to distinguish genuine pathology detection from
artefactual performance. This study undertakes an independent replication and
methodological evaluation of \citet{guha_automated_2025}, who reported near-perfect
deep learning classification of PMG on the Pediatric Polymicrogyria MRI (PPMR)
dataset \citep{zhang_novel_2024}. We identify three systematic errors in their
experimental setup: (1) all MRI slices from PMG patients are treated as positive
examples, despite only approximately 49\% of those slices carrying
neuroradiologist-confirmed PMG annotations (2,256 of 4,517); (2) PMG-patient images
are exported at a median resolution of $1508 \times 1727$\,pixels versus $260 \times
320$ to $512 \times 512$\,pixels for healthy controls, producing a class-correlated
texture confound that enables shortcut learning; and (3) healthy control slices are
globally downsampled to match the PMG class count prior to data splitting, producing
artificially balanced evaluation sets that do not reflect clinical prevalence. Under a
faithful replication of \citeauthor{guha_automated_2025}'s setup using 5-fold
cross-validation, we obtain $\text{F1} = 0.965 \pm 0.068$, consistent with their
reported values. Applying corrective measures --- restricting the positive class to
label\,=\,1 slices and enforcing patient-level splits --- reduces F1 to $0.755 \pm
0.096$, consistent with shortcut learning rather than genuine PMG detection. We
further evaluate the corrected pipeline on an independent clinical dataset of 183
T1-weighted MRI scans from 162 epilepsy patients collected in the Eastern Region of
Denmark.
```

---

## Introduction

```latex
Deep learning has transformed the analysis of medical images, achieving performance
comparable to clinical experts across a range of tasks including diabetic retinopathy
grading, skin lesion classification, and radiological triage \cite{CITATION_ML_REVIEW}.
In neuroimaging, convolutional neural networks have been applied to the automated
detection of structural brain abnormalities from MRI, with the promise of supporting
radiologists in high-volume screening and reducing diagnostic delays in underserved
settings. Realising this promise, however, requires that reported performance metrics
are grounded in sound methodology --- free from data leakage, label noise, and
low-level image confounds that can inflate apparent accuracy without reflecting genuine
pathology detection.

Polymicrogyria (PMG) is a malformation of cortical development characterised by
excessive cortical folding and abnormal lamination \cite{CITATION_1}. It is one of the
most common malformations of cortical development, accounting for approximately 20\% of
all such malformations \cite{stutterd_leventer_2014}. PMG is frequently encountered in
patients with drug-resistant epilepsy and is commonly associated with
neurodevelopmental delay, motor impairments, and cognitive deficits. Despite its
clinical importance, PMG remains challenging to characterise reliably due to its
heterogeneous radiological appearance, variable anatomical distribution, and the
limited availability of curated neuroimaging datasets.

Deep learning attempts to classify PMG have predominantly been conducted on small,
imbalanced cohorts without rigorous patient-level data splitting \cite{CITATION_2},
making it difficult to distinguish genuine classification performance from artefacts of
data leakage, class imbalance, or low-level image differences introduced during
acquisition. Furthermore, the use of models pretrained on natural image datasets ---
comprising everyday photographs of cars, birds, and household objects --- introduces a
substantial domain gap relative to clinical neuroimaging; ImageNet-pretrained features
have been shown to transfer poorly beyond the lowest convolutional layers, offering
little benefit over task-specific training \citep{raghu_transfusion_2019}, which may
further limit the sensitivity of these models to the subtle cortical morphology features
relevant to PMG classification.

Automated detection of PMG from MRI could nonetheless inform clinical decision-making
by reducing the diagnostic burden on specialist neuroradiologists and enabling earlier
identification of the condition. Fortunately, a publicly available benchmark dataset
--- the Pediatric Polymicrogyria MRI (PPMR) dataset \citep{zhang_novel_2024} --- has
been released, and two independent studies have applied deep learning methods to PMG
classification using this resource \citep{zhang_novel_2024, guha_automated_2025}.

The present study therefore undertakes an independent replication and methodological
evaluation of \citet{guha_automated_2025}. We examine the assumptions underlying their
experimental setup, assess whether the reported near-perfect classification accuracy
can be attributed to genuine PMG detection, and quantify the effect of applying
corrective measures to the identified methodological issues. In addition, we evaluate
the corrected pipeline on an independent clinical dataset of epilepsy patients,
providing an assessment of generalisation beyond the PPMR benchmark. We identify three
methodological errors in \citeauthor{guha_automated_2025}: (1) label noise from
including PMG-negative slices in the positive class; (2) a systematic class-correlated
resolution difference that enables shortcut learning; and (3) pre-split healthy
control downsampling that produces artificially balanced evaluation sets, obscuring the
true class distribution in validation and test. We quantify the effect of each error
and evaluate the corrected pipeline on an independent clinical dataset.
```

---

## Section 3.1 — Preprocessing as a Confound Amplifier

```latex
\subsection{Preprocessing pipeline and scanner-induced confounds}

The preprocessing pipeline described by \citet{guha_automated_2025} --- comprising
min--max normalisation, contrast-limited adaptive histogram equalisation (CLAHE),
bilateral filtering, and Canny edge detection --- does not correct for the
scanner-induced resolution differences identified above. Spatial operations with
fixed kernel sizes (e.g.\ bilateral filter radius, Canny thresholds) behave
differently depending on the effective resolution of the input image: applied to
PMG images at median $1508 \times 1727$\,pixels versus control images at
$512 \times 512$\,pixels or below, these operations extract texture at systematically
different spatial scales before any resizing step equalises the pixel grid. Rather than
harmonising the two classes, the pipeline therefore risks amplifying the
class-correlated texture differences introduced by the PACS export disparity into a
stronger, more learnable signal. This may account in part for the near-perfect
classification accuracy reported by \citet{guha_automated_2025}, which cannot be
attributed to PMG pathology alone. Visual inspection of preprocessed images
corroborates this interpretation: after applying the full pipeline, PMG and healthy
control images remain visually distinguishable by texture alone, without reference to
cortical morphology.

To validate findings against an independent clinical sample, we included 183
T1-weighted MRI scans from 162 unique patients evaluated for epilepsy (77 females;
mean $\pm$ SD age at acquisition: $25.8 \pm 17.4$ years). Of these, 110 scans (49
females; $24.7 \pm 15.9$ years) were reported to contain radiological evidence of PMG
based on clinical radiology reports. The non-PMG group (37 females; $27.5 \pm 19.6$
years) consisted of scans in which PMG-related terminology appeared in the radiology
report, but a subsequent double-rater assessment confirmed the absence of PMG. Of
these non-PMG scans, 56 cases had other radiological abnormalities and 17 were
reported as having a normal MRI.
```

---

## Issues to Resolve Before Submitting

1. **Placeholder citation keys** — replace before compiling:
   - `CITATION_ML_REVIEW`: a review of deep learning in medical imaging (e.g. Litjens et al. 2017, *Medical Image Analysis*).
   - `CITATION_1`: PMG definition / cortical folding (e.g. Severino et al. 2020, *Brain*).
   - `CITATION_2`: prior ML/MRI studies on PMG with limited sample sizes (e.g. Zhang et al. `\cite{zhang_novel_2024}` or a broader review).

2. **The ~49% and ~19% slice figures** — confirm exact values from `JPEG_exploration.ipynb`.
   Exact counts: 2,256 PMG-positive, 1,386 PMG-negative, 875 uncertain, out of 4,517 total PMG-patient slices.

3. **Resolution figures** (median 1508×1727 px vs 260–512 px) — from `JPEG_exploration.ipynb`.
   Clarify in methods that all images are resized to 224×224 for training, but the original export resolution creates systematic compression differences between classes that affect texture statistics before and during preprocessing.

4. **F1 drop figures** — from `Metrics_exploration.ipynb`:
   - ResNet-101: 0.97 → 0.74 (used in abstract, consistent with Guha et al.'s own best-reported architecture).
   - DenseNet-201 5-fold CV: 0.965±0.068 → 0.755±0.096 (cite in results/supplementary).

5. **Third error (data leakage)** — confirm exact description of how controls were globally downsampled in Guha et al.'s setup and cite the relevant section of their paper or supplementary if available.

6. **Section 3.1 placement** — confirm with supervisor whether the clinical dataset paragraph belongs in 3.1 (methods preamble) or in a dedicated participants/dataset subsection.

7. **Old P4** ("regional cortical surface-based features...") has been permanently removed. Do not reintroduce it.
