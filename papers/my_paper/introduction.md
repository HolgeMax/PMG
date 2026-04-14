# Introduction — Draft Versions

Three LaTeX versions of the introduction with different paragraph orderings.
Paragraph 1 is fixed in all versions. The old P4 ("surface-based features...") has been
replaced with a new P4 that correctly states the actual study aim.

**Legend:**
- P1 = Replication framing (FIXED)
- P2 = PMG clinical background
- P3 = Gap in prior work
- P4 = Actual study aim (revised — see note below)

**Empirical support for P4 (from `JPEG_exploration.ipynb` and `Metrics_exploration.ipynb`):**
- PMG images are exported from PACS at ~1508×1727 px; HC images range from 260×320 to 512×512 px
  — a systematic, class-correlated resolution difference independent of pathology.
- Annotator-uncertain slices (label=3) account for ~19% of PMG-patient slices and were
  included as positives in Guha et al.'s setup without justification.
- Correcting data usage (patient-level splits, proper label exclusion) drops ResNet-101 F1
  from 0.968 ± 0.056 to 0.737 ± 0.107 — a 23 percentage-point collapse.

---

## Version A — P1 → P2 → P3 → P4 *(Recommended)*

**Rationale:** Classic funnel — establishes the disease and the gap in prior work,
then lands on the specific failure this study documents.

```latex
This project is an attempt to reproduce Guha et al.\ \cite{guha_automated_2025} paper
``Automated detection of polymicrogyria in pediatric patients using deep learning''.
The article published by Scientific Reports utilised a publicly available Pediatric
Polymicrogyria MRI (PPMR) dataset \cite{zhang_novel_2024} on Kaggle to develop an
algorithm for the detection of polymicrogyria (PMG) in epilepsy patients.
This dataset was itself introduced by \citet{zhang_novel_2024}, who also applied it to
a separate PMG detection algorithm.
Both attempts lacked the domain knowledge necessary to use this dataset correctly,
leading to improper data usage and misleading algorithm performance.

Polymicrogyria (PMG) is a malformation of cortical development characterised by
excessive cortical folding and abnormal layering \cite{CITATION_1}. PMG is frequently
observed in patients with drug-resistant epilepsy and is commonly associated with
neurodevelopmental delay, motor impairments, and cognitive deficits. Despite its
clinical importance, PMG remains difficult to characterise reliably due to its
heterogeneous radiological appearance, variable anatomical distribution, and the
limited availability of curated neuroimaging datasets.

Previous deep learning studies on PMG have predominantly trained on small, imbalanced
paediatric cohorts without rigorous patient-level data splitting \cite{CITATION_2}.
This makes it difficult to distinguish genuine classification performance from
artifacts of data leakage, class imbalance, or low-level image confounds present in
the acquisition pipeline.

In this study, we demonstrate that the high classification accuracy reported by
\citet{guha_automated_2025} is an artifact of two methodological failures rooted in
insufficient domain knowledge of the PPMR dataset. First, all MRI slices from PMG
patients are treated as positive examples, despite the fact that only a subset of
slices per patient show PMG-related cortical abnormalities; PMG-negative slices and
annotator-uncertain slices — the latter comprising approximately 19\% of PMG-patient
slices — are incorrectly included as pathological examples. Second, PMG patient images
are exported from the picture archiving and communication system (PACS) at
approximately $1508 \times 1727$\,px, whereas healthy control images range from
$260 \times 320$\,px to $512 \times 512$\,px, providing a trivial, class-correlated
low-level signal that is independent of any cortical pathology. Correcting these errors
reduces the F1 score of the best-performing architecture from 0.97 to 0.74, consistent
with shortcut learning rather than genuine PMG detection.
```

---

## Version B — P1 → P3 → P2 → P4

**Rationale:** Problem-first — immediately follows the replication framing with the
methodological gap, then provides clinical context, then lands on the specific findings.

```latex
This project is an attempt to reproduce Guha et al.\ \cite{guha_automated_2025} paper
``Automated detection of polymicrogyria in pediatric patients using deep learning''.
The article published by Scientific Reports utilised a publicly available Pediatric
Polymicrogyria MRI (PPMR) dataset \cite{zhang_novel_2024} on Kaggle to develop an
algorithm for the detection of polymicrogyria (PMG) in epilepsy patients.
This dataset was itself introduced by \citet{zhang_novel_2024}, who also applied it to
a separate PMG detection algorithm.
Both attempts lacked the domain knowledge necessary to use this dataset correctly,
leading to improper data usage and misleading algorithm performance.

Previous deep learning studies on PMG have predominantly trained on small, imbalanced
paediatric cohorts without rigorous patient-level data splitting \cite{CITATION_2}.
This makes it difficult to distinguish genuine classification performance from
artefacts of data leakage, class imbalance, or low-level image confounds present in
the acquisition pipeline.

Polymicrogyria (PMG) is a malformation of cortical development characterised by
excessive cortical folding and abnormal layering \cite{CITATION_1}. PMG is frequently
observed in patients with drug-resistant epilepsy and is commonly associated with
neurodevelopmental delay, motor impairments, and cognitive deficits. Despite its
clinical importance, PMG remains difficult to characterise reliably due to its
heterogeneous radiological appearance, variable anatomical distribution, and the
limited availability of curated neuroimaging datasets.

In this study, we demonstrate that the high classification accuracy reported by
\citet{guha_automated_2025} is an artefact of two methodological failures rooted in
insufficient domain knowledge of the PPMR dataset. First, all MRI slices from PMG
patients are treated as positive examples, despite the fact that only a subset of
slices per patient show PMG-related cortical abnormalities; PMG-negative slices and
annotator-uncertain slices — the latter comprising approximately 19\% of PMG-patient
slices — are incorrectly included as pathological examples. Second, PMG patient images
are exported from the picture archiving and communication system (PACS) at
approximately $1508 \times 1727$\,px, whereas healthy control images range from
$260 \times 320$\,px to $512 \times 512$\,px, providing a trivial, class-correlated
low-level signal that is independent of any cortical pathology. Correcting these errors
reduces the F1 score of the best-performing architecture from 0.97 to 0.74, consistent
with shortcut learning rather than genuine PMG detection.
```

---

## Version C — P1 → P2 → P4 → P3

**Rationale:** Aim-early variant — states the specific failures immediately after
clinical context. The gap paragraph then serves as broader framing for why this
matters beyond the single paper being critiqued.

```latex
This project is an attempt to reproduce Guha et al.\ \cite{guha_automated_2025} paper
``Automated detection of polymicrogyria in pediatric patients using deep learning''.
The article published by Scientific Reports utilised a publicly available Pediatric
Polymicrogyria MRI (PPMR) dataset \cite{zhang_novel_2024} on Kaggle to develop an
algorithm for the detection of polymicrogyria (PMG) in epilepsy patients.
This dataset was itself introduced by \citet{zhang_novel_2024}, who also applied it to
a separate PMG detection algorithm.
Both attempts lacked the domain knowledge necessary to use this dataset correctly,
leading to improper data usage and misleading algorithm performance.

Polymicrogyria (PMG) is a malformation of cortical development characterised by
excessive cortical folding and abnormal layering \cite{CITATION_1}. PMG is frequently
observed in patients with drug-resistant epilepsy and is commonly associated with
neurodevelopmental delay, motor impairments, and cognitive deficits. Despite its
clinical importance, PMG remains difficult to characterise reliably due to its
heterogeneous radiological appearance, variable anatomical distribution, and the
limited availability of curated neuroimaging datasets.

In this study, we demonstrate that the high classification accuracy reported by
\citet{guha_automated_2025} is an artefact of two methodological failures rooted in
insufficient domain knowledge of the PPMR dataset. First, all MRI slices from PMG
patients are treated as positive examples, despite the fact that only a subset of
slices per patient show PMG-related cortical abnormalities; PMG-negative slices and
annotator-uncertain slices — the latter comprising approximately 19\% of PMG-patient
slices — are incorrectly included as pathological examples. Second, PMG patient images
are exported from the picture archiving and communication system (PACS) at
approximately $1508 \times 1727$\,px, whereas healthy control images range from
$260 \times 320$\,px to $512 \times 512$\,px, providing a trivial, class-correlated
low-level signal that is independent of any cortical pathology. Correcting these errors
reduces the F1 score of the best-performing architecture from 0.97 to 0.74, consistent
with shortcut learning rather than genuine PMG detection.

Previous deep learning studies on PMG have predominantly trained on small, imbalanced
paediatric cohorts without rigorous patient-level data splitting \cite{CITATION_2}.
This makes it difficult to distinguish genuine classification performance from
artefacts of data leakage, class imbalance, or low-level image confounds present in
the acquisition pipeline.
```

---

## Issues to Resolve Before Submitting

1. **Placeholder citation keys** — `CITATION_1` and `CITATION_2` must be replaced with
   real `.bib` keys before compiling.
   - `CITATION_1`: reference for PMG definition / cortical folding (e.g. Barkovich et al.).
   - `CITATION_2`: reference for prior ML/MRI studies on PMG with limited sample sizes
     (could be Zhang et al. \cite{zhang_novel_2024} or a broader review).

2. **The 19% uncertain slice figure** comes from `JPEG_exploration.ipynb`. Confirm
   the exact value before submitting; cite the dataset annotation protocol
   (Zhang et al. 2024, Section 4.1 / thesis Section 5.1.2) as the source for the
   three-label scheme (1=PMG present, 2=absent, 3=uncertain).

3. **The resolution figures** (1508×1727 vs 260–512 px) come from `JPEG_exploration.ipynb`.
   These are PACS export dimensions. Clarify in the methods section that this is the
   pre-resizing dimension — all images are resized to 256×256 for training, but the
   original export resolution creates systematic downsampling vs upsampling differences
   between classes that affect texture statistics.

4. **The F1 drop figure** (0.97 → 0.74) comes from `Metrics_exploration.ipynb`,
   ResNet-101, 5-fold CV. State the architecture and setup clearly if citing this
   number in the introduction.

5. **The old P4** ("In this study, we investigated whether regional cortical
   surface-based features can distinguish PMG cases from non-PMG epilepsy cases...")
   has been removed entirely — it described a different study aim and was inconsistent
   with the project's actual purpose. Do not reintroduce it.
```
