# 5. Results

Replicating the setup of Guha et al.\ (2025) (Condition~5: preprocessed images, paper-defined labels, pre-split downsampling) yields a mean F1-score of $0.969 \pm 0.059$ for ResNet101 and $0.967 \pm 0.064$ for DenseNet201 across five cross-validation folds — closely matching the reported performance of $0.965$ in the original paper. However, correcting all three identified methodological issues simultaneously (Condition~3: raw images, corrected labels, post-split downsampling) reduces F1 to $0.449 \pm 0.071$ for ResNet101 and $0.414 \pm 0.172$ for DenseNet201. The following subsections characterise each source of this performance gap in turn.

---

## 5.1 Dataset Characterisation and Methodological Discrepancies

### 5.1.1 Label Distribution

The dataset comprises 14,181 slices drawn from 162 patients: 23 PMG cases with three age- and sex-matched healthy controls each, yielding a 3:1 HC-to-PMG ratio at patient level. Each patient corresponds to a unique individual; the three controls per PMG patient are three separate people (Zhang et al., 2024, Section 4.1).

Radiological labels assigned to slices within the PMG cohort fall into three categories: PMG-positive (label = 1), PMG-negative (label = 2), and uncertain (label = 3). Under the Guha et al.\ protocol, the entire PMG cohort is treated as a single positive class, regardless of slice-level label, producing approximately 4,517 positive slices (30.8% of total). Under the corrected protocol, only label = 1 slices are counted as positive, yielding 2,256 positive slices (15.9%). The full label distribution is shown in Figure~X (dataset\_dist\_pie.svg).

This difference in label definition constitutes a 2.0× inflation of the reported positive class and introduces noise into the training signal: slices explicitly marked as PMG-negative or uncertain by the radiologist are treated as evidence of pathology.

The class balance comparison between the two labelling approaches is shown in Figure~Y (class\_balance\_pie.svg). Under the corrected protocol the positive class is substantially smaller, increasing the class imbalance from approximately 30:70 (Guha: 4,517 PMG / 10,539 controls) to 16:84 (corrected: 2,255 PMG-positive / 11,926 controls, excluding uncertain slices).

### 5.1.2 Resolution Confound

PMG scans in this dataset were acquired at a substantially different native resolution than healthy controls. PMG cases present at approximately $1{,}508 \times 1{,}727$ pixels, whereas control images range from $260 \times 260$ to $512 \times 512$ pixels — a roughly six-fold difference in linear resolution (Figure~Z: scatter.svg). After resizing all images to the model input size of $224 \times 224$ pixels, PMG images are downsampled by a factor of approximately six, while control images are upsampled or only modestly downsampled. This creates a systematic artefact in the spatial frequency and texture content of the two classes that is independent of pathology.

The resolution confound is present in all seven experimental conditions because it is a property of the raw data: the systematic size difference between PMG and control scans is encoded in every JPEG slice. Correcting it would require access to the original DICOM files in order to resample all volumes to a common voxel spacing before conversion to JPEG — a step that cannot be applied retrospectively to the existing slices. Beyond the data acquisition stage, the preprocessing pipeline introduces a second, compounding layer: because CLAHE and the bilateral filter operate with fixed hyperparameters regardless of input resolution, their effect on image appearance is not uniform across classes (see Section~5.2 and Figure~A). Conditions 1–3 (raw JPEG) therefore carry the confound in its base form, while Conditions 4–7 (preprocessed) carry both the base confound and its pipeline amplification. Condition~3 is the most conservative evaluation in this study because it avoids the amplification introduced by preprocessing, while also applying corrected labels and post-split downsampling.

<!-- ── PROPOSED NEW TEXT — start ────────────────────────────────────────────── -->
To quantify this effect, Laplacian variance — a measure of high-frequency image content — was computed for a random sample of 200 images per class at three successive stages: (1) native resolution, (2) after resizing to $224 \times 224$ pixels, and (3) after the full preprocessing pipeline followed by resizing. Results are shown in Figure~\ref{fig:laplacian_variance}. At native resolution, PMG images exhibited markedly lower Laplacian variance than HC images ($26.2 \pm 62.3$ vs $513.4 \pm 549.8$). After resizing, PMG variance increased by $+524.6$ to $550.8 \pm 692.1$, while HC variance increased by only $+124.1$ to $637.5 \pm 348.8$. Following the preprocessing pipeline, PMG variance increased a further $+99.0$ to $649.8 \pm 1{,}114.5$, whereas HC variance decreased slightly by $-16.3$ to $621.2 \pm 273.3$.
<!-- ── PROPOSED NEW TEXT — end ──────────────────────────────────────────────── -->

---

## 5.2 Classification Performance Across Experimental Conditions

Table~\ref{tab:conditions} defines the seven experimental conditions. Table~\ref{tab:results} reports five-fold cross-validation performance for both ResNet101 and DenseNet201 across all conditions.

### Effect of preprocessing alone (Conditions 1 vs.\ 4)

Applying the preprocessing pipeline (skull-stripping, CLAHE, normalisation) to paper-labelled data substantially increases performance. ResNet101 F1 rises from $0.708 \pm 0.091$ (raw, Condition~1) to $0.965 \pm 0.058$ (preprocessed, Condition~4). The same pattern holds for DenseNet201: $0.665 \pm 0.174$ to $0.967 \pm 0.061$. The preprocessing pipeline applies fixed hyperparameters — CLAHE tile grid size, clip limit, and bilateral filter kernel — uniformly across all images, regardless of their native resolution. Figure~A (preprocessing\_detail\_zoom\_default.png) illustrates the qualitatively different output this produces: on a high-resolution PMG scan the CLAHE enhancement is subtle, whereas on a lower-resolution control scan the same parameters produce markedly stronger local contrast amplification, sharpening structural edges throughout the image. This means the pipeline does not merely standardise image appearance; it also systematically differentiates the two classes along a texture axis tied to native resolution rather than pathology.

<!-- ── PROPOSED NEW TEXT — start ────────────────────────────────────────────── -->
The three-stage Laplacian variance analysis (Section~5.1.2, Figure~\ref{fig:laplacian_variance}) quantifies this asymmetry: preprocessing raised mean PMG variance by $+99.0$ relative to the resized-only baseline, while reducing mean HC variance by $-16.3$.
<!-- ── PROPOSED NEW TEXT — end ──────────────────────────────────────────────── -->

### Effect of pre-split downsampling (Conditions 4 vs.\ 5; Conditions 6 vs.\ 7)

Under paper labels, switching from no downsampling (Condition~4) to pre-split downsampling (Condition~5) yields a negligible change: ResNet-101 F1 moved from $0.965 \pm 0.058$ to $0.969 \pm 0.059$; DenseNet-201 from $0.967 \pm 0.061$ to $0.967 \pm 0.064$. The effect of pre-split downsampling is markedly larger under corrected labels. Comparing Condition~6 (preprocessed, corrected, no downsampling) to Condition~7 (preprocessed, corrected, pre-split), ResNet-101 F1 rose from $0.728 \pm 0.105$ to $0.868 \pm 0.097$ and DenseNet-201 from $0.739 \pm 0.091$ to $0.887 \pm 0.106$ — an increase of approximately 0.14 and 0.15 F1 points respectively. Pre-split downsampling selects a balanced subset before the train/val/test split is defined, meaning different slices from the same patient may appear in both training and evaluation; this is consistent with the larger inflation observed when the positive class is smaller and each correctly labelled PMG-positive slice carries more weight.

### Effect of label correction (Conditions 4 vs.\ 6; Conditions 5 vs.\ 7)

Correcting labels while keeping preprocessed images and no downsampling (Condition 4 → 6) reduced ResNet-101 F1 from $0.965 \pm 0.058$ to $0.728 \pm 0.105$ and DenseNet-201 from $0.967 \pm 0.061$ to $0.739 \pm 0.091$. When downsampling strategy is held fixed at pre-split (Condition 5 → 7), label correction reduced ResNet-101 F1 from $0.969 \pm 0.059$ to $0.868 \pm 0.097$ and DenseNet-201 from $0.967 \pm 0.064$ to $0.887 \pm 0.106$. In both cases label correction produced a substantial drop, though the drop is smaller when pre-split downsampling is retained (approximately 0.10 F1 points) than when no downsampling is used (approximately 0.23 F1 points), consistent with the leakage introduced by pre-split downsampling partially offsetting the label correction.

### Fully corrected protocol (Condition 3)

Condition~3 applies corrected labels and post-split downsampling to raw images, making it the most methodologically conservative configuration. ResNet101 achieves F1 $= 0.449 \pm 0.071$ and DenseNet201 achieves F1 $= 0.414 \pm 0.172$. Cohen's $\kappa$ falls to $0.336 \pm 0.079$ and $0.322 \pm 0.169$ respectively, indicating only fair agreement beyond chance. The high standard deviation for DenseNet201 across conditions involving corrected labels and raw data signals instability across folds.

---

## 5.3 Occlusion Ablation Study

To test whether model predictions depend on image content that is irrelevant to PMG pathology, a black-box occlusion was applied to the centre of each test image. The occlusion covers 25% of the image area (side length = 50% of the shorter image dimension). The best-epoch checkpoint from each cross-validation fold was evaluated on this modified test set; results are reported in Table~\ref{tab:ablation}.

Under the Guha replication conditions (Condition~5), ResNet101 retains an accuracy of $0.877 \pm 0.007$ with the centre masked. DenseNet201 achieves $0.797 \pm 0.093$. These figures are only marginally below the unoccluded cross-validation accuracies of $0.971$ and $0.970$ respectively. Under the fully corrected conditions (Condition~3), ResNet101 accuracy is $0.868 \pm 0.006$ with occlusion, compared to $0.806 \pm 0.041$ unoccluded — the occluded model performs similarly to the unoccluded model. DenseNet201 under Condition~3 shows high variance ($0.686 \pm 0.268$), with one fold collapsing to near-chance (fold~3: accuracy $= 0.221$), indicating instability under the balanced, corrected protocol.

The observation that masking the centre of the image does not substantially degrade performance — particularly under Condition~5 — is consistent with the hypothesis that the models rely on low-level image statistics present throughout the image (e.g., global texture, intensity distribution, or frequency content arising from the resolution confound) rather than on local cortical morphology.
