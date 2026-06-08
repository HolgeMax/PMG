# Style Guide — PMG Classification Paper

This file is the single source of truth for writing style across all sections of this paper.
It is loaded by the `/style-check` skill to audit prose for consistency.

---

## 1. Terminology Glossary

Use the canonical form on the left. Never use the variants on the right.

| Canonical | Never use |
|---|---|
| PMG | polymicrogyria (after first mention in a section) |
| PMG-positive | label=1, label 1, PMG positive, PMG+ |
| PMG-negative | label=2, label 2, PMG negative, PMG- |
| uncertain | label=3, ambiguous, excluded |
| healthy controls | normal controls, non-PMG, HC subjects |
| HC | healthy control (after "healthy controls (HC)" is established) |
| ResNet-101 | ResNet101, resnet101, ResNet 101 |
| DenseNet-201 | DenseNet201, densenet201, DenseNet 201 |
| 5-fold cross-validation | five-fold cross-validation, 5 fold cross-validation |
| mean ± SD | mean ± std, mean ± s.d., M ± SD |
| pre-split | presplit, pre split |
| post-split | postsplit, post split |
| patient-level split | patient level split, subject-level split |
| PPMR dataset | Pediatric Polymicrogyria MRI dataset (after first mention) |
| PACS | picture archiving and communication system (after first mention) |
| CLAHE | Contrast Limited Adaptive Histogram Equalisation (after first mention) |
| 224 × 224 pixels | 224x224, 224×224 pixels (no space around ×) |
| $1{,}508 \times 1{,}727$\,pixels | 1508×1727, 1508 × 1727 (LaTeX thousand-separators required) |
| paediatric | pediatric (British spelling throughout) |
| artefact | artifact |
| normalisation | normalization (British spelling throughout) |
| Guha et al.\ (2025) | Guha & Bhandage (2025), Guha et al. 2025 |
| Zhang et al.\ (2024) | Zhang (2024) |
| shortcut learning | shortcut-learning (no hyphen when used as noun phrase) |

---

## 2. Tense and Voice

**Methods section:** Past tense throughout.
> ✓ "Models were trained for 20 epochs."
> ✗ "Models are trained for 20 epochs."

**Results section:** Past tense for reported findings; present tense only for referring to figures/tables.
> ✓ "ResNet-101 achieved F1 = 0.969 ± 0.059 under Condition 5."
> ✓ "Table~\ref{tab:results} reports five-fold cross-validation performance."
> ✗ "ResNet-101 achieves F1 = 0.969."

**Discussion section:** Present tense for interpretation and general claims; past tense when referring back to specific results.
> ✓ "These findings suggest that the models rely on global texture statistics."
> ✓ "F1 dropped from 0.969 to 0.449 when labels were corrected."

**Voice:** Prefer active where possible. Reserve passive for methods steps where the agent is obvious.
> ✓ "We applied 5-fold cross-validation." (active, preferred)
> ✓ "Preprocessing was applied identically across all conditions." (passive acceptable — agent obvious)
> ✗ "It was found that performance dropped." (passive obscures agency unnecessarily)

---

## 3. Citation Format (APA / natbib)

Use `natbib` commands throughout.

| Use case | Command | Output |
|---|---|---|
| Author-year, in-text subject | `\citet{key}` | Guha et al. (2025) |
| Author-year, parenthetical | `\citep{key}` | (Guha et al., 2025) |
| Author name only | `\citeauthor{key}` | Guha et al. |
| Possessive | `\citeauthor{key}'s` | Guha et al.'s |
| Year only | `\citeyear{key}` | 2025 |

**Citation key conventions:**
- `guha_automated_2025`
- `zhang_novel_2024`
- `raghu_transfusion_2019`
- `stutterd_leventer_2014`

**Do not** write citations manually in prose (e.g. "(Guha et al., 2025)" in plain text).
Use `\citet` / `\citep` so the bibliography auto-generates.

**Reference list format (APA 7th):**
```
Author, A. A., & Author, B. B. (Year). Title of article. *Journal Name*, volume(issue), pages. https://doi.org/...
```

---

## 4. Figure and Table References

Always use `\ref{}` labels — never refer to figures by filename or number alone.

| Correct | Incorrect |
|---|---|
| `Figure~\ref{fig:scatter}` | "Figure Z (scatter.svg)" |
| `Table~\ref{tab:results}` | "Table Y" |
| `(see Figure~\ref{fig:scatter})` | "(see scatter.svg)" |

**Figure label convention:** `fig:` prefix + short descriptive slug.
> `fig:scatter`, `fig:dataset_pie`, `fig:pipeline_steps`, `fig:laplacian_variance`

**Table label convention:** `tab:` prefix.
> `tab:conditions`, `tab:results`, `tab:ablation`

**Caption style:** Full sentence ending with a period. First sentence states what the figure shows. Second sentence (if needed) highlights the key observation.
> ✓ "Scatter plot of image dimensions for all PPMR slices, coloured by class. PMG-patient images cluster at ${\approx}1508 \times 1727$\,px; healthy control images cluster at ${\leq}512 \times 512$\,px."

---

## 5. Academic Register

**Results section — do not interpret.** State numbers and observations only. Reserve causal claims and explanations for Discussion.
> ✓ "ResNet-101 F1 decreased from 0.965 ± 0.058 (Condition 4) to 0.728 ± 0.105 (Condition 6)."
> ✗ "This shows that label noise degrades performance." (interpretation → Discussion)

**Hedging:** Use appropriately in Discussion, not in Results or Methods.
> ✓ "These results are consistent with shortcut learning." (Discussion)
> ✗ "This might suggest the model could possibly be learning shortcuts." (over-hedged)
> ✗ "The model clearly learns shortcuts." (over-stated in Results)

**No colloquialisms.**
> ✗ "smoking gun", "nailed it", "basically", "pretty much"

**Numbers in prose:** Spell out one through nine; use numerals for 10 and above, and always for measurements.
> ✓ "three age-matched controls", "23 PMG patients", "F1 = 0.969"
> ✗ "3 age-matched controls", "twenty-three PMG patients"

**Thousands separator:** Use `{,}` in LaTeX math mode; commas in plain prose.
> ✓ "$4{,}517$ slices" in LaTeX; "4,517 slices" in markdown

**Percentages:** Use the `\%` symbol in LaTeX; no space before it.
> ✓ "approximately 49\% of slices"
> ✗ "approximately 49 % of slices"

**Uncertainty notation:** Always `mean ± SD` with a thin space in LaTeX: `$0.969 \pm 0.059$`.

---

## 6. Section-Specific Rules

**Abstract:** Present tense for background; past tense for what this study did and found.

**Introduction:** Present tense for established facts and motivation; past tense for prior work.

**Methods:** Past tense throughout. No results, no interpretation.

**Results:** Past tense for findings. Present tense only for figure/table references. No interpretation.

**Discussion:** Present tense for interpretation. Past tense when referring to specific results.
