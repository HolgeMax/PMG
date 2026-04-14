# Methods / Data Section — Draft Versions

**Preamble additions required:**
```latex
\usepackage{float}   % enables [H] for exact figure placement
\usepackage{svg}     % enables \includesvg
```

**Working notes mapped to academic points:**
- "check slices randomly chosen middle/start/end" → position-biased downsampling concern
- "fold-cross / middle slices" → cross-validation + anatomically-informed slice selection
- "print shape / check channels after grayscale" → preprocessing verification (methods detail, not main text)
- Red `\textcolor{red}{...}` text → replaced with precise citation to Guha et al. methods section

---

## Version A — Sequential: Label error → Resolution error → Downsampling error → Correct method

**Rationale:** Each issue is introduced, supported by a figure, then resolved. Clean for a reader
encountering the dataset for the first time.

```latex
% PREAMBLE (add if not present):
% \usepackage{float}
% \usepackage{svg}

\subsection*{Dataset}

This project uses the publicly available Pediatric Polymicrogyria MRI (PPMR) dataset
introduced by \citet{zhang_novel_2024} and subsequently used by \citet{guha_automated_2025}.
The dataset consists of coronal 3D gradient-echo T1-weighted MRI slices from 23 paediatric
epilepsy patients with confirmed polymicrogyria and three age- and gender-matched healthy
controls per patient, yielding a 3:1 control-to-case patient ratio.

\subsection*{Issue 1 — Mislabelled positive class}

A critical property of the dataset is that not all MRI slices from PMG patients contain
visible malformation. Each slice was individually annotated by a paediatric neuroradiologist
as PMG-positive (label~1), PMG-negative (label~2), or uncertain (label~3)
\cite{zhang_novel_2024}. As shown in Figure~\ref{fig:pmg_slices}, only approximately 49\%
of all PMG-patient slices are labelled as PMG-positive; approximately 30\% are PMG-negative,
and approximately 19\% are annotator-uncertain.

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/bar_pmgcases}
    \caption{Top: Number of 2D slices per PMG patient labelled as PMG-positive (red),
    PMG-negative (blue), and uncertain (grey), based on the Kaggle release of the PPMR
    dataset \cite{zhang_novel_2024}. Bottom: The same distribution excluding uncertain
    slices, replicating the figure published in the original dataset paper \cite{zhang_novel_2024}.}
    \label{fig:pmg_slices}
\end{figure}

Aggregating across all labels, the dataset contains 4{,}517 slices attributed to PMG
patients and 10{,}539 slices attributed to healthy controls. \citet{guha_automated_2025}
report using precisely these counts in their analysis, treating all 4{,}517 PMG-patient
slices as a single positive class and all 10{,}539 control slices as a single negative
class. This is the first methodological error: the positive class contains a large
proportion of non-pathological slices, diluting the PMG-specific signal and introducing
label noise into both training and evaluation.

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/bar_pmg+controls}
    \caption{Full slice-level class distribution of the PPMR dataset \cite{zhang_novel_2024},
    showing PMG-positive, PMG-negative, uncertain, and healthy control slices. The counts
    reported by \citet{guha_automated_2025} (4{,}517 PMG vs.\ 10{,}539 HC) correspond to
    collapsing all PMG-patient slices into a single positive class regardless of individual
    slice label.}
    \label{fig:paper_bar_plot}
\end{figure}

\subsection*{Issue 2 — Systematic resolution confound}

A second confound is a systematic difference in image resolution between classes.
As shown in Figure~\ref{fig:scatter}, PMG-patient images are consistently exported from
the picture archiving and communication system (PACS) at approximately
$1508 \times 1727$\,pixels, whereas healthy control images range from
$260 \times 320$\,pixels to $512 \times 512$\,pixels. This resolution difference is
class-correlated and independent of cortical pathology: a classifier could achieve
above-chance accuracy by exploiting pixel-level texture statistics arising from
heavy downsampling versus near-native-resolution artefacts due resizing to the model input dimension,
without learning any PMG-specific features.

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/scatter}
    \caption{Scatter plot of image dimensions (width $\times$ height in pixels) for all
    slices in the PPMR dataset, coloured by class. PMG-patient images cluster at
    ${\approx}1508 \times 1727$\,px; healthy control images cluster at
    ${\leq}512 \times 512$\,px. This systematic difference constitutes a low-level
    confound that is exploitable by a convolutional classifier without any learning of
    cortical morphology.}
    \label{fig:scatter}
\end{figure}

\subsection*{Issue 3 — Pre-split downsampling introduces data leakage}

\citet{guha_automated_2025} acknowledge the 3:1 natural class imbalance and address it
by globally downsampling the 10{,}539 control slices to 4{,}517 before splitting the data
into training, validation, and test sets. This procedure introduces data leakage.
Specifically, the 6{,}022 discarded control slices may have belonged to patients who would
otherwise have been represented in the validation or test set; after downsampling, those
patients are absent or underrepresented in evaluation. Consequently, the composition of
the test set is determined jointly by the downsampling step and the split step, rather
than by the split alone. Furthermore, because balancing is applied globally before the
split, the validation and test sets are also balanced at a 1:1 ratio, which is not
representative of the true clinical prevalence of PMG. This artificially inflates
evaluation metrics — accuracy, recall, and F1 — relative to what would be observed in a
real-world deployment where PMG is rare.

A further concern is whether the randomly retained control slices are drawn uniformly
across brain positions (anterior, central, posterior) or are disproportionately drawn
from the central region, which contains the most anatomically distinctive slices.
Position-biased retention could introduce an additional confound, as central slices
systematically differ in cortical appearance from peripheral slices regardless of
pathology.

\subsection*{Corrected data handling}

The methodologically correct procedure is as follows. First, splits are defined at the
\emph{patient level}, ensuring that all slices from a given patient appear exclusively
in one partition. Second, downsampling of the healthy control class is applied
\emph{within the training split only}, after the patient-level division is complete.
Third, the validation and test splits are left at their natural, imbalanced
distributions. Evaluation metrics computed on imbalanced held-out sets reflect the
performance a model would achieve in clinical deployment; balanced evaluation sets
conceal the high false-positive rate that would arise when PMG prevalence is low.
Additionally, cross-validation at the patient level is recommended given the small
cohort size (23 PMG patients), and anatomically-informed slice selection — prioritising
central slices where PMG is most likely to be visible — may further reduce the variance
of training signal.
```

---

## Version B — Two-issue thesis upfront, then detail

**Rationale:** States both errors as a thesis claim in the opening paragraph, then
unpacks each with figures. Better for a reader who already knows the dataset.

```latex
\subsection*{Dataset and methodological errors}

This project uses the publicly available Pediatric Polymicrogyria MRI (PPMR) dataset
\cite{zhang_novel_2024}. The dataset contains coronal T1-weighted MRI slices from 23
paediatric PMG patients and three matched healthy controls per patient (3:1 patient
ratio; 4{,}517 PMG-patient slices and 10{,}539 control slices at the slice level).
We identify two methodological errors in the analysis by \citet{guha_automated_2025}
that together render their reported performance metrics uninterpretable as a measure of
genuine PMG detection: (1) the positive class is contaminated with PMG-negative and
annotator-uncertain slices, and (2) a systematic resolution confound between classes
provides a trivially exploitable low-level signal.

\subsubsection*{Error 1: Label contamination of the positive class}

Individual slices from PMG patients were annotated as PMG-positive (label~1),
PMG-negative (label~2), or uncertain (label~3) \cite{zhang_novel_2024}. As shown in
Figure~\ref{fig:pmg_slices}, the 4{,}517 PMG-patient slices break down as approximately
49\% PMG-positive, 30\% PMG-negative, and 19\% uncertain. \citet{guha_automated_2025}
treat the full 4{,}517 as a homogeneous positive class, introducing substantial label
noise into training and evaluation (Figure~\ref{fig:paper_bar_plot}).

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/bar_pmgcases}
    \caption{Slice-level label distribution per PMG patient. Top: all three labels
    (PMG-positive in red, PMG-negative in blue, uncertain in grey). Bottom: excluding
    uncertain slices, replicating the figure in \citet{zhang_novel_2024}.}
    \label{fig:pmg_slices}
\end{figure}

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/bar_pmg+controls}
    \caption{Full slice-level class distribution of the PPMR dataset \cite{zhang_novel_2024}.
    The counts used by \citet{guha_automated_2025} (4{,}517 PMG vs.\ 10{,}539 HC) collapse
    all PMG-patient slices into a single positive class, conflating pathological and
    non-pathological slices.}
    \label{fig:paper_bar_plot}
\end{figure}

\subsubsection*{Error 2: Resolution confound}

PMG-patient images are exported at approximately $1508 \times 1727$\,px; healthy control
images range from $260 \times 320$\,px to $512 \times 512$\,px (Figure~\ref{fig:scatter}).
This class-correlated resolution difference is a consequence of different PACS export
settings across scanner types, not of cortical pathology. When all images are resized to
the model input dimension (224$\times$224\,px), PMG slices are consistently downsampled
by a factor of approximately 7$\times$ while control slices are resized near 1:1,
producing systematic differences in texture statistics that a convolutional network can
exploit without learning any PMG-specific morphology.

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/scatter}
    \caption{Image dimensions for all PPMR slices. PMG-patient images (red) cluster at
    ${\approx}1508 \times 1727$\,px; healthy control images (blue) cluster at
    ${\leq}512 \times 512$\,px, constituting a class-correlated low-level confound.}
    \label{fig:scatter}
\end{figure}

\subsubsection*{Error 3: Pre-split downsampling}

To address the 3:1 imbalance, \citet{guha_automated_2025} downsample the 10{,}539
control slices to 4{,}517 globally, prior to splitting. This introduces data leakage:
the identity of patients present in the evaluation sets is partly determined by the
downsampling step. Additionally, the resulting validation and test sets are balanced
at 1:1, inflating all reported metrics relative to a clinically realistic imbalanced
evaluation. A further concern is positional bias in the retained slices: uniform random
retention may over-represent central brain slices, which are visually more distinctive
than peripheral slices, independently of PMG status.

\subsubsection*{Corrected procedure}

Splits should be defined at the patient level before any downsampling. HC downsampling
should then be applied within the training partition only. Validation and test sets
should preserve the natural class distribution to ensure that reported metrics are
interpretable as estimates of real-world clinical performance. Patient-level
cross-validation is recommended to account for the small cohort size, and selecting
central brain slices preferentially for training may reduce label noise by
concentrating the training signal on slices where PMG is most likely to be visible.
```

---

## Version C — Integrated narrative (most concise)

**Rationale:** Presents all issues as a single, flowing critique without subheadings.
Suitable if this section is part of a longer methods section rather than a standalone critique.

```latex
\subsection*{Dataset}

This project uses the publicly available Pediatric Polymicrogyria MRI (PPMR) dataset
\cite{zhang_novel_2024}, consisting of coronal T1-weighted MRI slices from 23 paediatric
PMG patients and three matched healthy controls per patient. Individual slices from PMG
patients were annotated as PMG-positive (label~1), PMG-negative (label~2), or uncertain
(label~3), reflecting the focal and variable nature of the malformation within a single
brain (Figure~\ref{fig:pmg_slices}). At the slice level, the dataset contains 4{,}517
PMG-patient slices — of which only approximately 49\% are labelled PMG-positive, with
the remaining 30\% PMG-negative and 19\% uncertain — and 10{,}539 healthy control slices
(Figure~\ref{fig:paper_bar_plot}).

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/bar_pmgcases}
    \caption{Slice-level label distribution per PMG patient (PMG-positive: red;
    PMG-negative: blue; uncertain: grey). Only a minority of slices per patient
    contain visible malformation, confirming the focal nature of PMG.}
    \label{fig:pmg_slices}
\end{figure}

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/bar_pmg+controls}
    \caption{Full slice-level distribution of the PPMR dataset \cite{zhang_novel_2024}.
    \citet{guha_automated_2025} use all 4{,}517 PMG-patient slices as a single positive
    class and all 10{,}539 HC slices as a single negative class, conflating PMG-positive,
    PMG-negative, and uncertain slices within the positive class.}
    \label{fig:paper_bar_plot}
\end{figure}

The analysis by \citet{guha_automated_2025} is affected by three compounding
methodological errors. First, by treating all PMG-patient slices as positive examples,
the positive class contains a large proportion of non-pathological and ambiguous slices,
introducing label noise throughout training and evaluation. Second, as shown in
Figure~\ref{fig:scatter}, PMG-patient images are exported from the PACS system at
approximately $1508 \times 1727$\,pixels, whereas healthy control images range from
$260 \times 320$\,px to $512 \times 512$\,px. This class-correlated resolution
difference is an acquisition artefact unrelated to cortical pathology; once all images
are resized to the model input dimension, it manifests as systematic differences in
texture statistics that a convolutional network can exploit via shortcut learning.

\begin{figure}[H]
    \centering
    \includesvg[width=\linewidth]{plot/data_dist/scatter}
    \caption{Image export dimensions for all PPMR slices. PMG-patient images consistently
    cluster at ${\approx}1508 \times 1727$\,px; healthy control images at
    ${\leq}512 \times 512$\,px. This class-correlated resolution difference constitutes
    a trivially exploitable low-level confound.}
    \label{fig:scatter}
\end{figure}

Third, \citet{guha_automated_2025} address the class imbalance by globally downsampling
the 10{,}539 control slices to 4{,}517 \emph{before} splitting into training, validation,
and test sets. This violates the independence of the evaluation sets in two ways: the
identities of patients present in evaluation are partly determined by the downsampling
step, and the resulting validation and test sets are artificially balanced at 1:1,
inflating reported metrics — accuracy, recall, and F1 — relative to clinically realistic
conditions where PMG is rare. An additional concern is whether the retained control
slices are drawn uniformly across brain positions or are disproportionately central,
which could introduce a further positional confound independent of pathology.

The methodologically sound approach splits the data at the patient level first, then
downsamples HC slices within the training partition only, leaving validation and test
sets at their natural imbalanced distributions. This ensures that evaluation metrics
reflect the performance achievable in clinical deployment. Patient-level cross-validation
is recommended given the small cohort, and selecting central brain slices — where PMG is
most likely visible — may further sharpen the training signal.
```

---

## Notes

- All figures use `[H]` (requires `\usepackage{float}` in preamble) — figures will not
  drift on recompile.
- `.svg` extension is omitted from `\includesvg` paths (required by the `svg` package).
- Filenames containing `+` (e.g. `bar_pmg+controls`) may cause LaTeX parsing errors.
  Rename to `bar_pmg_controls` and update paths, or wrap in `\detokenize{}`.
- The `\textcolor{red}{...}` working note has been replaced with a precise claim in all
  versions: Guha et al. report using 4,517 PMG and 10,539 HC slices in their methods
  section, which corresponds to collapsing all PMG-patient labels into a single class.
- The debugging notes ("print shape", "check channels") are methods-verification steps,
  not paper text — not included here.
