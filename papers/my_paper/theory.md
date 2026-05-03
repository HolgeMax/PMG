# Theory

## 2.1 Polymicrogyria

### 2.1.1 Neuropathology

Polymicrogyria (PMG) is a malformation of the developing brain characterized by abnormal cortical lamination and an excessive number of abnormally small gyri, resulting in an irregular folding pattern across all or part of the cerebral cortex (Stutterd & Leventer, 2014; MedlinePlus Genetics, 2024). PMG involves disruption of cortical organisation with or without fusion of the overlying molecular layer, and is frequently characterised by defects in the pial limiting membrane (Severino et al., 2020). As a malformation of cortical development (MCD), PMG arises during the post-migrational stage of cortical development, distinguishing it from other MCDs such as lissencephaly, which affects the migration stage (Severino et al., 2020).

Multiple forms of PMG have been identified. Unilateral PMG affects one hemisphere only, whereas bilateral PMG involves both hemispheres; both have further subcategories. Symptoms depend on how much of the brain is affected: the mildest form, unilateral focal PMG, affects a relatively small area on one side and may cause few or no symptoms, while bilateral forms tend to cause more severe neurological problems (Stutterd & Leventer, 2014).

Approximately 20% of all malformations of cortical development are accounted for by PMG, making it one of the most common brain malformations (Stutterd & Leventer, 2014).

### 2.1.2 Aetiology

PMG is aetiologically heterogeneous, arising from both genetic and environmental factors during the period of cortical organisation. More than 40 genes have been implicated in PMG (Stutterd & Leventer, 2014; MedlinePlus Genetics, 2024). Among these, mutations in *ADGRG1* cause bilateral frontoparietal PMG with cobblestone-like cortical features and cerebellar dysplasia (Piao et al., 2004).

Infectious causes — most notably congenital cytomegalovirus (cCMV) infection — represent the most common environmental trigger of PMG. The birth prevalence of cCMV is approximately 0.5–1% (Grosse et al., 2008; Kenneson & Cannon, 2007); of these, approximately 10–15% are symptomatic at birth, and among symptomatic cases, approximately 20% develop PMG (Lawson et al., 2025).

Focal intrauterine ischaemia, particularly in monochorionic twin pregnancies with twin-to-twin transfusion, can produce unilateral or focal PMG through disruption of the vascular supply to the developing cortex (Stutterd & Leventer, 2014; Park et al., 2021). However, in a substantial proportion of cases the aetiology remains unknown despite genetic testing (MedlinePlus Genetics, 2024).

### 2.1.3 Clinical Presentation

The clinical phenotype depends on the extent and location of the malformation.

- **Epilepsy** is the most frequent presenting feature and is often refractory to antiseizure medication; onset typically occurs in childhood.
- **Intellectual disability** affects the majority of patients with bilateral forms; up to 75% of bilateral perisylvian PMG cases have significant cognitive impairment (Stutterd & Leventer, 2014).
- **Oromotor dysfunction** — including dysarthria, dysphagia, and drooling — is prominent in perisylvian variants due to involvement of the opercular cortex.
- Focal unilateral PMG may present with mild or no symptoms and is often detected incidentally.

### 2.1.4 Diagnosis

Magnetic Resonance Imaging (MRI) provides high resolution and adequate soft-tissue contrast to identify the small folds that define PMG, which other imaging modalities lack — most notably Computed Tomography (CT). MRI is the most important imaging method in the evaluation of MCDs, owing to its optimal delineation of grey and white matter structures (Severino et al., 2020).

## 2.2 MRI in PMG Assessment

Magnetic Resonance Imaging (MRI) is the diagnostic gold standard for PMG. T1-weighted sequences provide high soft-tissue contrast — white matter appears bright and grey matter darker — enabling delineation of cortical thickness, sulcal morphology, and the gray-white junction irregularities characteristic of PMG. Computed Tomography (CT) is insufficient, as its lower soft-tissue contrast and spatial resolution fail to resolve the fine gyral morphology required for reliable PMG detection (Barkovich, 2010).

The standard sequence for structural cortical assessment is the Magnetisation Prepared Rapid Gradient Echo (MPRAGE) protocol, a three-dimensional T1-weighted gradient echo sequence. An inversion pulse is applied prior to data acquisition; the inversion time (TI) determines when signal is sampled relative to the longitudinal recovery of different tissues, optimising gray-white matter contrast. Thin-slice isotropic acquisitions (typically 1 mm³ voxels) allow multiplanar reformatting without loss of resolution. Our dataset, acquired at Hvidovre Hospital, Copenhagen on a 3 T Siemens Verio scanner, is summarised in Table 1. The Pediatric Polymicrogyria MRI (PPMR) dataset (Zhang et al., 2023), acquired at the Children's Hospital of Eastern Ontario on two scanner types — a 3 T Siemens Skyra and a 1.5 T General Electric Cigna magnet — is summarised in Table 2.

```latex
\begin{table}[h]
    \centering
    \begin{tabular}{ll}
      \toprule
      \textbf{Parameter} & \textbf{Value} \\
      \midrule
      Scanner              & Siemens Verio, 3\,T \\
      Sequence             & MPRAGE (T1-weighted) \\
      Repetition time (TR) & 1900\,ms \\
      Echo time (TE)       & 2.23\,ms \\
      Inversion time (TI)  & 900\,ms \\
      Flip angle           & 9° \\
      Voxel spacing        & $1 \times 1 \times 1$\,mm$^3$ \\
      \bottomrule
    \end{tabular}
    \caption{Acquisition parameters --- Hvidovre Hospital dataset.}
    \label{tab:hvidovre_params}
\end{table}

\begin{table}[h]
    \centering
    \begin{tabular}{lll}
      \toprule
      \textbf{Parameter} & \textbf{3\,T Siemens Skyra} & \textbf{1.5\,T GE Cigna} \\
      \midrule
      Sequence             & Coronal 3D GRE T1  & Coronal 3D GRE T1 \\
      Repetition time (TR) & 2200\,ms           & 10.44\,ms \\
      Inversion time (TI)  & 1030\,ms           & 450\,ms \\
      Echo time (TE)       & 2.63\,ms           & 4.3\,ms \\
      Matrix               & $320 \times 260$   & $512 \times 512$ \\
      Slice thickness      & 1.2\,mm            & 1.2\,mm \\
      Field of view (FOV)  & $20 \times 23$\,cm & $22 \times 27$\,cm \\
      \bottomrule
    \end{tabular}
    \caption{Acquisition parameters --- PPMR dataset \cite{zhang_novel_2024}.}
    \label{tab:ppmr_params}
\end{table}
```

Scanner variability is a practical concern in multi-site or retrospective datasets. Differences in field strength (1.5 T vs. 3 T), field of view (FOV), flip angle, and voxel spacing alter absolute signal intensities and spatial resolution, producing systematic differences in image appearance unrelated to underlying pathology. The PPMR dataset exemplifies this directly, having been acquired on two scanners with markedly different field strengths and acquisition parameters. When PMG and healthy control (HC) cohorts are not matched for acquisition parameters, a classifier may exploit these low-level statistical artefacts rather than genuine cortical morphology — a confounding risk that must be addressed in experimental design. The inherent subtlety of PMG imaging findings, combined with this acquisition variability, makes visual diagnosis challenging and motivates the development of automated detection tools.

---

## 2.3 Deep Learning for Medical Image Classification

### 2.3.1 Convolutional Neural Networks

Convolutional neural networks (CNNs) learn a hierarchical decomposition of visual information through successive layers of learnable filters. Early layers capture low-level features (edges, textures); intermediate layers encode mid-level patterns (shapes, object parts); and deeper layers represent high-level semantic concepts. This inductive bias — translation equivariance and local connectivity — makes CNNs well suited to image classification tasks where spatially localised features carry diagnostic information (LeCun et al., 1998).

### 2.3.2 Residual Networks (ResNet)

He et al. (2016) introduced residual learning to address the degradation problem in very deep networks, where accuracy saturates or degrades as depth increases. Each residual block learns a residual mapping F(x) relative to the identity shortcut x, so the block output is H(x) = F(x) + x. The skip connection allows gradients to flow directly through the network during backpropagation, enabling stable training of networks with over 100 layers. **ResNet-101**, comprising 101 weight layers organised in four residual stages, achieved state-of-the-art performance on ImageNet ILSVRC 2015 and has become a widely adopted backbone for medical image classification tasks.

### 2.3.3 Densely Connected Networks (DenseNet)

Huang et al. (2017) proposed DenseNet, in which every layer receives the concatenated feature maps of all preceding layers as input. With L layers, there are L(L+1)/2 direct connections. This dense connectivity encourages feature reuse, strengthens gradient flow, and substantially reduces the number of trainable parameters compared to networks of equivalent depth. **DenseNet-201**, the 201-layer variant, has demonstrated competitive performance on medical image benchmarks and was recognised with the CVPR 2017 Best Paper Award.

### 2.3.4 Transfer Learning in Medical Imaging

Training deep CNNs from random initialisation requires large labelled datasets, which are rarely available in clinical neuroimaging. Transfer learning mitigates this by initialising network weights from a model pre-trained on a large general-purpose dataset (typically ImageNet, ~1.2 million images, 1000 classes) and then fine-tuning on the target medical dataset (Litjens et al., 2017). The lower convolutional layers, which encode generic edge and texture detectors, transfer well across domains; the final classification layers are replaced and retrained to fit the target task. Despite the domain gap between natural and medical images, ImageNet-pretrained models consistently outperform random initialisation under limited data regimes (Raghu et al., 2019), making transfer learning the de facto standard for small-cohort medical imaging studies such as PMG classification.

---

## References

- Barkovich, A.J. (2010). Pediatric neuroimaging (5th ed.). Lippincott Williams & Wilkins.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K.Q. (2017). Densely connected convolutional networks. *CVPR 2017*, 4700–4708.
- Kwak, M. et al. (2018). Congenital cytomegalovirus and polymicrogyria. *Cited in GeneReviews NBK1329.*
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Litjens, G. et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88.
- MedlinePlus Genetics. (2024). Polymicrogyria. U.S. National Library of Medicine. https://medlineplus.gov/genetics/condition/polymicrogyria/
- Piao, X. et al. (2005). G protein-coupled receptor-dependent development of human frontal cortex. *Science*, 308(5729), 1923–1927.
- Raghu, M. et al. (2019). Transfusion: Understanding transfer learning for medical imaging. *NeurIPS 2019*.
- Park, K.B., Chapman, T., Aldinger, K.A., et al. (2021). The spectrum of brain malformations and disruptions in twins. *American Journal of Medical Genetics Part A*, 185(4), 1091–1104. DOI: 10.1002/ajmg.a.61972.
- Stutterd, C.A., & Leventer, R.J. (2014). Polymicrogyria: A common and heterogeneous malformation of cortical development. *American Journal of Medical Genetics Part C*, 166C(2), 227–239.
