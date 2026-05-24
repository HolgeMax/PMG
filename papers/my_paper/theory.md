# Theory

**Supervisor comments addressed:**
- Comment 1: Neuropathology trimmed to 2 concise paragraphs.
- Comment 2: No isolated single sentences — every section has complete paragraphs.
- Comment 3: Aetiology section removed entirely.
- Comment 4: Clinical Manifestations section removed entirely.
- Comment 5: Diagnosis folded into the PMG section as a closing paragraph.
- Comment 6: MRI section rewritten to focus on MRI in neurological disease broadly; acquisition parameter tables moved to methods.
- Comment 7: Deep Learning subsections flattened — bold paragraph titles replace numbered subsections (4.3.1 etc.); introductory paragraph added at section level.

---

## 2.1 Polymicrogyria

Polymicrogyria (PMG) is a malformation of cortical development characterised by
abnormal lamination and an excessive number of abnormally small gyri, resulting in an
irregular folding pattern across all or part of the cerebral cortex (Stutterd &
Leventer, 2014; Severino et al., 2020). PMG arises during the post-migrational stage
of cortical development, distinguishing it from other malformations of cortical
development (MCDs) such as lissencephaly, which affects the earlier migration stage.
Multiple forms exist: unilateral PMG affects one hemisphere, bilateral PMG affects
both, and severity ranges from mildly focal to extensive bilateral involvement. PMG
accounts for approximately 20% of all MCDs, making it one of the most common brain
malformations of this class (Stutterd & Leventer, 2014).

Despite its prevalence among MCDs, PMG remains challenging to diagnose reliably.
MRI is the only imaging modality that provides sufficient soft-tissue contrast and
spatial resolution to identify the fine cortical folds characteristic of PMG; computed
tomography (CT) is inadequate for this task (Severino et al., 2020). The radiological
appearance of PMG is heterogeneous: features include an irregular or bumpy cortical
surface, a stippled gray-white junction, and apparent cortical thickening. In some
cases, adjacent microgyri fuse their overlying molecular layers, producing a
deceptively smooth outer surface that can be mistaken for pachygyria. MRI appearance
also changes with myelination, making diagnosis particularly difficult in infants
during the first year of life, when low gray-white contrast may obscure the subtle
cortical irregularities of the condition. Inter-rater variability among specialist
neuroradiologists is well documented, underscoring both the difficulty of the
diagnostic task and the potential value of automated detection methods.

---

## 2.2 MRI in Neurological Assessment

Magnetic Resonance Imaging (MRI) is the principal structural neuroimaging modality
across a wide range of neurological conditions, including epilepsy, brain tumours,
white matter diseases, and cortical malformations. Its clinical dominance in neurology
stems from its capacity to provide high soft-tissue contrast without ionising
radiation. T1-weighted sequences in particular offer excellent differentiation between
grey and white matter, enabling delineation of cortical thickness, sulcal morphology,
and the gray-white junction architecture that is central to identifying MCDs such as
PMG. In epilepsy workups — the clinical context of both datasets used in this project
— structural MRI is mandatory and serves as the primary tool for identifying
potentially resectable lesions.

A key challenge in applying deep learning to retrospective or multi-site neuroimaging
data is scanner variability. Differences in field strength, scanner vendor, pulse
sequence design, and field of view (FOV) produce systematic differences in signal
intensity, spatial resolution, and image contrast that are unrelated to underlying
pathology. In datasets where patient groups were not acquired under matched conditions,
these low-level statistical differences can act as confounds, allowing classifiers to
exploit acquisition-related signals rather than genuine anatomical features. This risk
is directly relevant to the present project: the PPMR dataset was acquired on two
scanners at different field strengths, and the clinical dataset was collected
retrospectively across multiple protocols and vendors. The specific acquisition
parameters for both datasets are detailed in the Methods section.

---

## 2.3 Deep Learning for Medical Image Classification

Deep learning methods — particularly convolutional neural networks (CNNs) and their
variants — have become the dominant approach in medical image classification, achieving
expert-level performance in tasks ranging from retinal disease grading to histological
cancer detection (Litjens et al., 2017). These models learn discriminative image
representations directly from labelled data, removing the need for hand-crafted
feature extractors. In the context of PMG classification from MRI, deep learning
offers a principled way to extract subtle cortical morphology features from 2D slices
that would be difficult to specify analytically. The following subsections describe the
specific architectures and training strategies used in this project.

**Convolutional Neural Networks.** CNNs learn a hierarchical decomposition of visual
information through successive layers of learnable filters. Early layers capture
low-level features such as edges and textures; intermediate layers encode mid-level
patterns such as shapes and local structures; and deeper layers represent high-level
semantic concepts relevant to the classification task. Two architectural properties
underlie this: local connectivity, where each filter operates on a small spatial
neighbourhood, and weight sharing, which produces translation equivariance — the same
feature detector is applied at every spatial position. Together, these properties make
CNNs particularly effective for tasks where the diagnostically relevant signal is
spatially localised, as is the case for cortical fold morphology in PMG (LeCun et
al., 1998).

**Residual Networks (ResNet-101).** He et al. (2016) introduced residual learning to
address the degradation problem in very deep networks, where training accuracy
saturates or degrades as depth increases. Each residual block learns a residual
mapping F(x) relative to an identity shortcut, so the block output is H(x) = F(x) +
x. The skip connection allows gradients to flow directly through the network during
backpropagation, enabling stable training beyond 100 layers. ResNet-101 uses
bottleneck blocks (1×1, 3×3, 1×1 convolution sequences) organised in four stages
with [3, 4, 23, 3] blocks, totalling 101 weight layers. Pretrained on ImageNet,
ResNet-101 achieved state-of-the-art performance on ILSVRC 2015 and has since become
a widely adopted backbone for medical image classification tasks. In this project,
ResNet-101 is one of two architectures evaluated for PMG classification.

**Densely Connected Networks (DenseNet-201).** Huang et al. (2017) proposed DenseNet,
in which each layer receives the concatenated feature maps of all preceding layers as
input, yielding L(L+1)/2 direct connections in a network of L layers. Dense
connectivity encourages feature reuse across network depth, strengthens gradient flow
to early layers, and reduces the number of trainable parameters relative to networks
of equivalent depth with strictly sequential connections. DenseNet-201, the 201-layer
variant organised in four dense blocks, has demonstrated competitive performance on
medical image benchmarks. In this project it serves as the second evaluated
architecture and was reported by Guha et al. (2025) as the best-performing model on
the PPMR dataset under their experimental setup.

**Transfer Learning.** Training deep CNNs from scratch requires large labelled
datasets, which are rarely available in clinical neuroimaging. Transfer learning
addresses this by initialising network weights from a model pretrained on a large
general-purpose dataset — typically ImageNet (approximately 1.2 million images, 1000
classes) — and fine-tuning on the target medical data (Litjens et al., 2017). Lower
convolutional layers encode generic edge and texture detectors that transfer well
across domains; the final classification layers are replaced with a task-specific head
and retrained on the target task. Despite the apparent domain gap between natural
photographs and clinical MRI, ImageNet-pretrained models consistently outperform
random initialisation in limited-data regimes (Raghu et al., 2019), making transfer
learning the standard approach for small-cohort studies such as PMG classification. In
this project, both ResNet-101 and DenseNet-201 are initialised with ImageNet weights,
with the final fully connected layer replaced by a binary classifier (PMG vs. healthy
control).

---

## 2.4 Guha et al. (2025)

Guha et al. (2025) applied fine-tuned convolutional neural networks to the automated
detection of PMG using the PPMR dataset. Their pipeline consisted of a five-step
preprocessing sequence — grayscale conversion, min-max normalisation,
contrast-limited adaptive histogram equalisation (CLAHE), bilateral filtering, and
Canny edge detection — applied to all 2D MRI slices before training. Five
architectures were evaluated, all initialised with ImageNet weights: ResNet-50,
ResNet-101, VGG-16, MobileNetV2, and DenseNet-201. The final classification layers
were replaced with a dense head (256 units, ReLU activation, L2 regularisation,
dropout = 0.5), with only the classification head trainable, and models were trained
using the Adam optimiser (learning rate = 0.0005) for up to ten epochs with a batch
size of 32. The data were split 60\% / 20\% / 20\% into training, validation, and test
sets. Guha et al.\ report using 4,517 PMG-patient images and 10,539 healthy control
images, and addressed the resulting class imbalance by randomly sampling 4,517 control
images, yielding a balanced 1:1 dataset for training and evaluation. The
best-performing model, DenseNet-201, achieved test accuracy 0.9967, precision 0.9933,
recall 1.000, F1 0.993, and Cohen's $\kappa$ 0.993. The authors attributed this
performance to the preprocessing pipeline and the representational capacity of deep
CNNs.

The first methodological error in Guha et al.\ is the definition of the positive
class. The PPMR dataset provides slice-level annotations by a fellowship-trained
paediatric neuroradiologist: each MRI slice from a PMG patient is individually
labelled as PMG-positive (label\,=\,1), PMG-negative (label\,=\,2), or uncertain
(label\,=\,3), reflecting the focal and variable nature of the malformation within a
single patient's brain. Of the 4,517 slices attributed to PMG patients, only
approximately 2,256 (49\%) carry label\,=\,1 (PMG-positive); approximately 1,386
(30\%) carry label\,=\,2 (PMG-negative, i.e.\ no visible malformation in that slice);
and approximately 875 (19\%) carry label\,=\,3 (uncertain). Guha et al.\ treated all
4,517 PMG-patient slices as the positive class, irrespective of these annotations.
Training a classifier with PMG-negative slices included in the positive class
introduces substantial label noise: the model must learn to distinguish slices that
are visually indistinguishable from healthy controls from actual healthy controls,
diluting the PMG-specific signal and rendering the task definition inconsistent with
the neuroradiological annotations provided in the dataset.

The second methodological error is a systematic, class-correlated difference in image
resolution. PMG-patient images have a median resolution of $1508 \times 1727$\,pixels,
while healthy control images range from $260 \times 320$ to $512 \times 512$\,pixels
(median $512 \times 512$). When all images are resized to the $224 \times 224$\,pixel
input required by the CNN architectures, PMG images are downsampled by approximately
6$\times$, producing smooth, low-variance textures as many original pixels are
averaged into each output pixel. Control images are resized by only 1.1--2.3$\times$,
preserving the coarser, higher-variance texture of the original acquisition. This
systematic texture difference is present in every image patch, regardless of whether
cortical pathology is visible in that region. A classifier can therefore solve the
task by learning that smooth texture implies PMG and coarse texture implies healthy
control --- a signal entirely unrelated to cortical morphology. Critically, the
preprocessing pipeline applied by Guha et al.\ amplifies rather than harmonises this
resolution-induced texture difference: spatial operations with fixed kernel sizes
(CLAHE tile size, bilateral filter diameter, Canny thresholds) behave differently
depending on the effective resolution of the input image, extracting texture at
systematically different spatial scales before the resizing step equalises the pixel
grid. This shortcut learning mechanism provides a parsimonious explanation for the
near-100\% accuracy reported by Guha et al.

The third methodological error concerns the handling of class imbalance and the
integrity of the evaluation sets. To address the 10,539:4,517 imbalance, Guha et al.\
randomly sampled 4,517 control slices before splitting the data. Pre-split
downsampling causes the train, validation, and test sets to all inherit the
artificially balanced 1:1 class ratio, so reported metrics are computed against a
class distribution that does not exist in clinical practice. Only the training set
should be balanced; validation and test sets should reflect the natural prevalence so
that reported performance metrics generalise to real-world deployment. Additionally,
slice-level random sampling without regard to patient identity risks excluding entire
patients from the evaluation partitions and undermines patient-level data splitting.
Guha et al.\ do not describe patient-level splitting; without it, slices from the same
patient can appear in both training and test sets, inflating apparent generalisation
through data leakage. In contrast, Zhang et al.\ (2024), who introduced the PPMR
dataset, explicitly employed patient-level splits and 5-fold double cross-validation.
The present study applies corrective measures for all three errors and quantifies
their collective and individual effects on classification performance.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K.Q. (2017). Densely connected convolutional networks. *CVPR 2017*, 4700–4708.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Litjens, G. et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88.
- Raghu, M. et al. (2019). Transfusion: Understanding transfer learning for medical imaging. *NeurIPS 2019*.
- Severino, M. et al. (2020). Definitions and classification of malformations of cortical development: Practical guidelines. *Brain*, 143(10), 2874–2894.
- Stutterd, C.A., & Leventer, R.J. (2014). Polymicrogyria: A common and heterogeneous malformation of cortical development. *American Journal of Medical Genetics Part C*, 166C(2), 227–239.
