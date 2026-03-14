import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

from src.func.utils.loader import collect_input_files

# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

def _parse_raw_label(path: Path) -> int | None:
    """
    Extract the numeric label embedded in a PMG-case filename.

    The filename encodes the per-slice annotation as the 4th
    underscore-delimited field (index 3), e.g.::

        10cor_1_42_1_preprocessed_default.jpg
                  ^
                  raw label = 1  (PMG visible)

    Parameters
    ----------
    path : Path
        Full path to the slice file.

    Returns
    -------
    int or None
        The raw integer label, or ``None`` if parsing fails (malformed name).
    """
    try:
        return int(path.stem.split("_")[3])
    except (IndexError, ValueError):
        return None


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class PMGDataset(Dataset):
    """
    PyTorch Dataset for the PMG / HC coronal-slice classification task.

    Iterates recursively over *data_dir*, assigns a binary label to every
    slice, and optionally skips slices whose annotation is ambiguous.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing ``PMGcases/`` and ``controlcases/``
        sub-trees (see module docstring for the expected layout).
    transform : callable, optional
        A torchvision transform (or ``transforms.Compose``) applied to
        each PIL image before it is returned.  Build one with
        :func:`data_augmentation`.
    pmg_negative_mode : {"paper", "correct"}
        Controls how label-2 (PMG patient, no PMG visible) and label-3
        (uncertain) slices are treated:

        ``"paper"``
            Replicates Guha & Bhandage (2025).  **All** slices from the
            PMG folder are positive (label=1), including label=2 and
            label=3.  This conflates "patient has PMG" with "this slice
            shows PMG", which is methodologically incorrect.

        ``"correct"`` *(default)*
            Only label-1 slices (PMG actually visible) are positive.
            Label-2 slices become HC (0); label-3 slices are excluded.

    Attributes
    ----------
    samples : list of (Path, int)
        Collected (image_path, binary_label) pairs after filtering.

    Notes
    -----
    Images are opened as RGB PIL images inside ``__getitem__``.
    No tensors are pre-allocated; memory usage is constant
    regardless of dataset size.
    """

    # Raw label values embedded in PMG filenames
    _RAW_PMG_POS   = 1   # PMG visible in this slice
    _RAW_PMG_NEG   = 2   # PMG patient but no PMG visible here
    _RAW_UNCERTAIN = 3   # uncertain / ambiguous annotation

    def __init__(
        self,
        data_dir: str = None,
        transform=None,
        pmg_negative_mode: str = "correct",
        samples: list = None,           # pre-built list from split_dataset()
    ):
        if pmg_negative_mode not in ("paper", "correct"):
            raise ValueError("pmg_negative_mode must be 'paper' or 'correct'")

        self.transform = transform
        self.pmg_negative_mode = pmg_negative_mode

        if samples is not None:
            # fast path: use a pre-built (path, label) list from split_dataset()
            self.samples = samples
            return

        self.samples: list[tuple[Path, int]] = []
        data_dir = Path(data_dir)
        for file in collect_input_files(data_dir, recursive=True):
            label = self._assign_label(file)
            if label is not None:
                self.samples.append((file, label))

    def _assign_label(self, path: Path) -> int | None:
        """
        Map a file path to a binary label, or ``None`` to skip the slice.

        Parameters
        ----------
        path : Path
            Path to a single slice file.

        Returns
        -------
        int or None
            ``0`` (HC), ``1`` (PMG), or ``None`` (exclude from dataset).
        """
        # Control cases: folder membership determines the label
        if "controlcases" in path.parts:
            return 0

        # PMG cases: decode from filename
        raw = _parse_raw_label(path)
        if raw is None:
            return None

        if raw == self._RAW_PMG_POS:
            return 1

        if raw == self._RAW_PMG_NEG:
            # "paper": all PMG-folder slices are positive
            # "correct": a normal-looking slice from a PMG patient is HC
            return 1 if self.pmg_negative_mode == "paper" else 0

        if raw == self._RAW_UNCERTAIN:
            # "paper": uncertain slices are still counted as positive
            # "correct": uncertain slices are excluded entirely
            return 1 if self.pmg_negative_mode == "paper" else None

        return None  # unknown raw label — skip

    def __len__(self) -> int:
        """Return the number of (image, label) pairs in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return one (image, label) pair.

        Parameters
        ----------
        idx : int
            Index into ``self.samples``.

        Returns
        -------
        image : torch.Tensor
            Float tensor of shape ``(C, H, W)`` after transforms.
        label : torch.Tensor
            Scalar ``torch.long`` tensor — 0 (HC) or 1 (PMG).

        Notes
        -----
        ``BCEWithLogitsLoss`` requires the label as ``torch.float``.
        Cast it in the training loop::

            loss = criterion(logit.squeeze(), label.float())
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------------------------
# Transform factory
# ------------------------------------------------------------------------------

def data_augmentation( # add cfg instead of many args
    crop_size: int = None,
    scale: tuple = None,
    mean=None,
    std=None,
    is_training: bool = True,
) -> transforms.Compose:
    """
    Build a torchvision preprocessing / augmentation pipeline.

    Training pipeline
        RandomResizedCrop → RandomHorizontalFlip → ToTensor → Normalize

    Validation / inference pipeline
        Resize → CenterCrop → ToTensor → Normalize

    Parameters
    ----------
    crop_size : int
        Target spatial size (height = width) after cropping.
    scale : tuple of (float, float)
        Lower and upper bounds for the random crop area ratio (training only).
        E.g. ``(0.8, 1.0)`` crops between 80 % and 100 % of the image area.
    mean : sequence of float
        Per-channel normalisation means, e.g. ``[0.485, 0.456, 0.406]``.
    std : sequence of float
        Per-channel normalisation standard deviations.
    is_training : bool
        ``True`` → augmented training pipeline; ``False`` → deterministic
        evaluation pipeline.

    Returns
    -------
    transforms.Compose
        Ready-to-use transform callable.
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ------------------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------------------

def get_dataloader( # add cfg instead of many args
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap a Dataset in a DataLoader with sensible defaults.

    Parameters
    ----------
    dataset : Dataset
        Any ``torch.utils.data.Dataset`` instance (e.g. ``PMGDataset``).
    batch_size : int
        Number of samples per batch.
    num_workers : int, optional
        Number of parallel data-loading worker processes.  Default: 4.
    shuffle : bool, optional
        Whether to shuffle the dataset each epoch.  Should be ``True``
        for training and ``False`` for validation / test.  Default: ``True``.

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ------------------------------------------------------------------------------
# Patient-level train / val / test split
# ------------------------------------------------------------------------------

def split_dataset(
    data_dir: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    pmg_negative_mode: str = "correct",
) -> tuple[list, list, list]:
    """
    Split all samples in *data_dir* into train / val / test at the patient level.

    Slices from the same patient always land in the same split, preventing
    data leakage between the sets.

    Patient identity is read from the first ``_``-delimited field of each
    filename stem (e.g. ``10cor`` from ``10cor_1_42_1_preprocessed.jpg``).

    Parameters
    ----------
    data_dir : str
        Root directory passed to :class:`PMGDataset`.
    val_frac, test_frac : float
        Fraction of *patients* (not slices) assigned to each held-out split.
    seed : int
        Random seed for reproducible shuffling.
    pmg_negative_mode : str
        Passed to :class:`PMGDataset` for label assignment.

    Returns
    -------
    train_samples, val_samples, test_samples : list of (Path, int)
        Three (path, label) lists ready to pass as ``samples=`` to
        :class:`PMGDataset`.
    """
    full = PMGDataset(data_dir=data_dir, pmg_negative_mode=pmg_negative_mode)

    # group sample indices by patient ID (first "_"-field of the stem)
    patient_map: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(full.samples):
        pid = path.stem.split("_")[0]
        patient_map[pid].append(idx)

    patients = list(patient_map.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = max(1, round(n * test_frac))
    n_val  = max(1, round(n * val_frac))

    test_pids  = set(patients[:n_test])
    val_pids   = set(patients[n_test:n_test + n_val])
    train_pids = set(patients[n_test + n_val:])

    def _collect(pids):
        return [full.samples[i] for pid in pids for i in patient_map[pid]]

    return _collect(train_pids), _collect(val_pids), _collect(test_pids)


# ------------------------------------------------------------------------------
# One-call convenience factory
# ------------------------------------------------------------------------------

def create_pmg_dataloader( ## add cfg instead of many args
    data_dir: str,
    batch_size: int,
    crop_size: int,
    scale: tuple,
    mean=None,
    std=None,
    pmg_negative_mode: str = "correct",
    num_workers: int = 4,
    is_training: bool = True,
) -> DataLoader:
    """
    Build the full data pipeline in one call.

    Combines :func:`data_augmentation`, :class:`PMGDataset`, and
    :func:`get_dataloader` into a single convenience function suitable
    for use inside a training script or Hydra config.

    Parameters
    ----------
    data_dir : str
        Root directory of the processed dataset (see module docstring).
    batch_size : int
        Samples per batch.
    crop_size : int
        Spatial size passed to :func:`data_augmentation`.
    scale : tuple of (float, float)
        Random-crop scale range (training only).
    mean : sequence of float, optional
        Normalisation means.  Defaults to ImageNet values if ``None``
        is passed through to ``transforms.Normalize``.
    std : sequence of float, optional
        Normalisation standard deviations.
    pmg_negative_mode : {"paper", "correct"}
        Passed directly to :class:`PMGDataset`.  Default: ``"correct"``.
    num_workers : int, optional
        DataLoader worker count.  Default: 4.
    is_training : bool, optional
        Selects the augmentation pipeline and disables shuffle for eval.
        Default: ``True``.

    Returns
    -------
    DataLoader
        Fully configured dataloader ready for a training / eval loop.

    Example
    -------
    >>> train_loader = create_pmg_dataloader(
    ...     data_dir="data/PPMR_default",
    ...     batch_size=32,
    ...     crop_size=224,
    ...     scale=(0.8, 1.0),
    ...     mean=[0.485, 0.456, 0.406],
    ...     std=[0.229, 0.224, 0.225],
    ...     pmg_negative_mode="correct",
    ...     is_training=True,
    ... )
    """
    transform = data_augmentation(
        crop_size=crop_size,
        scale=scale,
        mean=mean,
        std=std,
        is_training=is_training,
    )
    dataset = PMGDataset(
        data_dir=data_dir,
        transform=transform,
        pmg_negative_mode=pmg_negative_mode,
    )
    return get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_training,
    )
