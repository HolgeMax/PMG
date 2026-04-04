import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from sklearn.model_selection import StratifiedKFold

from src.func.data.get_loader import PMGDataset, _undersample_to_minority


def _patient_class(paths: list[Path]) -> int:
    """Determine the biological class (PMG=1, HC=0) of a patient from their slice paths."""
    for p in paths:
        if "PMGcases" in p.parts:
            return 1
        if "controlcases" in p.parts or "PMGControlsEditedDec2021" in p.parts:
            return 0
    raise ValueError(f"Cannot determine patient class from path: {paths[0]}")


def kfold_split_patients(
    data_dir: str,
    n_folds: int = 5,
    val_frac_of_train: float = 0.15,
    seed: int = 42,
    pmg_negative_mode: str = "correct",
    balance_mode: str | None = None,
) -> Iterator[tuple[list, list, list, int]]:
    """Patient-level stratified k-fold generator.

    Yields one (train_samples, val_samples, test_samples, fold_idx) tuple per
    fold. All slices from one patient always land in the same split.
    Stratification preserves the HC:PMG ratio across folds.

    Parameters
    ----------
    data_dir : str
        Root directory passed to PMGDataset.
    n_folds : int
        Number of folds.
    val_frac_of_train : float
        Fraction of the non-test patients held out as validation.
    seed : int
        Random seed for StratifiedKFold and val sampling.
    pmg_negative_mode : str
        Passed to PMGDataset.
    balance_mode : {None, "pre_split", "post_split"}
        ``None``         — no balancing.
        ``"pre_split"``  — undersample HC before building patient groups.
        ``"post_split"`` — undersample HC in train split only after splitting.

    Yields
    ------
    train_samples, val_samples, test_samples : list of (Path, int)
    fold_idx : int  (0-based)
    """
    rng = random.Random(seed)

    full = PMGDataset(data_dir=data_dir, pmg_negative_mode=pmg_negative_mode)

    if balance_mode == "pre_split":
        full.samples = _undersample_to_minority(full.samples, rng)

    # Group sample indices by patient ID
    patient_map: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(full.samples):
        pid = path.stem.split("_")[0]
        patient_map[pid].append(idx)

    patient_ids = sorted(patient_map.keys())
    patient_paths = {pid: [full.samples[i][0] for i in patient_map[pid]] for pid in patient_ids}
    patient_labels = [_patient_class(patient_paths[pid]) for pid in patient_ids]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        skf.split(patient_ids, patient_labels)
    ):
        test_pids  = {patient_ids[i] for i in test_idx}
        train_val_pids = [patient_ids[i] for i in train_val_idx]

        # Carve out val from train_val patients
        n_val = max(1, round(len(train_val_pids) * val_frac_of_train))
        rng_fold = random.Random(seed + fold_idx)
        shuffled = train_val_pids.copy()
        rng_fold.shuffle(shuffled)
        val_pids   = set(shuffled[:n_val])
        train_pids = set(shuffled[n_val:])

        def _collect(pids: set) -> list:
            return [full.samples[i] for pid in pids for i in patient_map[pid]]

        train_samples = _collect(train_pids)
        if balance_mode == "post_split":
            train_samples = _undersample_to_minority(train_samples, rng_fold)

        val_samples  = _collect(val_pids)
        test_samples = _collect(test_pids)

        print(
            f"Fold {fold_idx + 1}/{n_folds} — "
            f"train: {len(train_samples)} slices ({len(train_pids)} patients), "
            f"val: {len(val_samples)} slices ({len(val_pids)} patients), "
            f"test: {len(test_samples)} slices ({len(test_pids)} patients)"
        )

        yield train_samples, val_samples, test_samples, fold_idx
