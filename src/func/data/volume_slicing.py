"""Logic for slicing 3-D NIfTI volumes into 2-D JPEG slices.

All helpers and the orchestrating ``run_slice_volumes`` function live here;
the CLI wrapper in ``src/cli/slice_volumes.py`` only holds the Hydra decorator.
"""

import random
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _session_id(subject: str, session: str) -> str:
    """sub-01 + ses-001 -> sub01-ses001  (no underscores, safe for field[0])."""
    return f"{subject.replace('-', '')}-{session.replace('-', '')}"


def _label_int(label_str: str) -> int:
    return 1 if str(label_str).strip() == "PMG" else 0


def _output_subfolder(label: int) -> str:
    return "PMGcases" if label == 1 else "controlcases"


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------

def _select_indices(n_slices: int, cfg_vs) -> list[int]:
    selection = cfg_vs.slice_selection

    if selection == "all":
        return list(range(n_slices))

    if selection == "central":
        frac = float(cfg_vs.central_fraction)
        margin = int(n_slices * (1.0 - frac) / 2)
        start, end = margin, n_slices - margin
        return list(range(start, end)) if start < end else list(range(n_slices))

    if selection == "random":
        n = int(cfg_vs.n_random_slices)
        rng = random.Random(int(cfg_vs.seed))
        return sorted(rng.sample(range(n_slices), min(n, n_slices)))

    raise ValueError(
        f"Unknown slice_selection: {selection!r}. Must be 'all', 'central', or 'random'."
    )


# ---------------------------------------------------------------------------
# Volume loading & normalization
# ---------------------------------------------------------------------------

def _load_volume(path: Path) -> np.ndarray | None:
    """Load NIfTI, orient to RAS canonical, return float64 array (H, W, D)."""
    if not path.exists():
        print(f"  Warning: file not found: {path}")
        return None
    try:
        nii = nib.load(str(path))
        nii = nib.as_closest_canonical(nii)
        return nii.get_fdata()
    except Exception as exc:
        print(f"  Warning: failed to load {path}: {exc}")
        return None


def _volume_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Global min/max normalisation to uint8 (preserves relative intensities)."""
    v_min, v_max = float(volume.min()), float(volume.max())
    if v_max > v_min:
        normed = (volume - v_min) / (v_max - v_min) * 255.0
    else:
        normed = np.zeros_like(volume, dtype=np.float32)
    return normed.astype(np.uint8)


def _extract_slice(volume_uint8: np.ndarray, idx: int, axis: int) -> np.ndarray:
    if axis == 0:
        return volume_uint8[idx, :, :]
    if axis == 1:
        return volume_uint8[:, idx, :]  # axis == 1 (coronal, default)
    return volume_uint8[:, :, idx]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_slice_volumes(cfg: DictConfig) -> None:
    """Slice all volumes listed in the metadata CSV and write JPEG files.

    Args:
        cfg: Hydra config containing a ``volume_slicing`` sub-config with
            keys ``metadata_file``, ``output_dir``, ``axis``,
            ``slice_selection``, ``central_fraction``, ``n_random_slices``,
            ``jpeg_quality``, and ``seed``.
    """
    cwd = Path(get_original_cwd())
    vs = cfg.volume_slicing

    metadata_path = cwd / vs.metadata_file
    output_dir = cwd / vs.output_dir
    axis = int(vs.axis)

    if not metadata_path.exists():
        print(f"Error: metadata CSV not found: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    missing = {"subject", "session", "label", "path"} - set(df.columns)
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {metadata_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Axis         : {axis}  |  Slice selection: {vs.slice_selection}")
    print()

    total_slices = 0
    class_counts = {0: 0, 1: 0}
    failed = []

    for _, row in df.iterrows():
        subject = str(row["subject"]).strip()
        session = str(row["session"]).strip()
        label = _label_int(str(row["label"]))
        volume_path = Path(str(row["path"]).strip())

        sess_id = _session_id(subject, session)
        dest_dir = (
            output_dir
            / _output_subfolder(label)
            / subject.replace("-", "")
            / session.replace("-", "")
        )

        volume = _load_volume(volume_path)
        if volume is None:
            failed.append(str(volume_path))
            continue

        volume_uint8 = _volume_to_uint8(volume)
        n_slices = volume_uint8.shape[axis]
        indices = _select_indices(n_slices, vs)

        dest_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for idx in indices:
            slice_2d = _extract_slice(volume_uint8, idx, axis)
            out_path = dest_dir / f"{sess_id}_{idx:03d}_0_{label}.jpg"
            if cv2.imwrite(
                str(out_path),
                slice_2d,
                [cv2.IMWRITE_JPEG_QUALITY, int(vs.jpeg_quality)],
            ):
                n_saved += 1
            else:
                print(f"  Warning: cv2.imwrite failed for {out_path}")

        total_slices += n_saved
        class_counts[label] += n_saved
        label_name = "PMG" if label == 1 else "HC"
        print(f"  [{label_name}] {subject}/{session}  ->  {n_saved} slices")

    print()
    print("=" * 60)
    print(f"Volumes processed : {len(df) - len(failed)}")
    print(
        f"Total slices saved: {total_slices}"
        f"  (PMG={class_counts[1]}, HC={class_counts[0]})"
    )
    if failed:
        print(f"Failed ({len(failed)}):")
        for v in failed:
            print(f"  {v}")
    print(f"Output            : {output_dir}")
    print("=" * 60)
