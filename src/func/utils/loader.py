from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

from src.config.preprocessing_config import PreprocessingConfig
from src.main.configurable_pipeline import preprocess_image
from src.func.evaluation.preprocessing_metrics import evaluate_preprocessing


_NIFTI_SUFFIXES = {".nii", ".nii.gz"}
_JPEG_SUFFIXES = {".jpg", ".jpeg"}
_SUPPORTED = {".jpg", ".jpeg", ".nii", ".nii.gz"}

_PPMR_SOURCE_MAP: dict[str, str] = {
    "PMGstudycaseslabelled": "PMGcases",
    "PMGControlsEditedDec2021": "controlcases",
}


def collect_input_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return every supported file at *input_path*.

    If *input_path* is a file it is returned as-is (single-file mode).
    If it is a directory, all .jpg / .jpeg / .nii / .nii.gz files inside are
    collected.  Set *recursive=True* to also search sub-directories.
    """
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        files = sorted(
            f
            for f in iterator
            if f.is_file() and "".join(f.suffixes).lower() in _SUPPORTED
        )
        if not files:
            print(f"Warning: no supported files found in {input_path}")
        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_nifti_slice(
    nii_path: str,
    slice_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load a 2D axial slice from a NIfTI file as uint8.

    Args:
        nii_path: Path to .nii or .nii.gz file.
        slice_idx: Axial slice index. If None, uses the middle slice.

    Returns:
        Tuple of (slice as uint8 grayscale array, slice index used).
    """
    nii = nib.load(nii_path)
    volume = nii.get_fdata()

    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")

    if slice_idx is None:
        slice_idx = volume.shape[2] // 2

    slice_2d = volume[:, :, slice_idx]

    s_min, s_max = float(slice_2d.min()), float(slice_2d.max())
    if s_max > s_min:
        slice_uint8 = ((slice_2d - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        slice_uint8 = np.zeros_like(slice_2d, dtype=np.uint8)

    return slice_uint8, slice_idx


def load_jpeg(jpeg_path: str) -> np.ndarray:
    """Load a JPEG file as a uint8 grayscale array.

    Args:
        jpeg_path: Path to a .jpg or .jpeg file.

    Returns:
        2-D uint8 array (grayscale).

    Raises:
        FileNotFoundError: If the file cannot be read by OpenCV.
    """
    img = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {jpeg_path}")
    return img


def load_image(
    path: str,
    slice_idx: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Load either a JPEG or a NIfTI file and return a uint8 grayscale array.

    Dispatches to :func:`load_jpeg` or :func:`load_nifti_slice` based on the
    file extension.  Callers no longer need to know the source format.

    Args:
        path: Path to a .jpg / .jpeg / .nii / .nii.gz file.
        slice_idx: For NIfTI files, the axial slice to extract (None → middle
            slice).  Ignored for JPEG inputs.

    Returns:
        Tuple of ``(image, metadata)`` where *image* is a 2-D uint8 array and
        *metadata* is a dict with keys:

        - ``source_type``: ``'nifti'`` or ``'jpeg'``
        - ``slice_idx``: the slice index used (``None`` for JPEG)

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file cannot be found or read.
    """
    p = Path(path)
    # Collapse compound suffixes (.nii.gz) into a single lower-case string
    full_suffix = "".join(p.suffixes).lower()
    ext = full_suffix if full_suffix in _NIFTI_SUFFIXES else p.suffix.lower()

    if ext in _NIFTI_SUFFIXES:
        arr, used_idx = load_nifti_slice(path, slice_idx)
        return arr, {"source_type": "nifti", "slice_idx": used_idx}

    if ext in _JPEG_SUFFIXES:
        arr = load_jpeg(path)
        return arr, {"source_type": "jpeg", "slice_idx": None}

    raise ValueError(
        f"Unsupported file format '{p.suffix}'. "
        f"Expected one of: {sorted(_NIFTI_SUFFIXES | _JPEG_SUFFIXES)}"
    )


def _resolve_ppmr_output_dir(
    file_path: Path,
    output_dir: Path,
) -> Path | None:
    """Map a PPMR source file to its canonical output directory.

    Mirrors the full relative path from the PPMR source folder so that
    the subject-number folder and scan subfolder are both preserved, e.g.:
        PMGstudycaseslabelled/34/34cor_1/file.jpg
        → output_dir/PMGcases/34/34cor_1/

    Args:
        file_path: Path to the source file.
        output_dir: Root output directory (e.g. data/PPMR_processed).

    Returns:
        Resolved directory when file belongs to a known PPMR source folder,
        None otherwise.
    """
    parts = file_path.parts
    for source_name, dest_name in _PPMR_SOURCE_MAP.items():
        if source_name in parts:
            idx = parts.index(source_name)
            source_dir = Path(*parts[: idx + 1])
            rel = file_path.parent.relative_to(source_dir)
            return output_dir / dest_name / rel
    return None


def _resolve_output_dir(
    file_path: Path,
    output_dir: Path,
    input_root: Path | None,
) -> Path:
    """Compute the destination directory for one file.

    Delegates to PPMR-specific logic when the file originates from a known
    PPMR source directory; otherwise mirrors the source tree under output_dir.

    Args:
        file_path: Path to the source file.
        output_dir: Root output directory for this run.
        input_root: If provided, mirror subdirectories relative to this root
            when no PPMR mapping applies.

    Returns:
        Resolved output directory for this specific file.
    """
    ppmr_dir = _resolve_ppmr_output_dir(file_path, output_dir)
    if ppmr_dir is not None:
        return ppmr_dir
    if input_root is None:
        return output_dir
    try:
        return output_dir / file_path.parent.relative_to(input_root)
    except ValueError:
        return output_dir


def process_one(
    file_path: Path,
    preprocess_config: PreprocessingConfig,
    slice_idx: int | None,
    output_dir: Path,
    preset_name: str,
    input_root: Path | None = None,
) -> dict:
    """Load, preprocess, evaluate, and save one file.

    Args:
        file_path: Path to the source image file.
        preprocess_config: Pipeline configuration.
        slice_idx: Axial slice index for NIfTI files; ignored for JPEG.
        output_dir: Root directory for processed outputs.
        preset_name: Preset label appended to the output filename.
        input_root: If provided, mirror subdirectory structure under output_dir.

    Returns:
        Metrics dict with keys ``psnr``, ``ssim``, ``entropy_change``,
        ``source_type``, and ``file``.
    """
    img, meta = load_image(str(file_path), slice_idx)
    result, _log = preprocess_image(img, preprocess_config)
    metrics = evaluate_preprocessing(img, result)

    stem = file_path.name.split(".")[0]
    dest_dir = _resolve_output_dir(file_path, output_dir, input_root)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / f"{stem}_preprocessed_{preset_name}.jpg"
    output_8bit = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), output_8bit, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {**metrics, "source_type": meta["source_type"], "file": file_path.name}
