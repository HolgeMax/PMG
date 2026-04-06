import logging
from pathlib import Path

import torch
import torchvision.utils as vutils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.func.models.get_models import build_densenet201, build_resnet101

_LOG = logging.getLogger(__name__)

# -------------------------------------------------------
# Public API
# -------------------------------------------------------


def run_all_ckpts_ablation_study(
    cfg,
    modified_data: list,
    checkpoint_dir: str,
    device: str = "cpu",
) -> dict:
    """Evaluate every checkpoint in *checkpoint_dir* on pre-occluded data.

    Architecture is inferred from each checkpoint's filename prefix
    (``resnet101_*`` or ``densenet201_*``), falling back to ``cfg.model.name``.

    Args:
        cfg: Hydra DictConfig with ``model.dropout_p``, ``model.freeze_backbone``,
            and ``model.name``.
        modified_data: List of ``(images, labels)`` tuples from :func:`make_black_box`.
        checkpoint_dir: Directory containing ``.pt`` / ``.pth`` checkpoint files.
        device: PyTorch device string.

    Returns:
        Mapping of relative checkpoint path → metric dict.
    """
    ckpt_root = Path(checkpoint_dir)
    ckpt_paths = sorted(
        p for p in ckpt_root.rglob("*") if p.suffix in {".pt", ".pth"}
    )
    metrics_dict: dict = {}
    for ckpt_path in tqdm(ckpt_paths, desc="checkpoints"):
        rel_key = str(ckpt_path.relative_to(ckpt_root))
        _LOG.info("Evaluating %s ...", rel_key)
        model = _infer_model(cfg, ckpt_path.name)
        model = _load_model_params(model, ckpt_path, device)
        metrics_dict[rel_key] = _evaluate_on_modified(model, modified_data, device)
    return metrics_dict


def make_black_box(
    data_loader: DataLoader,
    device: str = "cpu",
    box_size_frac: float = 0.2,
    save_example: bool = False,
    example_output_dir: Path | None = None,
) -> list:
    """Apply a random square black-box occlusion to every image in *data_loader*.

    Args:
        data_loader: DataLoader yielding ``(images, labels)`` batches.
        device: PyTorch device string used during occlusion.
        box_size_frac: Side length of the occlusion box as a fraction of
            ``min(height, width)``. Must be in ``(0, 1)``.
        save_example: If ``True``, save the first occluded batch as a JPEG.
        example_output_dir: Directory for the example image.  Required when
            ``save_example=True``.

    Returns:
        List of ``(images_cpu, labels)`` tuples with the occlusion applied.

    Raises:
        ValueError: If ``box_size_frac`` is not in ``(0, 1)``.
        ValueError: If ``save_example=True`` but ``example_output_dir`` is ``None``.
    """
    if not 0.0 < box_size_frac < 1.0:
        raise ValueError(
            f"box_size_frac must be in (0, 1), got {box_size_frac}"
        )
    if save_example and example_output_dir is None:
        raise ValueError(
            "example_output_dir must be provided when save_example=True"
        )

    modified_data: list = []
    for images, labels in data_loader:
        images = images.to(device)
        occluded = _apply_occlusion(images, box_size_frac)
        modified_data.append((occluded.cpu(), labels))

        if save_example and len(modified_data) == 1:
            _save_example(occluded, example_output_dir)  # type: ignore[arg-type]
            save_example = False  # only once

    return modified_data


# -------------------------------------------------------
# Internal helpers
# -------------------------------------------------------


def _apply_occlusion(images: torch.Tensor, box_size_frac: float) -> torch.Tensor:
    """Zero-out a random square patch in each image of a batch.

    Args:
        images: Float tensor of shape ``(B, C, H, W)``.
        box_size_frac: Occlusion side as a fraction of ``min(H, W)``.

    Returns:
        Tensor of the same shape with occlusion applied in-place.
    """
    _b, _c, height, width = images.shape
    box = int(min(height, width) * box_size_frac)
    x_starts = torch.randint(0, width - box, (_b,))
    y_starts = torch.randint(0, height - box, (_b,))

    # Vectorised mask: (B, 1, H, W)
    xs = torch.arange(width, device=images.device).view(1, 1, 1, -1)
    ys = torch.arange(height, device=images.device).view(1, 1, -1, 1)
    x0 = x_starts.view(-1, 1, 1, 1).to(images.device)
    y0 = y_starts.view(-1, 1, 1, 1).to(images.device)
    mask = (xs >= x0) & (xs < x0 + box) & (ys >= y0) & (ys < y0 + box)
    images[mask.expand_as(images)] = 0.0
    return images


def _infer_model(cfg, checkpoint_file: str) -> torch.nn.Module:
    """Build the correct architecture from a checkpoint filename.

    Args:
        cfg: Hydra DictConfig with ``model.dropout_p``, ``model.freeze_backbone``,
            and ``model.name``.
        checkpoint_file: Basename of the checkpoint (used to detect prefix).

    Returns:
        Uninitialised model with the correct architecture.
    """
    kwargs = dict(
        dropout_p=cfg.model.dropout_p,
        freeze_backbone=cfg.model.freeze_backbone,
    )
    if checkpoint_file.startswith("densenet201"):
        return build_densenet201(**kwargs)
    if checkpoint_file.startswith("resnet101"):
        return build_resnet101(**kwargs)

    _LOG.warning(
        "Cannot infer architecture from '%s'; using cfg.model.name='%s'",
        checkpoint_file,
        cfg.model.name,
    )
    if cfg.model.name == "densenet201":
        return build_densenet201(**kwargs)
    return build_resnet101(**kwargs)


def _load_model_params(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load weights from *checkpoint_path* into *model*.

    Args:
        model: Uninitialised model instance.
        checkpoint_path: Path to a ``.pt`` / ``.pth`` file.
        device: Map-location for :func:`torch.load`.

    Returns:
        The same model with weights loaded.

    Raises:
        ValueError: If the checkpoint does not contain a state-dict.
    """
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise ValueError(
            f"Unexpected checkpoint format in '{checkpoint_path}': {type(raw)}"
        )
    model.load_state_dict(state)
    return model


def _evaluate_on_modified(
    model: torch.nn.Module,
    modified_data: list,
    device: str = "cpu",
) -> dict:
    """Run inference on pre-occluded batches and return classification metrics.

    Args:
        model: Trained model.
        modified_data: List of ``(images_cpu, labels)`` tuples.
        device: PyTorch device string.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``, ``f1_score``.
    """
    model.to(device)
    model.eval()
    all_labels: list = []
    all_pred: list = []
    with torch.no_grad():
        for images, labels in tqdm(modified_data, desc="eval", leave=False):
            logits = model(images.to(device)).squeeze(1)
            predicted = (torch.sigmoid(logits) >= 0.5).long()
            all_pred.append(predicted.cpu())
            all_labels.append(labels)
    return _calculate_metrics(
        torch.cat(all_pred).tolist(),
        torch.cat(all_labels).tolist(),
    )


def _calculate_metrics(predictions: list, labels: list) -> dict:
    """Compute weighted classification metrics.

    Args:
        predictions: List of integer predicted labels.
        labels: List of integer ground-truth labels.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1_score``.
    """
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
    }


def _save_example(images: torch.Tensor, output_dir: Path) -> None:
    """Save the first image of *images* as a JPEG after un-normalising.

    Args:
        images: Float tensor of shape ``(B, C, H, W)``, normalised with
            ImageNet mean/std.
        output_dir: Directory in which ``black_box_example.jpg`` is written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (images[0].cpu() * std + mean).clamp(0, 1)
    dest = output_dir / "black_box_example.jpg"
    vutils.save_image(img, dest)
    _LOG.info("Saved black-box example image to %s", dest)
