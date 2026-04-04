import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def run_all_ckpts_ablation_study(
    model: torch.nn.Module,
    modified_data: list,
    checkpoint_dir: str,
    device: str = "cpu",
) -> dict:
    """Evaluate model on pre-occluded data for every checkpoint in *checkpoint_dir*.

    Args:
        model: Model instance whose weights are replaced per checkpoint.
        modified_data: List of (images, labels) tuples returned by :func:`make_black_box`.
        checkpoint_dir: Directory containing ``.pt`` / ``.pth`` checkpoint files.
        device: PyTorch device string.

    Returns:
        Mapping of checkpoint filename → metric dict.
    """
    metrics_dict = {}
    for checkpoint_file in os.listdir(checkpoint_dir):
        if checkpoint_file.endswith(".pt") or checkpoint_file.endswith(".pth"):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            model = _load_model_params(model, checkpoint_path, device)
            metrics_dict[checkpoint_file] = _evaluate_on_modified(model, modified_data, device)
    return metrics_dict


def make_black_box(
    data_loader: DataLoader,
    device: str = "cpu",
    box_size_frac: float = 0.2,
) -> list:
    """Apply a random square black-box occlusion to every image in *data_loader*.

    Args:
        data_loader: DataLoader yielding (images, labels) batches.
        device: PyTorch device string used during occlusion.
        box_size_frac: Side length of the occlusion box as a fraction of
            ``min(height, width)``.

    Returns:
        List of ``(images_cpu, labels)`` tuples with the occlusion applied.
    """
    modified_data = []
    for images, labels in data_loader:
        images = images.to(device)
        batch_size, _ch, height, width = images.size()
        box_size = int(min(height, width) * box_size_frac)

        for i in range(batch_size):
            x_start = torch.randint(0, width - box_size, (1,)).item()
            y_start = torch.randint(0, height - box_size, (1,)).item()
            images[i, :, y_start:y_start + box_size, x_start:x_start + box_size] = 0

        modified_data.append((images.cpu(), labels))

        if len(modified_data) == 1:
            _save_example(images)

    return modified_data


# -------------------------------------------------------
# Internal helpers
# -------------------------------------------------------

def _evaluate_on_modified(
    model: torch.nn.Module,
    modified_data: list,
    device: str = "cpu",
) -> dict:
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in modified_data:
            images = images.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs.squeeze(1)) >= 0.5).long()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return _calculate_metrics(all_predictions, all_labels)


def _load_model_params(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def _calculate_metrics(predictions: list, labels: list) -> dict:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1_score": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def _save_example(images: torch.Tensor) -> None:
    BASE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results", "ablation_study")
    )
    Path(BASE).mkdir(parents=True, exist_ok=True)
    vutils.save_image(images[0].cpu(), os.path.join(BASE, "black_box_example.jpg"))
    print(f"Saved black-box example image to {BASE}/black_box_example.jpg")
