# filename: classification_metrics.py
"""
Classification metrics for PMG / HC binary evaluation.

Functions
---------
collect_predictions  — run model inference and return labels + logits
compute_metrics      — accuracy, precision, recall, F1, Cohen's Kappa
print_metrics        — pretty-print a metrics dict
evaluate_model       — convenience wrapper: collect → compute → print
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# Inference
# =============================================================================


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[list[int], list[int]]:
    """
    Run the model over *dataloader* and collect ground-truth / predicted labels.

    Parameters
    ----------
    model : nn.Module
        Trained model.  Must output a raw logit of shape ``(B, 1)`` or ``(B,)``.
    dataloader : DataLoader
        Yields ``(images, labels)`` batches.
    device : torch.device
        Device to run inference on.
    threshold : float
        Sigmoid threshold for converting logits to binary predictions.

    Returns
    -------
    y_true, y_pred : list of int
        Flat lists of ground-truth and predicted class indices (0 or 1).
    """
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images).squeeze(1)  # (B,)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            all_labels.append(labels)
            all_preds.append(preds.cpu())
        
        y_true = torch.cat(all_labels).tolist()
        y_pred = torch.cat(all_preds).tolist()
    return y_true, y_pred


# =============================================================================
# Metric computation
# =============================================================================


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    """
    Compute binary classification metrics from label lists.

    Parameters
    ----------
    y_true : list of int
        Ground-truth binary labels (0 or 1).
    y_pred : list of int
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, cohen_kappa
    """
    # 2×2 confusion matrix: cm[true_label, pred_label]
    y_t = torch.tensor(y_true)
    y_p = torch.tensor(y_pred)
    cm = torch.bincount(y_t * 2 + y_p, minlength=4).reshape(2, 2)     
    # cm:  [[TN, FP],
    #      [FN, TP]] 
    tn, fp, fn, tp = cm[0, 0].item(), cm[0, 1].item(), cm[1, 0].item(), cm[1, 1].item()

    n = len(y_true)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Cohen's Kappa
    # p_e = expected agreement by chance
    p_pos = ((tp + fp) / n) * ((tp + fn) / n)
    p_neg = ((tn + fn) / n) * ((tn + fp) / n)
    p_e = p_pos + p_neg
    kappa = (accuracy - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =============================================================================
# Pretty-print
# =============================================================================


def print_metrics(metrics: dict[str, float], split: str = "") -> None:
    """Print a metrics dict in a readable table."""
    header = f"--- Metrics{' (' + split + ')' if split else ''} ---"
    print(header)
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(
        f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}"
    )


# =============================================================================
# Convenience wrapper
# =============================================================================


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    split: str = "",
) -> dict[str, float]:
    """
    Collect predictions, compute metrics, and print results.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataloader : DataLoader
        Evaluation dataloader.
    device : torch.device
    threshold : float
        Sigmoid threshold.
    split : str
        Label shown in the printed header (e.g. "test", "val").

    Returns
    -------
    dict with accuracy, precision, recall, f1, cohen_kappa, tp, tn, fp, fn.
    """
    y_true, y_pred = collect_predictions(model, dataloader, device, threshold)
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, split=split)
    return metrics
