"""Post-hoc evaluation script for trained PMG checkpoints.

Scans a checkpoint directory, infers the data configuration from each
checkpoint filename, reconstructs the matching test split, and evaluates
with balanced accuracy.  Results are saved to results/post-hoc_metrics/.

Filename inference rules
------------------------
  *resnet*          → ResNet-101 backbone
  *densenet*        → DenseNet-201 backbone
  *_raw_*           → raw (unprocessed) data
  *_preprocessed_*  → preprocessed data (default)
  *_paper*          → pmg_negative_mode = "paper"
  *_correct*        → pmg_negative_mode = "correct" (default)
  *_downsampled*    → balance_mode = "post_split"
  *fold<N>*         → k-fold cross-validation, fold N (1-based)

Usage
-----
  python -m src.cli.posthoc_eval                          # defaults
  python -m src.cli.posthoc_eval posthoc.only_best=false  # include final ckpts
"""

import csv
import re
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.data.crossval_split import kfold_split_patients
from src.func.data.get_loader import (
    PMGDataset,
    data_augmentation,
    get_dataloader,
    split_dataset,
)
from src.func.evaluation.classification_metrics import evaluate_model
from src.func.models.get_models import build_densenet201, build_resnet101


# =============================================================================
# Helpers
# =============================================================================


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_builder(name: str):
    name = name.lower()
    if "resnet" in name:
        return build_resnet101
    if "densenet" in name:
        return build_densenet201
    raise ValueError(f"Cannot infer model architecture from filename: {name!r}")


def _parse_ckpt(ckpt_path: Path) -> dict:
    """Derive data config from the checkpoint filename."""
    name = ckpt_path.stem  # strip .pt

    train_raw = "_raw_" in name

    pmg_mode = "paper" if "_paper" in name else "correct"

    balance_mode = "post_split" if "_downsampled" in name else None

    fold_match = re.search(r"fold(\d+)", name)
    if fold_match:
        return {
            "train_raw": train_raw,
            "pmg_mode": pmg_mode,
            "balance_mode": balance_mode,
            "is_crossval": True,
            "fold_idx": int(fold_match.group(1)) - 1,  # 0-based
        }

    return {
        "train_raw": train_raw,
        "pmg_mode": pmg_mode,
        "balance_mode": balance_mode,
        "is_crossval": False,
        "fold_idx": None,
    }


def _get_test_samples(cfg: DictConfig, parsed: dict) -> list:
    """Reconstruct the same test split that was used during training."""
    data_dir = (
        cfg.data_loader.raw_data_dir if parsed["train_raw"] else cfg.data_loader.data_dir
    )

    if parsed["is_crossval"]:
        target = parsed["fold_idx"]
        for _, _, test_samples, fold_i in kfold_split_patients(
            data_dir=data_dir,
            n_folds=cfg.crossval.n_folds,
            val_frac_of_train=cfg.crossval.val_frac_of_train,
            seed=cfg.train.seed,
            pmg_negative_mode=parsed["pmg_mode"],
            balance_mode=parsed["balance_mode"],
        ):
            if fold_i == target:
                return test_samples
        raise ValueError(f"Fold {target + 1} not found in {cfg.crossval.n_folds}-fold split")

    _, _, test_samples = split_dataset(
        data_dir=data_dir,
        val_frac=cfg.train.val_frac,
        test_frac=cfg.train.test_frac,
        seed=cfg.train.seed,
        pmg_negative_mode=parsed["pmg_mode"],
        balance_mode=parsed["balance_mode"],
    )
    return test_samples


# =============================================================================
# Main evaluation loop
# =============================================================================


def run_posthoc_eval(cfg: DictConfig) -> None:
    device = _select_device(cfg.train.device)
    cwd = Path(get_original_cwd())

    ckpt_dir = Path(cfg.posthoc.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = cwd / ckpt_dir

    pattern = "*_best.pt" if cfg.posthoc.only_best else "*.pt"
    ckpts = sorted(ckpt_dir.rglob(pattern))

    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir} (pattern={pattern!r})")
        return

    print(f"Found {len(ckpts)} checkpoint(s) under {ckpt_dir}\n")

    transform = data_augmentation(
        crop_size=cfg.data_loader.crop_size,
        scale=tuple(cfg.data_loader.scale),
        mean=list(cfg.data_loader.mean),
        std=list(cfg.data_loader.std),
        is_training=False,
    )

    results = []
    for ckpt_path in ckpts:
        print("=" * 60)
        rel = str(ckpt_path.relative_to(cwd))
        print(f"Checkpoint: {rel}")

        try:
            builder = _infer_builder(ckpt_path.name)
        except ValueError as exc:
            print(f"  Skipping — {exc}")
            continue

        parsed = _parse_ckpt(ckpt_path)
        print(
            f"  Inferred config: raw={parsed['train_raw']}, "
            f"pmg_mode={parsed['pmg_mode']}, "
            f"balance={parsed['balance_mode']}, "
            f"crossval={parsed['is_crossval']}"
            + (f", fold={parsed['fold_idx'] + 1}" if parsed["is_crossval"] else "")
        )

        try:
            test_samples = _get_test_samples(cfg, parsed)
        except Exception as exc:
            print(f"  Skipping — could not reconstruct test split: {exc}")
            continue

        n_pmg = sum(1 for _, lbl in test_samples if lbl == 1)
        n_hc = sum(1 for _, lbl in test_samples if lbl == 0)
        print(f"  Test set: {len(test_samples)} slices  (PMG={n_pmg}, HC={n_hc})")

        test_loader = get_dataloader(
            PMGDataset(samples=test_samples, transform=transform),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )

        model = builder(dropout_p=cfg.model.dropout_p, freeze_backbone=False)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)

        metrics = evaluate_model(model, test_loader, device, split="test")
        results.append(
            {
                "checkpoint": rel,
                "pmg_mode": parsed["pmg_mode"],
                "train_raw": parsed["train_raw"],
                "balance_mode": parsed["balance_mode"] or "none",
                **metrics,
            }
        )

    if not results:
        print("No checkpoints were successfully evaluated.")
        return

    out_dir = Path(cfg.posthoc.output_dir)
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "posthoc_summary.csv"
    fieldnames = [
        "checkpoint",
        "pmg_mode",
        "train_raw",
        "balance_mode",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "cohen_kappa",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Evaluated {len(results)} model(s).  Summary → {csv_path}")


# =============================================================================
# Hydra entry point
# =============================================================================


@hydra.main(
    version_base=None,
    config_path=str(project_root / "hydra"),
    config_name="posthoc_eval_config",
)
def main(cfg: DictConfig) -> None:
    run_posthoc_eval(cfg)


def posthoc_eval_cli() -> None:
    main()


if __name__ == "__main__":
    posthoc_eval_cli()
