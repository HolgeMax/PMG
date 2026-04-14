import csv
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.data.get_loader import PMGDataset, data_augmentation, get_dataloader, split_dataset
from src.func.evaluation.classification_metrics import evaluate_model
from src.func.models.get_models import build_densenet201, build_resnet101


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_builder(ckpt_name: str):
    """Return the model builder function inferred from the checkpoint filename."""
    name = ckpt_name.lower()
    if "resnet" in name:
        return build_resnet101
    if "densenet" in name:
        return build_densenet201
    raise ValueError(f"Cannot infer model architecture from filename: {ckpt_name!r}")


def run_evaluate(cfg: DictConfig) -> None:
    device = _select_device(cfg.train.device)
    print(f"Using device: {device}")

    # --- Test dataloader ---
    data_dir = (
        cfg.data_loader.raw_data_dir
        if cfg.data_loader.train_raw
        else cfg.data_loader.data_dir
    )
    _, _, test_samples = split_dataset(
        data_dir=data_dir,
        val_frac=cfg.train.val_frac,
        test_frac=cfg.train.test_frac,
        seed=cfg.train.seed,
        pmg_negative_mode=cfg.data_loader.pmg_negative_mode,
        balance_mode=cfg.data_loader.balance_mode,
    )
    transform = data_augmentation(
        crop_size=cfg.data_loader.crop_size,
        scale=tuple(cfg.data_loader.scale),
        mean=list(cfg.data_loader.mean),
        std=list(cfg.data_loader.std),
        is_training=False,
    )
    test_loader = get_dataloader(
        PMGDataset(samples=test_samples, transform=transform),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    print(f"Test set: {len(test_samples)} samples  ({cfg.data_loader.pmg_negative_mode} mode)")

    # --- Discover checkpoints ---
    cwd = Path(get_original_cwd())
    ckpt_dir = Path(cfg.evaluate.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = cwd / ckpt_dir

    pattern = "*_best.pt" if cfg.evaluate.only_best else "*.pt"
    ckpts = sorted(ckpt_dir.rglob(pattern))

    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir} (pattern={pattern!r})")
        return

    print(f"\nFound {len(ckpts)} checkpoint(s) under {ckpt_dir}\n")

    # --- Evaluate each checkpoint ---
    results = []
    for ckpt_path in ckpts:
        print(f"{'=' * 60}")
        print(f"Checkpoint: {ckpt_path.relative_to(cwd)}")
        try:
            builder = _infer_builder(ckpt_path.name)
        except ValueError as exc:
            print(f"  Skipping — {exc}")
            continue

        model = builder(dropout_p=cfg.model.dropout_p, freeze_backbone=False)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)

        metrics = evaluate_model(model, test_loader, device, split="test")
        results.append({"checkpoint": str(ckpt_path.relative_to(cwd)), **metrics})

    if not results:
        print("No checkpoints were successfully evaluated.")
        return

    # --- Save CSV summary ---
    out_dir = Path(cfg.evaluate.output_dir)
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "evaluation_summary.csv"
    fieldnames = [
        "checkpoint",
        "accuracy", "precision", "recall", "f1", "cohen_kappa",
        "tp", "tn", "fp", "fn",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"Evaluated {len(results)} model(s).  Summary → {csv_path}")


@hydra.main(
    version_base=None,
    config_path=str(project_root / "hydra"),
    config_name="evaluate_config",
)
def main(cfg: DictConfig) -> None:
    run_evaluate(cfg)


def evaluate_cli() -> None:
    main()


if __name__ == "__main__":
    evaluate_cli()
