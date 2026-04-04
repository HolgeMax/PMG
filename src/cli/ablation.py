import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.data.get_loader import PMGDataset, data_augmentation, get_dataloader, split_dataset
from src.func.evaluation.ablation_study import make_black_box, run_all_ckpts_ablation_study
from src.func.models.get_models import build_densenet201, build_resnet101


def run_ablation(cfg: DictConfig) -> None:
    device = "cuda" if cfg.ablation.device == "cuda" and torch.cuda.is_available() else "cpu"

    name = cfg.model.name
    if name == "resnet101":
        model = build_resnet101(dropout_p=cfg.model.dropout_p, freeze_backbone=cfg.model.freeze_backbone)
    elif name == "densenet201":
        model = build_densenet201(dropout_p=cfg.model.dropout_p, freeze_backbone=cfg.model.freeze_backbone)
    else:
        raise ValueError(f"Unknown model '{name}'. Expected 'resnet101' or 'densenet201'.")

    _train, _val, test_samples = split_dataset(
        cfg.data_loader.data_dir,
        cfg.train.val_frac,
        cfg.train.test_frac,
        cfg.train.seed,
        cfg.data_loader.pmg_negative_mode,
        cfg.data_loader.balance_mode,
    )

    transform = data_augmentation(
        cfg.data_loader.crop_size,
        cfg.data_loader.scale,
        cfg.data_loader.mean,
        cfg.data_loader.std,
        is_training=False,
    )

    test_loader = get_dataloader(
        PMGDataset(samples=test_samples, transform=transform),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )

    modified_data = make_black_box(
        test_loader,
        device=device,
        box_size_frac=cfg.ablation.box_size_frac,
    )

    results = run_all_ckpts_ablation_study(
        model,
        modified_data,
        cfg.ablation.checkpoint_dir,
        device,
    )

    output_dir = Path(cfg.ablation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "checkpoint"
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")

    yaml_path = output_dir / "ablation_config.yaml"
    yaml_path.write_text(OmegaConf.to_yaml(cfg))
    print(f"Config saved to {yaml_path}")


@hydra.main(
    version_base=None,
    config_path=str(project_root / "hydra"),
    config_name="ablation_config",
)
def main(cfg: DictConfig) -> None:
    run_ablation(cfg)


def ablation_cli() -> None:
    main()


if __name__ == "__main__":
    ablation_cli()
