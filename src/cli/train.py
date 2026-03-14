"""CLI entry point for model training with Hydra.

Usage:
    uv run train
    uv run train model.name=densenet201
    uv run train train.num_epochs=10 train.learning_rate=1e-3
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.models.get_train import train


@hydra.main(version_base=None, config_path=str(project_root / "hydra"), config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


def train_cli():
    main()


if __name__ == "__main__":
    train_cli()
