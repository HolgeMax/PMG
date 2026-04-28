import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.evaluation.run_ablation import run_ablation


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
