import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.models.get_crossval import run_crossval


@hydra.main(
    version_base=None,
    config_path=str(project_root / "hydra"),
    config_name="crossval_config",
)
def main(cfg: DictConfig) -> None:
    run_crossval(cfg)


def crossval_cli() -> None:
    main()


if __name__ == "__main__":
    crossval_cli()
