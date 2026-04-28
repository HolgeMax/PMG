"""Slice 3D NIfTI volumes into 2D JPEG slices.

Paths in the metadata CSV are resolved as local filesystem paths.
Run this script on the server where the data lives (e.g. via ThinLinc).

Output filename convention: {session_id}_{slice_idx:03d}_0_{label_int}.jpg
  where session_id = strip-hyphens(subject) + '-' + strip-hyphens(session)
  e.g. sub-01 + ses-001  ->  sub01-ses001_042_0_1.jpg

Output folder structure mirrors PPMR so PMGDataset works unchanged:
  <output_dir>/PMGcases/<subject_sanitized>/<session_sanitized>/...jpg
  <output_dir>/controlcases/<subject_sanitized>/<session_sanitized>/...jpg

Usage:
    uv run slice-volumes
    uv run slice-volumes volume_slicing.metadata_file=data/pmg_labels.csv
    uv run slice-volumes volume_slicing.slice_selection=random
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.data.volume_slicing import run_slice_volumes


@hydra.main(
    config_path=str(project_root / "hydra"),
    config_name="volume_slicing_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_slice_volumes(cfg)


def slice_volumes_cli():
    main()


if __name__ == "__main__":
    slice_volumes_cli()
