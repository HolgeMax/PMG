"""CLI entry point for preprocessing with Hydra.

Accepts a single file **or** a directory as input_path.  When a directory is
given, every supported file inside it is processed in one run.

Usage — single file:
    uv run preprocess input_path=data/nii_test/file.nii preprocessing=light
    uv run preprocess input_path=data/PPMR/.../slice.jpg preprocessing=light

Usage — directory (flat):
    uv run preprocess input_path=data/nii_test preprocessing=light

Usage — directory (recursive, e.g. PPMR patient tree):
    uv run preprocess input_path=data/PPMR/PMGstudycaseslabelled preprocessing=light recursive=true
"""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.func.utils.loader import collect_input_files, process_one
from src.func.utils.cfg import config_to_preprocessing_config


# ── Entry point
@hydra.main(
    version_base=None, config_path=str(project_root / "hydra"), config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Run the preprocessing pipeline on a file or directory."""
    # Suppress per-step INFO logs from the pipeline during processing
    import logging

    logging.getLogger("src.main.configurable_pipeline").setLevel(logging.WARNING)

    # check if input path exists
    input_path = Path(cfg.input_path)
    if not input_path.exists():
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    # check if output path exists
    output_path = Path(cfg.output_path)
    if not output_path.exists():
        print(f"Output path does not exist, creating: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.get("recursive", False))
    files = collect_input_files(input_path, recursive=recursive)

    if not files:
        sys.exit(0)

    preset_name = HydraConfig.get().runtime.choices.get("preprocessing", "default")

    preprocess_config = config_to_preprocessing_config(cfg)

    print(f"Output path: {output_path}")
    # Save config once per run
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / f"config_{preset_name}.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ── Process files
    print(f"\nInput  : {input_path}  ({'recursive' if recursive else 'flat'})")
    print(f"Files  : {len(files)}")
    print(f"Output : {cfg.output_path}\n")

    results, failed = [], []

    for file_path in tqdm(files, desc="Preprocessing", unit="file"):
        try:
            metrics = process_one(
                file_path,
                preprocess_config,
                cfg.slice_idx,
                output_path,
                preset_name,
                input_root=input_path if input_path.is_dir() else None,
            )
            results.append(metrics)
        except Exception as exc:
            tqdm.write(f"  FAILED {file_path.name}: {exc}")
            failed.append(file_path.name)

    # ── Summary
    print(f"\n{'─' * 70}")
    print(f"Done — {len(results)} succeeded, {len(failed)} failed")
    if results:
        psnr_vals = [r["psnr"] for r in results if np.isfinite(r["psnr"])]
        ssim_vals = [r["ssim"] for r in results]
        print(
            f"  PSNR  mean ± std : {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB"
        )
        print(
            f"  SSIM  mean ± std : {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}"
        )
    if failed:
        print("\nFailed files:")
        for name in failed:
            print(f"  {name}")
    print(f"Config saved to : {config_path}")
    print(f"Outputs in      : {output_path}")
    print(f"\n{'─' * 70}")
    print("Config used:")
    print(OmegaConf.to_yaml(cfg))


def preprocess_cli():
    """Entry point for the preprocess command."""
    main()


if __name__ == "__main__":
    preprocess_cli()
