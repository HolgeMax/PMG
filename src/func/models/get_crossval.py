import csv
import numpy as np
from pathlib import Path

from src.func.data.crossval_split import kfold_split_patients
from src.func.models.get_train import train_one_fold


def run_crossval(cfg) -> None:
    """Run 5-fold (or n-fold) cross-validation and save per-fold + summary CSVs.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config. Must include a ``crossval`` block with
        ``n_folds`` and ``val_frac_of_train``.
    """
    data_dir = (
        cfg.data_loader.raw_data_dir
        if cfg.data_loader.train_raw
        else cfg.data_loader.data_dir
    )
    print(
        f"Cross-validation on {'raw' if cfg.data_loader.train_raw else 'preprocessed'} data: {data_dir}"
    )

    n_folds = cfg.crossval.n_folds
    fold_results = []

    for train_samples, val_samples, test_samples, fold_idx in kfold_split_patients(
        data_dir=data_dir,
        n_folds=n_folds,
        val_frac_of_train=cfg.crossval.val_frac_of_train,
        seed=cfg.train.seed,
        pmg_negative_mode=cfg.data_loader.pmg_negative_mode,
        balance_mode=cfg.data_loader.balance_mode,
    ):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'=' * 60}")

        metrics = train_one_fold(
            cfg,
            train_samples,
            val_samples,
            test_samples,
            fold_tag=f"fold{fold_idx + 1}",
        )
        fold_results.append({"fold": fold_idx + 1, **metrics})

    _save_results(cfg, fold_results)


def _save_results(cfg, fold_results: list[dict]) -> None:
    metrics_dir = Path("results/metrics") / "crossvalidation"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.model.name
    suffix = "raw" if cfg.data_loader.train_raw else "preprocessed"
    base = f"{model_name}_{suffix}_crossval"

    metric_keys = ["accuracy", "precision", "recall", "f1", "cohen_kappa"]

    # Per-fold CSV
    per_fold_path = metrics_dir / f"{base}_per_fold.csv"
    with open(per_fold_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fold"] + metric_keys)
        writer.writeheader()
        for row in fold_results:
            writer.writerow(
                {
                    "fold": row["fold"],
                    "accuracy": round(row.get("accuracy", 0.0), 6),
                    "precision": round(row.get("precision", 0.0), 6),
                    "recall": round(row.get("recall", 0.0), 6),
                    "f1": round(row.get("f1", 0.0), 6),
                    "cohen_kappa": round(row.get("cohen_kappa", 0.0), 6),
                }
            )

    # Pre-compute one array per metric — used for both CSV and console output
    cols = {
        k: np.array([r.get(k, 0.0) for r in fold_results]) for k in metric_keys
    }

    #    Summary CSV (mean ± std)
    summary_path = metrics_dir / f"{base}_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stat"] + metric_keys)
        writer.writeheader()
        for stat in ("mean", "std"):
            fn = np.mean if stat == "mean" else lambda c: np.std(c, ddof=1)
            writer.writerow(
                {"stat": stat, **{k: round(float(fn(cols[k])), 6) for k in metric_keys}}
            )

    print("\nCross-validation complete.")
    print(f"  Per-fold results → {per_fold_path}")
    print(f"  Summary          → {summary_path}")

    print(f"\n  Mean test metrics across {len(fold_results)} folds:")
    for k in metric_keys:
        print(f"    {k:<14} {round(float(np.mean(cols[k])), 4)}")
