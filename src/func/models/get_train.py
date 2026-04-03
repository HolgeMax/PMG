import csv
import torch
from torch import optim
import torch.nn as nn
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim import Adam

from src.func.models.get_models import PMGHead, build_resnet101, build_densenet201
from src.func.data.get_loader import PMGDataset, data_augmentation, get_dataloader, split_dataset
from src.func.evaluation.classification_metrics import compute_metrics

criterion = nn.BCEWithLogitsLoss()

# =============================================================================
# Training loop
# =============================================================================

def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optimizer | None = None,
    threshold: float = 0.5,
) -> tuple[float, dict]:
    """
    Run one epoch (train if *optimizer* is given, eval otherwise).

    Returns
    -------
    avg_loss : float
    metrics  : dict  (accuracy, precision, recall, f1, cohen_kappa, tp, tn, fp, fn)
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            print(images.shape)
            if training:
                optimizer.zero_grad()

            logits = model(images).squeeze(1)          # (B,)
            loss   = criterion(logits, labels.float())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= threshold).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics  = compute_metrics(y_true, y_pred)
    return avg_loss, metrics

def train(cfg):

    requested = cfg.train.device
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # --- Build model ---
    model_name      = cfg.model.name
    dropout_p       = cfg.model.dropout_p
    freeze_backbone = cfg.model.freeze_backbone

    if model_name == "resnet101":
        model = build_resnet101(dropout_p=dropout_p, freeze_backbone=freeze_backbone)
        print(f"Built ResNet-101 with dropout_p={dropout_p} and freeze_backbone={freeze_backbone}")
    elif model_name == "densenet201":
        model = build_densenet201(dropout_p=dropout_p, freeze_backbone=freeze_backbone)
        print(f"Built DenseNet-201 with dropout_p={dropout_p} and freeze_backbone={freeze_backbone}")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    model.to(device)

    # --- Patient-level split ---
    data_dir = cfg.data_loader.raw_data_dir if cfg.data_loader.train_raw else cfg.data_loader.data_dir
    print(f"Training on {'raw' if cfg.data_loader.train_raw else 'preprocessed'} data: {data_dir}")
    train_samples, val_samples, test_samples = split_dataset(
        data_dir          = data_dir,
        val_frac          = cfg.train.val_frac,
        test_frac         = cfg.train.test_frac,
        seed              = cfg.train.seed,
        pmg_negative_mode = cfg.data_loader.pmg_negative_mode,
        balance_mode      = cfg.data_loader.balance_mode,
    )
    print(f"Split dataset into {len(train_samples)} train, {len(val_samples)} val, and {len(test_samples)} test samples")

    # --- Build dataloaders ---
    transform_kwargs = dict(
        crop_size = cfg.data_loader.crop_size,
        scale     = tuple(cfg.data_loader.scale),
        mean      = list(cfg.data_loader.mean) if cfg.data_loader.mean is not None else [0.485, 0.456, 0.406],
        std       = list(cfg.data_loader.std)  if cfg.data_loader.std  is not None else [0.229, 0.224, 0.225],
    )
    augment = cfg.data_loader.get("augment", True)
    train_transform = data_augmentation(**transform_kwargs, is_training=augment)
    eval_transform  = data_augmentation(**transform_kwargs, is_training=False)
    if not augment:
        print("Data augmentation disabled — using deterministic transform for training")

    train_loader = get_dataloader(PMGDataset(samples=train_samples, transform=train_transform), batch_size=cfg.train.batch_size, shuffle=True,  num_workers=cfg.train.num_workers)
    val_loader   = get_dataloader(PMGDataset(samples=val_samples,   transform=eval_transform),  batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader  = get_dataloader(PMGDataset(samples=test_samples,  transform=eval_transform),  batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    print(f"Created dataloaders with batch size {cfg.train.batch_size} and num_workers {cfg.train.num_workers}")

    # --- Optimizer ---
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    print(f"Initialized Adam optimizer with learning rate {cfg.train.learning_rate}")

    # --- Output directory ---
    run_tag  = f"{model_name}_{'raw' if cfg.data_loader.train_raw else 'preprocessed'}"
    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = ckpt_dir / f"{run_tag}_best.pt"
    final_ckpt = ckpt_dir / f"{run_tag}_final.pt"

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"{run_tag}_metrics.csv"

    csv_columns = [
        "epoch",
        "train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "train_kappa",
        "val_loss",   "val_acc",   "val_precision",   "val_recall",   "val_f1",   "val_kappa",
        "test_loss",  "test_acc",  "test_precision",  "test_recall",  "test_f1",  "test_kappa",
    ]

    best_val_loss = float("inf")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()

        for epoch in range(cfg.train.num_epochs):
            train_loss, train_m = _run_epoch(model, train_loader, device, optimizer=optimizer)
            val_loss,   val_m   = _run_epoch(model, val_loader,   device)
            test_loss,  test_m  = _run_epoch(model, test_loader,  device)

            print(f"Epoch {epoch+1}/{cfg.train.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_m['accuracy']:.4f} - "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_m['accuracy']:.4f} - "
                  f"Test Loss: {test_loss:.4f}  Acc: {test_m['accuracy']:.4f}")
            # write metrics 
            writer.writerow({
                "epoch":            epoch + 1,
                "train_loss":       round(train_loss, 6),
                "train_acc":        round(train_m["accuracy"],    6),
                "train_precision":  round(train_m["precision"],   6),
                "train_recall":     round(train_m["recall"],      6),
                "train_f1":         round(train_m["f1"],          6),
                "train_kappa":      round(train_m["cohen_kappa"], 6),
                "val_loss":         round(val_loss, 6),
                "val_acc":          round(val_m["accuracy"],      6),
                "val_precision":    round(val_m["precision"],     6),
                "val_recall":       round(val_m["recall"],        6),
                "val_f1":           round(val_m["f1"],            6),
                "val_kappa":        round(val_m["cohen_kappa"],   6),
                "test_loss":        round(test_loss, 6),
                "test_acc":         round(test_m["accuracy"],     6),
                "test_precision":   round(test_m["precision"],    6),
                "test_recall":      round(test_m["recall"],       6),
                "test_f1":          round(test_m["f1"],           6),
                "test_kappa":       round(test_m["cohen_kappa"],  6),
            })
            fh.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)

    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved best checkpoint  → {best_ckpt}")
    print(f"Saved final checkpoint → {final_ckpt}")
    print(f"Saved metrics CSV      → {csv_path}")

# ==================================<============================================
# Test code to verify output shapes and parameter freezing
# ==============================================================================

if __name__ == "__main__":
    print("Running shape tests (downloads pretrained weights on first run)...\n")
    batch = torch.zeros(2, 3, 224, 224)

    # Test PMGHead standalone
    print("--- PMGHead (standalone) ---")
    head = PMGHead(in_features=2048, dropout_p=0.5)
    dummy_features = torch.zeros(2, 2048)
    out = head(dummy_features)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  Output shape: {out.shape}  [PASS]\n")

    # Test ResNet-101
    print("--- ResNet-101 (full model, frozen backbone) ---")
    resnet = build_resnet101(dropout_p=0.5, freeze_backbone=True)
    out = resnet(batch)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  Output shape: {out.shape}  [PASS]")
    trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in resnet.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  (only head should be trainable)\n")

    # Test DenseNet-201
    print("--- DenseNet-201 (full model, frozen backbone) ---")
    densenet = build_densenet201(dropout_p=0.5, freeze_backbone=True)
    out = densenet(batch)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  Output shape: {out.shape}  [PASS]")
    trainable = sum(p.numel() for p in densenet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in densenet.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  (only head should be trainable)\n")

    print("All tests passed.")
