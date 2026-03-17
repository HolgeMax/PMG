import torch
from torch import optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim import Adam

from src.func.models.get_models import PMGHead, build_resnet101, build_densenet201
from src.func.data.get_loader import PMGDataset, data_augmentation, get_dataloader, split_dataset

criterion = nn.BCEWithLogitsLoss()

# =============================================================================
# Training loop
# =============================================================================
def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, device: torch.device):
    model.train()
    total_loss= 0.0

    for (images, labels) in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def validate_one_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits.squeeze(), labels.float())
            total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def test_one_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits.squeeze(), labels.float())
            total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

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
        data_dir         = data_dir,
        val_frac         = cfg.train.val_frac,
        test_frac        = cfg.train.test_frac,
        seed             = cfg.train.seed,
        pmg_negative_mode= cfg.data_loader.pmg_negative_mode,
    )
    print(f"Split dataset into {len(train_samples)} train, {len(val_samples)} val, and {len(test_samples)} test samples")
    # --- Build dataloaders ---
    transform_kwargs = dict(
        crop_size = cfg.data_loader.crop_size,
        scale     = tuple(cfg.data_loader.scale),
        mean      = list(cfg.data_loader.mean) if cfg.data_loader.mean is not None else [0.485, 0.456, 0.406],
        std       = list(cfg.data_loader.std)  if cfg.data_loader.std  is not None else [0.229, 0.224, 0.225],
    )
    train_transform = data_augmentation(**transform_kwargs, is_training=True)
    eval_transform  = data_augmentation(**transform_kwargs, is_training=False)

    train_loader = get_dataloader(PMGDataset(samples=train_samples, transform=train_transform), batch_size=cfg.train.batch_size, shuffle=True,  num_workers=cfg.train.num_workers)
    val_loader   = get_dataloader(PMGDataset(samples=val_samples,   transform=eval_transform),  batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader  = get_dataloader(PMGDataset(samples=test_samples,  transform=eval_transform),  batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    print(f"Created dataloaders with batch size {cfg.train.batch_size} and num_workers {cfg.train.num_workers}")

    # --- Optimizer ---
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate)
    print(f"Initialized Adam optimizer with learning rate {cfg.train.learning_rate}")

    for epoch in range(cfg.train.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = validate_one_epoch(model, val_loader, device)
        test_loss  = test_one_epoch(model, test_loader, device)

        print(f"Epoch {epoch+1}/{cfg.train.num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Test Loss: {test_loss:.4f}")

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
