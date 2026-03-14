import torch
import torch.nn as nn

from src.func.models.get_models import PMGHead, build_resnet101, build_densenet201

criterion = nn.BCEWithLogitsLoss()


# =============================================================================
# Training loop
# =============================================================================

def train(model, dataloader, optimizer, device):
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

# ==============================================================================
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
