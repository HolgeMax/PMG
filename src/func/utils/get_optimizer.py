import torch

# ==============================================================================
# Optimizer helper function
# ==============================================================================

def get_optimizer(model, option="A"):
    if option == "A":
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam([
            {"params": model.classifier.parameters(),  "lr": 1e-3},
            {"params": [p for n,p in model.named_parameters() if not n.startswith("classifier")], "lr": 1e-5}
        ])
    return optimizer