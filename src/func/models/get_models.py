
from xml.parsers.expat import model

import torch
import torch.nn as nn
import torchvision.models as models

#=============================================================================
# Custom head and model-building functions
#=============================================================================

class PMGHead(nn.Module):
    """
    Custom binary classification head.

    Args:
        in_features : size of the feature vector from the backbone
                      (2048 for ResNet-101, 1920 for DenseNet-201)
        dropout_p   : dropout probability (0 = disabled)
    """

    def __init__(self, in_features, dropout_p=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),  
        )

    def forward(self, x):
        fc = self.head(x)
        return fc  # raw logit 


def build_resnet101(dropout_p=0.5, freeze_backbone=False):
    """
    Load pretrained ResNet-101 and replace its classification head.

    Args:
        dropout_p       : dropout probability in the custom head
        freeze_backbone : if True, freeze all layers except the new head
                          (useful when your dataset is very small)

    Returns:
        model : modified ResNet-101 ready for binary classification
    """
    model = models.resnet101(pretrained=True) 

    in_features = model.fc.in_features
    model.fc = PMGHead(in_features=in_features, dropout_p=dropout_p)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True  # ensure head is trainable

    return model

# ==============================================================================
# Attach the head to DenseNet-201
# ==============================================================================
    
def build_densenet201(dropout_p=0.5, freeze_backbone=False):
    """
    Load pretrained DenseNet-201 and replace its classification head.

    Same structure as build_resnet101, but:
      - the head attribute is model.classifier  (not model.fc)
      - in_features is 1920

    Returns:
        model : modified DenseNet-201 ready for binary classification
    """

    model = models.densenet201(pretrained=True)

    in_features = model.classifier.in_features

    model.classifier = PMGHead(in_features=in_features, dropout_p=dropout_p)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True  # ensure head is trainable

    return model
