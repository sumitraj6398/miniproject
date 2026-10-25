import torch.nn as nn
from torchvision.models import efficientnet_b0


def load_model(num_classes: int = 4):
    """Build EfficientNet-B0 with the custom head used during training.

    weights=None because the trained checkpoint is loaded afterwards,
    so there is no need to download ImageNet weights at startup.
    """
    model = efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )

    return model
