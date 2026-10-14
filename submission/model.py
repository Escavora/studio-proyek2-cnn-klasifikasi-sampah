
# Submission Studio 2 - EfficientNet-B0 pretrained ImageNet.

import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model():
    # Arsitektur saja: tanpa mengunduh bobot (evaluate.py yang memuat model.pth).
    model = models.efficientnet_b0(weights=None)
    model.classifier[0] = nn.Dropout(p=0.3, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def get_transform():
    # Harus sama dengan transform validasi saat training.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
