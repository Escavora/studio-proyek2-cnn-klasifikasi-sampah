# Submission Studio 2 - ConvNeXt-Tiny (timm convnext_tiny.fb_in22k_ft_in1k), resep notebook 06.

import timm
from torchvision import transforms

NUM_CLASSES = 5


def get_model():
    # Arsitektur saja: tanpa mengunduh bobot (evaluate.py yang memuat model.pth).
    return timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=NUM_CLASSES)


def get_transform():
    # Harus sama dengan transform validasi saat training (get_eval_transform(224)).
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
