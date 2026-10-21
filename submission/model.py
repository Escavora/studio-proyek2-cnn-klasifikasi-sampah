# Submission Studio 2 - ensemble 2 ConvNeXt-Tiny (timm convnext_tiny.fb_in22k_ft_in1k).
# Anggota 0: resep notebook 06. Anggota 1: resep notebook 07 (layer-wise LR decay + drop path).

import timm
import torch
import torch.nn as nn
from torchvision import transforms

NUM_CLASSES = 5
ARCH = "convnext_tiny.fb_in22k_ft_in1k"
N_MEMBERS = 2


class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.members = nn.ModuleList(
            timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES) for _ in range(N_MEMBERS))

    def forward(self, x):
        # Rata-rata probabilitas anggota; log() supaya output tetap logits [batch, 5] dan argmax tidak berubah.
        probs = torch.stack([m(x).softmax(1) for m in self.members]).mean(0)
        return probs.clamp_min(1e-12).log()


def get_model():
    # Arsitektur saja: tanpa mengunduh bobot (evaluate.py yang memuat model.pth).
    return Ensemble()


def get_transform():
    # Kedua anggota dilatih dengan transform validasi yang sama (get_eval_transform(224)).
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
