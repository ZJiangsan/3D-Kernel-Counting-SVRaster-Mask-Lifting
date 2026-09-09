from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BCEDiceLoss(nn.Module):
    """BCE + soft Dice loss used by the experimental U-Net script."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        prob = torch.sigmoid(pred)
        inter = (prob * target).sum()
        dice = 1.0 - (2.0 * inter + self.smooth) / (
            prob.sum() + target.sum() + self.smooth
        )
        return bce + dice


def build_unet(encoder_weights: str | None = "imagenet") -> nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )
