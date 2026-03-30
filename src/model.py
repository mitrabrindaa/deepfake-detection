"""Xception backbone (ImageNet pretrained) + strong classifier head; optional EfficientNet-B0 branch."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import models as tv_models


def _replace_xception_head(model: nn.Module, num_classes: int, hidden: int, dropout: float) -> nn.Module:
    in_f = model.output.in_features
    model.output = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, hidden),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(hidden),
        nn.Dropout(dropout * 0.6),
        nn.Linear(hidden, num_classes),
    )
    return model


def _replace_efficient_head(model: nn.Module, num_classes: int, hidden: int, dropout: float) -> nn.Module:
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout, inplace=True),
        nn.Linear(in_f, hidden),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(hidden),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden, num_classes),
    )
    return model


class XceptionDeepFake(nn.Module):
    """Primary model: Xception + MLP head (matches common DF literature)."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden: int = 512,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = ptcv_get_model("xception", pretrained=pretrained)
        self.backbone = _replace_xception_head(self.backbone, num_classes, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, p in self.backbone.named_parameters():
            if name.startswith("output"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze_last_stages(self, unfreeze_stages: Tuple[str, ...] = ("stage4", "final_block")) -> None:
        for name, p in self.backbone.features.named_parameters():
            if any(s in name for s in unfreeze_stages):
                p.requires_grad = True


class AuxEfficientNet(nn.Module):
    """Auxiliary branch: EfficientNet-B0 (complementary inductive bias for ensemble)."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden: int = 256,
        dropout: float = 0.45,
        pretrained: bool = True,
    ):
        super().__init__()
        w = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.efficientnet_b0(weights=w)
        self.backbone = _replace_efficient_head(self.backbone, num_classes, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, p in self.backbone.named_parameters():
            if name.startswith("classifier"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze_last_blocks(self) -> None:
        for name, p in self.backbone.features.named_parameters():
            if name.startswith("7.") or name.startswith("8."):
                p.requires_grad = True


class EnsembleDetector(nn.Module):
    """Xception + EfficientNet; train with sum of CE losses; infer with weighted logits."""

    def __init__(
        self,
        num_classes: int = 2,
        x_hidden: int = 512,
        e_hidden: int = 256,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.xcep = XceptionDeepFake(
            num_classes=num_classes,
            hidden=x_hidden,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.eff = AuxEfficientNet(
            num_classes=num_classes,
            hidden=e_hidden,
            dropout=dropout * 0.9,
            pretrained=pretrained,
        )

    def forward(
        self, x: torch.Tensor, return_both: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        lx = self.xcep(x)
        le = self.eff(x)
        if return_both:
            return lx, le
        # default: equal fusion for loss; train script can use both separately
        return 0.5 * (lx + le)

    def freeze_backbones(self) -> None:
        self.xcep.freeze_backbone()
        self.eff.freeze_backbone()

    def unfreeze_fine_tune(self) -> None:
        self.xcep.unfreeze_last_stages()
        self.eff.unfreeze_last_blocks()


def build_model(
    use_ensemble: bool,
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    from . import config

    if use_ensemble:
        return EnsembleDetector(
            num_classes=num_classes,
            x_hidden=config.HIDDEN_DIM,
            e_hidden=max(256, config.HIDDEN_DIM // 2),
            dropout=config.DROPOUT,
            pretrained=pretrained,
        )
    return XceptionDeepFake(
        num_classes=num_classes,
        hidden=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        pretrained=pretrained,
    )
