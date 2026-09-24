"""Model construction via timm.

EfficientNet-B0 is a strong, lightweight backbone (~5M params) that trains
fast on a single Colab GPU and pairs well with 224x224 inputs.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    """Create an EfficientNet-B0 classifier from the config.

    timm handles downloading pretrained weights and swapping the classifier
    head to `num_classes` for us.
    """
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.drop_rate,
    )
    return model
