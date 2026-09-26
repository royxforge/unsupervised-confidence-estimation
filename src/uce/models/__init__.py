"""Model architectures for UCE."""

from __future__ import annotations

from .resnet import (
    ResNetCIFAR,
    ResNetCIFARWithDropout,
    EvidentialResNet,
    create_model,
)

__all__ = [
    "ResNetCIFAR",
    "ResNetCIFARWithDropout",
    "EvidentialResNet",
    "create_model",
]