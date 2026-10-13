"""Model architectures for UCE.

The committed checkpoints are ResNet-18 (the ``pretrained=True`` path of
:func:`create_model`), so those classes are part of the public API too.
"""

from __future__ import annotations

from .resnet import (
    BasicBlock,
    EvidentialResNet,
    EvidentialResNet18,
    ResNet18,
    ResNetCIFAR,
    ResNetCIFARWithDropout,
    create_model,
    load_pretrained,
)

__all__ = [
    "BasicBlock",
    "EvidentialResNet",
    "EvidentialResNet18",
    "ResNet18",
    "ResNetCIFAR",
    "ResNetCIFARWithDropout",
    "create_model",
    "load_pretrained",
]