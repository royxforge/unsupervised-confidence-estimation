"""ResNet architectures for CIFAR-10 with UQ method variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResNetCIFAR",
    "ResNetCIFARWithDropout",
    "EvidentialResNet",
    "create_model",
    "load_pretrained",
]


class BasicBlock(nn.Module):
    """Basic ResNet block for CIFAR (3x3 conv, no bottleneck)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """Standard ResNet for CIFAR-10 (ResNet-20/32/44/56/110)."""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features for contrastive/feature-space methods."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class ResNetCIFARWithDropout(ResNetCIFAR):
    """ResNet with dropout layers for MC Dropout uncertainty estimation."""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
        dropout_rate: float = 0.1,
        dropout_layers: list[str] | None = None,
    ):
        super().__init__(block, num_blocks, num_classes, in_channels)
        self.dropout_rate = dropout_rate
        self.dropout_layers = dropout_layers or ["layer1", "layer2", "layer3"]
        self.dropouts = nn.ModuleDict()

        # Add dropout after each specified layer
        for layer_name in self.dropout_layers:
            if hasattr(self, layer_name):
                self.dropouts[layer_name] = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))

        # Layer 1 with optional dropout
        out = self.layer1(out)
        if "layer1" in self.dropouts:
            out = self.dropouts["layer1"](out)

        # Layer 2 with optional dropout
        out = self.layer2(out)
        if "layer2" in self.dropouts:
            out = self.dropouts["layer2"](out)

        # Layer 3 with optional dropout
        out = self.layer3(out)
        if "layer3" in self.dropouts:
            out = self.dropouts["layer3"](out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features for contrastive/feature-space methods."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

        return out

    def mc_dropout_forward(
        self, x: torch.Tensor, n_samples: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout forward pass for uncertainty estimation.

        Returns:
            (mean_logits, variance_logits) over MC samples
        """
        self.train()  # Enable dropout
        logits_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                logits_list.append(logits)

        logits_stack = torch.stack(logits_list, dim=1)  # (B, N, C)
        mean_logits = logits_stack.mean(dim=1)
        var_logits = logits_stack.var(dim=1)
        return mean_logits, var_logits



class EvidentialResNet(nn.Module):
    """
    Evidential Deep Learning ResNet for CIFAR-10.

    Outputs Dirichlet distribution parameters (alpha) for Dirichlet distribution
    over class probabilities. Uses ReLU + 1 for evidence, alpha = evidence + 1.
    """
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 3, stride=1)
        self.layer2 = self._make_layer(block, 32, 3, stride=2)
        self.layer3 = self._make_layer(block, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        evidence = self.fc(out)
        evidence = F.relu(evidence) + 1e-6
        alpha = evidence + 1
        return alpha
    def get_evidence(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        evidence = F.relu(self.fc(out)) + 1e-6
        return evidence

    def get_alpha(self, x):
        """Get Dirichlet concentration parameters alpha."""
        return self.get_evidence(x) + 1

    def predict_proba(self, x):
        """Predict class probabilities from Dirichlet mean."""
        alpha = self.get_alpha(x)
        return alpha / alpha.sum(dim=1, keepdim=True)


class ResNet18(nn.Module):
    """Standard ResNet-18 for CIFAR-10 (64, 64, 128, 256 channels)."""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features for contrastive/feature-space methods."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class EvidentialResNet18(nn.Module):
    """
    Evidential Deep Learning ResNet-18 for CIFAR-10.

    Outputs Dirichlet distribution parameters (alpha) for Dirichlet distribution
    over class probabilities. Uses ReLU + 1 for evidence, alpha = evidence + 1.
    Uses evidence_head instead of fc to match pretrained checkpoints.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Evidential output: predict evidence (K parameters) -> alpha = evidence + 1
        self.evidence_head = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        evidence = self.evidence_head(out)
        # Evidence must be non-negative: ReLU + 1e-6
        evidence = F.relu(evidence) + 1e-6
        alpha = evidence + 1  # Dirichlet concentration parameters
        return alpha

    def get_evidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw evidence (before +1)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        evidence = F.relu(self.evidence_head(out)) + 1e-6
        return evidence

    def get_alpha(self, x: torch.Tensor) -> torch.Tensor:
        """Get Dirichlet concentration parameters alpha."""
        return self.get_evidence(x) + 1

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities from Dirichlet mean."""
        alpha = self.get_alpha(x)
        return alpha / alpha.sum(dim=1, keepdim=True)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features for contrastive/feature-space methods."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def create_model(
    model_type: str,
    num_classes: int = 10,
    in_channels: int = 3,
    depth: int = 20,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "baseline", "mc_dropout", "evidential", "ensemble"
        num_classes: Number of classes (default 10 for CIFAR-10)
        in_channels: Input channels (3 for CIFAR)
        depth: ResNet depth (20, 32, 44, 56, 110 for CIFAR ResNet; 18 for ResNet-18)
        pretrained: If True, use ResNet-18 architecture matching pretrained checkpoints
        **kwargs: Additional arguments for specific model types

    Returns:
        Initialized model
    """
    # Map depth to num_blocks per layer for ResNet on CIFAR
    depth_to_blocks = {
        18: [2, 2, 2, 2],  # ResNet-18
        20: [3, 3, 3],     # ResNet-20
        32: [5, 5, 5],     # ResNet-32
        44: [7, 7, 7],     # ResNet-44
        56: [9, 9, 9],     # ResNet-56
        110: [18, 18, 18], # ResNet-110
    }
    
    if pretrained or depth == 18:
        # Use ResNet-18 architecture for pretrained models (all checkpoints are ResNet-18)
        num_blocks = depth_to_blocks.get(18, [2, 2, 2, 2])
        if model_type == "evidential":
            return EvidentialResNet18(BasicBlock, num_blocks, num_classes, in_channels)
        return ResNet18(BasicBlock, num_blocks, num_classes, in_channels)
    else:
        num_blocks = depth_to_blocks.get(depth, [3, 3, 3])
        if model_type == "baseline":
            return ResNetCIFAR(BasicBlock, num_blocks, num_classes, in_channels)
        elif model_type == "mc_dropout":
            dropout_rate = kwargs.get("dropout_rate", 0.1)
            return ResNetCIFARWithDropout(
                BasicBlock, num_blocks, num_classes, in_channels, dropout_rate
            )
        elif model_type == "evidential":
            return EvidentialResNet(BasicBlock, num_blocks, num_classes, in_channels)
        elif model_type == "ensemble":
            return ResNetCIFAR(BasicBlock, num_blocks, num_classes, in_channels)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


def load_pretrained(
    model_type: str,
    checkpoint_path: str,
    num_classes: int = 10,
    in_channels: int = 3,
    depth: int = 18,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """Load a pretrained model from checkpoint using ResNet-18 architecture."""
    model = create_model(model_type, num_classes, in_channels, depth, pretrained=True, **kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model