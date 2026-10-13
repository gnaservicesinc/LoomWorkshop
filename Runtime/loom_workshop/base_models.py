"""Architecture definitions for the new Factory base; Workshop freezes these weights."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value):
        return F.relu(value + self.conv2(F.relu(self.conv1(value))))


class MaterialMapNetwork(nn.Module):
    """RGB [0,1] -> linear depth/roughness or OpenGL normal RGB [0,1]."""
    architecture = "loom-factory-material-residual-v1"

    def __init__(self, map_kind: str, width: int = 24, blocks: int = 4):
        super().__init__()
        self.map_kind = map_kind
        self.width = width
        self.blocks = blocks
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.output = nn.Conv2d(width, 3 if map_kind == "normal" else 1, 3, padding=1)

    def forward(self, rgb):
        value = torch.sigmoid(self.output(self.body(F.relu(self.input(rgb)))))
        if self.map_kind == "normal":
            value = F.normalize(value * 2.0 - 1.0, dim=1, eps=1e-6) * 0.5 + 0.5
        return value

    def descriptor(self):
        return {"id": self.architecture, "width": self.width, "blocks": self.blocks,
                "input_channels": 3, "output_channels": 3 if self.map_kind == "normal" else 1,
                "input": "RGB sRGB float32 NCHW [0,1]", "output": "float32 NCHW [0,1]",
                "padding": "zero", "normal_convention": "OpenGL" if self.map_kind == "normal" else None}
