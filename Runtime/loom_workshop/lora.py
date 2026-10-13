import math

import torch
from torch import nn


class LoRAConv2d(nn.Module):
    """Frozen convolution plus low-rank kernel update, initially exactly zero."""
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        if base.groups != 1:
            raise ValueError("This LoRA architecture requires ordinary ungrouped convolutions.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.down = nn.Conv2d(base.in_channels, rank, base.kernel_size, stride=base.stride,
                              padding=base.padding, dilation=base.dilation, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, image):
        return self.base(image) + self.up(self.down(image)) * (self.alpha / self.rank)

    def fused(self):
        convolution = nn.Conv2d(self.base.in_channels, self.base.out_channels, self.base.kernel_size,
                                stride=self.base.stride, padding=self.base.padding,
                                dilation=self.base.dilation, bias=self.base.bias is not None)
        update = (self.up.weight[:, :, 0, 0] @ self.down.weight.flatten(1)).reshape_as(self.base.weight)
        convolution.weight.data.copy_(self.base.weight.detach().cpu() + update.detach().cpu() * (self.alpha / self.rank))
        if self.base.bias is not None:
            convolution.bias.data.copy_(self.base.bias.detach().cpu())
        return convolution


def attach_lora(model, rank, alpha):
    model.requires_grad_(False)
    targets = []
    def visit(parent, prefix=""):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                setattr(parent, name, LoRAConv2d(child, rank, alpha))
                targets.append(path)
            else:
                visit(child, path)
    visit(model)
    if not targets:
        raise ValueError("This base architecture has no compatible convolution layers.")
    return targets


def adapter_weights(model):
    return {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
            if key.endswith((".down.weight", ".up.weight"))}


def frozen_base_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if ".base." in key}


def fuse_lora(model):
    # The deployment copy contains ordinary convolutions. Original installed
    # weights remain untouched; adapter safetensors contain only down/up deltas.
    import copy
    result = copy.deepcopy(model).cpu()
    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRAConv2d):
                setattr(parent, name, child.fused())
            else:
                visit(child)
    visit(result)
    return result.eval()
