"""Convolutional Block Attention Module (CBAM) implementation.

Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
This file implements a lightweight CBAM compatible with feature tensors
extracted from backbone networks. It exposes a simple constructor:

    CBAM(channels, reduction=16)

and a forward(tensor) that returns a tensor of the same shape.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        b, c, _, _ = x.size()
        # B x C x 1 x 1
        avg_pool_out = self.avg_pool(x).view(b, c)
        max_pool_out = self.max_pool(x).view(b, c)
        # MLP shared
        avg_out = self.mlp(avg_pool_out)
        max_out = self.mlp(max_pool_out)
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_pool, max_pool], dim=1)  # B x 2 x H x W
        attn = self.sigmoid(self.conv(cat))
        return x * attn.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
