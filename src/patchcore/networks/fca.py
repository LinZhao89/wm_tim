"""Frequency & Channel Attention (approximate) module.

This implements a lightweight approximation of the Frequency & Channel
Attention concept: a channel attention (SE-like) and a frequency-derived
spatial attention (via FFT magnitude processed by a small conv). It's
intended to be applied to feature maps (B x C x H x W) before patchification.

This is a pragmatic implementation to improve robustness to noisy images.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, freq_conv_channels: int = 16):
        super(FrequencyChannelAttention, self).__init__()
        # Channel (SE-like) attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

        # Frequency-based spatial attention: small conv that processes FFT magnitudes
        # We'll map a reduced-frequency map to a single-channel spatial mask.
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, freq_conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(freq_conv_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # Channel attention (SE)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_channel = x * y

        # Frequency attention: compute FFT magnitude per-sample (sum over channels)
        # Use real FFT and magnitude; result shape [B, H, W] (complex -> magnitude)
        # We'll compute over spatial dims and collapse channels by energy.
        # Convert to float32 if needed
        xf = x_channel
        # compute rfft2 across spatial dims for each channel, take magnitude and sum over channels
        # xf: [B,C,H,W] -> fft: [B,C,Hf,Wf] complex
        try:
            fft = torch.fft.rfft2(xf, dim=(-2, -1))
            mag = torch.abs(fft)
            # sum over channels -> [B, Hf, Wf]
            mag = mag.sum(dim=1, keepdim=True)
        except Exception:
            # Fallback: use a simple spatial energy map (channel-wise mean abs)
            mag = xf.abs().mean(dim=1, keepdim=True)

        # Normalize magnitude to [0,1]
        mag_min = mag.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        mag_max = mag.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        denom = (mag_max - mag_min).clamp(min=1e-6)
        mag = (mag - mag_min) / denom

        # mag is [B,1,Hf,Wf] (Hf and Wf might be different due to rfft); resize to input spatial
        mag_spatial = F.interpolate(mag, size=(h, w), mode="bilinear", align_corners=False)
        # run small conv to produce spatial attention mask
        att_map = self.freq_conv(mag_spatial)

        out = x_channel * att_map
        return out + x * 0.1
