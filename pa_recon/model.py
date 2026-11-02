"""Residual U-Net with a mean reconstruction and per-pixel log variance."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = lambda channels: next(g for g in range(min(8, channels), 0, -1) if channels % g == 0)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups(out_channels), out_channels),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip(x)


class SparseViewRecon(nn.Module):
    """Residual U-Net mapping a 2-D measurement representation to an image."""

    def __init__(self, in_channels: int = 1, width: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.pool = nn.AvgPool2d(2)
        self.middle = ConvBlock(width * 4, width * 8)
        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.dec3 = ConvBlock(width * 8, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, in_channels * 2, 1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        original_size = x.shape[-2:]
        pad_h, pad_w = (-original_size[0]) % 8, (-original_size[1]) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.middle(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(z), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        if output_size is not None:
            if tuple(output_size) != tuple(d1.shape[-2:]):
                d1 = F.interpolate(d1, size=output_size, mode="bilinear", align_corners=False)
        elif pad_h or pad_w:
            d1 = d1[..., :original_size[0], :original_size[1]]
        mean, log_variance = self.head(d1).chunk(2, dim=1)
        return mean, log_variance.clamp(-8.0, 4.0)


def gaussian_l1_loss(mean: torch.Tensor, log_variance: torch.Tensor,
                     target: torch.Tensor, l1_weight: float = 0.1) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL plus a small L1 term."""
    nll = 0.5 * (torch.exp(-log_variance) * (mean - target).square() + log_variance)
    return nll.mean() + l1_weight * (mean - target).abs().mean()
