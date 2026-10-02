# Ultralytics YOLO, AGPL-3.0 license
"""Adaptive parallel attention module."""

import torch
import torch.nn as nn


class APAM(nn.Module):
    """Refine local features with residual channel and spatial attention."""

    def __init__(self, c1, c2=None, reduction=16, spatial_kernel=7, residual_scale=0.01):
        """Initialize an identity-friendly APAM block."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        if spatial_kernel not in {3, 7}:
            raise ValueError(f"spatial_kernel must be 3 or 7, but got {spatial_kernel}")

        hidden_channels = max(c2 // reduction, 1)
        self.shortcut = (
            nn.Identity()
            if c1 == c2
            else nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2))
        )
        self.local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c2, hidden_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, c2, 1, bias=False),
        )
        padding = spatial_kernel // 2
        self.spatial_attention = nn.Conv2d(2, 1, spatial_kernel, padding=padding, bias=False)
        self.scale = nn.Parameter(torch.full((1, c2, 1, 1), residual_scale))

    def forward(self, x):
        """Apply local refinement and parallel channel-spatial attention."""
        x = self.shortcut(x)
        features = self.local(x)

        avg_context = torch.mean(features, dim=(2, 3), keepdim=True)
        max_context = torch.amax(features, dim=(2, 3), keepdim=True)
        channel_weights = torch.sigmoid(self.channel_mlp(avg_context) + self.channel_mlp(max_context))

        spatial_context = torch.cat(
            (torch.mean(features, dim=1, keepdim=True), torch.amax(features, dim=1, keepdim=True)), dim=1
        )
        spatial_weights = torch.sigmoid(self.spatial_attention(spatial_context))

        return x + self.scale * features * channel_weights * spatial_weights
