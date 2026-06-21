from __future__ import annotations

import torch
from torch import nn


class CovCNNRegressor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        num_targets: int = 1,
        conv_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] | None = None,
        conv_strides: list[int] | None = None,
        conv_paddings: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        conv_channels = conv_channels or [32, 64, 128, 256]
        conv_kernel_sizes = conv_kernel_sizes or [5, 5, 3, 3]
        conv_strides = conv_strides or [2, 2, 2, 2]
        conv_paddings = conv_paddings or [2, 2, 1, 1]

        lengths = {
            len(conv_channels),
            len(conv_kernel_sizes),
            len(conv_strides),
            len(conv_paddings),
        }
        if len(lengths) != 1:
            raise ValueError(
                "conv_channels, conv_kernel_sizes, conv_strides, and conv_paddings must have the same length"
            )

        encoder_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel_size, stride, padding in zip(
            conv_channels,
            conv_kernel_sizes,
            conv_strides,
            conv_paddings,
            strict=True,
        ):
            encoder_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels

        encoder_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.encoder = nn.Sequential(*encoder_layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1], latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_targets)
        )

    def forward(self, cov):
        # cov: [B, 768, 768]
        cov = cov.unsqueeze(1)
        h = self.encoder(cov)
        output = self.head(h)
        return output