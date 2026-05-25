from __future__ import annotations

import torch
from torch import nn


def mean_pool(embeddings: list[torch.Tensor]) -> torch.Tensor:
    pooled = [tensor.mean(dim=0) for tensor in embeddings]
    return torch.stack(pooled, dim=0)


class MeanPoolMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        pooled = mean_pool(embeddings)
        return self.mlp(pooled)
