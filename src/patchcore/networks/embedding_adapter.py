"""Residual adapter for aggregated PatchCore embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualEmbeddingAdapter(nn.Module):
    def __init__(self, dimension: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(1, dimension // 4)
        self.dimension = dimension
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features):
        return F.normalize(features + self.scale * self.network(features), dim=-1)
