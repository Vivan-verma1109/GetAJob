from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TfidfTower(nn.Module):
    """
    Maps TF-IDF vectors -> L2-normalized embedding.
    """

    def __init__(self, tfidf_dim: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(tfidf_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)


class TwoTowerTfidf(nn.Module):
    def __init__(self, tfidf_dim: int, embed_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.resume = TfidfTower(tfidf_dim, embed_dim, dropout)
        self.job = TfidfTower(tfidf_dim, embed_dim, dropout)

    def encode_resume(self, r_tfidf: torch.Tensor) -> torch.Tensor:
        return self.resume(r_tfidf)

    def encode_job(self, j_tfidf: torch.Tensor) -> torch.Tensor:
        return self.job(j_tfidf)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # dot prod is cosine
        return (a * b).sum(dim=1)
