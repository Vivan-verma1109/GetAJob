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
        self.proj = nn.Linear(tfidf_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, tfidf_dim)
        x = self.drop(x)
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=1) # unit vectors
        return z


class TwoTowerTfidf(nn.Module):
    def __init__(self, tfidf_dim: int, embed_dim: int = 128, dropout: float = 0.1):
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
