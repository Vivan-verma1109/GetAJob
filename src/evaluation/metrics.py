from __future__ import annotations
from typing import Iterable, Dict
import numpy as np


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Recall@k for one resume
    labels: shape (M,), values 0/1
    scores: shape (M,), higher = more relevant from the tf-idf
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    number_positive = int(labels.sum())
    if number_positive == 0:
        return 0.0

    k = min(k, len(labels))
    topk_idx = np.argsort(-scores)[:k]
    hits = int(labels[topk_idx].sum())
    return hits / number_positive


def reciprocal_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Reciprocal Rank for one resume
    MRR is mean over resumes.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    ranked_idx = np.argsort(-scores)
    ranked_labels = labels[ranked_idx]
    pos = np.where(ranked_labels == 1)[0]
    if pos.size == 0:
        return 0.0

    first_rank_1based = int(pos[0]) + 1
    return 1.0 / first_rank_1based


def aggregate_metrics(per_query: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """
    Average metrics across resumes.
    """
    per_query = list(per_query)
    if not per_query:
        return {"num_queries": 0}

    keys = sorted({k for d in per_query for k in d.keys()})
    out = {"num_queries": len(per_query)}
    for k in keys:
        out[k] = float(np.mean([d.get(k, 0.0) for d in per_query]))
    return out
