from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterator
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class EvalGroup:
    resume_id: int
    job_ids: List[int]
    labels: np.ndarray
    scores: np.ndarray         


class ResumeJobEvalDataset:
    """
    Groups pairs_{dev,test}.csv into one item per resume_id.
    """

    def __init__(self, pairs_csv_path: str):
        df = pd.read_csv(pairs_csv_path)

        required = {"resume_id", "job_id", "label", "tfidf_cosine"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {pairs_csv_path}: {missing}")

        df["label"] = df["label"].astype(int)
        df["tfidf_cosine"] = df["tfidf_cosine"].astype(float)

        self._groups: List[EvalGroup] = []
        for rid, g in df.groupby("resume_id", sort=False):
            job_ids = g["job_id"].astype(int).tolist()
            labels = g["label"].to_numpy(dtype=int)
            scores = g["tfidf_cosine"].to_numpy(dtype=float)

            self._groups.append(
                EvalGroup(
                    resume_id=int(rid),
                    job_ids=job_ids,
                    labels=labels,
                    scores=scores,
                )
            )

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self) -> Iterator[EvalGroup]:
        return iter(self._groups)

    def __getitem__(self, idx: int) -> EvalGroup:
        return self._groups[idx]
