from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from joblib import load


@dataclass(frozen=True)
class TfidfTriplet:
    r: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor


class TfidfTripletDataset(Dataset):
    """
    Loads triplets_train.csv and converts into fitted vectorizer.
    """

    def __init__(self, triplets_csv: str, vectorizer_path: str):
        self.df = pd.read_csv(triplets_csv)

        required = {"resume_text", "pos_job_text", "neg_job_text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {triplets_csv}: {missing}")

        self.df["resume_text"] = self.df["resume_text"].astype(str)
        self.df["pos_job_text"] = self.df["pos_job_text"].astype(str)
        self.df["neg_job_text"] = self.df["neg_job_text"].astype(str)

        self.vectorizer = load(vectorizer_path)

        dim = len(self.vectorizer.get_feature_names_out())
        self.tfidf_dim = int(dim)

    def __len__(self) -> int:
        return len(self.df)

    def _tfidf_dense(self, text: str) -> torch.Tensor:
        # vectorizer returns a sparse row
        x = self.vectorizer.transform([text])
        # convert to dense float32 
        dense = x.toarray().astype("float32")[0]
        return torch.from_numpy(dense)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        r = self._tfidf_dense(row["resume_text"])
        p = self._tfidf_dense(row["pos_job_text"])
        n = self._tfidf_dense(row["neg_job_text"])
        return r, p, n

