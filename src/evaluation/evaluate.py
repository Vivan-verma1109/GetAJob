from __future__ import annotations

from typing import Dict, Sequence
import argparse
import os

import numpy as np
import pandas as pd
import torch
import joblib

from src.evaluation.metrics import recall_at_k, reciprocal_rank, aggregate_metrics
from src.modeling.twotower_tfidf import TwoTowerTfidf


def _compute_metrics_grouped(df: pd.DataFrame, ks: Sequence[int]) -> Dict[str, float]:
    per_query = []

    for _, g in df.groupby("resume_id"):
        labels = g["label"].to_numpy(np.int32)
        scores = g["score"].to_numpy(np.float32)

        if labels.sum() == 0:
            continue

        m = {f"Recall@{k}": recall_at_k(labels, scores, k) for k in ks}
        m["MRR"] = reciprocal_rank(labels, scores)
        per_query.append(m)

    return aggregate_metrics(per_query)


def evaluate_tfidf_baseline(pairs_csv: str, ks=(5, 10)) -> Dict[str, float]:
    df = pd.read_csv(pairs_csv)
    df["score"] = df["tfidf_cosine"].astype(float)
    return _compute_metrics_grouped(df, ks)


def evaluate_with_model(
    pairs_csv: str,
    model_ckpt: str,
    tfidf_dir: str,
    ks=(5, 10),
) -> Dict[str, float]:
    df = pd.read_csv(pairs_csv)

    # load resumes / jobs text
    resumes = pd.read_csv("data/processed/cleaned/resumes_clean.csv")
    jobs = pd.read_csv("data/processed/cleaned/jobs_model.csv")

    resume_text = dict(zip(resumes.resume_id, resumes.resume_text))
    job_text = dict(zip(jobs.job_id, jobs.job_text))

    # load vectorizer
    vectorizer = joblib.load(os.path.join(tfidf_dir, "tfidf_vectorizer.joblib"))

    Xr = vectorizer.transform(df["resume_id"].map(resume_text))
    Xj = vectorizer.transform(df["job_id"].map(job_text))


    # load model
    ckpt = torch.load(model_ckpt, map_location="cpu")
    model = TwoTowerTfidf(
        tfidf_dim=ckpt["tfidf_dim"],
        embed_dim=ckpt["embed_dim"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    @torch.no_grad()
    def score_batch(r, j):
        r = torch.from_numpy(r.toarray()).float()
        j = torch.from_numpy(j.toarray()).float()
        return torch.sum(model.resume(r) * model.job(j), dim=1).numpy()

    scores = []
    for i in range(0, len(df), 256):
        scores.append(score_batch(Xr[i:i+256], Xj[i:i+256]))

    df["score"] = np.concatenate(scores)
    return _compute_metrics_grouped(df, ks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument(
    "--scorer",
    choices=["tfidf", "model"],
    default="model",
    help="How to compute scores"
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--model_ckpt", default=None)
    parser.add_argument("--tfidf_dir", default="artifacts/tfidf_v1")
    args = parser.parse_args()

    if args.scorer == "model":
        if not args.model_ckpt:
            raise SystemExit("--model_ckpt is required when --scorer=model")

        results = evaluate_with_model(
            args.pairs_csv,
            args.model_ckpt,
            args.tfidf_dir,
            ks=args.ks,
        )

    else: 
        results = evaluate_tfidf_baseline(args.pairs_csv, ks=args.ks)


    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
