"""
K-fold cross-validation for the two-tower TF-IDF model.
Splits by resume_id so no resume leaks between train and test.

Usage:
    python -m src.evaluation.kfold_eval \
        --pairs_csv data/processed/pairs_sbert_k5_hard50_easy50.csv \
        --vectorizer_path artifacts/tfidf_v1/tfidf_vectorizer.joblib \
        --k 5
"""
from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import recall_at_k, reciprocal_rank, aggregate_metrics
from src.modeling.twotower_tfidf import TwoTowerTfidf
from src.preprocessing.triplets import generate_triplets


# ── training helpers ──────────────────────────────────────────────────────────

class TripletCosineLoss(nn.Module):
    def __init__(self, margin: float = 0.4):
        super().__init__()
        self.margin = margin

    def forward(self, q, p, n):
        s_pos = (q * p).sum(dim=1)
        s_neg = (q * n).sum(dim=1)
        return torch.relu(self.margin - s_pos + s_neg).mean()


def _make_triplet_tensors(merged_df: pd.DataFrame, vectorizer, triplets_per_resume: int = 20):
    triplets_df, _ = generate_triplets(merged_df, triplets_per_resume=triplets_per_resume)
    Xq = torch.from_numpy(vectorizer.transform(triplets_df["resume_text"].tolist()).toarray()).float()
    Xp = torch.from_numpy(vectorizer.transform(triplets_df["pos_job_text"].tolist()).toarray()).float()
    Xn = torch.from_numpy(vectorizer.transform(triplets_df["neg_job_text"].tolist()).toarray()).float()
    return Xq, Xp, Xn


def _train(Xq, Xp, Xn, tfidf_dim: int, embed_dim: int, epochs: int, lr: float, batch_size: int, margin: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTowerTfidf(tfidf_dim=tfidf_dim, embed_dim=embed_dim, dropout=0.1).to(device)
    criterion = TripletCosineLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = Xq.size(0)
    for epoch in range(1, epochs + 1):
        idx = torch.randperm(n)
        Xq, Xp, Xn = Xq[idx], Xp[idx], Xn[idx]
        for i in range(0, n, batch_size):
            q = Xq[i:i+batch_size].to(device)
            p = Xp[i:i+batch_size].to(device)
            n_ = Xn[i:i+batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.encode_resume(q), model.encode_job(p), model.encode_job(n_))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    return model


# ── eval helper ───────────────────────────────────────────────────────────────

def _eval_fold(model, test_pairs: pd.DataFrame, resumes: pd.DataFrame, jobs: pd.DataFrame,
               vectorizer, ks=(5, 10)):
    resume_text = dict(zip(resumes.resume_id, resumes.resume_text))
    job_text    = dict(zip(jobs.job_id, jobs.job_text))

    df = test_pairs.copy()
    Xr = vectorizer.transform(df["resume_id"].map(resume_text).fillna(""))
    Xj = vectorizer.transform(df["job_id"].map(job_text).fillna(""))

    scores = []
    with torch.no_grad():
        for i in range(0, len(df), 256):
            r = torch.from_numpy(Xr[i:i+256].toarray()).float()
            j = torch.from_numpy(Xj[i:i+256].toarray()).float()
            scores.append((model.encode_resume(r) * model.encode_job(j)).sum(dim=1).numpy())
    df["score"] = np.concatenate(scores)

    per_query = []
    for _, g in df.groupby("resume_id"):
        labels = g["label"].to_numpy(np.int32)
        sc     = g["score"].to_numpy(np.float32)
        if labels.sum() == 0:
            continue
        m = {f"Recall@{k}": recall_at_k(labels, sc, k) for k in ks}
        m["MRR"] = reciprocal_rank(labels, sc)
        per_query.append(m)

    return aggregate_metrics(per_query), len(per_query)


# ── merge helper ──────────────────────────────────────────────────────────────

def _merge(pairs: pd.DataFrame, resumes: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    resume_cols = [c for c in ["resume_id", "resume_text", "category"] if c in resumes.columns]
    df = pairs.merge(resumes[resume_cols], on="resume_id", how="left")
    df = df.merge(jobs[["job_id", "job_text"]], on="job_id", how="left")
    return df.dropna(subset=["resume_text", "job_text"]).reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", default="data/processed/pairs_sbert_k5_hard50_easy50.csv")
    parser.add_argument("--vectorizer_path", default="artifacts/tfidf_v1/tfidf_vectorizer.joblib")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--triplets_per_resume", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs   = pd.read_csv(args.pairs_csv)
    resumes = pd.read_csv("data/processed/cleaned/resumes_clean.csv")
    jobs    = pd.read_csv("data/processed/cleaned/jobs_model.csv")
    vectorizer = joblib.load(args.vectorizer_path)
    tfidf_dim = vectorizer.transform([""]).shape[1]

    resume_ids = np.array(pairs["resume_id"].unique())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(resume_ids)
    folds = np.array_split(resume_ids, args.k)

    all_metrics = []
    for fold_i, test_ids in enumerate(folds, start=1):
        train_ids = np.concatenate([folds[j] for j in range(args.k) if j != fold_i - 1])
        train_pairs = pairs[pairs["resume_id"].isin(train_ids)]
        test_pairs  = pairs[pairs["resume_id"].isin(test_ids)]

        train_merged = _merge(train_pairs, resumes, jobs)
        Xq, Xp, Xn  = _make_triplet_tensors(train_merged, vectorizer, args.triplets_per_resume)

        print(f"\n── Fold {fold_i}/{args.k}  train={len(train_ids)} test={len(test_ids)} resumes ──")
        model = _train(Xq, Xp, Xn, tfidf_dim, args.embed_dim, args.epochs, args.lr, args.batch_size, args.margin)

        fold_metrics, n_queries = _eval_fold(model, test_pairs, resumes, jobs, vectorizer, ks=args.ks)
        all_metrics.append(fold_metrics)
        print(f"  queries={n_queries}  " + "  ".join(f"{k}={v:.4f}" for k, v in fold_metrics.items()))

    # average across folds
    keys = all_metrics[0].keys()
    print("\n── Cross-validated results ──")
    for key in keys:
        vals = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
