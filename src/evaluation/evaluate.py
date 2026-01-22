from __future__ import annotations

from typing import Dict, Sequence, Optional
import argparse
import os

import numpy as np
import pandas as pd

from src.evaluation.metrics import recall_at_k, reciprocal_rank, aggregate_metrics


def _compute_metrics_grouped(df: pd.DataFrame, ks: Sequence[int]) -> Dict[str, float]:
    """
    df must contain: resume_id, label (0/1), and score (float)
    """
    per_query_metrics = []

    for resume_id, g in df.groupby("resume_id"):
        labels = g["label"].to_numpy(dtype=np.int32)
        scores = g["score"].to_numpy(dtype=np.float32)

        if labels.sum() == 0:
            continue

        m = {}
        for k in ks:
            m[f"Recall@{k}"] = recall_at_k(labels, scores, k)
        m["MRR"] = reciprocal_rank(labels, scores)
        per_query_metrics.append(m)

    return aggregate_metrics(per_query_metrics)


def evaluate_dataset_from_pairs_scores(pairs_csv_path: str, ks: Sequence[int] = (5, 10)) -> Dict[str, float]:
    """
    Baseline: uses existing columns in pairs CSV:
      - labels from 'label'
      - scores from 'tfidf_cosine'
    """
    df = pd.read_csv(pairs_csv_path)
    assert "resume_id" in df.columns, "pairs CSV must contain resume_id"
    assert "label" in df.columns, "pairs CSV must contain label"
    assert "tfidf_cosine" in df.columns, "pairs CSV must contain tfidf_cosine"

    df = df.copy()
    df["score"] = df["tfidf_cosine"].astype(float)

    return _compute_metrics_grouped(df, ks)


def evaluate_dataset_with_model(
    pairs_csv_path: str,
    model_ckpt: str,
    tfidf_dir: str,
    ks: Sequence[int] = (5, 10),
    resumes_csv: Optional[str] = None,
    jobs_csv: Optional[str] = None,
) -> Dict[str, float]:
    """
    Model eval: compute scores using learned towers.

    Requires pairs CSV to have either:
      (A) resume_text + job_text columns, OR
      (B) resume_id + job_id AND you pass resumes_csv/jobs_csv to join texts.

    Also requires TF-IDF vectorizer saved via joblib in tfidf_dir (e.g., artifacts/tfidf_v1).
    """
    import torch
    import joblib

    # --- load data ---
    df = pd.read_csv(pairs_csv_path)
    assert "resume_id" in df.columns, "pairs CSV must contain resume_id"
    assert "label" in df.columns, "pairs CSV must contain label"

    # Ensure we have resume_text/job_text
    if not {"resume_text", "job_text"}.issubset(df.columns):
        # need joins
        if resumes_csv is None or jobs_csv is None:
            raise ValueError(
                "pairs CSV missing resume_text/job_text. Provide --resumes_csv and --jobs_csv to join."
            )
        resumes = pd.read_csv(resumes_csv)[["resume_id", "resume_text"]]
        jobs = pd.read_csv(jobs_csv)[["job_id", "job_text"]]

        assert "job_id" in df.columns, "pairs CSV must contain job_id to join jobs_csv"
        df = df.merge(resumes, on="resume_id", how="left").merge(jobs, on="job_id", how="left")

    if df["resume_text"].isna().any() or df["job_text"].isna().any():
        raise ValueError("Found NaN resume_text/job_text after loading/joining. Check your CSVs/IDs.")

    # --- load TF-IDF vectorizer ---
    # common filenames; adjust if yours differs
    vec_path_candidates = [
        os.path.join(tfidf_dir, "vectorizer.joblib"),
        os.path.join(tfidf_dir, "tfidf_vectorizer.joblib"),
        os.path.join(tfidf_dir, "vectorizer.pkl"),
    ]
    vec_path = next((p for p in vec_path_candidates if os.path.exists(p)), None)
    if vec_path is None:
        raise FileNotFoundError(
            f"Could not find TF-IDF vectorizer in {tfidf_dir}. "
            f"Tried: {vec_path_candidates}"
        )
    vectorizer = joblib.load(vec_path)

    # --- vectorize ---
    # IMPORTANT: use same vectorizer used in training
    Xr = vectorizer.transform(df["resume_text"].tolist())  # sparse
    Xj = vectorizer.transform(df["job_text"].tolist())     # sparse

    # --- load model ---
    from src.modeling.twotower_tfidf import TwoTowerTfidf

    # infer tfidf_dim from vectorizer vocab size to avoid hardcoding
    tfidf_dim = len(vectorizer.vocabulary_)

    model = TwoTowerTfidf(tfidf_dim=tfidf_dim, embed_dim=128, dropout=0.0)
    ckpt = torch.load(model_ckpt, map_location="cpu")

    # If it's a checkpoint dict, pull the actual weights out
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    # Pull dims from checkpoint if present (safer than hardcoding)
    embed_dim = int(ckpt.get("embed_dim", 128)) if isinstance(ckpt, dict) else 128
    tfidf_dim_ckpt = int(ckpt.get("tfidf_dim", tfidf_dim)) if isinstance(ckpt, dict) else tfidf_dim

    model = TwoTowerTfidf(tfidf_dim=tfidf_dim_ckpt, embed_dim=embed_dim, dropout=0.0)
    model.load_state_dict(state_dict)
    model.eval()

    model.eval()

    # --- score in batches (sparse -> dense) ---
    @torch.no_grad()
    def score_batch(r_batch_sparse, j_batch_sparse) -> np.ndarray:
        # convert to dense float32 for Linear layer
        r = torch.from_numpy(r_batch_sparse.toarray()).float()
        j = torch.from_numpy(j_batch_sparse.toarray()).float()
        r_emb = model.resume(r)
        j_emb = model.job(j)
        # cosine if towers L2-normalize; dot product then equals cosine
        s = torch.sum(r_emb * j_emb, dim=1)
        return s.cpu().numpy()

    batch_size = 256
    scores = []
    n = df.shape[0]
    for i in range(0, n, batch_size):
        scores.append(score_batch(Xr[i:i+batch_size], Xj[i:i+batch_size]))
    scores = np.concatenate(scores, axis=0)

    df = df.copy()
    df["score"] = scores.astype(float)

    return _compute_metrics_grouped(df, ks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", type=str, required=True, help="Path to pairs_dev.csv or pairs_test.csv")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])

    # Optional: model-based eval
    parser.add_argument("--model_ckpt", type=str, default=None, help="Path to trained checkpoint .pt")
    parser.add_argument("--tfidf_dir", type=str, default="artifacts/tfidf_v1", help="Dir containing saved TF-IDF vectorizer (joblib)")

    # Only needed if pairs CSV lacks resume_text/job_text
    parser.add_argument("--resumes_csv", type=str, default=None, help="CSV with columns resume_id,resume_text")
    parser.add_argument("--jobs_csv", type=str, default=None, help="CSV with columns job_id,job_text")

    args = parser.parse_args()

    if args.model_ckpt is None:
        results = evaluate_dataset_from_pairs_scores(args.pairs_csv, ks=args.ks)
    else:
        results = evaluate_dataset_with_model(
            args.pairs_csv,
            model_ckpt=args.model_ckpt,
            tfidf_dir=args.tfidf_dir,
            ks=args.ks,
            resumes_csv=args.resumes_csv,
            jobs_csv=args.jobs_csv,
        )

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
