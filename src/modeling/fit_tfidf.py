from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_triplets_csv", required=True)
    ap.add_argument("--out_dir", default="artifacts/tfidf_v1")
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--ngram_max", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_triplets_csv)

    corpus = (
        df["resume_text"].astype(str).tolist()
        + df["pos_job_text"].astype(str).tolist()
        + df["neg_job_text"].astype(str).tolist()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        norm="l2",
    )
    X = vectorizer.fit_transform(corpus)

    dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")

    meta = {
        "tfidf_dim": int(X.shape[1]),
        "max_features": args.max_features,
        "min_df": args.min_df,
        "ngram_range": [1, args.ngram_max],
        "stop_words": "english",
        "num_fit_docs": int(X.shape[0]),
    }
    (out_dir / "tfidf_meta.json").write_text(json.dumps(meta, indent=2))

    print("Saved:")
    print(f"  {out_dir / 'tfidf_vectorizer.joblib'}")
    print(f"  {out_dir / 'tfidf_meta.json'}")
    print(f"TF-IDF dim = {meta['tfidf_dim']}")


if __name__ == "__main__":
    main()
