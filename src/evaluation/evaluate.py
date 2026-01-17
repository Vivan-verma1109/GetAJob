from __future__ import annotations

from typing import Dict, Sequence
from src.evaluation.dataset import ResumeJobEvalDataset
from src.evaluation.metrics import recall_at_k, reciprocal_rank, aggregate_metrics


def evaluate_dataset_from_pairs_scores(
    pairs_csv_path: str,
    ks: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """
    Evaluates ranking metrics using the existing scores in pairs CSV:
      - labels from 'label'
      - scores from 'tfidf_cosine'
    """
    eval_ds = ResumeJobEvalDataset(pairs_csv_path)

    per_query_metrics = []
    for g in eval_ds:
        labels = g.labels
        scores = g.scores

        num_pos = int(labels.sum())
        if num_pos == 0:
            continue

        m = {}
        for k in ks:
            m[f"Recall@{k}"] = recall_at_k(labels, scores, k)
        m["MRR"] = reciprocal_rank(labels, scores)

        per_query_metrics.append(m)

    return aggregate_metrics(per_query_metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", type=str, required=True, help="Path to pairs_dev.csv or pairs_test.csv")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    results = evaluate_dataset_from_pairs_scores(args.pairs_csv, ks=args.ks)

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
