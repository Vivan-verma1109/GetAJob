# src/preprocessing/merge_split.py

from pathlib import Path
import pandas as pd

RESUMES_PATH = Path("data/processed/resumes_clean.csv")
JOBS_PATH    = Path("data/processed/jobs_model.csv")
SPLITS_DIR   = Path("data/processed/splits")

TRAIN_PAIRS = SPLITS_DIR / "pairs_train.csv"
DEV_PAIRS   = SPLITS_DIR / "pairs_dev.csv"
TEST_PAIRS  = SPLITS_DIR / "pairs_test.csv"

OUT_DIR = Path("data/processed/merged")
OUT_TRAIN = OUT_DIR / "train.csv"
OUT_DEV   = OUT_DIR / "dev.csv"
OUT_TEST  = OUT_DIR / "test.csv"


def merge_one(pairs: pd.DataFrame, resumes: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    merged = pairs.merge(
        resumes[["resume_id", "resume_text", "category"]],
        on="resume_id",
        how="left"
    )
    merged = merged.merge(
        jobs[
            [
                "job_id",
                "job_text",
                "title",
                "company",
                "location",
                "description",
                "skills_str",
                "posted",
            ]
        ],
        on="job_id",
        how="left"
    )
    merged = merged.dropna(subset=["resume_text", "job_text"]).reset_index(drop=True)
    
    merged = merged[
        [
            "resume_id",
            "job_id",
            "label",
            "rank",
            "tfidf_cosine",
            "category",
            "title",
            "company",
            "location",
            "posted",
            "skills_str",
            "resume_text",
            "job_text",
        ]
    ]

    return merged


def main():
    resumes = pd.read_csv(RESUMES_PATH)
    jobs = pd.read_csv(JOBS_PATH)

    train_pairs = pd.read_csv(TRAIN_PAIRS)
    dev_pairs   = pd.read_csv(DEV_PAIRS)
    test_pairs  = pd.read_csv(TEST_PAIRS)
    
    resumes.columns = resumes.columns.str.strip()
    jobs.columns    = jobs.columns.str.strip()
    train_pairs.columns = train_pairs.columns.str.strip()
    dev_pairs.columns   = dev_pairs.columns.str.strip()
    test_pairs.columns  = test_pairs.columns.str.strip()

    train = merge_one(train_pairs, resumes, jobs)
    dev   = merge_one(dev_pairs, resumes, jobs)
    test  = merge_one(test_pairs, resumes, jobs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_TRAIN, index=False)
    dev.to_csv(OUT_DEV, index=False)
    test.to_csv(OUT_TEST, index=False)

    # prints
    print("Saved:", OUT_TRAIN, "rows:", len(train))
    print("Saved:", OUT_DEV, "rows:", len(dev))
    print("Saved:", OUT_TEST, "rows:", len(test))

    print("Pair rows:", len(train_pairs), len(dev_pairs), len(test_pairs))
    print("Merged rows:", len(train), len(dev), len(test))


if __name__ == "__main__":
    main()
