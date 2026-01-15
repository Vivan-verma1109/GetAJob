from pathlib import Path
import numpy as np
import pandas as pd

PAIRS_PATH = Path("data/processed/pairs_tfidf_k5_n20.csv")

OUT_DIR = Path("data/processed/splits")
TRAIN_PATH = OUT_DIR / "pairs_train.csv"
DEV_PATH   = OUT_DIR / "pairs_dev.csv"
TEST_PATH  = OUT_DIR / "pairs_test.csv"

# 70/15/15 split
TRAIN_FRAC = 0.70
DEV_FRAC   = 0.15
TEST_FRAC  = 0.15

SEED = 42

def main():
    pairs = pd.read_csv(PAIRS_PATH)

    # split by resume_id
    resume_ids = pairs["resume_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(resume_ids)

    n = len(resume_ids)
    n_train = int(n * TRAIN_FRAC)
    n_dev   = int(n * DEV_FRAC)
    n_test  = n - n_train - n_dev

    train_ids = set(resume_ids[:n_train])
    dev_ids   = set(resume_ids[n_train:n_train + n_dev])
    test_ids  = set(resume_ids[n_train + n_dev:])

    train_df = pairs[pairs["resume_id"].isin(train_ids)].reset_index(drop=True)
    dev_df   = pairs[pairs["resume_id"].isin(dev_ids)].reset_index(drop=True)
    test_df  = pairs[pairs["resume_id"].isin(test_ids)].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    dev_df.to_csv(DEV_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Saved:", TRAIN_PATH)
    print("Saved:", DEV_PATH)
    print("Saved:", TEST_PATH)

if __name__ == "__main__":
    main()
