import pandas as pd
import random

def generate_triplets(df, triplets_per_resume=20, seed=42):
    random.seed(seed)
    triplets = []
    skipped = 0

    for resume_id, g in df.groupby("resume_id"):
        resume_text = g["resume_text"].iloc[0]

        pos_jobs = g[g["label"] == 1]["job_text"].tolist()
        neg_jobs = g[g["label"] == 0]["job_text"].tolist()

        if len(pos_jobs) == 0 or len(neg_jobs) == 0:
            skipped += 1
            continue

        for _ in range(triplets_per_resume):
            triplets.append({
                "resume_text": resume_text,
                "pos_job_text": random.choice(pos_jobs),
                "neg_job_text": random.choice(neg_jobs),
            })

    triplets_df = pd.DataFrame(triplets)
    triplets_df = triplets_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return triplets_df, skipped


def make_split(split_name, triplets_per_resume=20, seed=42):
    in_path = f"data/processed/merged/{split_name}.csv"
    out_path = f"data/processed/merged/{split_name}_triplets.csv"

    df = pd.read_csv(in_path)
    triplets_df, skipped = generate_triplets(df, triplets_per_resume=triplets_per_resume, seed=seed)
    triplets_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_split("train", triplets_per_resume=20, seed=42)
    make_split("dev",   triplets_per_resume=20, seed=42)
