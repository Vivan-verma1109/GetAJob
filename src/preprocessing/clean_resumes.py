from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/Resume.csv")
OUT_PATH = Path("data/processed/resumes_clean.csv")

TARGET_CATEGORY = "INFORMATION-TECHNOLOGY"

def clean_resumes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[["ID", "Resume_str", "Category"]].rename(columns={
        "ID": "resume_id",
        "Resume_str": "resume_text",
        "Category": "category",
    })

    df["resume_id"] = pd.to_numeric(df["resume_id"], errors="coerce")
    df["resume_text"] = (
        df["resume_text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"] == TARGET_CATEGORY]

    df = df.dropna(subset=["resume_id", "resume_text"])
    df = df[df["resume_text"].str.len() >= 100]

    df = df.drop_duplicates(subset=["resume_text"]).reset_index(drop=True)

    return df

def main():
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_resumes(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Raw shape:", df.shape)
    print("Clean shape:", cleaned.shape)
    print("Category counts:")
    print(cleaned["category"].value_counts())

if __name__ == "__main__":
    main()
