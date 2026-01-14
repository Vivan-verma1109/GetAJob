from pathlib import Path
import ast
import pandas as pd

RAW_PATH = Path("data/raw/postings2.csv")
OUT_PATH = Path("data/processed/jobs_model.csv")

def clean_skills(x) -> str:
    try:
        items = ast.literal_eval(x) if isinstance(x, str) else []
        return ", ".join(s.strip() for s in items if s and str(s).strip())
    except Exception:
        return ""

def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={
        "job_title": "title",
        "company": "company",
        "job_location": "location",
        "job_summary": "description",
        "job_skills": "skills",
        "date_posted": "posted",
        "first_seen": "first_seen",
        "job_type": "job_type",
        "search_city": "search_city",
        "search_country": "search_country",
        "job_link": "job_link",
    })

    if "job_id" not in df.columns:
        df.insert(0, "job_id", range(1, len(df) + 1))

    keep = [
        "job_id", "title", "company", "location", "description", "skills",
        "posted", "first_seen", "job_type", "search_city", "search_country", "job_link"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    df["job_id"] = pd.to_numeric(df["job_id"], errors="coerce")

    text_cols = [c for c in ["title", "company", "location", "description", "skills"] if c in df.columns]
    for c in text_cols:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    meta_cols = [c for c in ["posted", "first_seen", "job_type", "search_city", "search_country", "job_link"] if c in df.columns]
    for c in meta_cols:
        df[c] = df[c].astype(str).str.strip()

    required = [c for c in ["job_id", "title", "description"] if c in df.columns]
    df = df.dropna(subset=required)

    df = df[df["description"].str.len() >= 200]

    df["skills_str"] = df["skills"].apply(clean_skills)

    df["job_text"] = (
        "TITLE: " + df["title"] + " "
        "COMPANY: " + df["company"] + " "
        "LOCATION: " + df["location"] + " "
        "SUMMARY: " + df["description"] + " "
        "SKILLS: " + df["skills_str"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df.drop_duplicates(subset=[c for c in ["title", "company", "location", "description"] if c in df.columns])
    df = df.reset_index(drop=True)

    out_cols = ["job_id", "title", "company", "location", "description", "skills_str", "posted", "job_text"]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols]

def main():
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_jobs(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Raw shape:", df.shape)
    print("Clean shape:", cleaned.shape)
    print("Null counts:\n", cleaned.isna().sum())

if __name__ == "__main__":
    main()
