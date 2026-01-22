import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RESUME_PATH = "data/processed/cleaned/resumes_clean.csv"
JOBS_PATH = "data/processed/cleaned/jobs_model.csv"

pos = 5
neg = 100

EXCLUDE_TOP_FOR_NEG = 200
LOW_SIM_PERCENTILE = 50  

resumes = pd.read_csv(RESUME_PATH)
jobs = pd.read_csv(JOBS_PATH)

resumes["text"] = resumes["resume_text"].fillna("").astype(str)  
jobs["text"] = jobs["job_text"].fillna("").astype(str)

# whitespace
resumes["text"] = resumes["text"].str.replace(r"\s+", " ", regex=True).str.strip()
jobs["text"] = jobs["text"].str.replace(r"\s+", " ", regex=True).str.strip()

resume_ids = resumes["resume_id"].tolist()
job_ids = jobs["job_id"].tolist()

corpus = pd.concat([resumes["text"], jobs["text"]], ignore_index=True)

vectorizer = TfidfVectorizer(
    stop_words="english",  
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\+\#\.]{1,}\b"  
)

X = vectorizer.fit_transform(corpus) 

X_res = X[:len(resumes)]
X_job = X[len(resumes):]

X_res = normalize(X_res) 
X_job = normalize(X_job)

pairs = []

for i, rid in enumerate(resume_ids):
    sims = (X_res[i] @ X_job.T).toarray().ravel()

    order = np.argsort(-sims)  

    # positives: top-K
    pos_idx = order[:pos]
    for rank, j in enumerate(pos_idx, start=1):
        pairs.append({
            "resume_id": rid,
            "job_id": job_ids[j],
            "label": 1,
            "rank": rank,
            "tfidf_cosine": float(sims[j]),
        })

    exclude = min(EXCLUDE_TOP_FOR_NEG, len(order) - 1)
    candidate = order[exclude:]

    cutoff = np.percentile(sims, LOW_SIM_PERCENTILE)
    low_pool = candidate[sims[candidate] <= cutoff]
    pool = low_pool if len(low_pool) >= neg else candidate

    rng = np.random.default_rng(42 + i) 

    replace = len(pool) < neg
    neg_idx = rng.choice(pool, size=neg, replace=replace)

    for j in neg_idx:
        pairs.append({
            "resume_id": rid,
            "job_id": job_ids[j],
            "label": 0,
            "rank": None,
            "tfidf_cosine": float(sims[j]),
        })

pairs_df = pd.DataFrame(pairs)

# Sanity check
check = pairs_df.groupby(["resume_id", "label"]).size().unstack(fill_value=0)
print(check.head())
print(f"Any resume missing {pos} positives?", (check.get(1, 0) != pos).any())
print(f"Any resume missing {neg} negatives?", (check.get(0, 0) != neg).any())

# Save
OUT_PATH = f"data/processed/pairs_tfidf_k{pos}_n{neg}.csv"
pairs_df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "rows:", len(pairs_df))
