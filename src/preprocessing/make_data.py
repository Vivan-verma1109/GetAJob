import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RESUME_PATH = "data/processed/resumes_clean.csv"
JOBS_PATH = "data/processed/jobs_model.csv" 
pos = 5
neg = 20

EXCLUDE_TOP_FOR_NEG = 200 # don't sample negatives from top-N ranked jobs
LOW_SIM_PERCENTILE = 50 # to get the bottom half

resumes = pd.read_csv(RESUME_PATH)
jobs = pd.read_csv(JOBS_PATH)

resumes["text"] = resumes["resume_text"].fillna("").astype(str) # so this gets the text from the resume file and fills in na with ""

jobs["text"] = jobs["job_text"].fillna("").astype(str)

# whitespace
resumes["text"] = resumes["text"].str.replace(r"\s+", " ", regex=True).str.strip()
jobs["text"] = jobs["text"].str.replace(r"\s+", " ", regex=True).str.strip()

resume_ids = resumes["resume_id"].tolist()
job_ids = jobs["job_id"].tolist()

corpus = pd.concat([resumes["text"], jobs["text"]], ignore_index=True) # Take all resume text and all job text, and put them into one big list of documents.

vectorizer = TfidfVectorizer(
    stop_words="english", # ignore stuff such as: like, or, the, as, removes noise yk?
    ngram_range=(1, 2),
    min_df=2, # Ignore words that appear in only one document
    max_df=0.9, # Ignore words that appear in >90% of documents, these are probably not important since its everywhere like: worked
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\+\#\.]{1,}\b" # removes weird dupe thing like red hat followed by red and hat in seperate lines
)

X = vectorizer.fit_transform(corpus) # convert to vector because cosine sim

X_res = X[:len(resumes)]
X_job = X[len(resumes):]

# X_res[i] = TF-IDF vector for resume i
# X_job[j] = TF-IDF vector for job j

X_res = normalize(X_res) # magnitude of these vectors are now 1, that way cosine(max) = 1
X_job = normalize(X_job)

pairs = []

for i, rid in enumerate(resume_ids):
    sims = (X_res[i] @ X_job.T).toarray().ravel() 
    # Takes resume i
    # Computes cosine similarity vs every job
    
    order = np.argsort(-sims) # sort reverse so order[0] is best, order[-1] worst
    
    pos_idx = order[:pos] # top 5 jobs
    for rank, j in enumerate(pos_idx, start=1):
        pairs.append({
            "resume_id": rid,
            "job_id": job_ids[j],
            "label": 1,
            "rank": rank,
            "tfidf_cosine": float(sims[j]),
        })
        
    candidate = order[EXCLUDE_TOP_FOR_NEG:]

    cutoff = np.percentile(sims, LOW_SIM_PERCENTILE)
    low_pool = candidate[sims[candidate] <= cutoff]
    pool = low_pool if len(low_pool) >= neg else candidate

    rng = np.random.default_rng(42 + i)  # deterministic
    neg_idx = rng.choice(pool, size=neg, replace=False)

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
print("Any resume missing 5 positives?", (check.get(1, 0) != pos).any())
print("Any resume missing 20 negatives?", (check.get(0, 0) != neg).any())

# Save
OUT_PATH = "data/processed/pairs_tfidf_k5_n20.csv"
pairs_df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "rows:", len(pairs_df))
