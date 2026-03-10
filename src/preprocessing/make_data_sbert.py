import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RESUME_PATH = "data/processed/cleaned/resumes_clean.csv"
JOBS_PATH = "data/processed/cleaned/jobs_model.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

pos = 5
neg = 100
HARD_NEG = 50   # hard negatives per resume (from TF-IDF ranking)
EASY_NEG = 50   # easy negatives per resume (random from bottom half)

EXCLUDE_TOP_FOR_NEG = 200
LOW_SIM_PERCENTILE = 50

resumes = pd.read_csv(RESUME_PATH)
jobs = pd.read_csv(JOBS_PATH)

resumes["text"] = resumes["resume_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
jobs["text"] = jobs["job_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

resume_ids = resumes["resume_id"].tolist()
job_ids = jobs["job_id"].tolist()

model = SentenceTransformer(MODEL_NAME)

print("Encoding resumes...")
res_embs = model.encode(
    resumes["text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)

print("Encoding jobs...")
job_embs = model.encode(
    jobs["text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)

# sims_matrix[i, j] = cosine similarity between resume i and job j
# embeddings are already L2-normalized so dot product == cosine similarity
sims_matrix = res_embs @ job_embs.T  # shape: (n_resumes, n_jobs)

# TF-IDF similarity matrix for hard negative mining
print("Building TF-IDF similarity matrix...")
tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True)
all_texts = resumes["text"].tolist() + jobs["text"].tolist()
tfidf.fit(all_texts)

res_tfidf = normalize(tfidf.transform(resumes["text"].tolist()))
job_tfidf = normalize(tfidf.transform(jobs["text"].tolist()))
tfidf_matrix = (res_tfidf @ job_tfidf.T).toarray()  # shape: (n_resumes, n_jobs)

pairs = []

for i, rid in enumerate(resume_ids):
    sims = sims_matrix[i]       # SBERT similarities for this resume
    tfidf_sims = tfidf_matrix[i]  # TF-IDF similarities for this resume
    order = np.argsort(-sims)

    # positives: top-K by SBERT similarity
    pos_idx = set(order[:pos].tolist())
    for rank, j in enumerate(list(pos_idx), start=1):
        pairs.append({
            "resume_id": rid,
            "job_id": job_ids[j],
            "label": 1,
            "rank": rank,
            "sbert_cosine": float(sims[j]),
        })

    # hard negatives: top of TF-IDF ranking, excluding SBERT positives
    tfidf_order = np.argsort(-tfidf_sims)
    hard_pool = [j for j in tfidf_order if j not in pos_idx][:HARD_NEG * 3]
    rng = np.random.default_rng(42 + i)
    hard_size = min(HARD_NEG, len(hard_pool))
    hard_idx = rng.choice(hard_pool, size=hard_size, replace=False)

    for j in hard_idx:
        pairs.append({
            "resume_id": rid,
            "job_id": job_ids[j],
            "label": 0,
            "rank": None,
            "sbert_cosine": float(sims[j]),
        })

    # easy negatives: random from bottom half of SBERT similarity
    exclude = min(EXCLUDE_TOP_FOR_NEG, len(order) - 1)
    candidate = order[exclude:]
    cutoff = np.percentile(sims, LOW_SIM_PERCENTILE)
    low_pool = candidate[sims[candidate] <= cutoff]
    easy_pool = low_pool if len(low_pool) >= EASY_NEG else candidate
    replace = len(easy_pool) < EASY_NEG
    easy_idx = rng.choice(easy_pool, size=EASY_NEG, replace=replace)

    for j in easy_idx:
        pairs.append({
            "resume_id": rid,
            "job_id": job_ids[j],
            "label": 0,
            "rank": None,
            "sbert_cosine": float(sims[j]),
        })

pairs_df = pd.DataFrame(pairs)

check = pairs_df.groupby(["resume_id", "label"]).size().unstack(fill_value=0)
print(check.head())
print(f"Any resume missing {pos} positives?", (check.get(1, 0) != pos).any())
print(f"Any resume missing {neg} negatives?", (check.get(0, 0) != neg).any())

OUT_PATH = f"data/processed/pairs_sbert_k{pos}_hard{HARD_NEG}_easy{EASY_NEG}.csv"
pairs_df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "rows:", len(pairs_df))
