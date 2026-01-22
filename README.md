# Resume–Job Matching with Two-Tower TF-IDF Embeddings

This project builds an end-to-end **resume–job matching system** that ranks job postings for a given resume using a learned embedding model. The system is trained under **weak supervision** derived from TF-IDF similarity and evaluated using standard **information-retrieval metrics**.

The goal is to demonstrate:
- clean data pipelines
- proper train/dev/test evaluation
- ranking-based metrics (Recall@K, MRR)
- a production-style two-tower model architecture

---

## Problem Setup

Given:
- a resume (free-form text)
- a pool of job descriptions

The system outputs:
- a ranked list of jobs
- similarity scores between the resume and each job

This mirrors real-world candidate–job retrieval systems used in recruiting platforms.

---

## Dataset

- **Resumes:** Cleaned U.S.-based resumes (`resumes_clean.csv`)
- **Jobs:** Cleaned job postings with descriptions (`jobs_model.csv`)
- **Representation:** TF-IDF (unigrams + bigrams)

### Weak Supervision
For each resume:
- Top **K = 5** jobs by TF-IDF cosine similarity → **positive**
- Remaining jobs → candidate pool for **negative sampling**

This avoids manual labeling while providing a consistent ranking signal.

---

## Pair & Triplet Construction

### Pair Generation
Each resume is paired with:
- **5 positive jobs**
- **N negative jobs**

Two evaluation regimes are used:
- **Easy:** N = 20 → 25 candidates per resume
- **Hard:** N = 100 → 105 candidates per resume

Pairs are split by `resume_id` into **train / dev / test** to avoid leakage.

### Triplets (for Training)
Training data is constructed as triplets:

