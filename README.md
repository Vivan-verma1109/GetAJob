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
- a resume
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

(resume, positive_job, negative_job)

and optimized using **triplet loss**.

---

## Model Architecture

### Two-Tower Network

- Separate towers for resumes and jobs
- Shared structure, separate weights

TF-IDF (50k dims) -> Linear projection -> 128-dim embedding -> L2 normalization

### Scoring

- Similarity = cosine similarity between resume and job embeddings
- Since embeddings are normalized, cosine = dot product

---

## Training

- **Loss:** Triplet loss
- **Optimizer:** Adam
- **Negative sampling:** Weakly supervised (TF-IDF-based)
- **Checkpointing:** Model saved every epoch

---

## Evaluation

Evaluation is **resume-centric**:
- Each resume is ranked against a fixed set of candidate jobs
- Metrics are computed per resume and averaged

### Metrics

- **MRR (Mean Reciprocal Rank)**
- **Recall@5**
- **Recall@10**

---

## Results

### Hard Evaluation (105 candidates per resume)

Held-out Test Set (18 resumes × 105 jobs):

| Metric     | Value |
|------------|-------|
| MRR        | **1.000** |
| Recall@5   | **0.811** |
| Recall@10  | **0.867** |

### Interpretation

- The model almost always ranks a relevant job at **position 1**
- Over **80%** of relevant jobs appear in the top-5 even with 100+ candidates
- Performance degrades gracefully as candidate pool size increases

This confirms the model is not overfitting to small candidate sets and generalizes within distribution.

## Why This Matters

This project demonstrates:
- weak supervision at scale
- retrieval-style evaluation (not just accuracy)
- clean separation of training vs evaluation data
- a deployable similarity model architecture

It reflects how **real production matching systems** are built, evaluated, and iterated on.

---

## Future Work

- Hard negative mining during training
- Larger candidate pools (500–1000 jobs)
- Human-annotated relevance evaluation
- Resume feedback and missing-skill explanations
- Replace TF-IDF with learned text encoders (e.g., transformers)
