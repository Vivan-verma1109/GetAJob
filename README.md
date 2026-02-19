# Resume–Job Matching with Two-Tower TF-IDF Embeddings

This project builds an end-to-end **resume–job matching system** that ranks job postings for a given resume using a learned embedding model. The system is trained under **weak supervision** derived from SBERT semantic similarity and evaluated using standard **information-retrieval metrics**.

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
- **Model input representation:** TF-IDF (unigrams + bigrams, 50k features)
- **Label generation:** SBERT cosine similarity (`all-MiniLM-L6-v2`)

### Weak Supervision

For each resume:
- Top **K = 5** jobs by **SBERT cosine similarity** → **positive**
- Remaining jobs → candidate pool for **negative sampling**

Using SBERT for label generation and TF-IDF for model inputs means the two signals are independent — the model cannot trivially memorize the labeling function. This produces a more honest evaluation.

---

## Pair & Triplet Construction

### Pair Generation

Each resume is paired with:
- **5 positive jobs**
- **100 negative jobs** (sampled from below the 50th similarity percentile, excluding the top 200)

This gives **105 candidates per resume** at evaluation time.

Pairs are split by `resume_id` into **train / dev / test** to avoid leakage.

### Triplets (for Training)

Training data is constructed as triplets:

```
(resume, positive_job, negative_job)
```

and optimized using **triplet loss**.

---

## Model Architecture

### Two-Tower Network

- Separate towers for resumes and jobs
- Shared structure, separate weights

```
TF-IDF (50k dims) → Linear projection → 128-dim embedding → L2 normalization
```

### Scoring

- Similarity = cosine similarity between resume and job embeddings
- Since embeddings are L2-normalized, cosine = dot product

---

## Training

- **Loss:** Triplet margin loss (margin = 0.2)
- **Optimizer:** AdamW
- **Labels:** SBERT-derived weak supervision
- **Negative sampling:** Low-similarity pool (bottom 50th percentile, excluding top 200)
- **Checkpointing:** Model saved every epoch

---

## Evaluation

Evaluation is **resume-centric**:
- Each resume is ranked against its fixed candidate set (105 jobs)
- Metrics are computed per resume and averaged across the test set

### Metrics

- **MRR (Mean Reciprocal Rank)**
- **Recall@5**
- **Recall@10**

---

## Results

### Hard Evaluation (105 candidates per resume)

Held-out Test Set (18 resumes × 105 jobs):

| Metric    | Value     |
|-----------|-----------|
| MRR       | **0.873** |
| Recall@5  | **0.622** |
| Recall@10 | **0.800** |

### Interpretation

- The model ranks a relevant job in the **top position** for ~87% of queries on average
- Over **80%** of relevant jobs appear in the **top 10** out of 105 candidates
- Labels are derived from SBERT semantic similarity; model inputs are TF-IDF keyword features — the two representations are independent, making the evaluation meaningful

---

## Why This Matters

This project demonstrates:
- weak supervision at scale (no manual labeling)
- decoupled label generation and feature representation to avoid circular evaluation
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
- Replace TF-IDF inputs with SBERT embeddings for a fully semantic two-tower model
