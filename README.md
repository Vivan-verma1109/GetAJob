# Resume–Job Matching with Two-Tower TF-IDF Embeddings

This project builds an end-to-end **resume–job matching system** that ranks job postings for a given resume using a learned embedding model. The system uses **SBERT-derived weak supervision** for labels and **TF-IDF** for input features, keeping the two signals independent to produce a meaningful evaluation.

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

---

## Pipeline

### Step 1 — Label Generation (SBERT)

SBERT (`all-MiniLM-L6-v2`) runs once offline as a ground truth oracle:

- Encodes every resume and job into dense 384-dim semantic vectors
- Computes cosine similarity between each resume and every job
- Top 5 jobs per resume → **label = 1** (positive)
- Sample from the bottom of the similarity distribution → **label = 0** (negative)

The float SBERT scores are discarded after ranking — only the binary labels survive into training data. SBERT never touches the model at training or inference time.

This step defines **what a good match looks like** using semantic understanding. The model's job is to recover that signal using a completely different representation.

### Step 2 — Text Vectorization (TF-IDF)

TF-IDF converts raw resume and job text into weighted numerical vectors:

- **TF (term frequency):** how often a word appears in this document
- **IDF (inverse document frequency):** how rare that word is across all documents

The result is a 50k-dim sparse vector where meaningful words like "Python" or "Kubernetes" are amplified and common noise like "the" or "worked" is dampened. TF-IDF is not scoring matches — it is removing noise and converting text to numbers the model can ingest. This gives the model a cleaner signal to work with so it doesn't waste capacity on basic filtering.

### Step 3 — Pair & Triplet Construction

Each resume is paired with:
- **5 positive jobs** (SBERT top-5)
- **100 negative jobs** (sampled from below the 50th similarity percentile, excluding the top 200)

This gives **105 candidates per resume** at evaluation time.

Pairs are split by `resume_id` into **train / dev / test** to avoid leakage, then converted to triplets:

```
(resume, positive_job, negative_job)
```

### Step 4 — Two-Tower Model

The model takes TF-IDF vectors and maps them into a **shared 128-dim meaning space**:

```
TF-IDF vector (50k) → Linear projection → 128-dim embedding → L2 normalize
```

Separate towers for resumes and jobs, same structure, separate learned weights.

The key insight: raw TF-IDF cosine similarity measures word overlap. The towers learn to map both resumes and jobs into a space where **proximity means semantic match quality**, not vocabulary overlap. "Python developer" and "software engineer, Python required" share few exact words but should be close — the towers learn that mapping from the training signal.

**Scoring:** cosine similarity between the resume and job embeddings. Since both are L2-normalized, this is a dot product. The same math as TF-IDF cosine similarity, but in a learned space where similar coordinates mean similar concepts rather than similar word counts.

### Step 5 — Training

- **Loss:** Triplet margin loss — push the positive job embedding closer to the resume than the negative job embedding
- **Optimizer:** AdamW
- **Checkpointing:** saved every epoch

The model never sees SBERT scores. It only sees the consequence of SBERT's judgment: which job is positive, which is negative. It learns whatever TF-IDF patterns tend to co-occur with SBERT-defined relevance.

---

## Why Decouple Labels and Features?

Using TF-IDF for both labels and inputs creates a circular evaluation — the model learns to approximate the labeling function using the same representation, so perfect metrics just mean it memorized the signal, not that it learned anything meaningful.

By using SBERT for labels and TF-IDF for inputs:

```
Ground truth:  SBERT semantic similarity  (what good looks like)
Model input:   TF-IDF keyword statistics  (what the model gets to work with)
```

The model must bridge the gap between keyword overlap and semantic relevance. That is a real learning problem — which is why MRR is 0.87 instead of 1.0, and why 0.87 is the honest number.

---

## Evaluation

Evaluation is **resume-centric**:
- Each resume is ranked against its 105 candidate jobs using the model's cosine similarity scores
- Metrics measure how well the model's ranking recovers SBERT's labeled positives
- Computed per resume and averaged across the test set

### Metrics

- **MRR (Mean Reciprocal Rank):** average of 1/rank of the first positive job
- **Recall@5:** fraction of positive jobs appearing in the top 5
- **Recall@10:** fraction of positive jobs appearing in the top 10

---

## Results

### Hard Evaluation (105 candidates per resume)

Held-out Test Set (18 resumes × 105 jobs):

| Metric    | Value     |
|-----------|-----------|
| MRR       | **0.873** |
| Recall@5  | **0.622** |
| Recall@10 | **0.800** |

The model — using only TF-IDF keyword features — agrees with SBERT's semantic judgment for the top position ~87% of the time. Over 80% of semantically relevant jobs appear in the top 10 out of 105 candidates.

---

## Why This Matters

This project demonstrates:
- weak supervision at scale without manual labeling
- decoupled label generation and feature representation to avoid circular evaluation
- retrieval-style evaluation (not classification accuracy)
- clean train/dev/test separation by resume to prevent leakage
- the two-tower architecture used in production retrieval systems at scale

The two-tower pattern is industry-standard for large-scale matching — jobs, ads, recommendations, search. Separate encoders allow job embeddings to be precomputed offline, so serving is just a dot product against a pre-built index. This project implements that pattern correctly; the current ceiling is the weakness of TF-IDF as an encoder, not the architecture.

---

## Future Work

- Replace TF-IDF inputs with SBERT embeddings for a fully semantic two-tower model
- Hard negative mining during training
- Larger candidate pools (500–1000 jobs)
- Human-annotated relevance labels
- Resume feedback and missing-skill explanations
