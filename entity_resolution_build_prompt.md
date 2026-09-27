# Build Prompt: Business Entity Resolution Pipeline (F₀.₅-Optimized)

## Objective

Build an end-to-end entity resolution pipeline that matches Source 2 and Source 3
business records to Source 1 reference entities, across noisy multilingual name/address
data spanning US, India, and (test-set-only, unseen-at-train-time) France. Optimize
specifically for **F₀.₅** (precision weighted 2× over recall), which means the pipeline
should be conservative by design — prefer predicting "no match" over a wrong match.

All models used must be **MIT or Apache 2.0 licensed and ≤8B parameters**. No external
API calls, databases, or geocoding services are permitted at any stage — only the
provided training/test data and locally-run open models/libraries.

---

## Architecture Overview

```
Source 1/2/3 records
        │
        ▼
┌───────────────────────┐
│ Stage 1: Blocking      │  → candidate_pairs.tsv
│ (multi-strategy union) │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 2: Feature       │
│ Engineering            │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 3: Base Models   │
│ (GBT + cross-encoder + │
│  embedding similarity) │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 4: Stacked       │
│ Meta-Learner +         │
│ Calibration            │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 5: F₀.₅-Tuned    │
│ Thresholding +         │
│ Margin-Based Abstention│
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 6: Post-         │
│ Processing & Output    │
│ Validation             │  → matching_results.tsv
└───────────────────────┘
```

---

## Stage 1: Blocking (Candidate Generation)

Use a **union of two complementary strategies** to maximize recall ceiling without
inflating the candidate set too much:

1. **Sorted-neighborhood / q-gram LSH blocking** on normalized business names
   (strip legal suffixes, lowercase, remove punctuation) and address tokens
   (street/city). Cheap, catches near-exact string overlaps.
2. **Multilingual embedding kNN blocking** — fine-tune `intfloat/multilingual-e5-large`
   or `BAAI/bge-m3` (both MIT, <1B params) via contrastive/triplet loss on labeled
   training pairs. Encode `business_name + business_address` per record, index with
   FAISS, retrieve top-k nearest neighbors per Source 1 entity from Source 2/3.

Take the union of both candidate sets. This embedding pass is the key generalization
lever for **France** in the test set, since it doesn't rely on country-specific string
patterns learned only from US/India training data.

Write this final candidate set (post any filtering) to `candidate_pairs.tsv`.

---

## Stage 2: Feature Engineering

Compute per-pair features across these groups:

**Name features**
- Legal-suffix-normalized exact match flag, plus similarity delta (pre- vs.
  post-normalization)
- Jaro-Winkler, Levenshtein ratio, token-sort ratio, token-set ratio
- Character n-gram (2,3-gram) TF-IDF cosine similarity
- Phonetic match flags (Soundex / Double Metaphone)
- Word-order-sensitivity delta (Levenshtein vs. token-sort ratio gap)

**Address features**
- Token Jaccard similarity, shared-token count normalized by length
- Digit-exact match flags: street number, PIN/postal code
- Missingness indicators: has PIN, has state, has landmark reference
- Component-level parsing (street/city/state/postal) via a **local** parser only
  (no external geocoding APIs) — treat parsing as a feature-extraction utility

**Cross-record / graph features**
- Embedding cosine similarity from Stage 1's fine-tuned encoder
- Margin feature: gap between this pair's score and the next-best competing
  candidate for the same Source 1 entity
- Node degree in the candidate graph (how many candidates does this S1/S2/S3
  record have overall — high-degree records are more prone to false merges)

**Categorical**
- Country: frequency-encoded with an explicit "unseen" bucket to handle France
  gracefully — never one-hot restricted to {US, India}

---

## Stage 3: Base Models

Train three complementary base models on the candidate pairs:

1. **LightGBM / CatBoost classifier** on all engineered features above. Fast,
   data-efficient, interpretable — the workhorse for surface-level noise.
2. **Fine-tuned cross-encoder** — `microsoft/deberta-v3-base` (MIT, ~184M params)
   over serialized pairs, e.g.:
   `[COL name] Acme Pvt Ltd [COL addr] 12 MG Road Pune [SEP] [COL name] ACME
   PRIVATE LIMITED [COL addr] MG Rd, Pune, near SBI ATM`
   Captures joint name+address interactions that hand-engineered features miss.
3. **Embedding cosine similarity** from the Stage 1 fine-tuned e5/bge-m3 model,
   used both as a blocking signal and as a standalone base-model score.

Optional fourth signal for the ambiguous middle band only (scores in a
mid-confidence range, e.g. 0.3–0.7): a few-shot **Qwen2.5-7B-Instruct** or
**Mistral-7B-Instruct-v0.3** (both Apache 2.0, ≤8B) verifier prompt to adjudicate
hard cases with reasoning. Restrict to this band to control inference cost.
(Note: Llama-family models are excluded — Meta's license is not MIT/Apache 2.0.)

---

## Stage 4: Stacked Meta-Learner + Calibration

- Generate out-of-fold predictions from each base model (K-fold, no leakage).
- Train a logistic regression (or shallow GBT) meta-learner on:
  base model scores + margin feature + graph-degree feature + country bucket.
- Calibrate final probabilities with isotonic or Platt scaling so the threshold
  in Stage 5 is meaningful and stable across country subgroups.

---

## Stage 5: F₀.₅-Tuned Thresholding

- On a held-out validation split (carved from training data, scored with the
  F₀.₅ formula), sweep the decision threshold and select the value that
  **maximizes macro-averaged F₀.₅** — this will generally sit above 0.5, since
  precision is weighted 2× over recall.
- Apply a **margin-based abstention rule**: if the top candidate's calibrated
  score minus the second-best candidate's score falls below a tuned gap, prefer
  predicting no match (singleton) rather than committing. This directly targets
  the metric's reward for correctly identifying singletons (full 1.0 credit)
  and its penalty for false merges.
- Re-validate the threshold across country subgroups if data allows — apply the
  same "unseen bucket" logic to France's threshold behavior (likely default to
  the global/cross-country threshold rather than a specialist one).

---

## Stage 6: Post-Processing & Output Validation

- Deduplicate IDs within each match list; enforce Source 2/3-only references
  (never Source 1 self-matches).
- Ensure **every** Source 1 test entity has exactly one output row, empty list
  when no match survives.
- Verify every ID in `matching_results.tsv` also appears in `candidate_pairs.tsv`
  (a mismatch signals a pipeline bug).
- Run `utils/validate_submission.py` locally before every leaderboard upload.

---

## Fair Play & Licensing Checklist

- [ ] No external API/database/geocoding calls anywhere in the pipeline
- [ ] All models MIT or Apache 2.0 licensed
- [ ] All models ≤8B parameters (e5/bge-m3 ~0.5–0.6B, DeBERTa-v3-base ~0.18B,
      Qwen2.5-7B-Instruct or Mistral-7B ~7B if used)
- [ ] `requirements.txt` pins exact versions for reproducibility
- [ ] `README.md` documents the full data → blocking → matching → output flow
- [ ] Methodology document covers blocking strategy, model architecture, feature
      engineering, and ensemble/calibration design

---

## Design Rationale Summary

- **Blocking recall ceiling matters most** — union of string-based and
  embedding-based blocking maximizes recall while the embedding pass is what
  lets the pipeline generalize to France without any France-specific training data.
- **Precision-first calibration** — because F₀.₅ penalizes false merges 2× more
  than misses, every design choice (margin abstention, conservative threshold,
  singleton-aware scoring) leans toward "don't guess unless confident."
- **Stacking over single-model reliance** — GBT (surface noise), cross-encoder
  (joint field interactions), and embeddings (semantic/transliteration
  robustness) each catch different failure modes; combining them outperforms
  any one in isolation, particularly on the unseen-country generalization test.
