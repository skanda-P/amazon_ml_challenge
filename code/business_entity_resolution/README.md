# Business Entity Resolution — Metric-Aware Probabilistic Entity Resolution

A precision-oriented, F0.5-optimized pipeline that matches Source-1 reference
businesses to noisy Source-2/Source-3 vendor records, without any external
data lookups.

Pipeline:

```
raw TSVs -> validation/parsing -> multi-view normalization
   -> multi-pass blocking (exact keys + char n-grams + TF-IDF + address structure)
   -> RRF candidate fusion -> pairwise feature engineering (+ IDF-weighted
      rarity features + Fellegi-Sunter LLR + cross-source triangulation)
   -> LightGBM pair classifier (GroupKFold OOF, hard-negative mining)
   -> isotonic probability calibration
   -> entity-level existence/singleton model
   -> metric-aware top-k F0.5 set decoder (+ confidence abstention)
   -> ownership/conflict arbitration
   -> matching_results.tsv + candidate_pairs.tsv
```

See `Documentation_template.md` (one level up, in the submission zip root)
for the full methodology write-up. This README only covers *how to run
the code*.

## 1. Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. All models are MIT/Apache-2.0 licensed
(LightGBM, scikit-learn) and well under the 8B-parameter limit — the
"model" here is a gradient-boosted tree ensemble of a few hundred KB,
not a neural network.

## 2. Directory layout expected by the code

```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Point `--data-dir` at the directory that *contains* `train/` and `test/`
(e.g. `dataset`).

## 3. Reproducing end-to-end: raw data -> blocking -> matching -> outputs

```bash
cd src

# 1) Train (runs blocking + feature engineering + LightGBM + calibration
#    + existence model + threshold tuning, all via GroupKFold OOF so there
#    is no entity leakage). Artifacts are written to --model-dir.
python3 main.py train --data-dir /path/to/dataset --model-dir ../models

# 2) Run inference on the test set. Writes BOTH required output files.
python3 main.py infer --data-dir /path/to/dataset --model-dir ../models \
    --output-dir ../../../output

# 3) Validate before submitting (stdlib only, no dependencies).
python3 ../scripts/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir /path/to/dataset/test
```

A `PASS` from step 3 means the files are safe to upload to the leaderboard.

## 4. Trying it immediately with generated sample data

The real challenge dataset is provided by the organizers and is **not**
included in this package. To let anyone verify the pipeline runs
end-to-end without that data, a synthetic generator is included that
mimics the problem statement's noise patterns (legal-suffix swaps,
abbreviations, typos, token reordering, missing address components,
landmark references, and a test-only unseen country):

```bash
cd src
python3 main.py generate-sample-data --out-dir ../../../dataset_sample \
    --n-train 500 --n-test 250 --seed 7

python3 main.py train --data-dir ../../../dataset_sample --model-dir ../models_sample
python3 main.py infer --data-dir ../../../dataset_sample --model-dir ../models_sample \
    --output-dir ../../../output_sample
python3 ../scripts/validate_submission.py \
    --matching ../../../output_sample/matching_results.tsv \
    --candidate ../../../output_sample/candidate_pairs.tsv \
    --test-dir ../../../dataset_sample/test
```

This is purely a smoke test / demo aid — it is not used anywhere in the
real pipeline logic, and the generator writes a `_hidden_test_ground_truth_
FOR_DEMO_ONLY.tsv` file used only for local sanity-checking (never read by
`train.py` / `infer.py`), exactly mirroring the "hold out your own
validation split" guidance in the problem statement.

To point the *real* pipeline at the *real* dataset, just pass
`--data-dir` at the actual `dataset/` directory containing the
organizer-provided files — nothing else changes.

## 5. Ablation flags

Every architectural component beyond the base pairwise classifier can be
toggled independently, both at train and inference time (pass the same
flags to both commands so artifacts and decoding stay consistent):

| Flag | Disables |
|---|---|
| `--no-llr` | Fellegi-Sunter LLR feature |
| `--no-triangulation` | cross-source triangulation feature |
| `--no-hard-negative-mining` | hard-negative reweighting/retraining |
| `--no-calibration` | isotonic probability calibration (uses raw LightGBM scores) |
| `--no-existence-model` | singleton/existence model (falls back to an independence-approximation proxy) |
| `--no-metric-decoder` | the F0.5-maximizing top-k decoder (falls back to a naive `p > 0.5` threshold per pair) |

This lets you reproduce the experiment-by-experiment ablation table
described in the methodology document by running train+infer once per
configuration and scoring each with `evaluation/evaluate.py` against a
held-out fold or your own validation split.

## 6. Design notes / scalability

* **Blocking first, always.** No stage ever attempts the full
  `|S1| x (|S2|+|S3|)` cross join. Every retrieval channel (exact keys,
  character n-gram inverted index, TF-IDF cosine, address-structure keys)
  is built once per split and queried per Source-1 record; RRF fusion
  caps the candidate set at `final_top_k` (default 25) per entity.
* **`candidate_pairs.tsv` is exactly what gets scored.** `infer.py`
  writes it straight from the fused candidate set *before* any feature
  computation or model inference — the same object the classifier scores
  — so it can never drift from what the model actually saw, and the
  validator's "matched IDs must be a subset of candidate IDs" check is a
  structural guarantee, not a coincidence.
* **Country is a generic string feature everywhere.** `normalize_country`
  never maps against a fixed allow-list; blocking keys, similarity
  features, and the classifier all treat country as an open categorical
  value, which is why the pipeline handles the test-only "France" records
  without any special-casing (see the sample-data country-sliced scoring
  in the methodology document).
* **No external lookups anywhere** — no geocoding APIs, no business
  registries, no web calls. All normalization dictionaries (legal
  suffixes, address abbreviations) are small, hard-coded linguistic aids
  bundled in `src/normalization/normalize.py`, not calls to an external
  service.
* **Scaling beyond the demo size.** For million-record inputs, the
  TF-IDF retrieval channel in `blocking.py` is the part most worth
  hardening further: replace the dense `matrix @ query_vec.T` per-query
  call with batched sparse matrix multiplication (query several S1
  records at once) or an approximate nearest-neighbor index (e.g.
  HNSW/FAISS-style, still purely local/offline — no external service),
  and shard the S2/S3 target pool by a coarse key (e.g. first
  postal-code digit, or country) before building per-shard indices so no
  single TF-IDF matrix has to hold the entire target pool in memory at
  once. The exact-key and address-structure channels (dict-based inverted
  indices) already scale linearly and need no changes.

## 7. Repository layout

```
src/
  data/ingest.py            Stage A: TSV loading + schema/prefix validation
  normalization/normalize.py Stage B: multi-view name/address normalization
  blocking/blocking.py       Stage D: multi-pass candidate retrieval + RRF fusion
  features/features.py       Stage E/F: pairwise + rarity-aware (IDF) features
  linkage/fellegi_sunter.py  Stage G: Fellegi-Sunter LLR
  linkage/triangulation.py   Stage P: cross-source triangulation evidence
  calibration/calibrate.py   Stage L: isotonic probability calibration
  decoding/decoder.py        Stage M/N/O: existence model + F0.5 set decoder + abstention
  decoding/ownership.py      Stage Q: conflict/ownership arbitration
  evaluation/evaluate.py     macro F0.5, candidate recall, reduction ratio, pair P/R
  pipeline/{config,common,train,infer}.py   orchestration
  data_synth/generate_sample_data.py        synthetic demo-data generator (NOT the real dataset)
  main.py                    CLI entrypoint
scripts/validate_submission.py   stdlib-only output validator
requirements.txt
```

## 8. Fair-play compliance

The pipeline never calls out to the network. Every similarity computation
(Levenshtein, Jaro-Winkler, Jaccard, TF-IDF cosine, character n-grams) is
computed locally from the fields the organizers provide, and every
normalization dictionary is a small bundled linguistic aid, not an
external database. The only trained artifacts are a LightGBM classifier,
an isotonic calibrator, and a logistic-regression existence model — all
open-source, MIT/Apache-2.0-licensed, and orders of magnitude under the
8B-parameter limit.
