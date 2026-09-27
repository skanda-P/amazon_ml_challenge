# Business Entity Resolution — Code README

## Reproducing Results End-to-End

### 1. Environment setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Directory layout expected

```
amazon_ml_challenge/
├── data/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv
│   │   │   ├── train_source2.tsv
│   │   │   ├── train_source3.tsv
│   │   │   └── train_ground_truth.tsv
│   │   └── test/
│   │       ├── test_source1.tsv
│   │       ├── test_source2.tsv
│   │       └── test_source3.tsv
│   └── utils/
│       └── validate_submission.py
└── code/
    └── business_entity_resolution/   ← you are here
        ├── src/
        ├── configs/config.yaml
        ├── models/                   ← created by train step
        ├── scripts/run_pipeline.py
        └── requirements.txt
```

Output files go to `../../output/` (i.e. `amazon_ml_challenge/output/`).

### 3. Run the full pipeline

```bash
# From code/business_entity_resolution/
python scripts/run_pipeline.py
```

This runs **train → predict → validate** in sequence.

### 4. Run steps individually

```bash
# Training only
python -m src.train --config configs/config.yaml

# Prediction only (requires trained model in models/)
python -m src.predict --config configs/config.yaml

# Validate output format
python ../../data/utils/validate_submission.py \
    --matching  ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir  ../../data/dataset/test
```

### 5. Configuration

Edit `configs/config.yaml` to tune:
- Blocking thresholds and top-k
- Model type (`xgboost` or `lightgbm`) and hyper-parameters
- Validation split size and random seed
- Decision threshold search range

### 6. Source layout

| File | Purpose |
|---|---|
| `src/preprocess.py` | Text normalisation (names, addresses) |
| `src/blocking.py` | Candidate generation (TF-IDF blocking) |
| `src/features.py` | Feature engineering per candidate pair |
| `src/train.py` | Training + threshold tuning |
| `src/predict.py` | Inference → output TSVs |
| `src/evaluate.py` | F_0.5 metric computation |
| `src/io_utils.py` | TSV loading and writing |
| `scripts/run_pipeline.py` | End-to-end runner |
| `configs/config.yaml` | All hyper-parameters |
