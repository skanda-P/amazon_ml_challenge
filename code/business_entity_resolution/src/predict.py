"""
predict.py
----------
Generate matching_results.tsv and candidate_pairs.tsv for the test set.

Usage
-----
    python -m src.predict [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.blocking import generate_candidates
from src.features import build_feature_matrix
from src.io_utils import load_source, write_candidate_pairs, write_matching_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_CFG = {
    "data": {
        "test_dir": "../../data/dataset/test",
    },
    "blocking": {
        "top_k_name": 10,
        "top_k_addr": 10,
        "name_min_score": 0.25,
        "addr_min_score": 0.20,
    },
    "output": {
        "model_path":     "models/matcher.pkl",
        "threshold_path": "models/threshold.txt",
        "matching_path":  "../../output/matching_results.tsv",
        "candidate_path": "../../output/candidate_pairs.tsv",
    },
}


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path and Path(path).exists():
        with open(path) as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            if k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def predict(cfg: dict) -> None:
    test_dir = Path(cfg["data"]["test_dir"])

    # ── Load test data ───────────────────────────────────────────────────────
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)

    # ── Blocking ─────────────────────────────────────────────────────────────
    b_cfg = cfg["blocking"]
    candidates = generate_candidates(s1, s2, s3, **b_cfg)

    # ── Write candidate_pairs.tsv ────────────────────────────────────────────
    write_candidate_pairs(candidates, cfg["output"]["candidate_path"])

    # ── Feature engineering ──────────────────────────────────────────────────
    pairs_df, X = build_feature_matrix(s1, pool, candidates)

    # ── Load model + threshold ────────────────────────────────────────────────
    model_path = Path(cfg["output"]["model_path"])
    threshold_path = Path(cfg["output"]["threshold_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run src.train first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5
    logger.info(f"Loaded model from {model_path}, threshold = {threshold:.4f}")

    # ── Score candidates ──────────────────────────────────────────────────────
    predictions: dict[str, list[str]] = {eid: [] for eid in s1["entity_id"]}

    if not pairs_df.empty:
        proba = model.predict_proba(X)[:, 1]
        pairs_df = pairs_df.assign(_proba=proba)

        for s1_id, grp in pairs_df.groupby("source1_entity_id"):
            matched = list(grp.loc[grp["_proba"] >= threshold, "candidate_entity_id"])
            predictions[s1_id] = matched

    # ── Write matching_results.tsv ────────────────────────────────────────────
    write_matching_results(predictions, cfg["output"]["matching_path"])

    n_matched = sum(1 for v in predictions.values() if v)
    logger.info(
        f"Done. {n_matched}/{len(predictions)} S1 entities have ≥1 match. "
        f"Output → {cfg['output']['matching_path']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity-resolution predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    predict(cfg)
