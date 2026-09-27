"""
predict.py  (v2)
----------------
Inference pipeline for the test set.

Loads trained artefacts, runs blocking + feature engineering,
applies meta-learner + threshold, writes both output TSVs.

Usage
-----
    cd code/business_entity_resolution
    python -m src.predict [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.blocking import generate_candidates
from src.features import FEATURE_NAMES, add_margin_feature, build_feature_matrix
from src.io_utils import load_source, write_candidate_pairs, write_matching_results
from src.models import load_artefact, meta_predict, safe_predict_proba
from src.threshold import apply_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_CFG: dict = {
    "data": {
        "test_dir": "../../data/dataset/test",
    },
    "blocking": {
        "top_k_name":       10,
        "top_k_addr":       10,
        "top_k_embed":      15,
        "name_min_score":   0.25,
        "addr_min_score":   0.20,
        "embed_min_score":  0.70,
        "embed_model":      "intfloat/multilingual-e5-large",
        "use_embeddings":   False,
    },
    "output": {
        "model_dir":      "models",
        "matching_path":  "../../output/matching_results.tsv",
        "candidate_path": "../../output/candidate_pairs.tsv",
    },
}


def load_config(path: str | None) -> dict:
    cfg = {k: dict(v) for k, v in DEFAULT_CFG.items()}
    if path and Path(path).exists():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def predict(cfg: dict) -> None:
    test_dir  = Path(cfg["data"]["test_dir"])
    model_dir = cfg["output"]["model_dir"]

    # ── Load test data ────────────────────────────────────────────────────
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)
    s1_ids = list(s1["entity_id"])

    # ── Blocking ──────────────────────────────────────────────────────────
    b_cfg = cfg["blocking"]
    candidates = generate_candidates(s1, s2, s3, **b_cfg)

    # Write candidate_pairs.tsv
    write_candidate_pairs(candidates, cfg["output"]["candidate_path"])

    # ── Feature engineering ───────────────────────────────────────────────
    pairs_df, X = build_feature_matrix(s1, pool, candidates)

    if pairs_df.empty:
        logger.warning("No candidate pairs found — all entities will be singletons.")
        predictions = {sid: [] for sid in s1_ids}
        write_matching_results(predictions, cfg["output"]["matching_path"])
        return

    # ── Load artefacts ────────────────────────────────────────────────────
    lgbm_final = load_artefact(model_dir, "lgbm_final")
    xgb_final  = load_artefact(model_dir, "xgb_final")
    meta       = load_artefact(model_dir, "meta")
    scaler     = load_artefact(model_dir, "scaler")

    threshold_file = Path(model_dir) / "threshold.txt"
    t_str = threshold_file.read_text().strip().split(",")
    threshold  = float(t_str[0])
    margin_gap = float(t_str[1]) if len(t_str) > 1 else 0.0
    logger.info(f"Threshold={threshold:.4f}  MarginGap={margin_gap:.4f}")

    # ── Score ─────────────────────────────────────────────────────────────
    lgbm_p = safe_predict_proba(lgbm_final, X)[:, 1]
    xgb_p  = safe_predict_proba(xgb_final, X)[:, 1]

    embed_idx = FEATURE_NAMES.index("embed_cosine")
    deg_idx   = FEATURE_NAMES.index("candidate_degree_log")
    cf_idx    = FEATURE_NAMES.index("country_freq_s1")

    embed_p = X[:, embed_idx]
    margin  = add_margin_feature(pairs_df, lgbm_p)

    final_proba = meta_predict(
        meta, scaler,
        [lgbm_p, xgb_p, embed_p],
        margin,
        X[:, deg_idx],
        X[:, cf_idx],
    )

    # ── Threshold + abstention ────────────────────────────────────────────
    predictions = apply_threshold(pairs_df, final_proba, threshold, margin_gap, s1_ids)

    # ── Write matching_results.tsv ────────────────────────────────────────
    write_matching_results(predictions, cfg["output"]["matching_path"])

    n_matched = sum(1 for v in predictions.values() if v)
    logger.info(
        f"Done: {n_matched}/{len(predictions)} S1 entities matched. "
        f"→ {cfg['output']['matching_path']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    predict(cfg)
