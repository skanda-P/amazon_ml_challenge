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
import time
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
    t_predict_start = time.time()
    test_dir  = Path(cfg["data"]["test_dir"])
    model_dir = cfg["output"]["model_dir"]

    # ── Pre-flight check: ensure models are trained ──────────────────────
    required_artefacts = ["lgbm_final.pkl", "xgb_final.pkl", "meta.pkl", "scaler.pkl", "threshold.txt"]
    missing = [f for f in required_artefacts if not (Path(model_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\n\n[ERROR] Missing required model artefacts in '{model_dir}': {missing}.\n"
            f"Please run the training pipeline first:\n"
            f"    python -m src.train --config configs/config.yaml\n"
        )

    # ── 1. Load test data ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 1/5] Ingesting Test Datasets (S1, S2, S3)")
    logger.info("=" * 65)
    t0 = time.time()
    s1 = load_source(test_dir / "test_source1.tsv")
    s2 = load_source(test_dir / "test_source2.tsv")
    s3 = load_source(test_dir / "test_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)
    s1_ids = list(s1["entity_id"])
    logger.info(
        f"✓ Test datasets loaded in {time.time() - t0:.1f}s:\n"
        f"    Query Entities (S1):  {len(s1):,}\n"
        f"    Candidate Pool (S2+S3): {len(pool):,} entities"
    )

    # ── 2. Blocking ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 2/5] Candidate Blocking Engine & candidate_pairs.tsv")
    logger.info("=" * 65)
    t0 = time.time()
    b_cfg = cfg["blocking"]
    candidates = generate_candidates(s1, s2, s3, **b_cfg)

    # Write candidate_pairs.tsv
    cand_path = cfg["output"]["candidate_path"]
    write_candidate_pairs(candidates, cand_path)
    logger.info(f"✓ candidate_pairs.tsv written → {cand_path} in {time.time() - t0:.1f}s")

    # ── 3. Feature engineering ────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 3/5] Extracting Feature Matrix for Test Pairs")
    logger.info("=" * 65)
    t0 = time.time()
    pairs_df, X = build_feature_matrix(s1, pool, candidates)

    if pairs_df.empty:
        logger.warning("No candidate pairs generated — all test entities default to singletons.")
        predictions = {sid: [] for sid in s1_ids}
        write_matching_results(predictions, cfg["output"]["matching_path"])
        return

    # ── 4. Load artefacts & Score ─────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 4/5] Scoring Test Pairs with Stacked Ensemble")
    logger.info("=" * 65)
    t0 = time.time()
    lgbm_final = load_artefact(model_dir, "lgbm_final")
    xgb_final  = load_artefact(model_dir, "xgb_final")
    meta       = load_artefact(model_dir, "meta")
    scaler     = load_artefact(model_dir, "scaler")

    threshold_file = Path(model_dir) / "threshold.txt"
    t_str = threshold_file.read_text().strip().split(",")
    threshold  = float(t_str[0])
    margin_gap = float(t_str[1]) if len(t_str) > 1 else 0.0
    logger.info(f"Loaded decision parameters: Threshold (t*)={threshold:.4f} | Margin Gap (g*)={margin_gap:.4f}")

    t_score = time.time()
    logger.info(f"Scoring {len(X):,} test pairs with LightGBM …")
    lgbm_p = safe_predict_proba(lgbm_final, X)[:, 1]
    logger.info(f"  ✓ LightGBM scored in {time.time() - t_score:.1f}s (mean prob: {np.mean(lgbm_p):.4f})")

    t_score = time.time()
    logger.info(f"Scoring {len(X):,} test pairs with XGBoost …")
    xgb_p  = safe_predict_proba(xgb_final, X)[:, 1]
    logger.info(f"  ✓ XGBoost scored in {time.time() - t_score:.1f}s (mean prob: {np.mean(xgb_p):.4f})")

    embed_idx = FEATURE_NAMES.index("embed_cosine")
    deg_idx   = FEATURE_NAMES.index("candidate_degree_log")
    cf_idx    = FEATURE_NAMES.index("country_freq_s1")

    embed_p = X[:, embed_idx]
    margin  = add_margin_feature(pairs_df, lgbm_p)

    t_score = time.time()
    logger.info("Applying calibrated meta-learner on stacked ensemble …")
    final_proba = meta_predict(
        meta, scaler,
        [lgbm_p, xgb_p, embed_p],
        margin,
        X[:, deg_idx],
        X[:, cf_idx],
    )
    logger.info(f"  ✓ Meta-learner scored in {time.time() - t_score:.1f}s")
    logger.info(f"✓ Ensemble scoring complete in {time.time() - t0:.1f}s")

    # ── 5. Threshold + abstention + write matching_results.tsv ────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 5/5] Thresholding, Abstention & Writing matching_results.tsv")
    logger.info("=" * 65)
    t0 = time.time()
    predictions = apply_threshold(pairs_df, final_proba, threshold, margin_gap, s1_ids)

    match_path = cfg["output"]["matching_path"]
    write_matching_results(predictions, match_path)

    # Breakdown statistics
    match_counts = [len(v) for v in predictions.values()]
    n_singletons = sum(1 for c in match_counts if c == 0)
    n_single_match = sum(1 for c in match_counts if c == 1)
    n_multi_match = sum(1 for c in match_counts if c > 1)
    n_total = len(predictions)

    logger.info(
        f"\nPrediction Distribution Across {n_total:,} S1 Queries:\n"
        f"    • Singletons (0 matches predicted): {n_singletons:,} ({100*n_singletons/n_total:5.1f}%)\n"
        f"    • Single Match (1 match predicted):  {n_single_match:,} ({100*n_single_match/n_total:5.1f}%)\n"
        f"    • Multi-Match (2+ matches predicted):{n_multi_match:,} ({100*n_multi_match/n_total:5.1f}%)"
    )

    total_time = time.time() - t_predict_start
    m_min, m_sec = divmod(int(total_time), 60)
    logger.info("\n" + "=" * 65)
    logger.info("  ✓ INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"  Total Duration: {m_min}m {m_sec:02d}s")
    logger.info(f"  Candidate Pairs:  {cand_path}")
    logger.info(f"  Matching Results: {match_path}")
    logger.info("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    predict(cfg)
