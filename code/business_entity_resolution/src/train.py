"""
train.py  (v2 — full stacked pipeline)
---------------------------------------
Full training pipeline:

1. Load + split data (train/val stratified on S1 entities)
2. Blocking (TF-IDF + optional embedding kNN)
3. Feature engineering (26 features per pair)
4. Base model OOF training (LightGBM + XGBoost)
5. Meta-learner training (LR + isotonic calibration)
6. Joint (threshold, margin_gap) tuning on val set to maximise F₀.₅
7. Save all artefacts to models/

Usage
-----
    cd code/business_entity_resolution
    python -m src.train [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.blocking import generate_candidates
from src.evaluate import macro_f05
from src.features import FEATURE_NAMES, add_margin_feature, build_feature_matrix
from src.io_utils import ground_truth_to_dict, load_ground_truth, load_source
from src.models import (
    load_artefact,
    make_lgbm,
    make_xgb,
    meta_predict,
    safe_fit,
    safe_predict_proba,
    save_artefacts,
    train_meta,
    train_oof,
)
from src.threshold import tune_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Default config ─────────────────────────────────────────────────────────

DEFAULT_CFG: dict = {
    "data": {
        "train_dir": "../../data/dataset/train",
    },
    "blocking": {
        "top_k_name":       10,
        "top_k_addr":       10,
        "top_k_embed":      15,
        "name_min_score":   0.25,
        "addr_min_score":   0.20,
        "embed_min_score":  0.70,
        "embed_model":      "intfloat/multilingual-e5-large",
        "use_embeddings":   False,   # set True to enable embedding kNN pass
    },
    "model": {
        "n_estimators":     500,
        "max_depth":        7,
        "num_leaves":       63,
        "learning_rate":    0.03,
        "scale_pos_weight": 15,
        "random_state":     42,
    },
    "stacking": {
        "n_splits":   5,
        "calibrate":  True,
    },
    "validation": {
        "val_size":     0.20,
        "random_state": 42,
    },
    "threshold": {
        "search_start": 0.10,
        "search_end":   0.95,
        "search_steps": 60,
        "margin_gaps":  [0.0, 0.05, 0.10, 0.15, 0.20],
    },
    "output": {
        "model_dir": "models",
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


def build_labeled_pairs(
    s1: pd.DataFrame,
    pool: pd.DataFrame,
    candidates: dict[str, list[str]],
    gt: dict[str, list[str]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pairs_df, X = build_feature_matrix(s1, pool, candidates)
    if pairs_df.empty:
        return pairs_df, X, np.array([], dtype=np.int8)
    gt_sets = {k: set(v) for k, v in gt.items()}
    y = np.fromiter(
        (int(r["candidate_entity_id"] in gt_sets.get(r["source1_entity_id"], set()))
         for _, r in pairs_df.iterrows()),
        dtype=np.int8, count=len(pairs_df),
    )
    return pairs_df, X, y


def train(cfg: dict) -> None:
    rng = np.random.default_rng(cfg["validation"]["random_state"])
    train_dir = Path(cfg["data"]["train_dir"])
    model_dir = cfg["output"]["model_dir"]

    # ── 1. Load data ──────────────────────────────────────────────────────
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt_df = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt = ground_truth_to_dict(gt_df)

    # ── 2. Train / val split ─────────────────────────────────────────────
    val_frac = cfg["validation"]["val_size"]
    s1_ids   = s1["entity_id"].tolist()
    val_mask = rng.random(len(s1_ids)) < val_frac
    val_ids  = set(np.array(s1_ids)[val_mask].tolist())
    tr_ids   = set(s1_ids) - val_ids

    s1_tr = s1[s1["entity_id"].isin(tr_ids)].copy().reset_index(drop=True)
    s1_va = s1[s1["entity_id"].isin(val_ids)].copy().reset_index(drop=True)
    pool  = pd.concat([s2, s3], ignore_index=True)

    logger.info(f"Train S1: {len(s1_tr):,} | Val S1: {len(s1_va):,}")

    # ── 3. Blocking ───────────────────────────────────────────────────────
    b_cfg = cfg["blocking"]
    cands_tr = generate_candidates(s1_tr, s2, s3, **b_cfg)
    cands_va = generate_candidates(s1_va, s2, s3, **b_cfg)

    # ── 4. Feature matrices ───────────────────────────────────────────────
    pairs_tr, X_tr, y_tr = build_labeled_pairs(s1_tr, pool, cands_tr, gt)
    pairs_va, X_va, y_va = build_labeled_pairs(s1_va, pool, cands_va, gt)

    pos_tr = int(y_tr.sum())
    logger.info(
        f"Train pairs: {len(X_tr):,}  positives: {pos_tr} ({100*pos_tr/max(len(X_tr),1):.2f}%)\n"
        f"Val pairs:   {len(X_va):,}  positives: {int(y_va.sum())}"
    )

    # ── 5. Base model OOF (on train set) ──────────────────────────────────
    m_cfg = cfg["model"]
    s_cfg = cfg["stacking"]

    logger.info("Training LightGBM (OOF) …")
    lgbm_base = make_lgbm({**m_cfg, **s_cfg})
    lgbm_oof, lgbm_folds = train_oof(
        lgbm_base, X_tr, y_tr,
        n_splits=s_cfg["n_splits"],
        random_state=m_cfg["random_state"],
    )

    logger.info("Training XGBoost (OOF) …")
    xgb_base = make_xgb({**m_cfg, **s_cfg})
    xgb_oof, xgb_folds = train_oof(
        xgb_base, X_tr, y_tr,
        n_splits=s_cfg["n_splits"],
        random_state=m_cfg["random_state"],
    )

    # ── 6. Val predictions from each base model (full train → val) ────────
    # Average fold models for val prediction
    lgbm_va_p = np.mean([safe_predict_proba(m, X_va)[:, 1] for m in lgbm_folds], axis=0)
    xgb_va_p  = np.mean([safe_predict_proba(m, X_va)[:, 1] for m in xgb_folds],  axis=0)

    # Embed cosine is already in features (index 21 in FEATURE_NAMES)
    embed_idx = FEATURE_NAMES.index("embed_cosine")
    embed_va  = X_va[:, embed_idx]
    embed_tr  = X_tr[:, embed_idx]

    # ── 7. Meta-learner (on OOF) ──────────────────────────────────────────
    logger.info("Training meta-learner …")
    margin_tr  = add_margin_feature(pairs_tr, lgbm_oof)
    deg_idx    = FEATURE_NAMES.index("candidate_degree_log")
    cf_idx     = FEATURE_NAMES.index("country_freq_s1")

    meta, scaler = train_meta(
        base_oofs=[lgbm_oof, xgb_oof, embed_tr],
        margin=margin_tr,
        degree_log=X_tr[:, deg_idx],
        country_freq=X_tr[:, cf_idx],
        y=y_tr,
        calibrate=s_cfg["calibrate"],
    )

    # Val meta prediction
    margin_va = add_margin_feature(pairs_va, lgbm_va_p)
    val_proba = meta_predict(
        meta, scaler,
        [lgbm_va_p, xgb_va_p, embed_va],
        margin_va,
        X_va[:, deg_idx],
        X_va[:, cf_idx],
    )

    # ── 8. Threshold tuning on val set ────────────────────────────────────
    gt_va = {k: v for k, v in gt.items() if k in val_ids}
    for sid in val_ids:
        gt_va.setdefault(sid, [])

    t_cfg = cfg["threshold"]
    best_t, best_mg, best_f05 = tune_threshold(
        pairs_va, val_proba, gt_va, list(val_ids),
        search_start=t_cfg["search_start"],
        search_end=t_cfg["search_end"],
        search_steps=t_cfg["search_steps"],
        margin_gaps=t_cfg["margin_gaps"],
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"  VAL F₀.₅ = {best_f05:.4f}")
    logger.info(f"  threshold = {best_t:.4f}  margin_gap = {best_mg:.4f}")
    logger.info(f"{'='*50}\n")

    # ── 9. Retrain base models on full training data ───────────────────────
    logger.info("Retraining base models on full training data …")
    lgbm_final = make_lgbm(m_cfg)
    safe_fit(lgbm_final, X_tr, y_tr)

    xgb_final = make_xgb(m_cfg)
    safe_fit(xgb_final, X_tr, y_tr)

    # ── 10. Save artefacts ────────────────────────────────────────────────
    save_artefacts(
        model_dir,
        lgbm_final=lgbm_final,
        lgbm_folds=lgbm_folds,
        xgb_final=xgb_final,
        xgb_folds=xgb_folds,
        meta=meta,
        scaler=scaler,
    )

    # Save threshold + margin gap
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / "threshold.txt").write_text(f"{best_t},{best_mg}")
    (Path(model_dir) / "val_f05.txt").write_text(str(best_f05))
    logger.info(f"All artefacts saved to ./{model_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)
