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
import time
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
        "batch_size":       256,
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
    "training": {
        "max_train_entities": 25000,
        "max_val_entities":   5000,
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
    t_pipeline_start = time.time()
    rng = np.random.default_rng(cfg["validation"]["random_state"])
    train_dir = Path(cfg["data"]["train_dir"])
    model_dir = cfg["output"]["model_dir"]

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 1/7] Ingesting Data & Ground Truth")
    logger.info("=" * 65)
    t0 = time.time()
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt_df = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt = ground_truth_to_dict(gt_df)
    logger.info(f"✓ Data ingestion completed in {time.time() - t0:.1f}s")

    # ── 2. Train / val split ─────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 2/7] Stratified Train/Val Partitioning")
    logger.info("=" * 65)
    t0 = time.time()
    val_frac = cfg["validation"]["val_size"]
    s1_ids   = s1["entity_id"].tolist()
    val_mask = rng.random(len(s1_ids)) < val_frac
    val_ids  = set(np.array(s1_ids)[val_mask].tolist())
    tr_ids   = set(s1_ids) - val_ids

    s1_tr = s1[s1["entity_id"].isin(tr_ids)].copy().reset_index(drop=True)
    s1_va = s1[s1["entity_id"].isin(val_ids)].copy().reset_index(drop=True)

    # Optional subsampling for training efficiency
    train_cfg = cfg.get("training", {})
    max_tr = train_cfg.get("max_train_entities")
    max_va = train_cfg.get("max_val_entities")
    if max_tr and len(s1_tr) > max_tr:
        logger.info(f"Sampling {max_tr:,} / {len(s1_tr):,} training S1 entities for efficient training")
        s1_tr = s1_tr.sample(n=max_tr, random_state=cfg["validation"]["random_state"]).reset_index(drop=True)
    if max_va and len(s1_va) > max_va:
        logger.info(f"Sampling {max_va:,} / {len(s1_va):,} validation S1 entities")
        s1_va = s1_va.sample(n=max_va, random_state=cfg["validation"]["random_state"]).reset_index(drop=True)
        val_ids = set(s1_va["entity_id"].tolist())

    pool  = pd.concat([s2, s3], ignore_index=True)
    logger.info(f"Split partition: Train S1: {len(s1_tr):,} | Val S1: {len(s1_va):,} | Candidate Pool: {len(pool):,} entities")
    logger.info(f"✓ Partitioning completed in {time.time() - t0:.1f}s")

    # ── 3. Blocking ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 3/7] Candidate Blocking Engine (Train & Validation)")
    logger.info("=" * 65)
    b_cfg = cfg["blocking"]
    logger.info("Running unified candidate blocking for train + validation entities …")
    s1_all = pd.concat([s1_tr, s1_va], ignore_index=True)
    all_cands = generate_candidates(s1_all, s2, s3, **b_cfg)

    # Split candidates into train and validation sets
    tr_ids_set = set(s1_tr["entity_id"])
    va_ids_set = set(s1_va["entity_id"])
    cands_tr = {k: v for k, v in all_cands.items() if k in tr_ids_set}
    cands_va = {k: v for k, v in all_cands.items() if k in va_ids_set}
    logger.info(
        f"✓ Unified candidates ready: {len(cands_tr):,} train S1 | {len(cands_va):,} val S1"
    )

    # ── 4. Feature matrices ───────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 4/7] Generating Feature Matrices & Labels")
    logger.info("=" * 65)
    t0 = time.time()
    logger.info("Building labeled training feature matrix …")
    pairs_tr, X_tr, y_tr = build_labeled_pairs(s1_tr, pool, cands_tr, gt)
    logger.info("Building labeled validation feature matrix …")
    pairs_va, X_va, y_va = build_labeled_pairs(s1_va, pool, cands_va, gt)

    pos_tr = int(y_tr.sum())
    pos_va = int(y_va.sum())
    logger.info(
        f"✓ Feature matrices ready in {time.time() - t0:.1f}s:\n"
        f"    Train pairs: {len(X_tr):,} | Positives: {pos_tr:,} ({100*pos_tr/max(len(X_tr),1):.2f}% pos rate)\n"
        f"    Val pairs:   {len(X_va):,} | Positives: {pos_va:,} ({100*pos_va/max(len(X_va),1):.2f}% pos rate)"
    )

    # ── 5. Base model OOF (on train set) ──────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 5/7] Out-of-Fold Base Model Training (LightGBM & XGBoost)")
    logger.info("=" * 65)
    m_cfg = cfg["model"]
    s_cfg = cfg["stacking"]

    t0 = time.time()
    logger.info(f"Training LightGBM {s_cfg['n_splits']}-Fold OOF …")
    lgbm_base = make_lgbm({**m_cfg, **s_cfg})
    lgbm_oof, lgbm_folds = train_oof(
        lgbm_base, X_tr, y_tr,
        n_splits=s_cfg["n_splits"],
        random_state=m_cfg["random_state"],
    )
    logger.info(f"✓ LightGBM OOF completed in {time.time() - t0:.1f}s")

    t0 = time.time()
    logger.info(f"Training XGBoost {s_cfg['n_splits']}-Fold OOF …")
    xgb_base = make_xgb({**m_cfg, **s_cfg})
    xgb_oof, xgb_folds = train_oof(
        xgb_base, X_tr, y_tr,
        n_splits=s_cfg["n_splits"],
        random_state=m_cfg["random_state"],
    )
    logger.info(f"✓ XGBoost OOF completed in {time.time() - t0:.1f}s")

    # ── 6. Val predictions from each base model (full train → val) ────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 6/7] Stacking Meta-Learner & F₀.₅ Threshold Tuning")
    logger.info("=" * 65)
    t0 = time.time()
    # Average fold models for val prediction
    lgbm_va_p = np.mean([safe_predict_proba(m, X_va)[:, 1] for m in lgbm_folds], axis=0)
    xgb_va_p  = np.mean([safe_predict_proba(m, X_va)[:, 1] for m in xgb_folds],  axis=0)

    embed_idx = FEATURE_NAMES.index("embed_cosine")
    embed_va  = X_va[:, embed_idx]
    embed_tr  = X_tr[:, embed_idx]

    logger.info("Fitting calibrated meta-learner on stacked OOF probabilities …")
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

    gt_va = {k: v for k, v in gt.items() if k in val_ids}
    for sid in val_ids:
        gt_va.setdefault(sid, [])

    t_cfg = cfg["threshold"]
    logger.info(
        f"Sweeping threshold in [{t_cfg['search_start']}, {t_cfg['search_end']}] "
        f"across margin gaps {t_cfg['margin_gaps']} to optimise macro-F₀.₅ …"
    )
    best_t, best_mg, best_f05 = tune_threshold(
        pairs_va, val_proba, gt_va, list(val_ids),
        search_start=t_cfg["search_start"],
        search_end=t_cfg["search_end"],
        search_steps=t_cfg["search_steps"],
        margin_gaps=t_cfg["margin_gaps"],
    )

    logger.info(f"\n{'='*65}")
    logger.info(f"  ★ OPTIMAL VALIDATION MACRO F₀.₅ = {best_f05:.4f}")
    logger.info(f"  ★ Threshold (t*) = {best_t:.4f}  |  Margin Gap (g*) = {best_mg:.4f}")
    logger.info(f"{'='*65}\n")

    # ── 7. Retrain base models on full training data ───────────────────────
    logger.info("=" * 65)
    logger.info("  [Step 7/7] Retraining Full-Dataset Production Models & Exporting")
    logger.info("=" * 65)
    t0 = time.time()
    logger.info("Retraining final LightGBM on all training pairs …")
    lgbm_final = make_lgbm(m_cfg)
    safe_fit(lgbm_final, X_tr, y_tr)

    logger.info("Retraining final XGBoost on all training pairs …")
    xgb_final = make_xgb(m_cfg)
    safe_fit(xgb_final, X_tr, y_tr)
    logger.info(f"✓ Production model retraining finished in {time.time() - t0:.1f}s")

    # Save artefacts
    save_artefacts(
        model_dir,
        lgbm_final=lgbm_final,
        lgbm_folds=lgbm_folds,
        xgb_final=xgb_final,
        xgb_folds=xgb_folds,
        meta=meta,
        scaler=scaler,
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / "threshold.txt").write_text(f"{best_t},{best_mg}")
    (Path(model_dir) / "val_f05.txt").write_text(str(best_f05))

    total_time = time.time() - t_pipeline_start
    m_min, m_sec = divmod(int(total_time), 60)
    logger.info("\n" + "=" * 65)
    logger.info("  ✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"  Total Duration: {m_min}m {m_sec:02d}s")
    logger.info(f"  Artefacts saved in: ./{model_dir}/")
    logger.info("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)
