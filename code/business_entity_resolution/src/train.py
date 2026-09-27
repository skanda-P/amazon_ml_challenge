"""
train.py
--------
Train the matching classifier on the training set.

Pipeline
--------
1. Load train_source1/2/3.tsv + train_ground_truth.tsv
2. Blocking → candidate pairs
3. Feature engineering → label assignment
4. Train XGBoost (or LightGBM) binary classifier
5. Threshold tuning on a held-out validation split (maximise F_0.5)
6. Save model artefact to models/

Usage
-----
    python -m src.train [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.blocking import generate_candidates
from src.evaluate import macro_f05
from src.features import FEATURE_NAMES, build_feature_matrix
from src.io_utils import (
    ground_truth_to_dict,
    load_ground_truth,
    load_source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Default config (overridden by YAML) ─────────────────────────────────────
DEFAULT_CFG = {
    "data": {
        "train_dir": "../../data/dataset/train",
    },
    "blocking": {
        "top_k_name": 10,
        "top_k_addr": 10,
        "name_min_score": 0.25,
        "addr_min_score": 0.20,
    },
    "model": {
        "type": "xgboost",         # "xgboost" | "lightgbm"
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": 10,    # adjust for class imbalance
    },
    "validation": {
        "val_size": 0.15,
        "random_state": 42,
    },
    "threshold": {
        "search_start": 0.1,
        "search_end": 0.9,
        "search_steps": 50,
    },
    "output": {
        "model_path": "models/matcher.pkl",
        "threshold_path": "models/threshold.txt",
    },
}


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path and Path(path).exists():
        with open(path) as f:
            overrides = yaml.safe_load(f)
        # shallow merge per top-level key
        for k, v in overrides.items():
            if k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def build_labeled_dataset(
    s1: pd.DataFrame,
    pool: pd.DataFrame,
    candidates: dict[str, list[str]],
    gt: dict[str, list[str]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build (pairs_df, X, y) for training.

    y = 1 if the candidate is a true match, else 0.
    """
    pairs_df, X = build_feature_matrix(s1, pool, candidates)
    if pairs_df.empty:
        return pairs_df, X, np.array([], dtype=np.int8)

    gt_set: dict[str, set[str]] = {s1_id: set(matched) for s1_id, matched in gt.items()}
    y = np.array(
        [
            int(row["candidate_entity_id"] in gt_set.get(row["source1_entity_id"], set()))
            for _, row in pairs_df.iterrows()
        ],
        dtype=np.int8,
    )
    return pairs_df, X, y


def tune_threshold(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    gt: dict[str, list[str]],
    search_start: float,
    search_end: float,
    search_steps: int,
) -> float:
    """Grid-search the decision threshold that maximises F_0.5 on validation data."""
    best_t, best_score = 0.5, -1.0
    thresholds = np.linspace(search_start, search_end, search_steps)
    for t in thresholds:
        preds: dict[str, list[str]] = {}
        for (s1_id, grp) in pairs_df.assign(_p=proba).groupby("source1_entity_id"):
            matched = list(grp.loc[grp["_p"] >= t, "candidate_entity_id"])
            preds[s1_id] = matched
        score = macro_f05(preds, gt)
        if score > best_score:
            best_score, best_t = score, t

    logger.info(f"Best threshold: {best_t:.3f} → F_0.5 = {best_score:.4f}")
    return best_t


def train(cfg: dict) -> None:
    train_dir = Path(cfg["data"]["train_dir"])

    # ── Load data ────────────────────────────────────────────────────────────
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt_df = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt = ground_truth_to_dict(gt_df)

    # ── Train / val split on S1 entities ────────────────────────────────────
    val_size = cfg["validation"]["val_size"]
    rng = np.random.default_rng(cfg["validation"]["random_state"])
    s1_ids = s1["entity_id"].tolist()
    val_mask = rng.random(len(s1_ids)) < val_size
    val_s1_ids = set(s = np.array(s1_ids)[val_mask])
    train_s1_ids = set(s1_ids) - val_s1_ids

    s1_tr = s1[s1["entity_id"].isin(train_s1_ids)].copy()
    s1_va = s1[s1["entity_id"].isin(val_s1_ids)].copy()
    pool = pd.concat([s2, s3], ignore_index=True)

    # ── Blocking ─────────────────────────────────────────────────────────────
    b_cfg = cfg["blocking"]
    candidates_tr = generate_candidates(s1_tr, s2, s3, **b_cfg)
    candidates_va = generate_candidates(s1_va, s2, s3, **b_cfg)

    # ── Features + labels ────────────────────────────────────────────────────
    pairs_tr, X_tr, y_tr = build_labeled_dataset(s1_tr, pool, candidates_tr, gt)
    pairs_va, X_va, y_va = build_labeled_dataset(s1_va, pool, candidates_va, gt)

    pos_tr = int(y_tr.sum())
    logger.info(f"Train: {len(X_tr):,} pairs, {pos_tr} positives ({100*pos_tr/max(len(X_tr),1):.2f}%)")
    logger.info(f"Val:   {len(X_va):,} pairs, {int(y_va.sum())} positives")

    # ── Model ─────────────────────────────────────────────────────────────────
    m_cfg = cfg["model"]
    if m_cfg["type"] == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=m_cfg["n_estimators"],
            max_depth=m_cfg["max_depth"],
            learning_rate=m_cfg["learning_rate"],
            scale_pos_weight=m_cfg["scale_pos_weight"],
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=cfg["validation"]["random_state"],
        )
    elif m_cfg["type"] == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=m_cfg["n_estimators"],
            max_depth=m_cfg["max_depth"],
            learning_rate=m_cfg["learning_rate"],
            scale_pos_weight=m_cfg["scale_pos_weight"],
            n_jobs=-1,
            random_state=cfg["validation"]["random_state"],
        )
    else:
        raise ValueError(f"Unknown model type: {m_cfg['type']}")

    logger.info(f"Training {m_cfg['type']} …")
    model.fit(X_tr, y_tr)

    # ── Threshold tuning on validation ───────────────────────────────────────
    proba_va = model.predict_proba(X_va)[:, 1]
    t_cfg = cfg["threshold"]

    # Build val ground truth (only for val S1 entities)
    gt_va = {k: v for k, v in gt.items() if k in val_s1_ids}
    # Also include singletons that have no candidates
    for s1_id in val_s1_ids:
        if s1_id not in gt_va:
            gt_va[s1_id] = []

    best_threshold = tune_threshold(
        pairs_va, proba_va, gt_va,
        t_cfg["search_start"], t_cfg["search_end"], t_cfg["search_steps"],
    )

    # ── Save artefacts ───────────────────────────────────────────────────────
    model_path = Path(cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {model_path}")

    threshold_path = Path(cfg["output"]["threshold_path"])
    threshold_path.write_text(str(best_threshold))
    logger.info(f"Threshold saved → {threshold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train entity-resolution matcher")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
