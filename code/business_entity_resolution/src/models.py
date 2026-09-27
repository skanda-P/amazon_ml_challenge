"""
models.py
---------
Base models and stacked meta-learner.

Base models
-----------
1. LightGBM  — on all engineered features
2. CatBoost  (optional, falls back to XGBoost)
3. Embedding cosine similarity (already in feature vector as embed_cosine)

Stacking
--------
- K-fold OOF predictions from each base model
- Logistic regression meta-learner on base scores + margin + degree + country
- Isotonic / Platt calibration of final probabilities

All models are trained as binary classifiers; positive = match.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def safe_fit(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit model with automatic fallback to CPU if GPU / CUDA fails."""
    try:
        model.fit(X, y)
        return model
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ("cuda", "gpu", "device", "memory", "out of memory", "tree learner")):
            logger.warning(f"GPU execution failed for {type(model).__name__}: {e}. Falling back to multi-core CPU...")
            if hasattr(model, "set_params"):
                try:
                    model.set_params(device="cpu", n_jobs=-1)
                except Exception:
                    try:
                        model.set_params(device="cpu")
                    except Exception:
                        pass
            model.fit(X, y)
            return model
        raise e


def safe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict probabilities with automatic fallback to CPU if GPU / CUDA fails."""
    try:
        return model.predict_proba(X)
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ("cuda", "gpu", "device", "memory", "out of memory")):
            logger.warning(f"GPU prediction failed for {type(model).__name__}: {e}. Retrying on CPU...")
            if hasattr(model, "set_params"):
                try:
                    model.set_params(device="cpu", n_jobs=-1)
                except Exception:
                    try:
                        model.set_params(device="cpu")
                    except Exception:
                        pass
            return model.predict_proba(X)
        raise e


# ── Base model factory ──────────────────────────────────────────────────────

def make_lgbm(cfg: dict) -> Any:
    device = cfg.get("device", "auto")
    # For LightGBM on Windows, binary packages often lack GPU/CUDA tree learner.
    # If device is explicitly 'cuda' or 'gpu', we configure it; otherwise default to multi-threaded CPU.
    lgb_device = "cpu"
    if device in ("cuda", "gpu"):
        lgb_device = device

    return LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 7),
        learning_rate=cfg.get("learning_rate", 0.03),
        num_leaves=cfg.get("num_leaves", 63),
        scale_pos_weight=cfg.get("scale_pos_weight", 15),
        colsample_bytree=0.8,
        subsample=0.8,
        min_child_samples=20,
        n_jobs=-1,
        device=lgb_device,
        random_state=cfg.get("random_state", 42),
        verbose=-1,
    )


def make_xgb(cfg: dict) -> Any:
    device = cfg.get("device", "auto")
    cuda_ok = (device == "cuda") or (device == "auto" and is_cuda_available())

    xgb_params: dict[str, Any] = dict(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.03),
        scale_pos_weight=cfg.get("scale_pos_weight", 15),
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=cfg.get("random_state", 42),
        eval_metric="logloss",
        verbosity=0,
    )
    if cuda_ok:
        logger.info("Configuring XGBoost with CUDA GPU acceleration (device='cuda').")
        xgb_params.update(tree_method="hist", device="cuda")
    else:
        logger.info("Configuring XGBoost on CPU (device='cpu', n_jobs=-1).")
        xgb_params.update(tree_method="hist", device="cpu", n_jobs=-1)

    return XGBClassifier(**xgb_params)


# ── OOF K-fold training ─────────────────────────────────────────────────────

def train_oof(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, list]:
    """
    Train model with K-fold OOF.  Returns:
      oof_proba : shape (N,)  — out-of-fold positive-class probabilities
      fold_models : list of trained fold models
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        m = pickle.loads(pickle.dumps(model))   # fresh copy each fold
        safe_fit(m, X_tr, y_tr)
        oof[va_idx] = safe_predict_proba(m, X_va)[:, 1]
        fold_models.append(m)
        logger.info(f"  Fold {fold}/{n_splits} done")

    return oof, fold_models


# ── Meta-learner ────────────────────────────────────────────────────────────

def train_meta(
    base_oofs: list[np.ndarray],
    margin: np.ndarray,
    degree_log: np.ndarray,
    country_freq: np.ndarray,
    y: np.ndarray,
    calibrate: bool = True,
) -> tuple[Any, StandardScaler]:
    """
    Train logistic regression meta-learner on stacked base OOF + extra features.

    Returns: (calibrated_meta, scaler)
    """
    X_meta = np.column_stack(base_oofs + [margin, degree_log, country_freq])
    scaler = StandardScaler()
    X_meta = scaler.fit_transform(X_meta)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    if calibrate:
        meta = CalibratedClassifierCV(lr, method="isotonic", cv=5)
    else:
        meta = lr

    meta.fit(X_meta, y)
    logger.info("Meta-learner trained.")
    return meta, scaler


def meta_predict(
    meta,
    scaler: StandardScaler,
    base_probas: list[np.ndarray],
    margin: np.ndarray,
    degree_log: np.ndarray,
    country_freq: np.ndarray,
) -> np.ndarray:
    X_meta = np.column_stack(base_probas + [margin, degree_log, country_freq])
    X_meta = scaler.transform(X_meta)
    return meta.predict_proba(X_meta)[:, 1]


# ── Save / load artefacts ───────────────────────────────────────────────────

def save_artefacts(path_dir: str, **objects) -> None:
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    for name, obj in objects.items():
        p = Path(path_dir) / f"{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved {name} → {p}")


def load_artefact(path_dir: str, name: str) -> Any:
    p = Path(path_dir) / f"{name}.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)
