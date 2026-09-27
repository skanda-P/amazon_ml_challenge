"""
Training orchestration.

    train data --> blocking --> features --> LightGBM (GroupKFold OOF)
        --> Fellegi-Sunter LLR --> isotonic calibration
        --> entity-existence model --> threshold tuning on OOF
        --> save artifacts

GroupKFold is grouped by source1_entity_id so every candidate pair for a
given Source-1 entity stays in the same fold -- this prevents entity
leakage between train/validation (Stage J).
"""
from __future__ import annotations

import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from data.ingest import load_split
from features.features import NUMERIC_FEATURE_NAMES
from linkage.fellegi_sunter import FellegiSunterModel
from calibration.calibrate import Calibrator
from decoding.decoder import ExistenceModel, build_entity_level_features, decode_entity, EXISTENCE_FEATURE_NAMES
from decoding.ownership import arbitrate
from evaluation.evaluate import macro_f_half, candidate_recall, reduction_ratio, pair_precision_recall, singleton_accuracy
from pipeline.common import prepare_all, build_target_pool, fit_feature_builder, build_candidates, build_pair_table, build_ground_truth_map
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_training(data_dir: str, model_dir: str, cfg: PipelineConfig = None) -> dict:
    cfg = cfg or PipelineConfig()
    os.makedirs(model_dir, exist_ok=True)
    t0 = time.time()

    raw, report = load_split(data_dir, "train")
    if not report.ok():
        logger.warning("Training data had %d validation issues (see log).", len(report.issues))

    s1, s2, s3 = prepare_all(raw)
    target_df = build_target_pool(s2, s3)
    gt_map = build_ground_truth_map(raw["ground_truth"])

    logger.info("Loaded train: |S1|=%d |S2|=%d |S3|=%d", len(s1), len(s2), len(s3))

    feature_builder = fit_feature_builder(s1, target_df)
    t_block = time.time()
    ti, candidates = build_candidates(s1, target_df, cfg)
    logger.info("Blocking done in %.1fs", time.time() - t_block)

    cand_sets = {sid: {target_df.iloc[i]["entity_id"] for i in info["candidates"]} for sid, info in candidates.items()}
    cr = candidate_recall(cand_sets, gt_map)
    n_cand_pairs = sum(len(v) for v in cand_sets.values())
    rr = reduction_ratio(n_cand_pairs, len(s1), len(s2), len(s3))
    logger.info("Candidate recall=%.4f | reduction ratio=%.6f | avg candidates/entity=%.2f",
                cr, rr, n_cand_pairs / max(len(s1), 1))

    t_feat = time.time()
    pairs = build_pair_table(s1, target_df, candidates, feature_builder, use_triangulation=cfg.use_triangulation_feature)
    logger.info("Feature table built in %.1fs (%d pairs)", time.time() - t_feat, len(pairs))

    pairs["label"] = pairs.apply(
        lambda r: int(r["target_entity_id"] in gt_map.get(r["source1_entity_id"], set())), axis=1
    )

    # Stage G: Fellegi-Sunter LLR (trained on the same candidate pairs' labels)
    fs_model = FellegiSunterModel()
    if cfg.use_llr_feature and pairs["label"].nunique() > 1:
        fs_model.fit(pairs, pairs["label"].values)
        pairs["llr"] = fs_model.score(pairs)
    else:
        pairs["llr"] = 0.0

    X_cols = NUMERIC_FEATURE_NAMES
    X = pairs[X_cols].astype(float).values
    y = pairs["label"].values
    groups = pairs["source1_entity_id"].values

    n_unique_groups = len(set(groups))
    n_splits = min(cfg.n_folds, max(n_unique_groups, 2))
    gkf = GroupKFold(n_splits=n_splits) if n_unique_groups >= 2 else None

    oof_raw = np.zeros(len(pairs))
    oof_llr = pairs["llr"].values.copy()

    if gkf is not None and pairs["label"].nunique() > 1:
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            sample_weight = np.ones(len(tr_idx))
            model = LGBMClassifier(**cfg.lightgbm_params)
            model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight)
            probs = model.predict_proba(X[va_idx])[:, 1]

            if cfg.use_hard_negative_mining:
                # Stage I: identify hard negatives (label=0, high predicted
                # prob) within the TRAIN fold itself and retrain with extra
                # weight on them, then re-score the held-out fold.
                train_probs = model.predict_proba(X[tr_idx])[:, 1]
                hard_neg_mask = (y[tr_idx] == 0) & (train_probs > cfg.hard_negative_prob_threshold)
                sample_weight2 = np.ones(len(tr_idx))
                sample_weight2[hard_neg_mask] = cfg.hard_negative_extra_weight
                model2 = LGBMClassifier(**cfg.lightgbm_params)
                model2.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight2)
                probs = model2.predict_proba(X[va_idx])[:, 1]

            oof_raw[va_idx] = probs
    else:
        # degenerate tiny-data fallback: no meaningful CV possible
        oof_raw[:] = y.astype(float)

    # Final model trained on ALL data (with hard-negative reweighting) for inference use
    final_sample_weight = np.ones(len(y))
    if cfg.use_hard_negative_mining and pairs["label"].nunique() > 1:
        pre_model = LGBMClassifier(**cfg.lightgbm_params)
        pre_model.fit(X, y)
        pre_probs = pre_model.predict_proba(X)[:, 1]
        hard_neg_mask = (y == 0) & (pre_probs > cfg.hard_negative_prob_threshold)
        final_sample_weight[hard_neg_mask] = cfg.hard_negative_extra_weight

    final_model = LGBMClassifier(**cfg.lightgbm_params)
    if pairs["label"].nunique() > 1:
        final_model.fit(X, y, sample_weight=final_sample_weight)
    else:
        final_model.fit(X, y)

    # Stage L: calibration on OOF predictions (never on in-fold predictions)
    calibrator = Calibrator()
    if cfg.use_calibration and pairs["label"].nunique() > 1:
        calibrator.fit(oof_raw, y)
        pairs["calibrated_prob"] = calibrator.transform(oof_raw)
    else:
        pairs["calibrated_prob"] = oof_raw

    # Stage M: entity-existence model, trained on OOF calibrated probs
    entity_ids = s1["entity_id"].tolist()
    X_exist, y_exist, s1_order = [], [], []
    grouped = pairs.groupby("source1_entity_id")
    for sid in entity_ids:
        if sid in grouped.groups:
            sub = grouped.get_group(sid).sort_values("calibrated_prob", ascending=False)
            probs_sorted = sub["calibrated_prob"].values
            llr_sorted = sub["llr"].values
            n_cand = len(sub)
        else:
            probs_sorted = np.array([])
            llr_sorted = np.array([])
            n_cand = 0
        X_exist.append(build_entity_level_features(probs_sorted, llr_sorted, n_cand))
        y_exist.append(int(len(gt_map.get(sid, set())) > 0))
        s1_order.append(sid)
    X_exist = np.array(X_exist)
    y_exist = np.array(y_exist)

    existence_model = ExistenceModel()
    if cfg.use_existence_model:
        existence_model.fit(X_exist, y_exist)
    existence_oof = existence_model.predict_proba(X_exist) if len(X_exist) else np.array([])

    # --- tune decision thresholds on OOF data to maximize macro F0.5 ------
    best_cfg = {"existence_threshold": cfg.existence_threshold, "margin_threshold": cfg.margin_threshold}
    best_macro = -1.0
    threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7] if cfg.use_existence_model else [0.0]
    margin_grid = [0.05, 0.08, 0.12, 0.18]
    for et in threshold_grid:
        for mt in margin_grid:
            preds = {}
            for i, sid in enumerate(s1_order):
                if sid not in grouped.groups:
                    preds[sid] = set()
                    continue
                sub = grouped.get_group(sid).sort_values("calibrated_prob", ascending=False)
                probs_sorted = sub["calibrated_prob"].values
                exist_p = existence_oof[i] if len(existence_oof) else 1.0
                if cfg.use_metric_aware_decoder:
                    k = decode_entity(probs_sorted, exist_p, et, mt)
                else:
                    k = int(np.sum(probs_sorted > 0.5))
                preds[sid] = set(sub["target_entity_id"].values[:k])
            macro, _ = macro_f_half(preds, gt_map, s1_order)
            if macro > best_macro:
                best_macro = macro
                best_cfg = {"existence_threshold": et, "margin_threshold": mt}

    cfg.existence_threshold = best_cfg["existence_threshold"]
    cfg.margin_threshold = best_cfg["margin_threshold"]
    logger.info("Tuned decision thresholds on OOF: %s -> macro F0.5=%.4f", best_cfg, best_macro)

    # final OOF metrics report at the tuned operating point
    final_preds = {}
    for i, sid in enumerate(s1_order):
        if sid not in grouped.groups:
            final_preds[sid] = set()
            continue
        sub = grouped.get_group(sid).sort_values("calibrated_prob", ascending=False)
        probs_sorted = sub["calibrated_prob"].values
        exist_p = existence_oof[i] if len(existence_oof) else 1.0
        k = decode_entity(probs_sorted, exist_p, cfg.existence_threshold, cfg.margin_threshold) if cfg.use_metric_aware_decoder else int(np.sum(probs_sorted > 0.5))
        final_preds[sid] = set(sub["target_entity_id"].values[:k])

    macro, _ = macro_f_half(final_preds, gt_map, s1_order)
    true_pairs = {(sid, t) for sid, tset in gt_map.items() for t in tset}
    pred_pairs = {(sid, t) for sid, tset in final_preds.items() for t in tset}
    p, r = pair_precision_recall(pred_pairs, true_pairs)
    sing_acc = singleton_accuracy(final_preds, gt_map, s1_order)

    metrics = {
        "candidate_recall": cr,
        "reduction_ratio": rr,
        "avg_candidates_per_entity": n_cand_pairs / max(len(s1), 1),
        "pair_precision": p,
        "pair_recall": r,
        "singleton_accuracy": sing_acc,
        "macro_f0.5_oof": macro,
        "n_train_pairs": len(pairs),
        "training_time_sec": time.time() - t0,
    }
    logger.info("Training metrics (OOF, tuned decoder): %s", json.dumps(metrics, indent=2))

    # --- persist artifacts -------------------------------------------------
    joblib.dump(final_model, os.path.join(model_dir, "lgbm_model.joblib"))
    joblib.dump(calibrator, os.path.join(model_dir, "calibrator.joblib"))
    joblib.dump(existence_model, os.path.join(model_dir, "existence_model.joblib"))
    joblib.dump(fs_model, os.path.join(model_dir, "fs_model.joblib"))
    joblib.dump(feature_builder, os.path.join(model_dir, "feature_builder.joblib"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)
    with open(os.path.join(model_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts saved to %s", model_dir)
    return metrics
