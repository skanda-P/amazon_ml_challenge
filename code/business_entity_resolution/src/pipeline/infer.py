"""
Inference orchestration.

    test data --> blocking --> candidate_pairs.tsv
        --> features --> LightGBM --> calibration
        --> existence model --> metric-aware set decoding
        --> ownership arbitration --> matching_results.tsv
"""
from __future__ import annotations

import csv
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from data.ingest import load_split
from features.features import NUMERIC_FEATURE_NAMES
from decoding.decoder import build_entity_level_features, decode_entity
from decoding.ownership import arbitrate
from pipeline.common import prepare_all, build_target_pool, build_candidates, build_pair_table
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def _load_artifacts(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "lgbm_model.joblib"))
    calibrator = joblib.load(os.path.join(model_dir, "calibrator.joblib"))
    existence_model = joblib.load(os.path.join(model_dir, "existence_model.joblib"))
    fs_model = joblib.load(os.path.join(model_dir, "fs_model.joblib"))
    feature_builder = joblib.load(os.path.join(model_dir, "feature_builder.joblib"))
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg = PipelineConfig()
    for k, v in cfg_dict.items():
        if k == "lightgbm_params":
            continue
        setattr(cfg, k, v)
    return model, calibrator, existence_model, fs_model, feature_builder, cfg


def _write_tsv(path: str, rows: list[tuple[str, str]], id_col: str, list_col: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow([id_col, list_col])
        for sid, joined in rows:
            writer.writerow([sid, joined])


def run_inference(data_dir: str, model_dir: str, output_dir: str) -> dict:
    model, calibrator, existence_model, fs_model, feature_builder, cfg = _load_artifacts(model_dir)

    raw, report = load_split(data_dir, "test")
    if not report.ok():
        logger.warning("Test data had %d validation issues (see log).", len(report.issues))

    s1, s2, s3 = prepare_all(raw)
    target_df = build_target_pool(s2, s3)
    logger.info("Loaded test: |S1|=%d |S2|=%d |S3|=%d", len(s1), len(s2), len(s3))

    ti, candidates = build_candidates(s1, target_df, cfg)

    # candidate_pairs.tsv is written from EXACTLY this candidate set, before
    # any ML scoring narrows it down.
    entity_ids_all = s1["entity_id"].tolist()
    cand_rows = []
    for sid in entity_ids_all:
        info = candidates.get(sid)
        if not info or not info["candidates"]:
            cand_rows.append((sid, ""))
            continue
        ids = [target_df.iloc[i]["entity_id"] for i in info["candidates"]]
        seen, ordered = set(), []
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        cand_rows.append((sid, ",".join(ordered)))
    _write_tsv(os.path.join(output_dir, "candidate_pairs.tsv"), cand_rows, "source1_entity_id", "candidate_entity_ids")

    pairs = build_pair_table(s1, target_df, candidates, feature_builder, use_triangulation=cfg.use_triangulation_feature)

    if len(pairs) == 0:
        # no candidates at all -- every entity is a singleton
        match_rows = [(sid, "") for sid in entity_ids_all]
        _write_tsv(os.path.join(output_dir, "matching_results.tsv"), match_rows, "source1_entity_id", "matched_entity_ids")
        return {"n_test_entities": len(entity_ids_all), "n_candidate_pairs": 0, "n_matched_entities": 0}

    if cfg.use_llr_feature:
        pairs["llr"] = fs_model.score(pairs)
    else:
        pairs["llr"] = 0.0

    X = pairs[NUMERIC_FEATURE_NAMES].astype(float).values
    raw_scores = model.predict_proba(X)[:, 1]
    pairs["calibrated_prob"] = calibrator.transform(raw_scores) if cfg.use_calibration else raw_scores

    grouped = pairs.groupby("source1_entity_id")

    proposed = {}  # sid -> [(target_entity_id, prob), ...]  (post metric-aware decode, pre-arbitration)
    for sid in entity_ids_all:
        if sid not in grouped.groups:
            proposed[sid] = []
            continue
        sub = grouped.get_group(sid).sort_values("calibrated_prob", ascending=False)
        probs_sorted = sub["calibrated_prob"].values
        llr_sorted = sub["llr"].values
        exist_feat = build_entity_level_features(probs_sorted, llr_sorted, len(sub)).reshape(1, -1)
        exist_p = existence_model.predict_proba(exist_feat)[0] if cfg.use_existence_model else 1.0

        if cfg.use_metric_aware_decoder:
            k = decode_entity(probs_sorted, exist_p, cfg.existence_threshold, cfg.margin_threshold)
        else:
            k = int(np.sum(probs_sorted > 0.5))

        top_ids = sub["target_entity_id"].values[:k]
        top_probs = probs_sorted[:k]
        proposed[sid] = list(zip(top_ids, top_probs))

    # Stage Q: ownership / conflict arbitration across Source-1 entities
    final_predictions = arbitrate(proposed, margin_delta=cfg.ownership_margin_delta)

    match_rows = []
    n_matched = 0
    for sid in entity_ids_all:
        matches = final_predictions.get(sid, [])
        ids = [m[0] for m in matches]
        # de-duplicate defensively (should already be unique)
        seen, ordered = set(), []
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        if ordered:
            n_matched += 1
        match_rows.append((sid, ",".join(ordered)))

    _write_tsv(os.path.join(output_dir, "matching_results.tsv"), match_rows, "source1_entity_id", "matched_entity_ids")

    summary = {
        "n_test_entities": len(entity_ids_all),
        "n_candidate_pairs": int(sum(len(v.split(",")) if v else 0 for _, v in cand_rows)),
        "n_matched_entities": n_matched,
    }
    logger.info("Inference summary: %s", summary)
    return summary
