"""
evaluate.py
-----------
F_0.5 evaluation utilities — matches the contest's exact metric.

F_0.5 is computed *per Source 1 entity* and then macro-averaged.
Singletons (no true matches) score:
  - 1.0 if predicted list is also empty
  - 0.0 if any prediction is made
"""

from __future__ import annotations

import numpy as np


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Generic F-beta score. Returns 0.0 when precision+recall == 0."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted: list[str], ground_truth: list[str]) -> float:
    """
    F_0.5 for a single Source 1 entity.

    Parameters
    ----------
    predicted    : list of predicted S2/S3 entity IDs
    ground_truth : list of true matching S2/S3 entity IDs (empty = singleton)
    """
    pred_set  = set(predicted)
    truth_set = set(ground_truth)

    # Singleton case
    if not truth_set:
        return 1.0 if not pred_set else 0.0

    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(truth_set)
    return f_beta(precision, recall)


def macro_f05(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> float:
    """
    Macro-averaged F_0.5 over all Source 1 entities in ground_truth.

    Parameters
    ----------
    predictions  : { source1_entity_id: [matched_entity_ids] }
    ground_truth : { source1_entity_id: [matched_entity_ids] }
    """
    scores = []
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, [])
        scores.append(score_entity(pred, truth))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_from_files(
    predictions_path: str,
    ground_truth_path: str,
) -> dict[str, float]:
    """
    Load two TSV files and return evaluation metrics.

    Returns dict with keys: macro_f05, precision_mean, recall_mean
    """
    import pandas as pd

    pred_df = pd.read_csv(predictions_path, sep="\t", dtype=str).fillna("")
    gt_df   = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")

    def parse(row: str) -> list[str]:
        return [x.strip() for x in row.split(",") if x.strip()] if row else []

    predictions  = {r["source1_entity_id"]: parse(r["matched_entity_ids"])  for _, r in pred_df.iterrows()}
    ground_truth = {r["source1_entity_id"]: parse(r["matched_entity_ids"])  for _, r in gt_df.iterrows()}

    scores = []
    precisions, recalls = [], []
    for s1_id, truth in ground_truth.items():
        pred     = predictions.get(s1_id, [])
        s        = score_entity(pred, truth)
        scores.append(s)

        truth_set = set(truth)
        pred_set  = set(pred)
        tp = len(pred_set & truth_set)
        precisions.append(tp / len(pred_set) if pred_set else (1.0 if not truth_set else 0.0))
        recalls.append(tp / len(truth_set) if truth_set else (1.0 if not pred_set else 0.0))

    return {
        "macro_f05":       float(np.mean(scores)),
        "precision_mean":  float(np.mean(precisions)),
        "recall_mean":     float(np.mean(recalls)),
        "n_entities":      len(scores),
    }
