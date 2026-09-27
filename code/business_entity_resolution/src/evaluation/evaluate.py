"""
Evaluation utilities: macro F0.5 (the leaderboard metric), candidate
recall (the blocking recall ceiling), and reduction ratio (blocking
efficiency). All computed entirely from local data -- no external
services involved.
"""
from __future__ import annotations

from collections import defaultdict


def per_entity_f_beta(pred: set, true: set, beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    beta2 = beta ** 2
    denom = (1 + beta2) * tp + beta2 * fn + fp
    if denom == 0:
        return 1.0 if not true else 0.0
    return (1 + beta2) * tp / denom


def macro_f_half(predictions: dict, ground_truth: dict, all_s1_ids) -> tuple[float, dict]:
    """predictions, ground_truth: {s1_id: set(matched_ids)}.
    Returns (macro_f0.5, per_entity_scores)."""
    scores = {}
    for sid in all_s1_ids:
        pred = predictions.get(sid, set())
        true = ground_truth.get(sid, set())
        scores[sid] = per_entity_f_beta(pred, true, beta=0.5)
    macro = sum(scores.values()) / len(scores) if scores else 0.0
    return macro, scores


def candidate_recall(candidates: dict, ground_truth: dict) -> float:
    """candidates: {s1_id: set(candidate_ids)}; ground_truth: {s1_id: set(true_ids)}."""
    total_true, found_true = 0, 0
    for sid, true_set in ground_truth.items():
        if not true_set:
            continue
        cand_set = candidates.get(sid, set())
        total_true += len(true_set)
        found_true += len(true_set & cand_set)
    return found_true / total_true if total_true else 1.0


def reduction_ratio(n_candidate_pairs: int, n_s1: int, n_s2: int, n_s3: int) -> float:
    all_possible = n_s1 * (n_s2 + n_s3)
    if all_possible == 0:
        return 0.0
    return 1.0 - (n_candidate_pairs / all_possible)


def pair_precision_recall(pred_pairs: set, true_pairs: set) -> tuple[float, float]:
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


def singleton_accuracy(predictions: dict, ground_truth: dict, all_s1_ids) -> float:
    correct, total = 0, 0
    for sid in all_s1_ids:
        true_is_singleton = len(ground_truth.get(sid, set())) == 0
        pred_is_singleton = len(predictions.get(sid, set())) == 0
        if true_is_singleton:
            total += 1
            correct += int(pred_is_singleton)
    return correct / total if total else 1.0
