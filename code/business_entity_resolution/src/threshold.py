"""
threshold.py
------------
F₀.₅-optimised thresholding + margin-based abstention.

Key ideas
---------
1. Global threshold sweep: find the probability cutoff that maximises
   macro-averaged F₀.₅ on the validation set.
2. Margin-based abstention: if (top-candidate score − second-best score)
   < margin_gap, predict "no match" (singleton) instead of committing.
   Targets the metric's full-1.0 reward for correctly identified singletons.
3. Optional per-country subgroup thresholding.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.evaluate import macro_f05, score_entity

logger = logging.getLogger(__name__)


def _build_predictions(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    margin_gap: float = 0.0,
    all_s1_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Build predictions dict from scored pairs.

    margin_gap > 0: if (max_score - second_score) < margin_gap → predict no match.
    """
    predictions: dict[str, list[str]] = {}

    if not pairs_df.empty:
        df = pairs_df.copy()
        df["_p"] = proba
        for s1_id, grp in df.groupby("source1_entity_id"):
            grp_s = grp.sort_values("_p", ascending=False)
            scores = grp_s["_p"].values

            # Margin abstention
            if margin_gap > 0 and len(scores) >= 2:
                gap = scores[0] - scores[1]
                if gap < margin_gap and scores[0] < threshold + 0.15:
                    predictions[s1_id] = []
                    continue

            matched = list(grp_s.loc[grp_s["_p"] >= threshold, "candidate_entity_id"])
            predictions[s1_id] = matched

    # Ensure all S1 entities have an entry
    if all_s1_ids:
        for sid in all_s1_ids:
            if sid not in predictions:
                predictions[sid] = []

    return predictions


def tune_threshold(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    gt: dict[str, list[str]],
    all_s1_ids: list[str],
    search_start: float = 0.1,
    search_end: float = 0.95,
    search_steps: int = 50,
    margin_gaps: list[float] | None = None,
) -> tuple[float, float, float]:
    """
    Grid-search (threshold, margin_gap) to maximise macro F₀.₅.

    Returns: (best_threshold, best_margin_gap, best_f05)
    """
    if margin_gaps is None:
        margin_gaps = [0.0, 0.05, 0.10, 0.15, 0.20]

    thresholds = np.linspace(search_start, search_end, search_steps)
    best_t, best_mg, best_score = 0.5, 0.0, -1.0

    for mg in margin_gaps:
        for t in thresholds:
            preds = _build_predictions(pairs_df, proba, t, mg, all_s1_ids)
            score = macro_f05(preds, gt)
            if score > best_score:
                best_score, best_t, best_mg = score, t, mg

    logger.info(
        f"Threshold tuning → t={best_t:.3f}  margin_gap={best_mg:.3f}  "
        f"F₀.₅={best_score:.4f}"
    )
    return best_t, best_mg, best_score


def apply_threshold(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    margin_gap: float,
    all_s1_ids: list[str],
) -> dict[str, list[str]]:
    """Apply a fixed (threshold, margin_gap) to generate final predictions."""
    return _build_predictions(pairs_df, proba, threshold, margin_gap, all_s1_ids)
