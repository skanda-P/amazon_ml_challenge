"""
Stage M -- Source-1 existence / singleton model.
Stage N -- Metric-aware F0.5 set decoder.
Stage O -- Confidence / abstention.

This is the key differentiator versus a plain pairwise classifier: the
final prediction for each Source-1 entity is chosen to maximize an
*expected* F0.5 over its own candidate set, not by thresholding each
pair independently at p > 0.5.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def build_entity_level_features(sorted_probs: np.ndarray, sorted_llr: np.ndarray, n_candidates: int) -> np.ndarray:
    """Aggregate per-entity features used by the existence model.
    sorted_probs / sorted_llr are sorted descending by probability."""
    p1 = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
    p2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    gap = p1 - p2
    n_high = float(np.sum(sorted_probs > 0.5))
    max_llr = sorted_llr[0] if len(sorted_llr) > 0 else 0.0
    mean_top3 = float(np.mean(sorted_probs[:3])) if len(sorted_probs) > 0 else 0.0
    return np.array([p1, p2, gap, n_high, max_llr, mean_top3, float(n_candidates)])


EXISTENCE_FEATURE_NAMES = ["top1_prob", "top2_prob", "prob_gap", "n_high_conf", "max_llr", "mean_top3_prob", "n_candidates"]


class ExistenceModel:
    """P(Z_i = 1) -- does this Source-1 entity have >=1 true match at all.

    A learned, calibrated logistic model over entity-level aggregate
    features, used *conservatively*: since a false positive on a true
    singleton drops that entity's score from 1.0 to 0.0, the decoder
    treats a low existence probability as a strong signal to abstain
    (predict empty set) even if some pairwise probabilities are
    moderately high.
    """

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExistenceModel":
        if len(np.unique(y)) < 2:
            self._fitted = False
            return self
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or len(X) == 0:
            # fall back: independence-approximation baseline
            # P(Z=1) ~= 1 - prod(1-p_ij) using top1_prob column as a proxy
            # when no learned model is available.
            return np.clip(X[:, 0] if X.shape[0] else np.array([]), 0, 1)
        return self.clf.predict_proba(X)[:, 1]


def expected_f_half(cumulative_prob: float, k: int, expected_total_true: float, beta: float = 0.5) -> float:
    beta2 = beta ** 2
    denom = k + beta2 * expected_total_true
    if denom <= 0:
        return 0.0
    return (1 + beta2) * cumulative_prob / denom


def decode_entity(
    calibrated_probs: list,
    existence_prob: float,
    existence_threshold: float,
    margin_threshold: float,
    max_k: int | None = None,
) -> int:
    """
    Returns k* -- the number of top-ranked candidates (by probability,
    already sorted descending) to predict as matches for this entity.

    Implements Stage N (metric-aware decoding) with a Stage O abstention
    guard: if the existence model is not confident the entity has ANY
    match, or the top candidates are not meaningfully separated from
    noise, prefer k=0 (since false merges are penalized 2x harder than
    misses under F0.5, and a true singleton predicted as singleton scores
    a full 1.0).
    """
    probs = np.asarray(calibrated_probs, dtype=float)
    if len(probs) == 0:
        return 0

    # Stage O: abstention guard on overall existence confidence.
    if existence_prob < existence_threshold:
        return 0

    expected_total_true = float(np.sum(probs))
    K = len(probs) if max_k is None else min(max_k, len(probs))

    best_k, best_score = 0, expected_f_half(0.0, 0, expected_total_true)
    cum = 0.0
    for k in range(1, K + 1):
        cum += probs[k - 1]
        score = expected_f_half(cum, k, expected_total_true)
        if score > best_score:
            best_score, best_k = score, k

    # Stage O: if best_k == 1 but the single top candidate isn't clearly
    # separated from the runner-up AND its absolute probability is weak,
    # abstain rather than risk a false merge on an ambiguous pair.
    if best_k >= 1:
        top1 = probs[0]
        top2 = probs[1] if len(probs) > 1 else 0.0
        if top1 < 0.5:
            return 0
        if best_k == 1 and (top1 - top2) < margin_threshold and top2 > (top1 * 0.85):
            # near-tie with weak absolute confidence -> abstain on this one
            if top1 < 0.75:
                return 0

    return best_k
