"""
Stage G -- Probabilistic record linkage (Fellegi-Sunter).

For a small set of *discretized* comparison features (agreement
patterns), estimate m_f = P(f | Match) and u_f = P(f | Non-match) from
labeled training pairs, and combine them into a log-likelihood ratio:

    LLR(a,b) = sum_f log( m_f / u_f )

The LLR is not used as a hard threshold decision by itself -- it is fed
into the LightGBM classifier as one more feature (Stage H), letting the
gradient-boosted model decide how much to trust it relative to the raw
similarity features.
"""
from __future__ import annotations

import numpy as np

# Each entry: (feature_column_name, bucketing function producing an int/str bucket id)
AGREEMENT_FEATURES = [
    "name_exact", "name_nosuffix_exact",
    "name_levenshtein_sim", "name_jarowinkler_sim", "name_weighted_jaccard",
    "addr_exact", "addr_levenshtein_sim",
    "house_number_match", "postal_code_match", "postal_prefix_match",
    "locality_match", "country_exact",
]

N_BUCKETS = 4  # continuous [0,1] features are bucketed into quartile-ish bins


def _bucket(value: float) -> int:
    if value <= 0.0:
        return 0
    if value < 0.5:
        return 1
    if value < 0.9:
        return 2
    return 3


class FellegiSunterModel:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.m_probs = {}  # feature -> {bucket: prob}
        self.u_probs = {}

    def fit(self, feature_frame, labels: np.ndarray):
        labels = np.asarray(labels)
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = max(pos_mask.sum(), 1)
        n_neg = max(neg_mask.sum(), 1)

        for feat in AGREEMENT_FEATURES:
            if feat not in feature_frame.columns:
                continue
            buckets = feature_frame[feat].map(_bucket).values
            m_counts = {b: self.smoothing for b in range(N_BUCKETS)}
            u_counts = {b: self.smoothing for b in range(N_BUCKETS)}
            for b, y in zip(buckets[pos_mask], labels[pos_mask]):
                m_counts[b] += 1
            for b, y in zip(buckets[neg_mask], labels[neg_mask]):
                u_counts[b] += 1
            m_total = sum(m_counts.values())
            u_total = sum(u_counts.values())
            self.m_probs[feat] = {b: c / m_total for b, c in m_counts.items()}
            self.u_probs[feat] = {b: c / u_total for b, c in u_counts.items()}
        return self

    def score(self, feature_frame) -> np.ndarray:
        n = len(feature_frame)
        llr = np.zeros(n)
        for feat in AGREEMENT_FEATURES:
            if feat not in feature_frame.columns or feat not in self.m_probs:
                continue
            buckets = feature_frame[feat].map(_bucket).values
            m = np.array([self.m_probs[feat][b] for b in buckets])
            u = np.array([self.u_probs[feat][b] for b in buckets])
            llr += np.log(m / u)
        return llr
