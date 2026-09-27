"""
Stage L -- Probability calibration.

Raw LightGBM scores are not guaranteed to be well-calibrated
probabilities. We fit isotonic regression on *out-of-fold* predictions
(never on in-fold predictions, to avoid leakage) to map raw scores to
calibrated probabilities used by the metric-aware set decoder.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, oof_scores: np.ndarray, oof_labels: np.ndarray) -> "Calibrator":
        self.iso.fit(oof_scores, oof_labels)
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            # fall back to raw scores if calibration wasn't fitted (e.g.
            # degenerate single-class training data in a tiny demo run)
            return scores
        return self.iso.predict(scores)
