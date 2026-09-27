"""Central, explicit configuration for the pipeline. Every knob mentioned
in the write-up (blocking depth, RRF fusion, decision thresholds, ablation
toggles) lives here so training and inference always agree, and so
ablations can be run by flipping one flag instead of hunting through code.
"""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- blocking / candidate generation ---
    top_k_per_channel: int = 40
    final_top_k: int = 25

    # --- feature engineering toggles (ablation hooks) ---
    use_llr_feature: bool = True
    use_triangulation_feature: bool = True

    # --- training ---
    n_folds: int = 4
    random_seed: int = 42
    use_hard_negative_mining: bool = True
    hard_negative_prob_threshold: float = 0.6
    hard_negative_extra_weight: float = 3.0

    # --- calibration / decision layer toggles ---
    use_calibration: bool = True
    use_existence_model: bool = True
    use_metric_aware_decoder: bool = True

    # --- decision thresholds (tuned on OOF validation in train.py, stored
    # alongside the model artifacts so infer.py reuses the fitted values) ---
    existence_threshold: float = 0.5
    margin_threshold: float = 0.08
    ownership_margin_delta: float = 0.10

    lightgbm_params: dict = None

    def __post_init__(self):
        if self.lightgbm_params is None:
            self.lightgbm_params = dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=400,
                min_child_samples=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=self.random_seed,
                verbosity=-1,
            )
