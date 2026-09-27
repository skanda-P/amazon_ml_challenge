from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from normalization.normalize import prepare_dataframe
from blocking.blocking import TargetIndex, generate_candidates_for_source1
from features.features import FeatureBuilder, NUMERIC_FEATURE_NAMES, build_idf
from linkage.triangulation import compute_triangulation

logger = logging.getLogger(__name__)


def build_target_pool(prepared_s2: pd.DataFrame, prepared_s3: pd.DataFrame) -> pd.DataFrame:
    target = pd.concat([prepared_s2, prepared_s3], ignore_index=True)
    return target


def fit_feature_builder(s1_prep: pd.DataFrame, target_df: pd.DataFrame) -> FeatureBuilder:
    all_name_tokens = list(s1_prep["name_tokens"]) + list(target_df["name_tokens"])
    all_addr_tokens = list(s1_prep["addr_tokens"]) + list(target_df["addr_tokens"])
    name_idf, name_n = build_idf(all_name_tokens)
    addr_idf, addr_n = build_idf(all_addr_tokens)
    return FeatureBuilder(name_idf, name_n, addr_idf, addr_n)


def prepare_all(raw_split: dict) -> tuple:
    s1 = prepare_dataframe(raw_split["source1"])
    s2 = prepare_dataframe(raw_split["source2"])
    s3 = prepare_dataframe(raw_split["source3"])
    return s1, s2, s3


def build_candidates(s1_prep: pd.DataFrame, target_df: pd.DataFrame, cfg) -> tuple[TargetIndex, dict]:
    ti = TargetIndex(target_df)
    candidates = generate_candidates_for_source1(
        s1_prep, ti, top_k_per_channel=cfg.top_k_per_channel, final_top_k=cfg.final_top_k,
    )
    return ti, candidates


def build_pair_table(
    s1_prep: pd.DataFrame,
    target_df: pd.DataFrame,
    candidates: dict,
    feature_builder: FeatureBuilder,
    use_triangulation: bool = True,
) -> pd.DataFrame:
    """Builds the full pairwise feature table for every (S1, candidate)
    pair currently in `candidates`. This table's (source1_entity_id,
    target_entity_id) set IS candidate_pairs.tsv's content, by construction."""
    rows = []
    target_records = target_df.to_dict("records")

    for s1_row in s1_prep.itertuples(index=False):
        info = candidates.get(s1_row.entity_id)
        if not info or not info["candidates"]:
            continue
        cand_idx = info["candidates"]
        n_channels_total = 4  # exact-key, ngram, tfidf, address-structure

        # first pass: base similarity features (llr placeholder = 0)
        sim_s1_to_target = {}
        target_name_ws = {}
        base_feats_by_idx = {}
        for idx in cand_idx:
            tgt = target_records[idx]
            tgt_row = _DictRow(tgt)
            feats = feature_builder.build(
                s1_row, tgt_row,
                rrf_score=info["rrf_score"].get(idx, 0.0),
                support=info["support"].get(idx, 0),
                best_rank=info["best_rank"].get(idx, 999),
                n_channels_total=n_channels_total,
                llr=0.0,
            )
            base_feats_by_idx[idx] = feats
            sim_s1_to_target[idx] = feats["name_levenshtein_sim"]
            target_name_ws[idx] = tgt["name_ws"]

        if use_triangulation:
            s2_idx = [i for i in cand_idx if target_records[i]["source"] == "source2"]
            s3_idx = [i for i in cand_idx if target_records[i]["source"] == "source3"]
            tri = compute_triangulation(sim_s1_to_target, target_name_ws, s2_idx, s3_idx)
        else:
            tri = {}

        for idx in cand_idx:
            feats = base_feats_by_idx[idx]
            feats["triangulation_score"] = tri.get(idx, 0.0)
            tgt = target_records[idx]
            row = {
                "source1_entity_id": s1_row.entity_id,
                "target_entity_id": tgt["entity_id"],
                "target_idx": idx,
                "target_source": tgt["source"],
            }
            for name in NUMERIC_FEATURE_NAMES:
                row[name] = feats.get(name, 0.0)
            rows.append(row)

    if not rows:
        cols = ["source1_entity_id", "target_entity_id", "target_idx", "target_source"] + NUMERIC_FEATURE_NAMES
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


class _DictRow:
    """Lightweight attribute-accessor wrapper so target dict records can be
    passed to FeatureBuilder.build the same way itertuples rows are."""
    __slots__ = ("_d",)

    def __init__(self, d):
        self._d = d

    def __getattr__(self, item):
        try:
            return self._d[item]
        except KeyError as e:
            raise AttributeError(item) from e


def build_ground_truth_map(gt_df: pd.DataFrame) -> dict:
    from data.ingest import parse_id_list
    return {
        row.source1_entity_id: set(parse_id_list(row.matched_entity_ids))
        for row in gt_df.itertuples(index=False)
    }
