"""
io_utils.py
-----------
Helpers for loading source files and writing submission TSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"entity_id", "business_name", "business_address", "country"}


def load_source(path: str | Path) -> pd.DataFrame:
    """Load a source TSV file, validate columns, return DataFrame."""
    path = Path(path)
    logger.info(f"Loading {path.name} …")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    logger.info(f"  → {len(df):,} records")
    return df


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load a ground truth TSV file."""
    path = Path(path)
    logger.info(f"Loading ground truth {path.name} …")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    logger.info(f"  → {len(df):,} rows")
    return df


def ground_truth_to_dict(gt_df: pd.DataFrame) -> dict[str, list[str]]:
    """Convert ground truth DataFrame → {s1_id: [matched_ids]}."""
    result: dict[str, list[str]] = {}
    for _, row in gt_df.iterrows():
        raw = str(row.get("matched_entity_ids", "")).strip()
        matched = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        result[str(row["source1_entity_id"])] = matched
    return result


def write_matching_results(
    predictions: dict[str, list[str]],
    output_path: str | Path,
) -> None:
    """
    Write matching_results.tsv.

    Format:
        source1_entity_id \\t matched_entity_ids
    where matched_entity_ids is comma-separated (empty for singletons).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, matched in sorted(predictions.items()):
        rows.append({
            "source1_entity_id":  s1_id,
            "matched_entity_ids": ",".join(matched),
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote matching_results.tsv → {output_path} ({len(df):,} rows)")


def write_candidate_pairs(
    candidates: dict[str, list[str]],
    output_path: str | Path,
) -> None:
    """
    Write candidate_pairs.tsv.

    Format:
        source1_entity_id \\t candidate_entity_ids
    where candidate_entity_ids is comma-separated.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, cands in sorted(candidates.items()):
        rows.append({
            "source1_entity_id":    s1_id,
            "candidate_entity_ids": ",".join(cands),
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote candidate_pairs.tsv → {output_path} ({len(df):,} rows)")
