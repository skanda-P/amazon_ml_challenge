"""
Stage A -- Data ingestion and validation.

Loads the tab-separated source files exactly as TSV (never inferring the
separator), validates schema / prefix consistency, and returns clean,
source-tagged dataframes. Source membership is ALWAYS derived from the
file the record came from (and cross-checked against the entity_id
prefix) -- never from business content.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

SOURCE_PREFIX = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


@dataclass
class ValidationReport:
    issues: list = field(default_factory=list)

    def add(self, msg: str):
        self.issues.append(msg)
        logger.warning("Validation issue: %s", msg)

    def ok(self) -> bool:
        return len(self.issues) == 0


def _read_tsv(path: str) -> pd.DataFrame:
    """Read a file as TSV explicitly. Never falls back to sep inference."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    return df


def load_source(path: str, source_name: str, report: ValidationReport) -> pd.DataFrame:
    df = _read_tsv(path)
    for col in REQUIRED_SOURCE_COLS:
        if col not in df.columns:
            report.add(f"{path}: missing required column '{col}'")
            df[col] = pd.NA

    # normalize whitespace-only fields to NA
    for col in ["business_name", "business_address", "country"]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA

    df["entity_id"] = df["entity_id"].astype(str).str.strip()

    # duplicate entity_id check
    dup_mask = df["entity_id"].duplicated(keep=False)
    if dup_mask.any():
        report.add(f"{path}: {dup_mask.sum()} duplicate entity_id rows found")

    # prefix consistency check -- source is defined by file, we only WARN
    # if the prefix looks inconsistent (helps catch mixed-up files early).
    expected_prefix = SOURCE_PREFIX[source_name]
    bad_prefix = ~df["entity_id"].str.startswith(expected_prefix)
    if bad_prefix.any():
        report.add(
            f"{path}: {bad_prefix.sum()} entity_id values do not start with "
            f"expected prefix '{expected_prefix}' for {source_name}"
        )

    df["source"] = source_name
    df = df.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)
    return df


def load_ground_truth(path: str, report: ValidationReport) -> pd.DataFrame:
    df = _read_tsv(path)
    for col in REQUIRED_GT_COLS:
        if col not in df.columns:
            report.add(f"{path}: missing required column '{col}'")
            df[col] = pd.NA
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str)
    dup_mask = df["source1_entity_id"].duplicated(keep=False)
    if dup_mask.any():
        report.add(f"{path}: {dup_mask.sum()} duplicate source1_entity_id rows")
    return df


def parse_id_list(cell: str) -> list:
    if not isinstance(cell, str) or cell.strip() == "":
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def load_split(data_dir: str, split: str) -> tuple[dict, ValidationReport]:
    """
    split: 'train' or 'test'
    Returns dict with keys 'source1','source2','source3' (+ 'ground_truth' for train)
    """
    report = ValidationReport()
    out = {}
    for src in ["source1", "source2", "source3"]:
        path = os.path.join(data_dir, split, f"{split}_{src}.tsv")
        if not os.path.exists(path):
            report.add(f"missing expected file: {path}")
            out[src] = pd.DataFrame(columns=REQUIRED_SOURCE_COLS + ["source"])
            continue
        out[src] = load_source(path, src, report)

    if split == "train":
        gt_path = os.path.join(data_dir, split, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            out["ground_truth"] = load_ground_truth(gt_path, report)
        else:
            report.add(f"missing ground truth file: {gt_path}")
            out["ground_truth"] = pd.DataFrame(columns=REQUIRED_GT_COLS)

    # schema consistency across the three sources
    cols_sets = [set(out[s].columns) for s in ["source1", "source2", "source3"]]
    if len(set(map(frozenset, cols_sets))) > 1:
        report.add("column schema differs across source1/source2/source3")

    # country distribution (informational only -- never used to filter)
    all_countries = pd.concat([out[s]["country"] for s in ["source1", "source2", "source3"]])
    logger.info("[%s] country distribution:\n%s", split, all_countries.value_counts(dropna=False))

    if not report.ok():
        for issue in report.issues:
            logger.warning(issue)

    return out, report
