"""
scripts/eda.py
--------------
Quick data exploration — prints key statistics without loading the full
datasets into memory all at once. Run before training to understand the data.

Usage (from code/business_entity_resolution/)
    python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "dataset"


def head_tsv(path: Path, n: int = 3):
    import pandas as pd
    return pd.read_csv(path, sep="\t", dtype=str, nrows=n).fillna("")


def main():
    import pandas as pd

    train_dir = DATA / "train"
    test_dir  = DATA / "test"

    print("=" * 60)
    print("LOADING (using chunksize for large files) …")

    # Use chunksize to avoid loading 500MB files fully for stats
    def count_rows(path: Path) -> int:
        n = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=50000):
            n += len(chunk)
        return n

    s1_n  = count_rows(train_dir / "train_source1.tsv")
    s2_n  = count_rows(train_dir / "train_source2.tsv")
    s3_n  = count_rows(train_dir / "train_source3.tsv")
    gt_n  = count_rows(train_dir / "train_ground_truth.tsv")
    ts1_n = count_rows(test_dir  / "test_source1.tsv")
    ts2_n = count_rows(test_dir  / "test_source2.tsv")
    ts3_n = count_rows(test_dir  / "test_source3.tsv")

    print(f"\n{'File':<35} {'Rows':>10}")
    print("-" * 46)
    print(f"{'train_source1.tsv':<35} {s1_n:>10,}")
    print(f"{'train_source2.tsv':<35} {s2_n:>10,}")
    print(f"{'train_source3.tsv':<35} {s3_n:>10,}")
    print(f"{'train_ground_truth.tsv':<35} {gt_n:>10,}")
    print(f"{'test_source1.tsv':<35} {ts1_n:>10,}")
    print(f"{'test_source2.tsv':<35} {ts2_n:>10,}")
    print(f"{'test_source3.tsv':<35} {ts3_n:>10,}")

    # Country distribution (read small sample)
    s1_full = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    ts1_full = pd.read_csv(test_dir  / "test_source1.tsv",  sep="\t", dtype=str).fillna("")

    print(f"\n\nTrain S1 country distribution:")
    print(s1_full["country"].value_counts().to_string())
    print(f"\nTest  S1 country distribution:")
    print(ts1_full["country"].value_counts().to_string())

    # Ground truth stats
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt["n_matches"] = gt["matched_entity_ids"].apply(
        lambda x: len([i for i in x.split(",") if i.strip()]) if x.strip() else 0
    )
    singletons = (gt["n_matches"] == 0).sum()
    print(f"\n\nGround Truth — match count distribution:")
    print(gt["n_matches"].value_counts().sort_index().head(15).to_string())
    print(f"\nSingletons: {singletons:,} / {len(gt):,} = {singletons/len(gt)*100:.1f}%")
    print(f"Max matches per entity: {gt['n_matches'].max()}")
    print(f"Mean matches per entity: {gt['n_matches'].mean():.3f}")

    # Sample records
    print("\n\nSample S1 records:")
    print(s1_full.head(3).to_string(index=False))

    print("\n\nSample GT rows:")
    print(gt.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
