#!/usr/bin/env python3
"""
scripts/run_pipeline.py
-----------------------
End-to-end convenience runner. Equivalent to:
    python -m src.train   --config configs/config.yaml
    python -m src.predict --config configs/config.yaml

Usage
-----
    cd code/business_entity_resolution
    python scripts/run_pipeline.py [--config configs/config.yaml] [--skip-train]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing model artefacts")
    args = parser.parse_args()

    base = [sys.executable, "-m"]

    if not args.skip_train:
        run(base + ["src.train", "--config", args.config])

    run(base + ["src.predict", "--config", args.config])

    # Run the official validator
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    matching  = cfg.get("output", {}).get("matching_path",  "../../output/matching_results.tsv")
    candidate = cfg.get("output", {}).get("candidate_path", "../../output/candidate_pairs.tsv")
    test_dir  = cfg.get("data", {}).get("test_dir", "../../data/dataset/test")

    validator = Path("../../data/utils/validate_submission.py")
    if validator.exists():
        run([
            sys.executable, str(validator),
            "--matching",   matching,
            "--candidate",  candidate,
            "--test-dir",   test_dir,
        ])
    else:
        print(f"[WARN] Validator not found at {validator} — skipping local validation.")


if __name__ == "__main__":
    main()
