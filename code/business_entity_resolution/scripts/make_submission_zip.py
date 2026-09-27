#!/usr/bin/env python3
"""
scripts/make_submission_zip.py
------------------------------
Packages the final submission zip as required by the contest:

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── configs/
    │       ├── scripts/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

Run from the repo root (amazon_ml_challenge/):
    python code/business_entity_resolution/scripts/make_submission_zip.py --team MyTeam
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # amazon_ml_challenge/


def add(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    if file_path.exists():
        zf.write(file_path, arcname)
        print(f"  + {arcname}")
    else:
        print(f"  ! MISSING: {arcname}  ({file_path})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="submission", help="Team name prefix for zip file")
    args = parser.parse_args()

    zip_path = ROOT / f"{args.team}_submission.zip"
    code_base = ROOT / "code" / "business_entity_resolution"

    print(f"\nBuilding {zip_path.name} …\n")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── output/ ────────────────────────────────────────────────────────
        add(zf, ROOT / "output" / "matching_results.tsv",  "output/matching_results.tsv")
        add(zf, ROOT / "output" / "candidate_pairs.tsv",   "output/candidate_pairs.tsv")

        # ── code/ ──────────────────────────────────────────────────────────
        for src_file in (code_base / "src").rglob("*.py"):
            rel = src_file.relative_to(ROOT)
            add(zf, src_file, f"code/{rel.as_posix()}")

        for cfg_file in (code_base / "configs").rglob("*"):
            if cfg_file.is_file():
                rel = cfg_file.relative_to(ROOT)
                add(zf, cfg_file, f"code/{rel.as_posix()}")

        for script in (code_base / "scripts").rglob("*.py"):
            rel = script.relative_to(ROOT)
            add(zf, script, f"code/{rel.as_posix()}")

        add(zf, code_base / "README.md",        "code/business_entity_resolution/README.md")
        add(zf, code_base / "requirements.txt", "code/business_entity_resolution/requirements.txt")

        # ── Documentation ─────────────────────────────────────────────────
        add(zf, ROOT / "Documentation_template.md", "Documentation_template.md")

    print(f"\nDone → {zip_path}")


if __name__ == "__main__":
    main()
