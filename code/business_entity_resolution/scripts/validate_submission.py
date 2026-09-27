#!/usr/bin/env python3
"""
Stdlib-only validator for matching_results.tsv and candidate_pairs.tsv,
mirroring every rule in the challenge problem statement's "Output Format"
and "Constraints" sections.

Usage:
    python3 validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Prints PASS (exit 0) if the files are safe to submit, or a numbered list
of issues (exit 1). Only reads the given output files and the test
source files -- it does NOT compute a score.
"""
import argparse
import csv
import os
import sys


def _read_tsv_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    return rows


def _load_entity_ids(test_dir, filename):
    path = os.path.join(test_dir, filename)
    if not os.path.exists(path):
        return set(), [f"required test file not found: {path}"]
    rows = _read_tsv_rows(path)
    if not rows:
        return set(), [f"{path} is empty"]
    header = [h.strip() for h in rows[0]]
    if "entity_id" not in header:
        return set(), [f"{path}: missing 'entity_id' column"]
    idx = header.index("entity_id")
    ids = set()
    for r in rows[1:]:
        if len(r) > idx and r[idx].strip():
            ids.add(r[idx].strip())
    return ids, []


def _load_output_tsv(path, id_col, list_col):
    issues = []
    if not os.path.exists(path):
        return None, [f"output file not found: {path}"]
    rows = _read_tsv_rows(path)
    if not rows:
        return None, [f"{path} is empty"]
    header = [h.strip() for h in rows[0]]
    if header != [id_col, list_col]:
        issues.append(f"{path}: header is {header}, expected [{id_col!r}, {list_col!r}]")
    data = {}
    dup_ids = []
    for line_no, r in enumerate(rows[1:], start=2):
        if len(r) < 1:
            issues.append(f"{path}: line {line_no} is malformed (no columns)")
            continue
        sid = r[0].strip()
        ids_cell = r[1].strip() if len(r) > 1 else ""
        if sid in data:
            dup_ids.append(sid)
        data[sid] = ids_cell
    if dup_ids:
        issues.append(f"{path}: duplicate source1_entity_id rows: {sorted(set(dup_ids))[:10]}"
                       f"{' ...' if len(set(dup_ids)) > 10 else ''}")
    return data, issues


def _parse_ids(cell):
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip() != ""]


def validate(matching_path, candidate_path, test_dir, verbose=True):
    issues = []

    s1_ids, err = _load_entity_ids(test_dir, "test_source1.tsv")
    issues += err
    s2_ids, err = _load_entity_ids(test_dir, "test_source2.tsv")
    issues += err
    s3_ids, err = _load_entity_ids(test_dir, "test_source3.tsv")
    issues += err
    valid_target_ids = s2_ids | s3_ids

    if issues:
        _report(issues, verbose)
        return False

    matching, m_issues = _load_output_tsv(matching_path, "source1_entity_id", "matched_entity_ids")
    issues += m_issues
    candidate, c_issues = _load_output_tsv(candidate_path, "source1_entity_id", "candidate_entity_ids")
    issues += c_issues

    if matching is None or candidate is None:
        _report(issues, verbose)
        return False

    # Rule: every S1 test entity must appear exactly once
    missing_in_matching = s1_ids - set(matching.keys())
    extra_in_matching = set(matching.keys()) - s1_ids
    if missing_in_matching:
        issues.append(f"matching_results.tsv: missing {len(missing_in_matching)} required S1 entities, "
                       f"e.g. {sorted(missing_in_matching)[:5]}")
    if extra_in_matching:
        issues.append(f"matching_results.tsv: {len(extra_in_matching)} rows reference S1 ids not in "
                       f"test_source1.tsv, e.g. {sorted(extra_in_matching)[:5]}")

    missing_in_candidate = s1_ids - set(candidate.keys())
    extra_in_candidate = set(candidate.keys()) - s1_ids
    if missing_in_candidate:
        issues.append(f"candidate_pairs.tsv: missing {len(missing_in_candidate)} required S1 entities, "
                       f"e.g. {sorted(missing_in_candidate)[:5]}")
    if extra_in_candidate:
        issues.append(f"candidate_pairs.tsv: {len(extra_in_candidate)} rows reference S1 ids not in "
                       f"test_source1.tsv, e.g. {sorted(extra_in_candidate)[:5]}")

    # Rule: matched_entity_ids / candidate_entity_ids must only contain
    # existing S2/S3 test ids, no self-references to S1, no duplicates
    # within a single row's list.
    def _check_id_lists(data, label):
        n_bad_prefix, n_nonexistent, n_dup_within = 0, 0, 0
        examples = []
        for sid, cell in data.items():
            ids = _parse_ids(cell)
            seen = set()
            for cid in ids:
                if not (cid.startswith("S2-") or cid.startswith("S3-")):
                    n_bad_prefix += 1
                    examples.append((sid, cid, "bad_prefix"))
                elif cid not in valid_target_ids:
                    n_nonexistent += 1
                    examples.append((sid, cid, "nonexistent"))
                if cid in seen:
                    n_dup_within += 1
                    examples.append((sid, cid, "duplicate_in_row"))
                seen.add(cid)
        if n_bad_prefix:
            issues.append(f"{label}: {n_bad_prefix} ids are not S2-/S3- prefixed "
                           f"(self-matches to S1 or malformed ids)")
        if n_nonexistent:
            issues.append(f"{label}: {n_nonexistent} ids do not exist in the test set")
        if n_dup_within:
            issues.append(f"{label}: {n_dup_within} duplicate ids within a single row's list")
        if examples and verbose:
            for ex in examples[:5]:
                issues[-1] += f"\n    e.g. source1={ex[0]} id={ex[1]} ({ex[2]})"

    _check_id_lists(matching, "matching_results.tsv")
    _check_id_lists(candidate, "candidate_pairs.tsv")

    # Rule: every predicted match must be a subset of that entity's candidate set
    n_not_subset = 0
    subset_examples = []
    for sid, cell in matching.items():
        matched = set(_parse_ids(cell))
        cand = set(_parse_ids(candidate.get(sid, "")))
        missing = matched - cand
        if missing:
            n_not_subset += 1
            subset_examples.append((sid, sorted(missing)[:3]))
    if n_not_subset:
        issues.append(f"{n_not_subset} Source1 entities have matched_entity_ids not present in their own "
                       f"candidate_entity_ids (pipeline bug signal) e.g. {subset_examples[:5]}")

    _report(issues, verbose)
    return len(issues) == 0


def _report(issues, verbose):
    if not issues:
        print("PASS")
        return
    print(f"FAIL -- {len(issues)} issue(s) found:")
    for i, msg in enumerate(issues, 1):
        print(f"  {i}. {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()
    ok = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
