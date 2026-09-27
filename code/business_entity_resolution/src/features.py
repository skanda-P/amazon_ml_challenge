"""
features.py
-----------
Feature engineering for (S1, S2/S3) candidate pairs.

For every candidate pair (s1_record, candidate_record) we compute a
fixed-width numeric feature vector that a downstream classifier uses.

Feature groups
--------------
1. Name similarity   — Jaccard (token), Levenshtein ratio, TF-IDF cosine,
                       character 3-gram Jaccard
2. Address similarity — same four metrics
3. Country match     — binary flag
4. Name × Address cross — composite score (geometric mean of name & addr)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocess import normalise_address, normalise_name, tokenise

# ── String similarity helpers ────────────────────────────────────────────────

def _jaccard_token(a: str, b: str) -> float:
    ta, tb = set(tokenise(a)), set(tokenise(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalised edit distance ratio: 1 - (edit_dist / max_len)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dl = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dl, sub))
        prev = curr

    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

    na, nb = ngrams(a), ngrams(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


# ── Public API ───────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Name features
    "name_jaccard_token",
    "name_levenshtein_ratio",
    "name_char3gram_jaccard",
    # Address features
    "addr_jaccard_token",
    "addr_levenshtein_ratio",
    "addr_char3gram_jaccard",
    # Country
    "country_match",
    # Cross
    "name_addr_geomean",
]


def compute_features(
    s1_row: dict[str, Any],
    cand_row: dict[str, Any],
) -> np.ndarray:
    """
    Compute the feature vector for a single (S1, candidate) pair.

    Parameters
    ----------
    s1_row, cand_row : dicts with keys
        entity_id, business_name, business_address, country,
        _norm_name, _norm_addr   (pre-computed by preprocess)

    Returns
    -------
    np.ndarray of shape (len(FEATURE_NAMES),)
    """
    n1 = s1_row.get("_norm_name", normalise_name(s1_row.get("business_name", "")))
    n2 = cand_row.get("_norm_name", normalise_name(cand_row.get("business_name", "")))
    a1 = s1_row.get("_norm_addr", normalise_address(s1_row.get("business_address", "")))
    a2 = cand_row.get("_norm_addr", normalise_address(cand_row.get("business_address", "")))

    name_jac = _jaccard_token(n1, n2)
    name_lev = _levenshtein_ratio(n1, n2)
    name_ng  = _char_ngram_jaccard(n1, n2)

    addr_jac = _jaccard_token(a1, a2)
    addr_lev = _levenshtein_ratio(a1, a2)
    addr_ng  = _char_ngram_jaccard(a1, a2)

    country_match = float(
        str(s1_row.get("country", "")).lower() == str(cand_row.get("country", "")).lower()
    )

    name_score = (name_jac + name_lev + name_ng) / 3.0
    addr_score = (addr_jac + addr_lev + addr_ng) / 3.0
    cross = np.sqrt(name_score * addr_score) if name_score * addr_score > 0 else 0.0

    return np.array(
        [
            name_jac,
            name_lev,
            name_ng,
            addr_jac,
            addr_lev,
            addr_ng,
            country_match,
            cross,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(
    s1: pd.DataFrame,
    pool: pd.DataFrame,
    candidates: dict[str, list[str]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the full feature matrix for all candidate pairs.

    Parameters
    ----------
    s1    : Source 1 DataFrame (indexed by entity_id)
    pool  : Combined S2+S3 DataFrame (indexed by entity_id)
    candidates : output of blocking.generate_candidates

    Returns
    -------
    pairs_df : DataFrame with columns [source1_entity_id, candidate_entity_id]
    X        : np.ndarray of shape (n_pairs, n_features)
    """
    s1_lookup   = s1.set_index("entity_id").to_dict("index")
    pool_lookup = pool.set_index("entity_id").to_dict("index")

    rows_meta: list[dict] = []
    rows_feat: list[np.ndarray] = []

    for s1_id, cand_ids in candidates.items():
        s1_row = s1_lookup.get(s1_id, {})
        for c_id in cand_ids:
            c_row = pool_lookup.get(c_id, {})
            feat  = compute_features(s1_row, c_row)
            rows_meta.append({"source1_entity_id": s1_id, "candidate_entity_id": c_id})
            rows_feat.append(feat)

    pairs_df = pd.DataFrame(rows_meta)
    X        = np.vstack(rows_feat) if rows_feat else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return pairs_df, X
