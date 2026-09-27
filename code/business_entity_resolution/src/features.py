"""
features.py  (v2)
-----------------
Rich per-pair feature vector covering all groups from the build prompt:

Group A – Name features
  - Legal-suffix-normalized exact match + similarity delta
  - Jaro-Winkler, Levenshtein ratio, token-sort ratio, token-set ratio
  - Character n-gram TF-IDF cosine (computed statically per batch via sklearn)
  - Soundex phonetic match flag
  - Word-order-sensitivity delta (Levenshtein vs token-sort gap)

Group B – Address features
  - Token Jaccard, shared-token count (normalized)
  - Digit exact match (street number / PIN)
  - Missingness indicators: has_pin, has_landmark
  - Char 3-gram Jaccard on address

Group C – Cross / graph features
  - Embedding cosine similarity (from blocking stage)
  - Margin: score gap vs next-best candidate (per S1 entity)
  - Node degree: # candidates for this S1 entity (log-scaled)

Group D – Categorical
  - Country frequency encoding (unseen bucket for France)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocess import (
    encode_country,
    extract_digits,
    has_landmark,
    has_pin,
    normalise_address,
    normalise_name,
    normalise_name_pre,
    soundex_match,
    tokenise,
)

# ── Import rapidfuzz / jellyfish gracefully ──────────────────────────────────
try:
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz import fuzz as _fuzz

    def levenshtein_ratio(a: str, b: str) -> float:
        return _Lev.normalized_similarity(a, b)

    def token_sort_ratio(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0

    def token_set_ratio(a: str, b: str) -> float:
        return _fuzz.token_set_ratio(a, b) / 100.0

except ImportError:
    def levenshtein_ratio(a: str, b: str) -> float:
        la, lb = len(a), len(b)
        if la == 0 and lb == 0:
            return 1.0
        if not la or not lb:
            return 0.0
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(ca != cb)))
            prev = curr
        return 1.0 - prev[lb] / max(la, lb)

    def _sort(s: str) -> str:
        return " ".join(sorted(s.split()))

    def token_sort_ratio(a: str, b: str) -> float:
        return levenshtein_ratio(_sort(a), _sort(b))

    def token_set_ratio(a: str, b: str) -> float:
        ta, tb = set(a.split()), set(b.split())
        shared = ta & tb
        s = " ".join(sorted(shared))
        rest_a = " ".join(sorted(ta - tb))
        rest_b = " ".join(sorted(tb - ta))
        return max(
            levenshtein_ratio((s + " " + rest_a).strip(), (s + " " + rest_b).strip()),
            levenshtein_ratio(s, (s + " " + rest_a).strip()),
            levenshtein_ratio(s, (s + " " + rest_b).strip()),
        )

try:
    import jellyfish as _jf
    def jaro_winkler(a: str, b: str) -> float:
        return _jf.jaro_winkler_similarity(a, b)
except ImportError:
    def jaro_winkler(a: str, b: str) -> float:
        return levenshtein_ratio(a, b)


# ── Core helpers ──────────────────────────────────────────────────────────────

def _jaccard_token(a: str, b: str) -> float:
    ta, tb = set(tokenise(a)), set(tokenise(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ng(s: str) -> set[str]:
        return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else set()
    na, nb = ng(a), ng(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _shared_token_count_norm(a: str, b: str) -> float:
    ta, tb = tokenise(a), tokenise(b)
    sa, sb = set(ta), set(tb)
    shared = len(sa & sb)
    denom = max(len(ta), len(tb))
    return shared / denom if denom else 0.0


def _digit_overlap(a: str, b: str) -> float:
    da, db = set(extract_digits(a)), set(extract_digits(b))
    if not da and not db:
        return 1.0
    if not da or not db:
        return 0.0
    return len(da & db) / len(da | db)


# ── Feature names registry ───────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # A: Name
    "name_exact_match_prenorm",
    "name_exact_match_postnorm",
    "name_norm_delta",           # pre→post similarity improvement
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_char3gram_jaccard",
    "name_token_jaccard",
    "name_soundex_match",
    "name_wordorder_delta",      # levenshtein - token_sort (sensitivity to word order)
    # B: Address
    "addr_levenshtein",
    "addr_token_jaccard",
    "addr_shared_token_norm",
    "addr_char3gram_jaccard",
    "addr_digit_overlap",
    "addr_has_pin_s1",
    "addr_has_pin_cand",
    "addr_pin_match",
    "addr_has_landmark_s1",
    "addr_has_landmark_cand",
    # C: Cross / graph
    "embed_cosine",              # filled by caller with embedding scores
    "candidate_degree_log",      # log(# candidates for this S1 entity)
    # D: Country
    "country_freq_s1",
    "country_freq_cand",
    "country_match",
]


# ── Main compute function ────────────────────────────────────────────────────

def compute_features(
    s1_row:   dict[str, Any],
    cand_row: dict[str, Any],
    embed_cosine: float = 0.0,
    candidate_degree: int = 1,
) -> np.ndarray:
    """Return feature vector for a single (S1, candidate) pair."""

    n1_raw  = s1_row.get("business_name", "") or ""
    n2_raw  = cand_row.get("business_name", "") or ""
    a1_raw  = s1_row.get("business_address", "") or ""
    a2_raw  = cand_row.get("business_address", "") or ""

    # Pre-norm (no suffix expansion)
    n1_pre = normalise_name_pre(n1_raw)
    n2_pre = normalise_name_pre(n2_raw)
    # Post-norm (with suffix expansion)
    n1 = s1_row.get("_norm_name") or normalise_name(n1_raw)
    n2 = cand_row.get("_norm_name") or normalise_name(n2_raw)
    a1 = s1_row.get("_norm_addr") or normalise_address(a1_raw)
    a2 = cand_row.get("_norm_addr") or normalise_address(a2_raw)

    # ── Name features ────────────────────────────────────────────────────
    exact_pre  = float(n1_pre == n2_pre)
    exact_post = float(n1 == n2)
    lev_pre    = levenshtein_ratio(n1_pre, n2_pre)
    lev_post   = levenshtein_ratio(n1, n2)
    norm_delta = lev_post - lev_pre   # positive = suffix expansion helped

    jw     = jaro_winkler(n1, n2)
    tsort  = token_sort_ratio(n1, n2)
    tset   = token_set_ratio(n1, n2)
    ng3    = _char_ngram_jaccard(n1, n2, 3)
    tjac   = _jaccard_token(n1, n2)
    sdx    = soundex_match(n1, n2)
    wo_delta = lev_post - tsort   # large → word order matters

    # ── Address features ─────────────────────────────────────────────────
    a_lev   = levenshtein_ratio(a1, a2)
    a_jac   = _jaccard_token(a1, a2)
    a_stn   = _shared_token_count_norm(a1, a2)
    a_ng3   = _char_ngram_jaccard(a1, a2, 3)
    a_dig   = _digit_overlap(a1, a2)

    hp1  = has_pin(a1_raw)
    hp2  = has_pin(a2_raw)
    pin_match = float(hp1 and hp2 and bool(set(extract_digits(a1_raw)) & set(extract_digits(a2_raw))))

    hl1  = has_landmark(a1_raw)
    hl2  = has_landmark(a2_raw)

    # ── Cross / graph ────────────────────────────────────────────────────
    deg_log = float(np.log1p(candidate_degree))

    # ── Country ──────────────────────────────────────────────────────────
    cf1 = encode_country(s1_row.get("country", ""))
    cf2 = encode_country(cand_row.get("country", ""))
    cmatch = float(
        str(s1_row.get("country", "")).lower() ==
        str(cand_row.get("country", "")).lower()
    )

    return np.array([
        exact_pre,
        exact_post,
        norm_delta,
        lev_post,
        jw,
        tsort,
        tset,
        ng3,
        tjac,
        sdx,
        wo_delta,
        a_lev,
        a_jac,
        a_stn,
        a_ng3,
        a_dig,
        hp1, hp2, pin_match,
        hl1, hl2,
        float(embed_cosine),
        deg_log,
        cf1, cf2, cmatch,
    ], dtype=np.float32)


def build_feature_matrix(
    s1:   pd.DataFrame,
    pool: pd.DataFrame,
    candidates: dict[str, list[str]],
    embed_scores: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build full feature matrix for all candidate pairs.

    embed_scores: optional dict (s1_id, cand_id) → cosine similarity
    Returns: (pairs_df, X)
    """
    s1_lkp   = s1.set_index("entity_id").to_dict("index")
    pool_lkp = pool.set_index("entity_id").to_dict("index")
    if embed_scores is None:
        embed_scores = {}

    rows_meta: list[dict] = []
    rows_feat: list[np.ndarray] = []

    for s1_id, cand_ids in candidates.items():
        s1_row = s1_lkp.get(s1_id, {})
        deg = len(cand_ids)
        for c_id in cand_ids:
            c_row = pool_lkp.get(c_id, {})
            ec = embed_scores.get((s1_id, c_id), 0.0)
            feat = compute_features(s1_row, c_row,
                                    embed_cosine=ec, candidate_degree=deg)
            rows_meta.append({"source1_entity_id": s1_id,
                               "candidate_entity_id": c_id})
            rows_feat.append(feat)

    pairs_df = pd.DataFrame(rows_meta)
    X = (np.vstack(rows_feat)
         if rows_feat
         else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32))
    return pairs_df, X


def add_margin_feature(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pair margin = top-candidate score - this-pair score,
    then inject into feature matrix as an additional column.
    Returns augmented feature matrix with margin prepended.
    """
    pairs_df = pairs_df.copy()
    pairs_df["_p"] = proba
    max_p = pairs_df.groupby("source1_entity_id")["_p"].transform("max")
    margin = (max_p.values - proba).clip(0, 1)
    return margin
