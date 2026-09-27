"""
Stage E -- Pairwise feature engineering.
Stage F -- Rarity-aware (IDF-weighted) similarity.

Builds a rich, source-pair-aware feature vector x_ab for every candidate
pair (a in Source1, b in Source2/Source3). Feature computation is kept
vectorizable / cache-friendly: normalized views are precomputed once per
record (see normalization.prepare_dataframe) and reused across every
pair that record participates in.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler

FEATURE_NAMES = None  # populated on first call, exported for consumers


def build_idf(token_sets: list) -> dict:
    """IDF(t) = log(N / (DF(t) + 1)) over a corpus of token sets (names or
    address tokens), used for rarity-aware weighted Jaccard (Stage F)."""
    N = len(token_sets)
    df = Counter()
    for toks in token_sets:
        for t in toks:
            df[t] += 1
    return {t: math.log(N / (c + 1)) for t, c in df.items()}, N


def _idf_lookup(idf_map: dict, default_idf: float, token: str) -> float:
    return idf_map.get(token, default_idf)


def weighted_jaccard(a: frozenset, b: frozenset, idf_map: dict, default_idf: float) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    inter = a & b
    denom = sum(_idf_lookup(idf_map, default_idf, t) for t in union)
    if denom <= 0:
        return 0.0
    numer = sum(_idf_lookup(idf_map, default_idf, t) for t in inter)
    return numer / denom


def rare_token_overlap(a: frozenset, b: frozenset, idf_map: dict, default_idf: float, thresh: float) -> float:
    """Sum of IDF weight of *shared* tokens whose IDF exceeds `thresh`
    (i.e. genuinely rare/discriminating tokens), a stronger signal than
    a shared common word like 'restaurant' or 'the'."""
    inter = a & b
    return sum(_idf_lookup(idf_map, default_idf, t) for t in inter if _idf_lookup(idf_map, default_idf, t) >= thresh)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def token_sort_ratio(tokens_a: tuple, tokens_b: tuple) -> float:
    sa = " ".join(sorted(tokens_a))
    sb = " ".join(sorted(tokens_b))
    if not sa and not sb:
        return 1.0
    return Levenshtein.normalized_similarity(sa, sb)


def ngram_cosine(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))


class FeatureBuilder:
    """Holds fitted IDF tables + default IDFs; exposes `build(pair)` that
    returns a dict of named features for one (s1_row, target_row) pair."""

    def __init__(self, name_idf: dict, name_n: int, addr_idf: dict, addr_n: int, rare_token_percentile: float = 0.85):
        self.name_idf = name_idf
        self.name_default_idf = math.log(max(name_n, 1) / 1.0)
        self.addr_idf = addr_idf
        self.addr_default_idf = math.log(max(addr_n, 1) / 1.0)
        vals = sorted(name_idf.values())
        self.rare_thresh = vals[int(len(vals) * rare_token_percentile)] if vals else 0.0

    def build(self, s1, tgt, rrf_score: float, support: int, best_rank: int, n_channels_total: int, llr: float = 0.0) -> dict:
        """s1, tgt: namedtuples/rows from the `prepare_dataframe` output."""
        name_a, name_b = s1.name_ws, tgt.name_ws
        toks_a, toks_b = s1.name_tokens, tgt.name_tokens
        addr_a, addr_b = s1.addr_ws, tgt.addr_ws
        atoks_a, atoks_b = s1.addr_tokens, tgt.addr_tokens

        name_exact = float(name_a == name_b and name_a != "")
        name_nosuffix_exact = float(s1.name_nosuffix == tgt.name_nosuffix and s1.name_nosuffix != "")
        name_lev = Levenshtein.normalized_similarity(name_a, name_b) if (name_a or name_b) else 1.0
        name_jw = JaroWinkler.normalized_similarity(name_a, name_b) if (name_a or name_b) else 1.0
        name_jac = jaccard(toks_a, toks_b)
        name_wjac = weighted_jaccard(toks_a, toks_b, self.name_idf, self.name_default_idf)
        name_tsr = token_sort_ratio(toks_a, toks_b)
        name_ngram_cos = ngram_cosine(s1.name_ngrams3, tgt.name_ngrams3)
        name_len_diff = abs(len(name_a) - len(name_b))
        name_tokcount_diff = abs(len(toks_a) - len(toks_b))
        rare_overlap = rare_token_overlap(toks_a, toks_b, self.name_idf, self.name_default_idf, self.rare_thresh)
        alnum_exact = float(s1.name_alnum == tgt.name_alnum and s1.name_alnum != "")

        addr_exact = float(addr_a == addr_b and addr_a != "")
        addr_lev = Levenshtein.normalized_similarity(addr_a, addr_b) if (addr_a or addr_b) else 1.0
        addr_jw = JaroWinkler.normalized_similarity(addr_a, addr_b) if (addr_a or addr_b) else 1.0
        addr_jac = jaccard(atoks_a, atoks_b)
        addr_ngram_cos = ngram_cosine(s1.addr_ngrams3, tgt.addr_ngrams3)
        house_match = float(bool(s1.house_number) and s1.house_number == tgt.house_number)
        postal_match = float(bool(s1.postal_code) and s1.postal_code == tgt.postal_code)
        postal_prefix_match = float(
            bool(s1.postal_code) and bool(tgt.postal_code) and
            s1.postal_code[:3] == tgt.postal_code[:3]
        )
        street_jac = jaccard(s1.street_tokens, tgt.street_tokens)
        locality_match = float(bool(s1.locality_guess) and s1.locality_guess == tgt.locality_guess)
        numeric_overlap = jaccard(s1.numeric_tokens, tgt.numeric_tokens)
        n_common_addr_components = sum([house_match, postal_match, locality_match, float(street_jac > 0.5)])
        missing_postal_either = float(not s1.postal_code or not tgt.postal_code)
        missing_house_either = float(not s1.house_number or not tgt.house_number)

        country_exact = float(bool(s1.country_norm) and s1.country_norm == tgt.country_norm)
        country_pair = f"{s1.country_norm}|{tgt.country_norm}"

        source_pair_type = 1.0 if tgt.source == "source2" else 0.0  # 1=S1-S2, 0=S1-S3

        feats = {
            "name_exact": name_exact,
            "name_nosuffix_exact": name_nosuffix_exact,
            "name_alnum_exact": alnum_exact,
            "name_levenshtein_sim": name_lev,
            "name_jarowinkler_sim": name_jw,
            "name_jaccard": name_jac,
            "name_weighted_jaccard": name_wjac,
            "name_token_sort_ratio": name_tsr,
            "name_ngram_cosine": name_ngram_cos,
            "name_len_diff": float(name_len_diff),
            "name_tokcount_diff": float(name_tokcount_diff),
            "name_rare_token_overlap": rare_overlap,
            "addr_exact": addr_exact,
            "addr_levenshtein_sim": addr_lev,
            "addr_jarowinkler_sim": addr_jw,
            "addr_jaccard": addr_jac,
            "addr_ngram_cosine": addr_ngram_cos,
            "house_number_match": house_match,
            "postal_code_match": postal_match,
            "postal_prefix_match": postal_prefix_match,
            "street_jaccard": street_jac,
            "locality_match": locality_match,
            "numeric_token_overlap": numeric_overlap,
            "n_common_addr_components": n_common_addr_components,
            "missing_postal_either": missing_postal_either,
            "missing_house_either": missing_house_either,
            "country_exact": country_exact,
            "retrieval_rrf_score": float(rrf_score),
            "retrieval_support_count": float(support),
            "retrieval_best_rank": float(best_rank),
            "retrieval_support_ratio": float(support) / max(n_channels_total, 1),
            "source_pair_is_s2": source_pair_type,
            "llr": float(llr),
            # interaction terms
            "int_namesim_housematch": name_lev * house_match,
            "int_namesim_postalmatch": name_lev * postal_match,
            "int_addrsim_namesim": addr_lev * name_lev,
            "int_raretoken_addrsim": rare_overlap * addr_lev,
            "int_wjac_countrymatch": name_wjac * country_exact,
        }
        feats["_country_pair"] = country_pair  # kept out of numeric matrix, useful for diagnostics
        return feats


NUMERIC_FEATURE_NAMES = [
    "name_exact", "name_nosuffix_exact", "name_alnum_exact", "name_levenshtein_sim",
    "name_jarowinkler_sim", "name_jaccard", "name_weighted_jaccard", "name_token_sort_ratio",
    "name_ngram_cosine", "name_len_diff", "name_tokcount_diff", "name_rare_token_overlap",
    "addr_exact", "addr_levenshtein_sim", "addr_jarowinkler_sim", "addr_jaccard",
    "addr_ngram_cosine", "house_number_match", "postal_code_match", "postal_prefix_match",
    "street_jaccard", "locality_match", "numeric_token_overlap", "n_common_addr_components",
    "missing_postal_either", "missing_house_either", "country_exact", "retrieval_rrf_score",
    "retrieval_support_count", "retrieval_best_rank", "retrieval_support_ratio",
    "source_pair_is_s2", "llr", "int_namesim_housematch", "int_namesim_postalmatch",
    "int_addrsim_namesim", "int_raretoken_addrsim", "int_wjac_countrymatch",
    "triangulation_score",
]
