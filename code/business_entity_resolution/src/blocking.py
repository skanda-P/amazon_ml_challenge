"""
blocking.py
-----------
Candidate generation (blocking) stage.

Goal: for each Source 1 entity, produce a small set of plausible
      Source 2 / Source 3 candidates so the expensive ML scorer only
      runs on O(k) pairs rather than O(N²).

Strategy (multi-pass union):
  1. TF-IDF cosine similarity on normalised names (top-k per S1 entity)
  2. TF-IDF cosine similarity on normalised addresses (top-k per S1 entity)
  3. Shared token prefix / 3-gram overlap fallback

Output: dict mapping source1_entity_id → list[candidate_entity_id]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import normalise_address, normalise_name

logger = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────────

def _ngrams(text: str, n: int = 3) -> set[str]:
    padded = f"_{text}_"
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def _tfidf_topk_candidates(
    source_texts: list[str],
    source_ids: list[str],
    query_texts: list[str],
    query_ids: list[str],
    top_k: int = 10,
    min_score: float = 0.2,
) -> dict[str, list[str]]:
    """
    Fit TF-IDF on source_texts, then for each query retrieve top-k
    source candidates by cosine similarity.

    Returns: { query_id: [source_id, ...] }
    """
    if not source_texts or not query_texts:
        return {}

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
    )
    source_matrix = vectorizer.fit_transform(source_texts)
    query_matrix = vectorizer.transform(query_texts)

    results: dict[str, list[str]] = defaultdict(list)
    batch_size = 512  # process queries in batches to keep memory low

    for start in range(0, len(query_ids), batch_size):
        end = start + batch_size
        q_batch = query_matrix[start:end]
        scores = cosine_similarity(q_batch, source_matrix)  # (batch, |source|)

        for i, q_id in enumerate(query_ids[start:end]):
            row = scores[i]
            # get indices of top_k scores above threshold
            top_idx = np.argpartition(row, -min(top_k, len(row)))[-top_k:]
            top_idx = top_idx[row[top_idx] >= min_score]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            results[q_id].extend(source_ids[j] for j in top_idx)

    return results


# ── public API ───────────────────────────────────────────────────────────────

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k_name: int = 10,
    top_k_addr: int = 10,
    name_min_score: float = 0.25,
    addr_min_score: float = 0.20,
) -> dict[str, list[str]]:
    """
    Generate candidate (S2 ∪ S3) matches for every S1 entity.

    Parameters
    ----------
    s1, s2, s3 : DataFrames with columns [entity_id, business_name, business_address, country]

    Returns
    -------
    dict: source1_entity_id → sorted list of unique candidate_entity_ids (S2-* or S3-*)
    """
    logger.info("Blocking: normalising texts …")

    # Normalise
    for df in (s1, s2, s3):
        df["_norm_name"] = df["business_name"].map(normalise_name)
        df["_norm_addr"] = df["business_address"].map(normalise_address)

    # Pool S2 + S3 as "source pool"
    pool = pd.concat([s2, s3], ignore_index=True)

    s1_ids = list(s1["entity_id"])
    pool_ids = list(pool["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)

    # ── Pass 1: name TF-IDF ────────────────────────────────────────────────
    logger.info("Blocking pass 1/2: name TF-IDF …")
    name_hits = _tfidf_topk_candidates(
        source_texts=list(pool["_norm_name"]),
        source_ids=pool_ids,
        query_texts=list(s1["_norm_name"]),
        query_ids=s1_ids,
        top_k=top_k_name,
        min_score=name_min_score,
    )
    for qid, cids in name_hits.items():
        candidates[qid].update(cids)

    # ── Pass 2: address TF-IDF ────────────────────────────────────────────
    logger.info("Blocking pass 2/2: address TF-IDF …")
    addr_hits = _tfidf_topk_candidates(
        source_texts=list(pool["_norm_addr"]),
        source_ids=pool_ids,
        query_texts=list(s1["_norm_addr"]),
        query_ids=s1_ids,
        top_k=top_k_addr,
        min_score=addr_min_score,
    )
    for qid, cids in addr_hits.items():
        candidates[qid].update(cids)

    # Ensure every S1 entity has an entry (even if empty)
    result = {eid: sorted(candidates.get(eid, set())) for eid in s1_ids}

    total_pairs = sum(len(v) for v in result.values())
    logger.info(
        f"Blocking complete: {len(s1_ids)} S1 entities → {total_pairs} candidate pairs"
        f" (avg {total_pairs / max(len(s1_ids), 1):.1f} per entity)"
    )
    return result
