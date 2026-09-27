"""
Stage D -- High-recall multi-pass candidate generation.

Candidate generation determines the hard recall ceiling of the whole
pipeline (Recall_final <= Recall_candidate), so several *independent*
retrieval channels are run and then fused with Reciprocal Rank Fusion
(RRF) rather than relying on any single blocking key.

Channels implemented:
  1. Exact normalized-key blocking   (name / country+postal / country+house+street / country+name+postal)
  2. Character n-gram inverted index (3-gram overlap counts)
  3. TF-IDF cosine retrieval          (name, address, and name+address combined)
  4. Address-structure retrieval      (postal code / house number / street-token / locality)

All candidates are restricted to the target pool (S2 + S3 records of the
SAME split, i.e. test candidates only ever come from the test S2/S3
files) so that `candidate_pairs.tsv` is deterministic and auditable, and
every predicted match is guaranteed to be a subset of the candidate set.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF smoothing constant


def _char_ngram_tokenizer(s: str, n: int = 3):
    s = s.replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


class TargetIndex:
    """Pre-built retrieval structures over the S2+S3 target pool for one
    split (train or test). Built once, reused for every S1 query."""

    def __init__(self, target_df: pd.DataFrame):
        self.df = target_df.reset_index(drop=True)
        self.n = len(self.df)
        self.entity_ids = self.df["entity_id"].values

        # --- exact-key inverted indices -------------------------------
        self.key_name = self._build_index(self.df["name_ws"])
        self.key_country_postal = self._build_index(
            self.df["country_norm"] + "||" + self.df["postal_code"]
        )
        self.key_country_house_street = self._build_index(
            self.df["country_norm"] + "||" + self.df["house_number"] + "||" +
            self.df["street_tokens"].map(lambda s: sorted(s)[0] if s else "")
        )
        self.key_country_name_postal = self._build_index(
            self.df["country_norm"] + "||" + self.df["name_ws"] + "||" + self.df["postal_code"]
        )

        # --- char n-gram inverted index (for overlap-count retrieval) -
        self.ngram_index = defaultdict(list)
        for i, ngset in enumerate(self.df["name_ngrams3"]):
            for g in ngset:
                self.ngram_index[g].append(i)

        # --- structural indices ---------------------------------------
        self.postal_index = self._build_index(self.df["postal_code"])
        self.house_index = self._build_index(self.df["house_number"])
        self.locality_index = self._build_index(self.df["locality_guess"])

        # --- TF-IDF retrieval matrices ----------------------------------
        self.tfidf_name_vec = TfidfVectorizer(
            tokenizer=lambda s: _char_ngram_tokenizer(s, 3), lowercase=False,
            token_pattern=None, min_df=1,
        )
        name_corpus = self.df["name_ws"].fillna("").tolist()
        self.tfidf_name_matrix = self.tfidf_name_vec.fit_transform(name_corpus) if self.n else None

        self.tfidf_addr_vec = TfidfVectorizer(
            tokenizer=lambda s: _char_ngram_tokenizer(s, 4), lowercase=False,
            token_pattern=None, min_df=1,
        )
        addr_corpus = self.df["addr_ws"].fillna("").tolist()
        self.tfidf_addr_matrix = self.tfidf_addr_vec.fit_transform(addr_corpus) if self.n else None

        combined_corpus = [f"{a} {b}" for a, b in zip(name_corpus, addr_corpus)]
        self.tfidf_combo_vec = TfidfVectorizer(
            tokenizer=lambda s: _char_ngram_tokenizer(s, 4), lowercase=False,
            token_pattern=None, min_df=1,
        )
        self.tfidf_combo_matrix = self.tfidf_combo_vec.fit_transform(combined_corpus) if self.n else None

    @staticmethod
    def _build_index(key_series) -> dict:
        idx = defaultdict(list)
        for i, k in enumerate(key_series):
            idx[k].append(i)
        # drop empty / all-separator keys -- these must never be treated as
        # a valid blocking match (an empty postal code shouldn't "match"
        # another empty postal code).
        for empty_key in ["", "||", "||||", "||||||"]:
            idx.pop(empty_key, None)
        idx = {k: v for k, v in idx.items() if k.strip("|") != ""}
        return idx

    def top_k_tfidf(self, query_vec, matrix, k: int):
        if matrix is None or matrix.shape[0] == 0:
            return [], []
        sims = (matrix @ query_vec.T).toarray().ravel()
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            order = part[np.argsort(-sims[part])]
        order = [i for i in order if sims[i] > 0]
        return order, [sims[i] for i in order]


def _rrf_fuse(ranklists: list[list[int]], k: int = RRF_K) -> dict:
    """ranklists: list of ordered candidate-index lists (best first) from
    each retrieval method. Returns {idx: rrf_score}."""
    scores = defaultdict(float)
    support = defaultdict(int)
    best_rank = defaultdict(lambda: 10 ** 9)
    for ranked in ranklists:
        for r, idx in enumerate(ranked):
            scores[idx] += 1.0 / (k + r + 1)
            support[idx] += 1
            best_rank[idx] = min(best_rank[idx], r + 1)
    return scores, support, best_rank


def generate_candidates_for_source1(
    s1_prepared: pd.DataFrame,
    target_index: TargetIndex,
    top_k_per_channel: int = 30,
    final_top_k: int = 25,
) -> dict:
    """
    Returns: {s1_entity_id: {
        'candidates': [target_idx, ...] (fused order, best first, len<=final_top_k),
        'rrf_score': {target_idx: score},
        'support': {target_idx: n_channels},
        'best_rank': {target_idx: rank},
    }}
    """
    results = {}
    ti = target_index

    for row in s1_prepared.itertuples(index=False):
        ranklists = []

        # Channel 1: exact keys (each exact-match bucket is one "rank-0" list)
        exact_hits = []
        for key_val, index in [
            (row.name_ws, ti.key_name),
            (row.country_norm + "||" + row.postal_code, ti.key_country_postal),
            (row.country_norm + "||" + row.house_number + "||" +
             (sorted(row.street_tokens)[0] if row.street_tokens else ""), ti.key_country_house_street),
            (row.country_norm + "||" + row.name_ws + "||" + row.postal_code, ti.key_country_name_postal),
        ]:
            if key_val in index:
                exact_hits.extend(index[key_val])
        if exact_hits:
            seen = []
            for h in exact_hits:
                if h not in seen:
                    seen.append(h)
            ranklists.append(seen[:top_k_per_channel])

        # Channel 2: character n-gram overlap counts
        counts = defaultdict(int)
        for g in row.name_ngrams3:
            for i in ti.ngram_index.get(g, []):
                counts[i] += 1
        if counts:
            ranked = sorted(counts.keys(), key=lambda i: -counts[i])[:top_k_per_channel]
            ranklists.append(ranked)

        # Channel 3: TF-IDF cosine retrieval (name / address / combined)
        if ti.n:
            for vec_attr, mat_attr in [
                ("tfidf_name_vec", "tfidf_name_matrix"),
                ("tfidf_addr_vec", "tfidf_addr_matrix"),
                ("tfidf_combo_vec", "tfidf_combo_matrix"),
            ]:
                vec = getattr(ti, vec_attr)
                mat = getattr(ti, mat_attr)
                if vec_attr == "tfidf_addr_vec":
                    q = vec.transform([row.addr_ws])
                elif vec_attr == "tfidf_combo_vec":
                    q = vec.transform([f"{row.name_ws} {row.addr_ws}"])
                else:
                    q = vec.transform([row.name_ws])
                order, _ = ti.top_k_tfidf(q, mat, top_k_per_channel)
                if order:
                    ranklists.append(list(order))

        # Channel 4: address-structure retrieval
        struct_hits = []
        for key_val, index in [
            (row.postal_code, ti.postal_index),
            (row.house_number, ti.house_index),
            (row.locality_guess, ti.locality_index),
        ]:
            if key_val in index:
                struct_hits.extend(index[key_val])
        if struct_hits:
            seen = []
            for h in struct_hits:
                if h not in seen:
                    seen.append(h)
            ranklists.append(seen[:top_k_per_channel])

        rrf_score, support, best_rank = _rrf_fuse(ranklists)
        fused_order = sorted(rrf_score.keys(), key=lambda i: -rrf_score[i])[:final_top_k]

        results[row.entity_id] = {
            "candidates": fused_order,
            "rrf_score": {i: rrf_score[i] for i in fused_order},
            "support": {i: support[i] for i in fused_order},
            "best_rank": {i: best_rank[i] for i in fused_order},
        }
    return results
