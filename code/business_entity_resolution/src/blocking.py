"""
blocking.py  (v2)
-----------------
Multi-strategy candidate generation.

Strategy 1 — TF-IDF / q-gram LSH (cheap, exact-overlap recall)
Strategy 2 — Embedding kNN (semantic / transliteration recall)
             Uses sentence-transformers; falls back to strategy 1 only
             if sentence-transformers is unavailable.

Union of both → candidate_pairs.tsv
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import normalise_address, normalise_name

logger = logging.getLogger(__name__)

# ── TF-IDF helpers ──────────────────────────────────────────────────────────

def _tfidf_topk(
    pool_texts: list[str],
    pool_ids: list[str],
    query_texts: list[str],
    query_ids: list[str],
    top_k: int,
    min_score: float,
    ngram_range: tuple[int, int] = (2, 4),
    batch_size: int = 256,
) -> dict[str, list[str]]:
    if not pool_texts or not query_texts:
        return {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                          min_df=1, sublinear_tf=True)
    pool_mat = vec.fit_transform(pool_texts)
    query_mat = vec.transform(query_texts)

    results: dict[str, list[str]] = defaultdict(list)
    pool_arr = np.array(pool_ids)

    for start in range(0, len(query_ids), batch_size):
        end = start + batch_size
        scores = cosine_similarity(query_mat[start:end], pool_mat)  # (batch, N_pool)
        for i, qid in enumerate(query_ids[start:end]):
            row = scores[i]
            k = min(top_k, len(row))
            idx = np.argpartition(row, -k)[-k:]
            idx = idx[row[idx] >= min_score]
            if len(idx):
                idx = idx[np.argsort(row[idx])[::-1]]
                results[qid].extend(pool_arr[idx].tolist())
    return results


# ── Embedding helpers ────────────────────────────────────────────────────────

def _build_embedding_index(
    pool_ids: list[str],
    pool_texts: list[str],
    model_name: str,
    batch_size: int = 64,
):
    """Encode pool records and build a FAISS flat-IP index. Returns (index, ids array)."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        return None, None

    logger.info(f"  Loading embedding model {model_name} …")
    model = SentenceTransformer(model_name)

    logger.info(f"  Encoding {len(pool_texts):,} pool records …")
    embs = model.encode(
        pool_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embs = embs.astype(np.float32)
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    return index, model


def _embedding_topk(
    s1_texts: list[str],
    s1_ids: list[str],
    pool_ids: list[str],
    index,
    model,
    top_k: int,
    min_score: float,
    batch_size: int = 64,
) -> dict[str, list[str]]:
    if index is None or model is None:
        return {}

    logger.info(f"  Encoding {len(s1_texts):,} query records …")
    q_embs = model.encode(
        s1_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    pool_arr = np.array(pool_ids)
    results: dict[str, list[str]] = defaultdict(list)

    scores, indices = index.search(q_embs, top_k)  # (N_s1, top_k)
    for i, qid in enumerate(s1_ids):
        for score, idx in zip(scores[i], indices[i]):
            if idx >= 0 and float(score) >= min_score:
                results[qid].append(pool_arr[int(idx)])
    return results


# ── Public API ───────────────────────────────────────────────────────────────

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k_name:  int   = 10,
    top_k_addr:  int   = 10,
    top_k_embed: int   = 15,
    name_min_score:  float = 0.25,
    addr_min_score:  float = 0.20,
    embed_min_score: float = 0.70,
    embed_model: str = "intfloat/multilingual-e5-large",
    use_embeddings: bool = True,
) -> dict[str, list[str]]:
    """
    Returns: { source1_entity_id → sorted unique list of candidate IDs (S2-*/S3-*) }
    """
    logger.info("Blocking: normalising …")
    for df in (s1, s2, s3):
        df["_norm_name"] = df["business_name"].map(normalise_name)
        df["_norm_addr"] = df["business_address"].map(normalise_address)
        df["_emb_text"]  = (
            "name: " + df["business_name"].fillna("") +
            " address: " + df["business_address"].fillna("")
        )

    pool = pd.concat([s2, s3], ignore_index=True).reset_index(drop=True)
    s1_ids   = list(s1["entity_id"])
    pool_ids = list(pool["entity_id"])

    candidates: dict[str, set[str]] = defaultdict(set)

    # ── TF-IDF pass 1: names ─────────────────────────────────────────────
    logger.info("Blocking pass 1/3: name TF-IDF …")
    for qid, cids in _tfidf_topk(
        list(pool["_norm_name"]), pool_ids,
        list(s1["_norm_name"]), s1_ids,
        top_k_name, name_min_score,
    ).items():
        candidates[qid].update(cids)

    # ── TF-IDF pass 2: addresses ─────────────────────────────────────────
    logger.info("Blocking pass 2/3: address TF-IDF …")
    for qid, cids in _tfidf_topk(
        list(pool["_norm_addr"]), pool_ids,
        list(s1["_norm_addr"]), s1_ids,
        top_k_addr, addr_min_score,
    ).items():
        candidates[qid].update(cids)

    # ── Embedding pass 3: semantic kNN ───────────────────────────────────
    if use_embeddings:
        logger.info(f"Blocking pass 3/3: embedding kNN ({embed_model}) …")
        index, model = _build_embedding_index(pool_ids, list(pool["_emb_text"]), embed_model)
        for qid, cids in _embedding_topk(
            list(s1["_emb_text"]), s1_ids,
            pool_ids, index, model,
            top_k_embed, embed_min_score,
        ).items():
            candidates[qid].update(cids)
    else:
        logger.info("Blocking: embedding pass skipped (use_embeddings=False).")

    result = {eid: sorted(candidates.get(eid, set())) for eid in s1_ids}
    total  = sum(len(v) for v in result.values())
    logger.info(
        f"Blocking complete: {len(s1_ids):,} S1 → {total:,} candidate pairs "
        f"(avg {total/max(len(s1_ids),1):.1f}/entity)"
    )
    return result


def store_embedding_model(
    pool: pd.DataFrame,
    model_name: str,
    save_dir: str,
) -> None:
    """Pre-encode pool once and save embeddings + FAISS index to disk."""
    import os
    import pickle
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(save_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    texts = list(pool["_emb_text"])
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(save_dir, "pool.faiss"))
    with open(os.path.join(save_dir, "pool_ids.pkl"), "wb") as f:
        pickle.dump(list(pool["entity_id"]), f)
    model.save(os.path.join(save_dir, "encoder"))
    logger.info(f"Embedding index saved to {save_dir}")
