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
import os
import pickle
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]
    _HAS_SENTENCE_TRANSFORMERS = False

from src.preprocess import fast_series_normalise, normalise_address, normalise_name

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
    max_features: int = 150000,
    max_df: float = 0.25,
) -> dict[str, list[str]]:
    """
    High-performance sparse TF-IDF candidate retriever.
    - Sets max_df=0.25 to prune ubiquitous, low-information character n-grams that cause
      combinatorial memory blowups across multi-million entity pools.
    - Fast batch_size=256 delivers ~145 queries/sec while keeping peak memory under ~60 MB.
    - Emits live verbatim progress updates every 5 seconds with rate and ETA.
    """
    if not pool_texts or not query_texts:
        return {}
    min_df_val = 2 if len(pool_texts) >= 1000 else 1
    max_df_val = max_df if len(pool_texts) >= 1000 else 1.0

    if batch_size is None or batch_size <= 0:
        batch_size = 256

    t_vec = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df_val,
        max_df=max_df_val,
        max_features=max_features,
        sublinear_tf=True,
    )
    pool_mat = vec.fit_transform(pool_texts)
    query_mat = vec.transform(query_texts)
    pool_mat_T = pool_mat.T.tocsc()
    vec_time = time.time() - t_vec
    logger.info(
        f"    TF-IDF Vectorized in {vec_time:.1f}s | Vocab: {len(vec.vocabulary_):,} features | "
        f"Batch size: {batch_size} queries"
    )

    results: dict[str, list[str]] = defaultdict(list)
    pool_arr = np.array(pool_ids)
    total_q = len(query_ids)
    t_start = time.time()
    last_log_time = t_start

    for start in range(0, total_q, batch_size):
        end = min(start + batch_size, total_q)
        sim_batch = query_mat[start:end].dot(pool_mat_T).tocsr()
        data = sim_batch.data
        indices = sim_batch.indices
        indptr = sim_batch.indptr

        for i, qid in enumerate(query_ids[start:end]):
            p0, p1 = indptr[i], indptr[i + 1]
            if p1 == p0:
                continue
            r_data = data[p0:p1]
            r_idx = indices[p0:p1]
            mask = r_data >= min_score
            if not np.any(mask):
                continue
            v_data = r_data[mask]
            v_idx = r_idx[mask]
            k = min(top_k, len(v_data))
            if len(v_data) > k:
                sub = np.argpartition(v_data, -k)[-k:]
                sub = sub[np.argsort(v_data[sub])[::-1]]
                results[qid].extend(pool_arr[v_idx[sub]].tolist())
            else:
                sub = np.argsort(v_data)[::-1]
                results[qid].extend(pool_arr[v_idx[sub]].tolist())

        now = time.time()
        if now - last_log_time >= 5.0 or end >= total_q:
            elapsed = now - t_start
            qps = end / max(elapsed, 0.001)
            rem = total_q - end
            eta_sec = rem / max(qps, 0.001)
            eta_m, eta_s = divmod(int(eta_sec), 60)
            logger.info(
                f"    Progress: {end:,}/{total_q:,} queries ({100*end/total_q:5.1f}%) | "
                f"{qps:5.1f} q/s | Elapsed: {int(elapsed)}s | ETA: {eta_m}m {eta_s:02d}s"
            )
            last_log_time = now

    return results


def _is_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# ── Embedding helpers ────────────────────────────────────────────────────────

def _build_embedding_index(
    pool_ids: list[str],
    pool_texts: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str = "auto",
):
    """Encode pool records and build a FAISS flat-IP index. Returns (index, model)."""
    if not _HAS_FAISS or not _HAS_SENTENCE_TRANSFORMERS or faiss is None or SentenceTransformer is None:
        logger.warning(
            "Embedding blocking requires 'sentence-transformers' and 'faiss-cpu'. "
            "Skipping embedding index."
        )
        return None, None

    # Resolve device with GPU preference and CPU backup
    target_device = "cpu"
    if device == "cuda" or (device == "auto" and _is_cuda_available()):
        target_device = "cuda"

    logger.info(f"  Loading embedding model {model_name} on {target_device} …")
    try:
        model = SentenceTransformer(model_name, device=target_device)
    except Exception as e:
        logger.warning(f"Failed to load {model_name} on {target_device}: {e}. Falling back to CPU.")
        target_device = "cpu"
        model = SentenceTransformer(model_name, device="cpu")

    logger.info(f"  Encoding {len(pool_texts):,} pool records on {target_device} …")
    try:
        embs = model.encode(
            pool_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=target_device,
        )
    except Exception as e:
        if target_device == "cuda":
            logger.warning(f"CUDA encoding failed ({e}); clearing cache and falling back to CPU.")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            model = model.to("cpu")
            target_device = "cpu"
            embs = model.encode(
                pool_texts,
                batch_size=max(16, batch_size // 2),
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                device="cpu",
            )
        else:
            raise e

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
    device: str = "auto",
) -> dict[str, list[str]]:
    if index is None or model is None:
        return {}

    target_device = "cpu"
    if device == "cuda" or (device == "auto" and _is_cuda_available()):
        target_device = "cuda"

    logger.info(f"  Encoding {len(s1_texts):,} query records on {target_device} …")
    try:
        q_embs = model.encode(
            s1_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=target_device,
        ).astype(np.float32)
    except Exception as e:
        if target_device == "cuda":
            logger.warning(f"CUDA query encoding failed ({e}); falling back to CPU.")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            model = model.to("cpu")
            q_embs = model.encode(
                s1_texts,
                batch_size=max(16, batch_size // 2),
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                device="cpu",
            ).astype(np.float32)
        else:
            raise e

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
    device: str = "auto",
    batch_size: int = 256,
    **kwargs: Any,
) -> dict[str, list[str]]:
    """
    Returns: { source1_entity_id → sorted unique list of candidate IDs (S2-*/S3-*) }
    """
    t_blocking_start = time.time()
    logger.info("Blocking: normalising names and addresses …")
    for df in (s1, s2, s3):
        if "_norm_name" not in df.columns:
            df["_norm_name"] = fast_series_normalise(df["business_name"], normalise_name)
        if "_norm_addr" not in df.columns:
            df["_norm_addr"] = fast_series_normalise(df["business_address"], normalise_address)
        if use_embeddings and "_emb_text" not in df.columns:
            df["_emb_text"]  = (
                "name: " + df["business_name"].fillna("") +
                " address: " + df["business_address"].fillna("")
            )

    pool = pd.concat([s2, s3], ignore_index=True).reset_index(drop=True)
    s1_ids = list(s1["entity_id"])
    candidates: dict[str, set[str]] = defaultdict(set)

    # Country-aware blocking (100% of matches are within same country)
    has_country = "country" in s1.columns and "country" in pool.columns
    if has_country:
        s1_countries = [c for c in s1["country"].dropna().unique() if str(c).strip()]
        country_groups = [
            (c, s1[s1["country"] == c], pool[pool["country"] == c])
            for c in s1_countries
        ]
        s1_other = s1[~s1["country"].isin(s1_countries)]
        if not s1_other.empty:
            country_groups.append(("OTHER", s1_other, pool))
    else:
        country_groups = [("ALL", s1, pool)]

    for c_name, s1_grp, pool_grp in country_groups:
        if s1_grp.empty or pool_grp.empty:
            continue
        c_s1_ids = list(s1_grp["entity_id"])
        c_pool_ids = list(pool_grp["entity_id"])
        logger.info(
            f"Blocking [{c_name}]: {len(c_s1_ids):,} S1 queries vs {len(c_pool_ids):,} candidate pool"
        )

        # ── TF-IDF pass 1: names ─────────────────────────────────────────────
        t_pass1 = time.time()
        logger.info(f"  Pass 1/3 ({c_name}): name TF-IDF on {len(c_s1_ids):,} queries vs {len(c_pool_ids):,} candidate pool (batch_size={batch_size}) …")
        for qid, cids in _tfidf_topk(
            list(pool_grp["_norm_name"]), c_pool_ids,
            list(s1_grp["_norm_name"]), c_s1_ids,
            top_k_name, name_min_score,
            ngram_range=(2, 4),
            batch_size=batch_size,
        ).items():
            candidates[qid].update(cids)
        logger.info(f"  ✓ Pass 1/3 ({c_name}) complete in {time.time() - t_pass1:.1f}s")

        # ── TF-IDF pass 2: addresses ─────────────────────────────────────────
        t_pass2 = time.time()
        logger.info(f"  Pass 2/3 ({c_name}): address TF-IDF on {len(c_s1_ids):,} queries vs {len(c_pool_ids):,} candidate pool (batch_size={batch_size}) …")
        for qid, cids in _tfidf_topk(
            list(pool_grp["_norm_addr"]), c_pool_ids,
            list(s1_grp["_norm_addr"]), c_s1_ids,
            top_k_addr, addr_min_score,
            ngram_range=(2, 4),
            batch_size=batch_size,
        ).items():
            candidates[qid].update(cids)
        logger.info(f"  ✓ Pass 2/3 ({c_name}) complete in {time.time() - t_pass2:.1f}s")

        # ── Embedding pass 3: semantic kNN ───────────────────────────────────
        if use_embeddings:
            t_pass3 = time.time()
            logger.info(f"  Pass 3/3 ({c_name}): embedding kNN ({embed_model}) …")
            index, model = _build_embedding_index(
                c_pool_ids, list(pool_grp["_emb_text"]), embed_model, device=device
            )
            for qid, cids in _embedding_topk(
                list(s1_grp["_emb_text"]), c_s1_ids,
                c_pool_ids, index, model,
                top_k_embed, embed_min_score,
                device=device,
            ).items():
                candidates[qid].update(cids)
            logger.info(f"  ✓ Pass 3/3 ({c_name}) complete in {time.time() - t_pass3:.1f}s")

    if not use_embeddings:
        logger.info("Blocking: embedding pass skipped (use_embeddings=False).")

    result = {eid: sorted(candidates.get(eid, set())) for eid in s1_ids}
    total  = sum(len(v) for v in result.values())
    matched_s1 = sum(1 for v in result.values() if len(v) > 0)
    logger.info(
        f"Blocking complete in {time.time() - t_blocking_start:.1f}s: {matched_s1:,}/{len(s1_ids):,} S1 entities have candidates "
        f"({total:,} total candidate pairs, avg {total/max(len(s1_ids),1):.1f}/entity)"
    )
    return result


def store_embedding_model(
    pool: pd.DataFrame,
    model_name: str,
    save_dir: str,
    device: str = "auto",
) -> None:
    """Pre-encode pool once and save embeddings + FAISS index to disk."""
    if not _HAS_FAISS or not _HAS_SENTENCE_TRANSFORMERS or faiss is None or SentenceTransformer is None:
        raise ImportError(
            "store_embedding_model requires 'sentence-transformers' and 'faiss-cpu' to be installed. "
            "Please install them via: pip install faiss-cpu sentence-transformers"
        )

    target_device = "cpu"
    if device == "cuda" or (device == "auto" and _is_cuda_available()):
        target_device = "cuda"

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Loading {model_name} on {target_device} for offline indexing...")
    try:
        model = SentenceTransformer(model_name, device=target_device)
    except Exception as e:
        logger.warning(f"Failed to load {model_name} on {target_device}: {e}. Falling back to CPU.")
        target_device = "cpu"
        model = SentenceTransformer(model_name, device="cpu")

    texts = list(pool["_emb_text"])
    try:
        embs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=target_device,
        ).astype(np.float32)
    except Exception as e:
        if target_device == "cuda":
            logger.warning(f"CUDA encoding failed ({e}); falling back to CPU.")
            model = model.to("cpu")
            embs = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                device="cpu",
            ).astype(np.float32)
        else:
            raise e

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(save_dir, "pool.faiss"))
    with open(os.path.join(save_dir, "pool_ids.pkl"), "wb") as f:
        pickle.dump(list(pool["entity_id"]), f)
    model.save(os.path.join(save_dir, "encoder"))
    logger.info(f"Embedding index saved to {save_dir}")
