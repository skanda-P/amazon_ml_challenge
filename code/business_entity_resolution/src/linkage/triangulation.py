"""
Stage P -- Cross-source triangulation.

Tri(S1_i, S2_j) = max_{k in S3-candidates} Sim(S1_i, S3_k) * Sim(S2_j, S3_k)

This is additional *evidence*, fed into the classifier as a feature --
never used to perform blind transitive closure (S1->S2, S2->S3 does NOT
imply S1->S3 by itself). Computation is restricted to each Source-1
entity's own (small) candidate set, so cost is O(|S2_cand| x |S3_cand|)
per entity, not O(|S2| x |S3|) globally.
"""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def compute_triangulation(sim_s1_to_target: dict, target_name_ws: dict, s2_idx: list, s3_idx: list) -> dict:
    """
    sim_s1_to_target: {target_idx: name_similarity(S1_i, target_idx)} for
        every candidate of S1_i (both S2 and S3 members).
    target_name_ws: {target_idx: normalized_name_string} lookup for the
        (small) set of candidate indices involved.
    Returns {target_idx: triangulation_score} for every idx in s2_idx + s3_idx.
    """
    tri = {}
    # cache cross similarities to avoid recomputation
    cross_cache = {}

    def cross_sim(j, k):
        key = (j, k) if j < k else (k, j)
        if key not in cross_cache:
            cross_cache[key] = Levenshtein.normalized_similarity(target_name_ws[j], target_name_ws[k])
        return cross_cache[key]

    for j in s2_idx:
        best = 0.0
        sim_j = sim_s1_to_target.get(j, 0.0)
        for k in s3_idx:
            val = sim_s1_to_target.get(k, 0.0) * cross_sim(j, k)
            if val > best:
                best = val
        tri[j] = best

    for k in s3_idx:
        best = 0.0
        for j in s2_idx:
            val = sim_s1_to_target.get(j, 0.0) * cross_sim(j, k)
            if val > best:
                best = val
        tri[k] = best

    return tri
