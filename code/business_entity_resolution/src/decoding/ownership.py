"""
Stage Q -- Target conflict / ownership arbitration.

This is a *conflict-resolution heuristic only*, applied after the
metric-aware decoder has proposed matches for every Source-1 entity. It
does NOT impose a one-to-one Source1<->target assumption in general
(one target can legitimately belong to exactly one S1 entity, and one
S1 entity can legitimately have many targets -- that part is untouched).
It only fires when the SAME target record was independently selected as
a match by more than one Source-1 entity, which should be rare with a
precision-oriented decoder and indicates genuine ambiguity worth
resolving conservatively.

owner(x) = argmax_i p_{ix}, subject to a confidence margin
p_best - p_second > delta; otherwise abstain (drop x from every
claimant) since predicting a wrong owner is a more expensive mistake
than predicting no owner at all under F0.5.
"""
from __future__ import annotations

from collections import defaultdict


def arbitrate(predictions: dict, margin_delta: float = 0.1) -> dict:
    """
    predictions: {s1_id: [(target_id, prob), ...]}  (already decoded, i.e.
        this is the *final* proposed match list per entity, not the full
        candidate list.)
    Returns a new predictions dict with the same shape, after resolving
    any target claimed by more than one Source-1 entity.
    """
    claims = defaultdict(list)  # target_id -> [(s1_id, prob), ...]
    for s1_id, matches in predictions.items():
        for target_id, prob in matches:
            claims[target_id].append((s1_id, prob))

    disputed = {t: c for t, c in claims.items() if len(c) > 1}
    if not disputed:
        return predictions

    drop_from = defaultdict(set)  # s1_id -> set(target_id) to remove
    for target_id, claimants in disputed.items():
        claimants_sorted = sorted(claimants, key=lambda x: -x[1])
        best_s1, best_p = claimants_sorted[0]
        second_p = claimants_sorted[1][1] if len(claimants_sorted) > 1 else 0.0
        if best_p - second_p > margin_delta:
            # keep for best_s1 only, drop for everyone else
            for s1_id, _ in claimants_sorted[1:]:
                drop_from[s1_id].add(target_id)
        else:
            # too ambiguous -- abstain for ALL claimants on this target
            for s1_id, _ in claimants_sorted:
                drop_from[s1_id].add(target_id)

    new_predictions = {}
    for s1_id, matches in predictions.items():
        to_drop = drop_from.get(s1_id, set())
        new_predictions[s1_id] = [(t, p) for t, p in matches if t not in to_drop]
    return new_predictions
