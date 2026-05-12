"""Strategy M — submodular / diversity-aware selection.

Optimizes:

    f(S) = α · sum(score(s) for s in S)  +  (1-α) · DIVERSITY_WEIGHT · |distinct_MELs(S)|

Greedy submodular maximization: at each step, pick the candidate with
the largest marginal gain. The score component is **lower-is-better**
(matches the FullLigand_Score / RTCNN convention) so we negate it to
get a "higher-is-better" gain that the marginal-gain comparison
expects.

V1 (this implementation): diversity = `|distinct_MELs(S)|`. No RDKit
dependency; the count is recomputed in O(1) per step. The diversity
term contributes a bonus the first time a MEL is included; zero
afterwards. This is exactly the structure of a coverage-style
submodular function.

V2 (deferred): scaffold-aware Tanimoto-based diversity via RDKit
fingerprints — adds a heavy dep and ~10× compute, defer until needed.

Score signal:

- Default: `RTCNN_Score` from the synthon row (raw, no model).
- Optional: `predicted_FullLigand_Score` from a `BaggedRegressor`
  trained on probe observations — same surrogate as L. Controlled
  by `use_learned_score=True`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from al_benchmark_gpr91._ml_common import (
    BaggedRegressor,
    joint_features,
    synthon_features,
)
from al_benchmark_gpr91.strategy_k_iterative_al import _initial_probe
from al_benchmark_gpr91.wenjin_strategies import StrategyResult


def _greedy_submodular(
    score: np.ndarray,
    key_norm: np.ndarray,
    n_take: int,
    alpha: float,
    diversity_weight: float,
    excluded_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy maximization of α·score + (1-α)·diversity_weight·|distinct_MELs|.

    `score` is in the higher-is-better orientation (caller has already
    negated FullLigand_Score / RTCNN). `n_take` is the budget.

    Returns selected row indices in selection order."""
    n = len(score)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    seen_mels: set = set()

    # The optimal greedy step: argmax over (not yet chosen, not excluded)
    # candidates of (α·score[i] + diversity_bonus_i).
    # diversity_bonus_i = (1-α)·diversity_weight if key_norm[i] is new, else 0.
    # We can pre-rank candidates by α·score (descending) and resolve
    # diversity on-the-fly: walk the score order, flagging each candidate
    # as it appears with its (per-step) diversity bonus.
    # The combined objective is NOT pure greedy on score, so a naive
    # ranking-only walk can be sub-optimal. For meaningful budget sizes
    # (n_take ≪ n) we afford one pass per pick — total O(n_take·n) =
    # O(1M · 10M) = 10^13, way too slow.
    # Compromise: precompute a candidate priority queue. At each step,
    # the top of the queue may be either a new-MEL row (gets diversity
    # bonus) or a known-MEL row (no bonus). A precomputed order using
    # the maximum-possible gain (= α·score + (1-α)·DW) gives an upper
    # bound; we descend and pick the first candidate whose ACTUAL gain
    # (= α·score + (1-α)·DW · I{new mel}) is ≥ the next one's upper
    # bound. This is essentially the lazy-greedy submodular trick.

    # Lazy-greedy: maintain a heap of (-upper_bound, idx) so we pop
    # the most-promising candidate first. Re-insert with the actual
    # post-MEL-status gain to verify ordering.
    import heapq
    bonus = (1.0 - alpha) * diversity_weight
    upper_bound = alpha * score + bonus       # max gain ignoring MEL-status
    heap = [(-ub, int(idx))
            for idx, ub in enumerate(upper_bound)
            if excluded_mask is None or not excluded_mask[idx]]
    heapq.heapify(heap)

    while heap and len(chosen) < n_take:
        # Peek at the top; compute its ACTUAL gain. If it's still the
        # largest, accept. Else re-insert with the actual gain.
        neg_ub, idx = heapq.heappop(heap)
        if idx in chosen_set:
            continue
        k = key_norm[idx]
        diversity_gain = 0.0 if k in seen_mels else bonus
        actual = alpha * score[idx] + diversity_gain

        # If actual gain == upper bound (i.e., this row is a new-MEL),
        # we can accept directly.
        # Else, re-insert with the actual gain and check the next item.
        if -actual <= heap[0][0] if heap else True:
            chosen.append(idx)
            chosen_set.add(idx)
            seen_mels.add(k)
        else:
            heapq.heappush(heap, (-actual, idx))

    return np.asarray(chosen, dtype=np.int64)


def strategy_m_submodular(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    budget: int = 1_000_000,
    alpha: float = 0.7,
    diversity_weight: float = 1.0,
    use_learned_score: bool = False,
    score_column: str = "RTCNN_Score",
    mel_features_df: pd.DataFrame | None = None,
    n_probe: int = 50_000,
    ensemble_size: int = 3,
    member_n_estimators: int = 30,
    seed: int = 42,
    **kwargs,
) -> StrategyResult:
    """Greedy submodular: maximize α·score + (1-α)·diversity_weight·n_distinct_MELs.

    Score signal modes (controlled by `score_column` + `use_learned_score`):

    - **default**: `score_column="RTCNN_Score"`, `use_learned_score=False`
      — use the synthon RTCNN as the cheap pre-computed signal. This is
      the "fair" comparison against C/D/E/F/G/H which all rank synthons
      by RTCNN.
    - **learned surrogate**: `use_learned_score=True` — train a
      `BaggedRegressor` on a probe of `n_probe` random (MEL, synthon)
      rows with `FullLigand_Score` as target. Use the predicted scores
      as the selection signal.
    - **upper-bound oracle (NOT a fair benchmark arm)**:
      `score_column="FullLigand_Score"`, `use_learned_score=False` —
      directly reads the oracle target. Establishes a *ceiling* on EF
      AUC if score prediction were perfect, but isn't a strategy that
      can be deployed (no model has access to FullLigand_Score before
      the docking is run).

    Args:
        alpha: weight on score component. 1.0 = pure score (no diversity);
            0.5 = balanced.
        diversity_weight: multiplier on |distinct_MELs| term.
        score_column: which column to use as the score signal (when
            `use_learned_score=False`). Defaults to `"RTCNN_Score"` for
            apples-to-apples comparison with the other strategies.
        use_learned_score: if True, train a small surrogate on a probe;
            use predicted FullLigand_Score as score signal.
    """
    rng = np.random.default_rng(seed)
    n = len(scored_df)
    key_norm = scored_df["key_norm"].astype(str).values
    selected_mask = np.zeros(n, dtype=bool)

    # Score signal.
    if use_learned_score:
        if mel_features_df is not None:
            X_all = joint_features(scored_df, mel_features_df)
        else:
            X_all = synthon_features(scored_df)
        y_all = scored_df["FullLigand_Score"].astype(np.float32).values
        probe_idx = _initial_probe(scored_df, n_probe, rng)
        selected_mask[probe_idx] = True
        # Drop NaN targets when fitting (~5% in GPR91 oracle).
        fit_idx = probe_idx[np.isfinite(y_all[probe_idx])]
        bag = BaggedRegressor(
            n_bags=ensemble_size, member_n_estimators=member_n_estimators, seed=seed,
        )
        bag.fit(X_all[fit_idx], y_all[fit_idx])
        mu = bag.predict(X_all)
        score = (-mu).astype(np.float32)
    else:
        # Read the chosen column directly. Default is RTCNN_Score (cheap,
        # pre-computed) — matches the information regime that C/D/E/F/G/H
        # operate under. Pass score_column="FullLigand_Score" only to
        # produce the oracle upper-bound reference (not a fair strategy).
        raw = scored_df[score_column].astype(np.float32).values
        score = (-raw).copy()
        score[~np.isfinite(score)] = -np.inf

    # Greedy submodular over the unselected pool.
    n_take = budget - int(selected_mask.sum())
    if n_take > 0:
        picked = _greedy_submodular(
            score, key_norm, n_take, alpha, diversity_weight,
            excluded_mask=selected_mask,
        )
        selected_mask[picked] = True

    selected = scored_df.iloc[np.flatnonzero(selected_mask)].copy()
    sel_keys = set(selected["key_norm"].unique())
    mel_ranks = [int(mel_ranked.loc[mel_ranked["key_norm"] == k, "mel_rank"].iloc[0])
                 for k in sel_keys
                 if (mel_ranked["key_norm"] == k).any()]
    rank_min = int(min(mel_ranks)) if mel_ranks else None
    rank_max = int(max(mel_ranks)) if mel_ranks else None

    return StrategyResult(
        selected=selected,
        n_mels=len(sel_keys),
        n_ligands=len(selected),
        rank_min=rank_min,
        rank_max=rank_max,
        extras={
            "policy": "m_submodular",
            "alpha": alpha,
            "diversity_weight": diversity_weight,
            "use_learned_score": use_learned_score,
            "uses_mel_features": mel_features_df is not None,
            "seed": seed,
        },
    )


__all__ = ["strategy_m_submodular"]
