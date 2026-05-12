"""Strategy J — per-synthon learned ranker (replaces softmax picker).

After the probe phase, train a GradientBoostingRegressor on the
probed synthons' joint features (synthon numeric features + per-MEL
chemistry features) → `FullLigand_Score`. For each unprobed (MEL,
synthon), predict the score. The MEL-level allocator (baseline-
dynamic OR UCB) decides per-MEL count n_i; the learned picker takes
the **top n_i unprobed synthons by predicted score** from each MEL.

Two variants:

- `strategy_j_synthon_ranker_baseline_alloc` — baseline-dynamic MEL
  allocator + learned synthon ranker. Tests whether replacing the
  softmax picker with a learned ranker beats the baseline-dynamic
  allocator's existing softmax picker.
- `strategy_j_synthon_ranker_ucb_alloc` — UCB MEL allocator + learned
  synthon ranker. Tests the same against the UCB allocator (which is
  E-S1 in the GPR91 leaderboard).

The training set ⊥ prediction set guarantee: probed synthons are
removed from the candidate pool before prediction, so the learned
ranker never trains AND predicts on the same row.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from al_benchmark_gpr91._ml_common import (
    BaggedRegressor,
    SYNTHON_NUMERIC_COLS,
    extract_probe_observations,
    joint_features,
    synthon_features,
)
from al_benchmark_gpr91.al_ext_strategies import (
    HIT_THRESHOLD,
    N_PROBE,
    SYNTHON_T,
    _build_probe_results,
    _instantiate_policy,
    _probe_each_mel,
)
from al_benchmark_gpr91.wenjin_strategies import StrategyResult


def _train_synthon_ranker(
    probes: dict[str, pd.DataFrame],
    mel_features_df: pd.DataFrame | None,
    seed: int,
) -> BaggedRegressor | None:
    """Fit a small ensemble on probe observations.

    Returns None if there's not enough labeled data to train (in which
    case the caller should fall back to softmax sampling on RTCNN)."""
    X, y = extract_probe_observations(probes, mel_features_df=mel_features_df)
    if len(X) < 20:           # safety: very small probes are useless
        return None
    bag = BaggedRegressor(n_bags=3, member_n_estimators=40, seed=seed)
    bag.fit(X, y)
    return bag


def _learned_synthon_picker(
    probes: dict[str, pd.DataFrame],
    remainders: dict[str, pd.DataFrame],
    allocations: dict[int, int],
    rank_by_key: dict[str, int],
    model: BaggedRegressor,
    mel_features_df: pd.DataFrame | None,
    target_budget: int,
    mel_rank_order: list[str],
) -> pd.DataFrame:
    """Per-MEL: rank unprobed synthons by model prediction (lower =
    better, matches FullLigand_Score), then take top n_i. Includes a
    Wenjin-style second-pass fill to hit the budget.

    Optimization: build the full leftover feature matrix ONCE up front
    (instead of once per MEL inside the loop) so the dominant cost is
    the single model.predict call on the full pool rather than
    per-MEL pandas merge overhead."""
    parts: list[pd.DataFrame] = []
    used_idx_per_mel: dict[str, set] = {}
    key_by_rank = {v: k for k, v in rank_by_key.items()}

    # Probe synthons are kept (already observed).
    for key, probe in probes.items():
        parts.append(probe)
        used_idx_per_mel[key] = set(probe.index)

    # Concatenate all per-MEL leftovers once → single big DataFrame.
    # Predict on the full thing in one shot, then index per-MEL.
    leftover_parts = []
    leftover_keys: list[str] = []
    for key, leftover in remainders.items():
        if leftover is None or len(leftover) == 0:
            continue
        leftover_parts.append(leftover)
        leftover_keys.extend([key] * len(leftover))
    if leftover_parts:
        big_leftover = pd.concat(leftover_parts, ignore_index=False)
        # Build features once.
        if mel_features_df is None:
            X_big = synthon_features(big_leftover)
        else:
            X_big = joint_features(big_leftover, mel_features_df)
        mu_big = model.predict(X_big)
        # Build a key → (positional slice) map for fast per-MEL lookup.
        mu_series = pd.Series(
            mu_big, index=big_leftover.index, name="mu",
        )
        # Drop duplicates if any (shouldn't be, but be defensive).
        if mu_series.index.has_duplicates:
            mu_series = mu_series[~mu_series.index.duplicated(keep="first")]
    else:
        big_leftover = None
        mu_series = None

    # First pass: per-MEL allocation from the policy.
    for row, commit_n in allocations.items():
        if commit_n <= 0:
            continue
        key = key_by_rank.get(row)
        if key is None or mu_series is None:
            continue
        leftover = remainders.get(key)
        if leftover is None or len(leftover) == 0:
            continue
        mu_here = mu_series.loc[leftover.index].values
        order = np.argsort(mu_here, kind="stable")
        n_take = min(int(commit_n), len(leftover))
        picked_idx = leftover.index.values[order[:n_take]]
        picked = leftover.loc[picked_idx]
        parts.append(picked)
        used_idx_per_mel.setdefault(key, set()).update(picked_idx.tolist())

    # Second pass: fill any remaining budget from per-MEL leftovers in
    # rank order (matches `_pick_synthons_softmax`'s convention).
    running_n = sum(len(p) for p in parts)
    remaining_budget = target_budget - running_n
    if remaining_budget > 0 and mu_series is not None:
        for key in mel_rank_order:
            if remaining_budget <= 0:
                break
            leftover = remainders.get(key)
            if leftover is None or len(leftover) == 0:
                continue
            already_taken = used_idx_per_mel.get(key, set())
            free_mask = ~leftover.index.isin(list(already_taken))
            free = leftover.iloc[free_mask.nonzero()[0]] if free_mask.any() else leftover.iloc[:0]
            if len(free) == 0:
                continue
            n_take = min(remaining_budget, len(free))
            mu_here = mu_series.loc[free.index].values
            order = np.argsort(mu_here, kind="stable")
            picked_idx = free.index.values[order[:n_take]]
            picked = free.loc[picked_idx]
            parts.append(picked)
            used_idx_per_mel.setdefault(key, set()).update(picked_idx.tolist())
            remaining_budget -= n_take

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _run_strategy_j(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    policy_name: str,
    budget: int = 1_000_000,
    n_probe: int = N_PROBE,
    hit_threshold: float = HIT_THRESHOLD,
    alpha: float = 1.0,
    min_commit: int = 50,
    seed: int = 42,
    mel_features_df: pd.DataFrame | None = None,
) -> StrategyResult:
    """Shared scaffold for J-baseline-alloc and J-UCB-alloc. The only
    difference is which MEL-level policy is invoked between probe and
    learned synthon-pick."""
    from al_policies import DictHistory

    rng = np.random.default_rng(seed)
    policy = _instantiate_policy(policy_name, seed=seed,
                                  mel_features_df=mel_features_df)

    # Phase 1 — probe each MEL.
    probes, remainders = _probe_each_mel(scored_df, mel_ranked, n_probe, rng)

    # Build passing list (MELs whose probe contains at least one obs).
    passing_objs, history_data = _build_probe_results(
        probes, mel_ranked, hit_threshold=hit_threshold
    )
    rank_by_key = dict(zip(mel_ranked["key_norm"], mel_ranked["mel_rank"]))
    passing = []
    for p in passing_objs:
        key = p.key_norm
        leftover_n = len(remainders.get(key, pd.DataFrame()))
        n_probe_scored = p.n_probe
        n_total = n_probe_scored + leftover_n
        expected_hits = (p.probe_hits * n_total / max(1, n_probe_scored))
        passing.append(SimpleNamespace(
            row=p.row, key_norm=key,
            remainder=leftover_n,
            expected_hits=expected_hits,
            probe_best=min(p.probe_scores) if p.probe_scores else float("inf"),
        ))
    passing = [p for p in passing if p.remainder > 0]

    # Phase 2 — compute remaining budget.
    n_probe_total = sum(len(p) for p in probes.values())
    remaining_budget = max(0, budget - n_probe_total)

    # Phase 3 — MEL allocation.
    history = DictHistory(history_data)
    allocations = policy.allocate(
        passing, budget=remaining_budget, history=history,
        alpha=alpha, min_commit=min_commit,
    )

    # Phase 4 — train synthon ranker.
    model = _train_synthon_ranker(probes, mel_features_df, seed=seed)
    if model is None:
        # Fall back: pick synthons by raw RTCNN (≈ Strategy C's softmax T→0).
        # We do a stable nsmallest on _score per MEL, matching what the
        # ML picker would do with a perfectly-trained model on RTCNN.
        from al_benchmark_gpr91.al_ext_strategies import _pick_synthons_softmax
        mel_rank_order = [
            r["key_norm"] for _, r in mel_ranked.iterrows()
            if r["key_norm"] in remainders
        ]
        selected = _pick_synthons_softmax(
            probes, remainders, allocations, rank_by_key,
            T=SYNTHON_T, rng=rng,
            target_budget=budget, mel_rank_order=mel_rank_order,
        )
    else:
        mel_rank_order = [
            r["key_norm"] for _, r in mel_ranked.iterrows()
            if r["key_norm"] in remainders
        ]
        selected = _learned_synthon_picker(
            probes, remainders, allocations, rank_by_key,
            model=model, mel_features_df=mel_features_df,
            target_budget=budget, mel_rank_order=mel_rank_order,
        )

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
            "policy": policy_name,
            "n_probe": n_probe,
            "alpha": alpha,
            "min_commit": min_commit,
            "n_probe_total": n_probe_total,
            "n_remaining_budget": remaining_budget,
            "seed": seed,
            "picker": "learned" if model is not None else "softmax_fallback",
            "uses_mel_features": mel_features_df is not None,
        },
    )


def strategy_j_synthon_ranker_baseline_alloc(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    budget: int = 1_000_000,
    **kwargs,
) -> StrategyResult:
    """J variant — baseline-dynamic allocator + learned synthon ranker."""
    return _run_strategy_j(
        scored_df, mel_ranked, policy_name="baseline", budget=budget, **kwargs,
    )


def strategy_j_synthon_ranker_ucb_alloc(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    budget: int = 1_000_000,
    **kwargs,
) -> StrategyResult:
    """J variant — UCB allocator + learned synthon ranker."""
    return _run_strategy_j(
        scored_df, mel_ranked, policy_name="ucb", budget=budget, **kwargs,
    )


__all__ = [
    "strategy_j_synthon_ranker_baseline_alloc",
    "strategy_j_synthon_ranker_ucb_alloc",
]
