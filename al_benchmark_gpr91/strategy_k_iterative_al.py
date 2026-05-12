"""Strategy K — iterative AL with model retraining.

Replaces the 2-phase (probe → commit) shape with an N-round loop:

    initial probe (uniform random across all MELs)
    while budget remaining:
        1. fit BaggedRegressor ensemble on observations so far
        2. predict (mu, sigma) for every unobserved (MEL, synthon)
        3. acquisition: a(M,S) = μ̂(M,S) − κ · σ̂(M,S)
           (lower = better, since FullLigand_Score is more-negative
            = better)
        4. pick `batch_size` candidates by acquisition (greedy, with
           optional per-MEL cap)
        5. "observe" their FullLigand_Score (oracle lookup)
        6. add to training set; loop

The full-information oracle lets us simulate this offline (each
"observation" is a row lookup in `scored_df`'s FullLigand_Score
column).

Defaults match `docs/AL_Pilot.md` and the user's pilot config:
n_initial=50k (≈ N₀=50/MEL × 1000 MELs), batch_size=100k → 10 rounds
to reach budget=1M, κ=1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from al_benchmark_gpr91._ml_common import (
    BaggedRegressor,
    joint_features,
    synthon_features,
)
from al_benchmark_gpr91.wenjin_strategies import StrategyResult


def _initial_probe(
    scored_df: pd.DataFrame, n_initial: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_initial row-indices from scored_df uniformly at random.
    The samples become the initial training set for the iterative loop."""
    n = len(scored_df)
    n_take = min(n_initial, n)
    return rng.choice(n, size=n_take, replace=False)


def _acquire(
    mu: np.ndarray, sigma: np.ndarray, kappa: float,
) -> np.ndarray:
    """UCB-style acquisition. Lower = better (matches FullLigand_Score
    convention: more-negative = better hit). Returns the acquisition
    score; the caller does `argsort` ascending to pick best."""
    return mu - kappa * sigma


def _greedy_select_with_per_mel_cap(
    acq: np.ndarray, key_norm: np.ndarray, per_mel_cap: int, n_take: int,
    excluded_mask: np.ndarray,
    cumulative_per_mel: dict[str, int] | None = None,
) -> np.ndarray:
    """Sort by acq ascending; take rows until n_take, respecting per-MEL
    cap. `excluded_mask` marks already-selected rows (excluded from
    consideration).

    `cumulative_per_mel` is the count-per-MEL of already-selected rows
    BEFORE this round; the cap applies to (cumulative + this-round)
    so that strategies retain a hard total-per-MEL cap across rounds.
    Pass an empty dict for the first round; the function mutates it
    in-place and returns it via reference.

    Returns array of selected row indices (into the full scored_df)."""
    order = np.argsort(acq, kind="stable")
    counts = dict(cumulative_per_mel) if cumulative_per_mel is not None else {}
    chosen = []
    for idx in order:
        if excluded_mask[idx]:
            continue
        k = key_norm[idx]
        if counts.get(k, 0) >= per_mel_cap:
            continue
        chosen.append(int(idx))
        counts[k] = counts.get(k, 0) + 1
        if len(chosen) >= n_take:
            break
    if cumulative_per_mel is not None:
        cumulative_per_mel.clear()
        cumulative_per_mel.update(counts)
    return np.asarray(chosen, dtype=np.int64)


def strategy_k_iterative_al(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    budget: int = 1_000_000,
    mel_features_df: pd.DataFrame | None = None,
    n_initial: int = 50_000,
    batch_size: int = 100_000,
    kappa: float = 1.0,
    per_mel_cap: int = 5_000,
    ensemble_size: int = 3,
    member_n_estimators: int = 30,
    seed: int = 42,
    **kwargs,
) -> StrategyResult:
    """Iterative AL: probe → fit → predict → pick batch → refit → repeat."""
    rng = np.random.default_rng(seed)

    # Pre-compute the joint feature matrix once. This is the hot path
    # (10M rows × ~1k features). After this the inner loop only does
    # indexing and prediction.
    if mel_features_df is not None:
        X_all = joint_features(scored_df, mel_features_df)
    else:
        X_all = synthon_features(scored_df)
    y_all = scored_df["FullLigand_Score"].astype(np.float32).values
    key_norm = scored_df["key_norm"].astype(str).values
    n = len(scored_df)

    selected_mask = np.zeros(n, dtype=bool)
    n_selected = 0
    rounds: list[dict] = []
    cumulative_per_mel: dict[str, int] = {}

    # Phase 1 — initial probe. Cap-aware: don't exceed per_mel_cap even
    # in the random initial draw, so the cumulative cap holds globally.
    raw_initial = _initial_probe(scored_df, n_initial, rng)
    initial_keep = []
    for idx in raw_initial:
        k = key_norm[idx]
        if cumulative_per_mel.get(k, 0) >= per_mel_cap:
            continue
        initial_keep.append(int(idx))
        cumulative_per_mel[k] = cumulative_per_mel.get(k, 0) + 1
        if len(initial_keep) >= n_initial:
            break
    initial_idx = np.asarray(initial_keep, dtype=np.int64)
    selected_mask[initial_idx] = True
    n_selected = len(initial_idx)
    rounds.append({"round": 0, "kind": "probe", "n_added": int(n_selected)})

    # Phase 2 — iterative loop.
    round_i = 0
    while n_selected < budget:
        round_i += 1
        # Drop NaN targets (~5% of GPR91 oracle has missing FullLigand_Score).
        train_idx = np.flatnonzero(selected_mask)
        finite = np.isfinite(y_all[train_idx])
        train_idx = train_idx[finite]
        if len(train_idx) < 2:
            break
        bag = BaggedRegressor(
            n_bags=ensemble_size,
            member_n_estimators=member_n_estimators,
            seed=seed + round_i,
        )
        bag.fit(X_all[train_idx], y_all[train_idx])
        mu, sigma = bag.predict_with_std(X_all)
        acq = _acquire(mu, sigma, kappa)

        n_take = min(batch_size, budget - n_selected)
        picked = _greedy_select_with_per_mel_cap(
            acq, key_norm, per_mel_cap, n_take,
            excluded_mask=selected_mask,
            cumulative_per_mel=cumulative_per_mel,
        )
        if len(picked) == 0:
            break
        selected_mask[picked] = True
        n_selected += len(picked)
        rounds.append({"round": round_i, "kind": "commit", "n_added": int(len(picked))})

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
            "policy": "k_iterative",
            "n_initial": n_initial,
            "batch_size": batch_size,
            "kappa": kappa,
            "per_mel_cap": per_mel_cap,
            "ensemble_size": ensemble_size,
            "n_rounds": len(rounds),
            "uses_mel_features": mel_features_df is not None,
            "seed": seed,
        },
    )


__all__ = ["strategy_k_iterative_al"]
