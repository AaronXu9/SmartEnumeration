"""Strategy L — multi-fidelity AL (RTCNN cheap, FullLigand_Score expensive).

Single-shot counterpart to Strategy K. Same surrogate (BaggedRegressor
ensemble on joint synthon+MEL features → FullLigand_Score) but **no
retraining**: probe once, fit once, pick the top-N unobserved by UCB
acquisition.

The "multi-fidelity" framing: RTCNN_Score is a *cheap pre-computed
feature* (in the synthon feature vector), and FullLigand_Score is the
*expensive labeled target* we want to predict. The surrogate explicitly
models the residual RTCNN → FullLigand relationship.

Key difference vs K:
- K: probe → fit → predict → pick batch → observe → REFIT → loop
- L: probe → fit → predict → pick TOP-N → done (one shot)

Key difference vs J:
- J: uses learned synthon ranker within a per-MEL-allocated framework
- L: uses learned ranker globally with a per-MEL cap (more like
  Strategy D but learned)

The strategies share `BaggedRegressor` from `_ml_common.py` and the
greedy selector with per-MEL cap from
`strategy_k_iterative_al._greedy_select_with_per_mel_cap`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from al_benchmark_gpr91._ml_common import (
    BaggedRegressor,
    joint_features,
    synthon_features,
)
from al_benchmark_gpr91.strategy_k_iterative_al import (
    _acquire,
    _greedy_select_with_per_mel_cap,
    _initial_probe,
)
from al_benchmark_gpr91.wenjin_strategies import StrategyResult


def strategy_l_multifidelity_al(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    budget: int = 1_000_000,
    mel_features_df: pd.DataFrame | None = None,
    n_probe: int = 50_000,
    kappa: float = 1.0,
    per_mel_cap: int = 5_000,
    ensemble_size: int = 5,
    member_n_estimators: int = 50,
    seed: int = 42,
    **kwargs,
) -> StrategyResult:
    """Multi-fidelity AL: probe, fit one surrogate, pick top-N by UCB
    globally with per-MEL cap."""
    rng = np.random.default_rng(seed)

    if mel_features_df is not None:
        X_all = joint_features(scored_df, mel_features_df)
    else:
        X_all = synthon_features(scored_df)
    y_all = scored_df["FullLigand_Score"].astype(np.float32).values
    key_norm = scored_df["key_norm"].astype(str).values
    n = len(scored_df)

    selected_mask = np.zeros(n, dtype=bool)
    cumulative_per_mel: dict[str, int] = {}

    # Phase 1 — probe. Sample uniformly across all (MEL, synthon) rows,
    # capping per-MEL at per_mel_cap.
    raw_probe = _initial_probe(scored_df, n_probe, rng)
    probe_keep = []
    for idx in raw_probe:
        k = key_norm[idx]
        if cumulative_per_mel.get(k, 0) >= per_mel_cap:
            continue
        probe_keep.append(int(idx))
        cumulative_per_mel[k] = cumulative_per_mel.get(k, 0) + 1
        if len(probe_keep) >= n_probe:
            break
    probe_idx = np.asarray(probe_keep, dtype=np.int64)
    selected_mask[probe_idx] = True

    # Phase 2 — fit a single surrogate on probe observations.
    # Drop NaN targets (~5% of GPR91 oracle has missing FullLigand_Score).
    fit_idx = probe_idx[np.isfinite(y_all[probe_idx])]
    if len(fit_idx) < 2:
        # Degenerate: trivial result with just the probe.
        return _build_result(scored_df, mel_ranked, selected_mask,
                              extras={"policy": "l_mf", "n_probe": int(len(probe_idx)),
                                       "fit": "skipped_no_data"})
    bag = BaggedRegressor(
        n_bags=ensemble_size, member_n_estimators=member_n_estimators, seed=seed,
    )
    bag.fit(X_all[fit_idx], y_all[fit_idx])

    # Phase 3 — predict + acquire on all rows.
    mu, sigma = bag.predict_with_std(X_all)
    acq = _acquire(mu, sigma, kappa)

    # Phase 4 — greedy global pick with cumulative per-MEL cap.
    n_take = budget - int(selected_mask.sum())
    if n_take > 0:
        picked = _greedy_select_with_per_mel_cap(
            acq, key_norm, per_mel_cap, n_take,
            excluded_mask=selected_mask,
            cumulative_per_mel=cumulative_per_mel,
        )
        selected_mask[picked] = True

    return _build_result(
        scored_df, mel_ranked, selected_mask,
        extras={
            "policy": "l_mf",
            "n_probe": int(len(probe_idx)),
            "kappa": kappa,
            "per_mel_cap": per_mel_cap,
            "ensemble_size": ensemble_size,
            "uses_mel_features": mel_features_df is not None,
            "seed": seed,
        },
    )


def _build_result(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    selected_mask: np.ndarray, extras: dict,
) -> StrategyResult:
    selected = scored_df.iloc[np.flatnonzero(selected_mask)].copy()
    sel_keys = set(selected["key_norm"].unique())
    mel_ranks = [int(mel_ranked.loc[mel_ranked["key_norm"] == k, "mel_rank"].iloc[0])
                 for k in sel_keys
                 if (mel_ranked["key_norm"] == k).any()]
    rank_min = int(min(mel_ranks)) if mel_ranks else None
    rank_max = int(max(mel_ranks)) if mel_ranks else None
    return StrategyResult(
        selected=selected, n_mels=len(sel_keys), n_ligands=len(selected),
        rank_min=rank_min, rank_max=rank_max, extras=extras,
    )


__all__ = ["strategy_l_multifidelity_al"]
