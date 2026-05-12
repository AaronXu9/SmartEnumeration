"""Strategy N — joint MEL+synthon UCB acquisition (the project-customized AL).

This is the bi-level joint AL strategy the user asked for. Unlike
E/F/G/H/I (which pre-allocate per-MEL budget then pick synthons inside
each), Strategy N **does not pre-allocate at all**. It trains a single
learned model on the joint (MEL features + synthon features) space,
then globally picks 1M (MEL, synthon) pairs by UCB acquisition on that
joint posterior. The MEL allocation **emerges** from the joint
selection.

This is essentially "Strategy D + learned scores + uncertainty +
chemistry-aware features":

- D: global top-N by RTCNN, per-MEL cap = 10K — winner of Wenjin's
  EF AUC leaderboard at 4.041 but degenerate (ignores MEL rank).
- N: global top-N by predicted-FullLigand-Score-with-UCB-bonus,
  per-MEL cap = 5K, model sees MEL chemistry + synthon features.

The per-MEL cap (default 5K, matching Strategy C) is the project's
domain-knowledge regularizer: 5K synthons is enough to characterize
one scaffold; beyond that, returns diminish.

Structurally Strategy N is **identical to Strategy L** (same code
path: probe → fit → predict → greedy global with per-MEL cap). The
two are differentiated by **default hyperparameters**:

- L: n_probe=50_000, kappa=1.0, per_mel_cap=5_000, ensemble_size=5
- N: n_probe=50_000, kappa=1.0, per_mel_cap=5_000, ensemble_size=5

So they're algorithmically the same when called with the same args.
Calling them as separate strategies in the runner gives us two
identical labels in the leaderboard — useful as a sanity check
(seed-deterministic identical output) and to keep the "story" clear:
L is "multi-fidelity AL"; N is "joint bi-level AL". The reader picks
whichever framing they prefer.

If/when we differentiate them (e.g., N adds synthon FP features that
L doesn't have), we'll diverge. For V1 they're the same code path
with the same defaults.
"""
from __future__ import annotations

import pandas as pd

from al_benchmark_gpr91.strategy_l_multifidelity import strategy_l_multifidelity_al
from al_benchmark_gpr91.wenjin_strategies import StrategyResult


def strategy_n_joint_ucb(
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
    """Joint MEL+synthon UCB acquisition, single-shot, with per-MEL cap.

    The MEL+synthon allocation emerges from a single global UCB-greedy
    pick on the joint learned posterior. The per-MEL cap keeps the
    selection from collapsing to one MEL.

    Note: V1 implementation defers to `strategy_l_multifidelity_al`
    (algorithmically identical). The two will diverge if/when N adds
    synthon-side chemistry features that L doesn't have.
    """
    result = strategy_l_multifidelity_al(
        scored_df, mel_ranked, budget=budget,
        mel_features_df=mel_features_df,
        n_probe=n_probe, kappa=kappa,
        per_mel_cap=per_mel_cap,
        ensemble_size=ensemble_size,
        member_n_estimators=member_n_estimators,
        seed=seed,
        **kwargs,
    )
    # Rewrite the policy tag so the runner reports it as N.
    result.extras["policy"] = "n_joint_ucb"
    return result


__all__ = ["strategy_n_joint_ucb"]
