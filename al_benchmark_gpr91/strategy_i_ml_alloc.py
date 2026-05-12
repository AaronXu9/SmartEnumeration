"""Strategy I — chemistry-aware ML allocator + softmax synthon picker.

The ML allocator here is the same `MLRegressionAllocator` from
[`../al_policies/ml.py`](../al_policies/ml.py) but constructed with
the per-MEL chemistry feature DataFrame from `_mel_features.py`
prepended to its feature vector. The softmax synthon picker is
unchanged from `al_ext_strategies._pick_synthons_softmax`.

Compared to Strategy E (UCB allocator + softmax pick) — the only
change is **which signal the allocator uses to decide per-MEL
budget**: UCB uses closed-form Gaussian posterior on probe scores;
I uses a learned model with chemistry context.

Compared to the V1 ML allocator (CB2 pilot, no chemistry features)
— this V2 prepends ~1037 chemistry/physchem/Stage-1 columns to the
9-dim probe-summary feature vector. The model now has access to:

- Morgan fingerprint of the MEL scaffold (1024 bits)
- MW / Tox_Score / molPAINS (3)
- 10 Stage-1 docking decomposition columns (Score, RTCNNscore,
  dEel, dEgrid, dEhb, dEhp, dEin, dEsurf, dTSsc, mfScore)
- 3 pool descriptors (mapped occupied, Nat, Nva)
- 7 probe summary stats (mean, min, median, p10, stdev, n, expected
  hits) — these come from `_features_for` in al_policies/ml.py.

Together that's the ~1047-dim feature vector tested in
[`tests/test_mel_features.py`](../tests/test_mel_features.py).
"""
from __future__ import annotations

import pandas as pd

from al_benchmark_gpr91.al_ext_strategies import _run_probe_alloc_pick
from al_benchmark_gpr91.wenjin_strategies import StrategyResult


def strategy_i_ml_alloc_softmax_pick(
    scored_df: pd.DataFrame,
    mel_ranked: pd.DataFrame,
    budget: int = 1_000_000,
    mel_features_df: pd.DataFrame | None = None,
    **kwargs,
) -> StrategyResult:
    """Run probe → ML allocator (with MEL chemistry features) → softmax
    synthon picker.

    `mel_features_df` should come from
    `al_benchmark_gpr91._mel_features.load_or_compute(...)`. If None,
    the ML allocator degenerates to the V1 features-only mode (probe
    summary stats only) — equivalent to Strategy E if you squint.
    """
    return _run_probe_alloc_pick(
        scored_df, mel_ranked,
        policy_name="ml",
        budget=budget,
        mel_features_df=mel_features_df,
        **kwargs,
    )


__all__ = ["strategy_i_ml_alloc_softmax_pick"]
