"""MLRegressionAllocator — gradient-boosting on per-MEL features.

Train a regressor whose target is the per-MEL **hit rate** observed
during the probe phase. Features: a slim set of cheap per-MEL
descriptors (synthon-pool size, probe scores summary statistics, and
optionally pocket-grid samples around the APO atom — disabled by
default to keep the first cut dependency-light).

Use the predicted hit rate × remaining synthons as the allocation
weight, then run the same cap-and-spill + floor logic as the baseline.

Optional dependency: `scikit-learn` for the regressor. If sklearn is
unavailable, this module raises ImportError at import time, and
`al_policies/__init__.py` reports the policy as unavailable.

Note on training data: the policy trains on whatever observations
arrive in `history` at call time — there's no separate offline
training step in the first cut. On Run #1 with empty history, the
regressor falls back to "predict the global mean hit rate," which
makes the policy degenerate into uniform allocation. By Run #2+
(within a multi-round simulation, or when the live runner has prior
probe data cached), real features kick in.

For the offline benchmark, the harness can choose to pre-warm the
policy with a training set drawn from the oracle (a "transfer
learning" mode — out of scope for the first cut, see
docs/AL_Benchmark.md §"Phase 3b" for the design).
"""
from __future__ import annotations

import math
import statistics

# Gated at import time. Falling through to ImportError lets
# al_policies/__init__.py mark the policy unavailable cleanly.
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

from al_policies.base import HistoryView, register


def _summarize(scores: list[float]) -> dict[str, float]:
    """Cheap summary statistics over a MEL's observed RTCNN scores."""
    if not scores:
        return {"n": 0.0, "mean": 0.0, "min": 0.0, "median": 0.0,
                "p10": 0.0, "stdev": 0.0}
    srt = sorted(scores)
    n = len(srt)
    p10_idx = max(0, int(round(0.1 * (n - 1))))
    return {
        "n": float(n),
        "mean": statistics.fmean(srt),
        "min": srt[0],
        "median": srt[n // 2],
        "p10": srt[p10_idx],
        "stdev": statistics.stdev(srt) if n > 1 else 0.0,
    }


def _features_for(p, history: HistoryView) -> list[float]:
    """Per-MEL feature vector. Lightweight: pool stats + probe summary."""
    s = _summarize(history.scores_for(p.row))
    return [
        float(p.remainder),
        math.log1p(float(p.remainder)),
        s["n"],
        s["mean"],
        s["min"],
        s["median"],
        s["p10"],
        s["stdev"],
        float(getattr(p, "expected_hits", 0.0)),
    ]


class MLRegressionAllocator:
    name = "ml"

    def __init__(self, hit_threshold: float = -25.0,
                 n_estimators: int = 50, random_state: int = 0) -> None:
        self.hit_threshold = hit_threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _train(self, passing, history: HistoryView) -> GradientBoostingRegressor | None:
        """Fit a regressor on (features → observed hit rate) using current
        observations. Returns None if too few labeled examples to train."""
        X, y = [], []
        for p in passing:
            scores = history.scores_for(p.row)
            if not scores:
                continue
            hit_rate = sum(1 for s in scores if s <= self.hit_threshold) / len(scores)
            X.append(_features_for(p, history))
            y.append(hit_rate)
        if len(X) < 2:
            return None
        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=2,
            random_state=self.random_state,
        )
        model.fit(X, y)
        return model

    def allocate(self, passing, budget: int, history: HistoryView,
                 alpha: float, min_commit: int) -> dict[int, int]:
        passing = list(passing)
        if not passing:
            return {}
        remainders = {p.row: int(p.remainder) for p in passing}
        model = self._train(passing, history)
        if model is None:
            # Degenerate case: fall back to uniform-by-remainder, which is
            # the maximum-entropy "I know nothing" choice.
            weights = {p.row: max(1, int(p.remainder)) for p in passing}
        else:
            predicted_rate = model.predict([_features_for(p, history) for p in passing])
            # weight = predicted_hit_rate × remaining synthons
            #        = predicted *number* of new hits if we spent the entire
            #          remainder on this MEL. Clip to a small positive lower
            #          bound so a zero-prediction MEL doesn't get -0 weight.
            weights = {
                p.row: max(predicted_rate[i] * p.remainder, 1e-6)
                for i, p in enumerate(passing)
            }
        # Re-use the same cap-spill + floor logic the bandit policies use.
        from al_policies.bandit import _cap_spill
        return _cap_spill(weights, remainders, budget, min_commit)


register(MLRegressionAllocator())
