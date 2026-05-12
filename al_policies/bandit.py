"""Bandit-style allocators: UCB1 and Thompson Sampling.

Both treat each passing MEL as an arm with an unknown score
distribution. The posterior is the per-MEL set of probe-phase score
observations (read from `history.scores_for(row)`). The two policies
differ in how they translate posterior into an allocation weight:

- **UCBAllocator**: weight ∝ -(μ̂_i - c·σ̂_i / √n_i). The
  exploration-bonus form rewards both low (= good) observed mean AND
  high uncertainty. We flip sign so "better" arms get higher weight,
  matching the existing `allocate_budget` convention where higher
  `expected_hits ** alpha` → more budget.

- **ThompsonSamplingAllocator**: for each MEL, sample one score from a
  Gaussian posterior on the mean (with empirical variance / n). The
  sampled scores define the weights. Across `n_rounds` of repeated
  sampling, the policy effectively spends MORE budget on the MELs whose
  posteriors give them realistic chances of being best.

Both produce a `{row: commit_n}` dict respecting per-MEL caps and the
floor — same shape as the baseline.

Posterior choice: empirical mean ± stderr with t-distribution-style
floor on n=1 (use `c·σ_prior` instead of √(1/0) blowup). Pure stdlib
+ random; no scipy.
"""
from __future__ import annotations

import math
import random
import statistics

from al_policies.base import HistoryView, register


# Prior std-dev on a MEL's score distribution when n_obs ≤ 1.
# Picked to roughly match the empirical spread of RTCNN_Score on the
# CB2_5ZTY debug fixture (range −44..+95, std ≈ 12; we use a slightly
# smaller value so exploration isn't dominant).
_PRIOR_SIGMA = 8.0
# UCB exploration constant. 2.0 is the textbook UCB1 default; smaller
# values are more exploit-heavy.
_UCB_C = 2.0


def _posterior_mean_std(scores: list[float]) -> tuple[float, float]:
    """Return (mean, stderr-of-mean). Defaults handle n=0 / n=1 robustly."""
    n = len(scores)
    if n == 0:
        return (0.0, _PRIOR_SIGMA)
    if n == 1:
        return (scores[0], _PRIOR_SIGMA)
    mu = statistics.fmean(scores)
    sd = statistics.stdev(scores)
    return (mu, sd / math.sqrt(n))


def _cap_spill(weights: dict[int, float], remainders: dict[int, int],
               budget: int, min_commit: int) -> dict[int, int]:
    """Iteratively allocate `budget` proportional to weights, capping each
    MEL at its remainder, then apply the floor. Equivalent logic to
    `allocate_budget` in run_srg_batch.py but accepts the weight dict
    directly so different policies can compute weights their own way."""
    commit_n: dict[int, int] = {r: 0 for r in remainders}
    active = set(remainders.keys())
    B = int(budget)
    while active:
        total_w = sum(weights[r] for r in active)
        if total_w <= 0:
            for r in active:
                commit_n[r] = 0
            break
        capped = False
        for r in list(active):
            raw = int(round(B * weights[r] / total_w))
            if raw >= remainders[r]:
                commit_n[r] = remainders[r]
                B -= remainders[r]
                active.discard(r)
                capped = True
        if not capped:
            for r in active:
                commit_n[r] = max(0, int(round(B * weights[r] / total_w)))
            break
    for r in remainders:
        commit_n[r] = min(max(commit_n[r], min_commit), remainders[r])
    return commit_n


class UCBAllocator:
    name = "ucb"

    def __init__(self, c: float = _UCB_C) -> None:
        self.c = c

    def allocate(self, passing, budget: int, history: HistoryView,
                 alpha: float, min_commit: int) -> dict[int, int]:
        passing = list(passing)
        if not passing:
            return {}
        remainders = {p.row: int(p.remainder) for p in passing}
        # weight = exp(-(mu - c*sigma)/_PRIOR_SIGMA); the exp keeps weights
        # positive and gives a smooth softmax over UCB scores.
        weights: dict[int, float] = {}
        for p in passing:
            mu, se = _posterior_mean_std(history.scores_for(p.row))
            ucb_score = mu - self.c * se     # lower = better
            weights[p.row] = math.exp(-ucb_score / _PRIOR_SIGMA)
        return _cap_spill(weights, remainders, budget, min_commit)


class ThompsonSamplingAllocator:
    name = "ts"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def allocate(self, passing, budget: int, history: HistoryView,
                 alpha: float, min_commit: int) -> dict[int, int]:
        passing = list(passing)
        if not passing:
            return {}
        remainders = {p.row: int(p.remainder) for p in passing}
        # Single sample per MEL from its Gaussian posterior, weight by
        # softmax. The "explore" effect: posteriors with low mean OR high
        # uncertainty can sample below other MELs' point estimates and
        # pull budget toward themselves.
        weights: dict[int, float] = {}
        for p in passing:
            mu, se = _posterior_mean_std(history.scores_for(p.row))
            sample = self._rng.gauss(mu, max(se, 1e-3))
            weights[p.row] = math.exp(-sample / _PRIOR_SIGMA)
        return _cap_spill(weights, remainders, budget, min_commit)


register(UCBAllocator())
register(ThompsonSamplingAllocator(seed=0))
