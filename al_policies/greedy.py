"""EpsilonGreedyAllocator — exploit-the-top-MEL with ε share to random.

Allocate `(1-ε)·budget` to the single MEL with the best (= lowest /
most-negative) observed top-K mean. Spread `ε·budget` uniformly across
the other passing MELs. Floor and cap-and-spill semantics match the
baseline policy.

`ε` defaults to 0.1. Override via the policy's `epsilon` attribute or
via `--al-epsilon` on the live runner / benchmark CLI.

Rationale (vs full greedy): pure top-1 allocation under-explores when
the probe sample was small (~50 synthons) — one lucky draw on MEL A
shouldn't permanently starve MEL B. The ε share is the standard fix.
"""
from __future__ import annotations

import statistics

from al_policies.base import HistoryView, register


_K_DEFAULT = 3


class EpsilonGreedyAllocator:
    name = "greedy"

    def __init__(self, epsilon: float = 0.1, top_k: int = _K_DEFAULT) -> None:
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(f"epsilon must be in [0, 1): got {epsilon!r}")
        self.epsilon = epsilon
        self.top_k = top_k

    def _topk_mean(self, scores: list[float]) -> float:
        """Mean of the top-K (most negative) scores. Returns +inf if no scores
        — guarantees a MEL with empty history is never the chosen exploit
        target (its top-K mean ranks last in min-is-best ordering)."""
        if not scores:
            return float("inf")
        srt = sorted(scores)[: self.top_k]
        return statistics.fmean(srt)

    def allocate(self, passing, budget: int, history: HistoryView,
                 alpha: float, min_commit: int) -> dict[int, int]:
        passing = list(passing)
        if not passing:
            return {}

        remainders = {p.row: int(p.remainder) for p in passing}
        # Score each MEL by top-K mean of observed scores (history). If
        # the history is empty for a MEL (live runner: this shouldn't
        # happen for *passing* MELs; offline harness with no probe yet:
        # falls back to a deterministic order via row index).
        topk = {p.row: self._topk_mean(history.scores_for(p.row)) for p in passing}
        # Pick the best MEL by min topk; ties broken by row for determinism.
        best_row = min(passing, key=lambda p: (topk[p.row], p.row)).row

        others = [p.row for p in passing if p.row != best_row]
        commit_n: dict[int, int] = {p.row: 0 for p in passing}

        if not others:
            # Lone passing MEL: no second target for the explore share, so
            # the entire budget goes here (capped by remainder).
            commit_n[best_row] = min(budget, remainders[best_row])
        else:
            exploit = int(round(budget * (1.0 - self.epsilon)))
            explore = budget - exploit  # avoid rounding drift
            commit_n[best_row] = min(exploit, remainders[best_row])
            # Spill any leftover from the cap on the exploit MEL back into the
            # explore pool.
            spill = exploit - commit_n[best_row]
            explore_total = explore + spill
            base = explore_total // len(others)
            extra = explore_total - base * len(others)
            for i, r in enumerate(others):
                share = base + (1 if i < extra else 0)
                commit_n[r] = min(share, remainders[r])

        # Apply the per-MEL floor, clipped by remainder.
        for r in remainders:
            commit_n[r] = min(max(commit_n[r], min_commit), remainders[r])

        return commit_n


register(EpsilonGreedyAllocator())
