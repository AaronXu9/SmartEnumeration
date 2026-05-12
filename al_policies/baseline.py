"""BaselineDynamicAllocator — the control arm.

Thin wrapper around `run_srg_batch.allocate_budget()` (which implements
the rule in docs/MELSelection.md: `weight_i = expected_hits ** alpha`
with cap-and-spill + per-MEL floor). Identical numerics to the
pre-AL behavior.

Ignores `history` — the baseline policy uses only the probe summary
in `passing`.
"""
from __future__ import annotations

from al_policies.base import HistoryView, register


class BaselineDynamicAllocator:
    name = "baseline"

    def allocate(self, passing, budget: int, history: HistoryView,
                 alpha: float, min_commit: int) -> dict[int, int]:
        # Late import to avoid an import cycle: run_srg_batch imports
        # al_policies (via the live runner's --al-policy flag), so the
        # other direction has to be deferred to call time.
        from run_srg_batch import allocate_budget
        return allocate_budget(passing, budget=budget, alpha=alpha,
                               min_commit=min_commit)


register(BaselineDynamicAllocator())
