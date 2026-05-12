"""Common interface for MEL-budget allocation policies.

The four policies in this package (`baseline`, `greedy`, `bandit`, `ml`)
all implement the same signature so they're drop-in interchangeable —
both inside the existing live runner (`run_srg_batch.py`) and inside
the offline benchmark harness (`al_benchmark/run_benchmark.py`).

The signature matches the existing `allocate_budget()` function in
[run_srg_batch.py](../run_srg_batch.py) plus one new arg: a
`HistoryView` exposing per-MEL observed scores so smarter policies
(UCB/TS, ML) can use them. The baseline policy ignores the history arg.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HistoryView(Protocol):
    """Per-MEL observed-score history.

    The probe phase emits one of these; smarter policies read from it.
    Implementations: `RunHistory` (live runner — wraps `ProbeResult.scores`)
    and `OfflineHistory` (benchmark harness — reads from oracle CSV
    slices). Both are duck-typed; this Protocol just documents the
    expected attributes/methods."""

    def scores_for(self, row: int) -> list[float]:
        """Observed RTCNN scores for this MEL so far (probe + any prior
        commits). Empty if the MEL hasn't been probed yet."""

    def rows(self) -> list[int]:
        """All MEL rows with at least one observation."""


class AllocationPolicy(Protocol):
    """A policy decides, given probe outcomes + observation history, how
    much commit budget each passing MEL gets. The returned dict's
    semantics match `allocate_budget()` in run_srg_batch.py:

    - Keyed by MEL row.
    - Values sum may slightly exceed `budget` after floor application.
    - Values may sum below `budget` if every MEL caps at its remainder.
    - Per-MEL floor `min_commit` is clipped by remainder.

    `passing` items are duck-typed: each must expose `.row`,
    `.remainder`, and `.expected_hits` (matches `ProbeResult` and any
    SimpleNamespace stand-in)."""

    name: str

    def allocate(
        self,
        passing,
        budget: int,
        history: HistoryView,
        alpha: float,
        min_commit: int,
    ) -> dict[int, int]: ...


class EmptyHistory:
    """HistoryView that reports no observations. Default for callers that
    don't track per-MEL scores (e.g., a baseline-only configuration)."""

    def scores_for(self, row: int) -> list[float]:
        return []

    def rows(self) -> list[int]:
        return []


class DictHistory:
    """HistoryView backed by an in-memory {row: list[float]} dict.

    Used by the offline benchmark harness — the replay loop appends
    drawn scores into this dict as the simulation proceeds. The live
    runner uses `RunHistory` (in run_srg_batch.py — wraps `ProbeResult`
    directly so no duplication is needed)."""

    def __init__(self, data: dict[int, list[float]] | None = None) -> None:
        self._data: dict[int, list[float]] = {} if data is None else dict(data)

    def observe(self, row: int, score: float) -> None:
        self._data.setdefault(row, []).append(score)

    def observe_many(self, row: int, scores) -> None:
        self._data.setdefault(row, []).extend(scores)

    def scores_for(self, row: int) -> list[float]:
        return list(self._data.get(row, ()))

    def rows(self) -> list[int]:
        return [r for r, vs in self._data.items() if vs]


POLICY_REGISTRY: dict[str, "AllocationPolicy"] = {}


def register(policy: "AllocationPolicy") -> "AllocationPolicy":
    """Register a policy in the global registry under its `.name`.

    Used by `run_srg_batch.py --al-policy <name>` and the benchmark
    harness to resolve a string CLI choice into an instance."""
    POLICY_REGISTRY[policy.name] = policy
    return policy


def get(name: str) -> "AllocationPolicy":
    if name not in POLICY_REGISTRY:
        raise KeyError(
            f"unknown AL policy: {name!r}. "
            f"Registered: {sorted(POLICY_REGISTRY)}"
        )
    return POLICY_REGISTRY[name]
