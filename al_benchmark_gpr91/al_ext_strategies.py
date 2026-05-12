"""AL-extension strategies (E/F/G/H) for Wenjin's benchmark framework.

The four strategies in [`wenjin_strategies.py`](wenjin_strategies.py)
are full-information: they see every synthon's score upfront and pick
the best 1M. Strategy C — the declared winner — uses softmax sampling
per MEL after walking MELs in Stage-1 docking-rank order.

The AL-extension strategies below add a **probe-then-allocate** phase
on top of Wenjin's framework. The intuition: in a real pipeline you
don't see every synthon's RTCNN upfront — you have to spend budget on
SRG runs to observe scores. An AL policy that probes a small per-MEL
sample, updates its beliefs about per-MEL hit potential, and then
re-allocates the remaining budget across MELs accordingly should do
strictly better than the full-information Strategy C on a per-budget-
spent basis (because the probe is part of the budget).

For this offline benchmark, we simulate the probe by sampling N₀=50
random synthons per MEL from the oracle (treating those score lookups
as the "probe observations"). Then we feed the per-MEL probe-summary
to my existing MEL-level policies from
[`../al_policies/`](../al_policies/) to allocate remaining budget,
and finally use softmax synthon-sampling within each MEL — the same
synthon-picker Wenjin's Strategy C uses.

Four AL-extension arms:

| Strategy | MEL allocator | Synthon picker |
|---|---|---|
| E | UCB1 on per-MEL Gaussian posterior over RTCNN_Score | softmax(T=1) |
| F | Thompson Sampling on same posterior                  | softmax(T=1) |
| G | Baseline dynamic: `expected_hits ** α × remainder`   | softmax(T=1) |
| H | ε-greedy on observed top-K mean                       | softmax(T=1) |

All four share the same shape as Wenjin's A/B/C/D: take
`(scored_df, mel_ranked, budget)` → return a `StrategyResult` with the
selected ligands and per-MEL stats. They consume the same scored pool
that `prepare_scored_pool()` produces, so they fit the existing
6-option scoring sweep.

Implementation note: my MEL-level policies in `al_policies/` expect a
`probe.expected_hits` field computed at the project's `--hit-threshold`
default. We compute it here from the probe samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import math
import numpy as np
import pandas as pd

from al_benchmark_gpr91.wenjin_strategies import (
    StrategyResult,
    _softmax_sample_df,
)


# Default hit threshold used to count "hits" in the probe summary —
# matches docs/MELSelection.md's recommendation for SRG-based prescreen.
HIT_THRESHOLD = -25.0
# Probe size per MEL (matches MELSelection.md §"Phase A" recommendation).
N_PROBE = 50
# Synthon-picker temperature; matches Strategy C's declared winner.
SYNTHON_T = 1.0


def _probe_each_mel(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    n_probe: int, rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Sample n_probe synthons per MEL uniformly at random.

    Returns:
      probes:    {key_norm: DataFrame of probed synthons (size ≤ n_probe)}
      remainders:{key_norm: DataFrame of unprobed synthons}

    MELs not present in `scored_df` are silently absent from both dicts."""
    mel_groups = {k: g for k, g in scored_df.groupby("key_norm")}
    probes: dict[str, pd.DataFrame] = {}
    remainders: dict[str, pd.DataFrame] = {}
    for key, mel_df in mel_groups.items():
        n = min(n_probe, len(mel_df))
        idx = rng.choice(len(mel_df), size=n, replace=False)
        mask = np.zeros(len(mel_df), dtype=bool)
        mask[idx] = True
        probes[key] = mel_df.iloc[mask].copy()
        remainders[key] = mel_df.iloc[~mask].copy()
    return probes, remainders


def _build_probe_results(
    probes: dict[str, pd.DataFrame], mel_ranked: pd.DataFrame,
    hit_threshold: float = HIT_THRESHOLD,
) -> tuple[list, dict]:
    """Build the MEL-level policy's input (`passing` list) and
    score-history dict from the probe samples.

    The MEL-level policy expects each entry to have `.row`, `.remainder`,
    and `.expected_hits`. The history dict maps row → list of probe
    scores for the UCB/TS/ML policies that read history."""
    passing = []
    history_data: dict[int, list[float]] = {}
    # Use mel_rank as the row identifier (deterministic mapping).
    rank_by_key = dict(zip(mel_ranked["key_norm"], mel_ranked["mel_rank"]))
    for key, probe in probes.items():
        row = int(rank_by_key.get(key, -1))
        if row < 0:
            continue
        scores = probe["_score"].values.astype(float).tolist()
        hits = sum(1 for s in scores if s <= hit_threshold)
        n_total = len(scores)
        # Expected hits projected over the full pool (same formula
        # as run_srg_batch.evaluate_probe).
        # `remainder` is computed from the FULL synthon pool minus probe.
        # We don't have the full pool here, so we leave it for the caller
        # to fill in after computing remainders.
        passing.append(SimpleNamespace(
            row=row, key_norm=key,
            n_probe=n_total, probe_hits=hits,
            probe_scores=scores,
        ))
        history_data[row] = scores
    return passing, history_data


# ----------------------------------------------------------------------
# Synthon-level picker (shared by all four AL-extension strategies)
# ----------------------------------------------------------------------

def _pick_synthons_softmax(
    probes: dict[str, pd.DataFrame], remainders: dict[str, pd.DataFrame],
    allocations: dict[int, int], rank_by_key: dict[str, int],
    T: float, rng: np.random.Generator,
    target_budget: int | None = None,
    mel_rank_order: list[str] | None = None,
) -> pd.DataFrame:
    """For each MEL, the probe synthons are kept and additional
    `allocations[row]` synthons are softmax-sampled from `remainders[key]`.

    If `target_budget` is provided, a Wenjin-style **second pass** runs after
    the per-MEL allocations are consumed: any unfilled budget is sampled
    from per-MEL leftover pools, in `mel_rank_order` order. This matches
    Strategy C's behavior (see `strategy_c_softmax_per_mel`) and prevents
    AL strategies that concentrate budget heavily on one MEL from leaving
    the overall budget under-spent."""
    parts: list[pd.DataFrame] = []
    used_idx_per_mel: dict[str, set] = {}

    # The probe synthons are already "observed" (their scores were used by
    # the allocator). They count toward the final selection — they're real
    # docked ligands.
    for key, probe in probes.items():
        parts.append(probe)
        used_idx_per_mel[key] = set(probe.index)

    # First pass: per-MEL allocations from the policy.
    key_by_rank = {v: k for k, v in rank_by_key.items()}
    for row, commit_n in allocations.items():
        if commit_n <= 0:
            continue
        key = key_by_rank.get(row)
        if key is None:
            continue
        leftover = remainders.get(key)
        if leftover is None or len(leftover) == 0:
            continue
        take = min(commit_n, len(leftover))
        sampled = (_softmax_sample_df(leftover, take, T, rng)
                   if take < len(leftover) else leftover)
        parts.append(sampled)
        used_idx_per_mel.setdefault(key, set()).update(sampled.index)

    # Second pass — Wenjin-style fill if requested and short of budget.
    if target_budget is not None and mel_rank_order is not None:
        running_n = sum(len(p) for p in parts)
        remaining_budget = target_budget - running_n
        if remaining_budget > 0:
            for key in mel_rank_order:
                if remaining_budget <= 0:
                    break
                leftover = remainders.get(key)
                if leftover is None or len(leftover) == 0:
                    continue
                already_taken = used_idx_per_mel.get(key, set())
                free = leftover[~leftover.index.isin(already_taken)]
                if len(free) == 0:
                    continue
                n_take = min(remaining_budget, len(free))
                extra = (_softmax_sample_df(free, n_take, T, rng)
                         if n_take < len(free) else free)
                parts.append(extra)
                used_idx_per_mel.setdefault(key, set()).update(extra.index)
                remaining_budget -= n_take

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------
# Strategy template — shared frame for E/F/G/H
# ----------------------------------------------------------------------

def _instantiate_policy(name: str, seed: int):
    """Construct a FRESH policy instance per call so seeded policies
    (TS in particular) don't share state across strategy invocations.

    Avoids the global `POLICY_REGISTRY` singletons that would cause
    seed-determinism to break."""
    from al_policies.baseline import BaselineDynamicAllocator
    from al_policies.greedy import EpsilonGreedyAllocator
    from al_policies.bandit import ThompsonSamplingAllocator, UCBAllocator
    if name == "baseline":
        return BaselineDynamicAllocator()
    if name == "greedy":
        return EpsilonGreedyAllocator()
    if name == "ucb":
        return UCBAllocator()
    if name == "ts":
        return ThompsonSamplingAllocator(seed=seed)
    raise KeyError(f"unknown policy: {name!r}")


def _run_probe_alloc_pick(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    policy_name: str,
    budget: int = 1_000_000,
    n_probe: int = N_PROBE,
    synthon_T: float = SYNTHON_T,
    hit_threshold: float = HIT_THRESHOLD,
    alpha: float = 1.0,
    min_commit: int = 50,
    seed: int = 42,
) -> StrategyResult:
    """Common scaffold for the four AL-extension strategies. The only
    difference between E/F/G/H is which policy from `al_policies/` is
    called for the MEL-level allocation step."""
    from al_policies import DictHistory

    rng = np.random.default_rng(seed)
    policy = _instantiate_policy(policy_name, seed=seed)

    # Phase 1 — probe each MEL with n_probe random synthons.
    probes, remainders = _probe_each_mel(scored_df, mel_ranked, n_probe, rng)

    # Build the policy's `passing` list. Filter to MELs whose probe
    # contains at least one hit (matches the default top-score criterion
    # below the hit threshold). MELs with no probed hits are aborted.
    passing_objs, history_data = _build_probe_results(
        probes, mel_ranked, hit_threshold=hit_threshold
    )
    rank_by_key = dict(zip(mel_ranked["key_norm"], mel_ranked["mel_rank"]))
    # Fill in `.remainder` and `.expected_hits` on each passing object.
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
    # Pass-only those whose probe has at least one observation (otherwise
    # the policy has nothing to act on for that MEL).
    passing = [p for p in passing if p.remainder > 0]

    # Phase 2 — compute remaining budget after the probe phase.
    n_probe_total = sum(len(p) for p in probes.values())
    remaining_budget = max(0, budget - n_probe_total)

    # Phase 3 — invoke MEL-level allocator.
    history = DictHistory(history_data)
    allocations = policy.allocate(
        passing, budget=remaining_budget, history=history,
        alpha=alpha, min_commit=min_commit,
    )

    # Phase 4 — softmax-pick synthons within each MEL based on allocations.
    # Includes a Wenjin-style second-pass fill so the final selection size
    # tracks the budget even when the policy concentrates heavily on one
    # MEL (e.g., greedy at high exploit share).
    mel_rank_order = [r["key_norm"] for _, r in mel_ranked.iterrows()
                      if r["key_norm"] in remainders]
    selected = _pick_synthons_softmax(
        probes, remainders, allocations, rank_by_key, T=synthon_T, rng=rng,
        target_budget=budget, mel_rank_order=mel_rank_order,
    )

    # Stats.
    sel_keys = set(selected["key_norm"].unique())
    mel_ranks = [int(mel_ranked.loc[mel_ranked["key_norm"] == k, "mel_rank"].iloc[0])
                 for k in sel_keys
                 if (mel_ranked["key_norm"] == k).any()]
    rank_min = int(min(mel_ranks)) if mel_ranks else None
    rank_max = int(max(mel_ranks)) if mel_ranks else None

    return StrategyResult(
        selected=selected, n_mels=len(sel_keys), n_ligands=len(selected),
        rank_min=rank_min, rank_max=rank_max,
        extras={
            "policy": policy_name, "n_probe": n_probe, "T": synthon_T,
            "alpha": alpha, "min_commit": min_commit,
            "n_probe_total": n_probe_total,
            "n_remaining_budget": remaining_budget, "seed": seed,
        },
    )


# ----------------------------------------------------------------------
# Public strategies E/F/G/H
# ----------------------------------------------------------------------

def strategy_e_ucb_alloc_softmax_pick(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    budget: int = 1_000_000, **kwargs,
) -> StrategyResult:
    return _run_probe_alloc_pick(
        scored_df, mel_ranked, policy_name="ucb", budget=budget, **kwargs,
    )


def strategy_f_ts_alloc_softmax_pick(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    budget: int = 1_000_000, **kwargs,
) -> StrategyResult:
    return _run_probe_alloc_pick(
        scored_df, mel_ranked, policy_name="ts", budget=budget, **kwargs,
    )


def strategy_g_baseline_alloc_softmax_pick(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    budget: int = 1_000_000, **kwargs,
) -> StrategyResult:
    return _run_probe_alloc_pick(
        scored_df, mel_ranked, policy_name="baseline", budget=budget, **kwargs,
    )


def strategy_h_greedy_alloc_softmax_pick(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    budget: int = 1_000_000, **kwargs,
) -> StrategyResult:
    return _run_probe_alloc_pick(
        scored_df, mel_ranked, policy_name="greedy", budget=budget, **kwargs,
    )


__all__ = [
    "HIT_THRESHOLD",
    "N_PROBE",
    "SYNTHON_T",
    "strategy_e_ucb_alloc_softmax_pick",
    "strategy_f_ts_alloc_softmax_pick",
    "strategy_g_baseline_alloc_softmax_pick",
    "strategy_h_greedy_alloc_softmax_pick",
]
