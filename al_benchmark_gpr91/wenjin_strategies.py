"""Strategies from Wenjin Liu's BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation notebook.

Ported from
[`../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb`](../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb)
into a clean, testable module so that:

1. We can re-run her benchmark on our infra and confirm we reproduce her
   declared winner (Strategy C, T=1.0, Option 1 = RTCNN softmax only).
2. We can plug additional AL policy arms into the same framework — see
   [`../al_policies/`](../al_policies/) for the MEL-level allocators
   from the original pilot and the new synthon-level policies that
   match this framework's shape.

Notebook → module mapping:

| Notebook cell | Function here | Notes |
|---|---|---|
| 4 (VS baseline)  | `vs_baseline_rank_walk()`            | walks MELs in docking-rank order, takes all synthons until budget |
| 10 (Strategy A)  | `strategy_a_global_hard_cutoff()`    | global score cutoff, then walk MELs |
| 14 (Strategy B)  | `strategy_b_greedy_per_mel()`        | top-X% per MEL ranked by score |
| 18 (Strategy C)  | `strategy_c_softmax_per_mel()`       | softmax sampling per MEL with PER_MEL_CAP. Wenjin's code calls this `run_strategy_d` (naming bug — it operates on `results_c`). |
| 22 (Strategy D)  | `strategy_d_global_rank_per_mel_cap()` | per-MEL cap then global rank |
| 10/14/26         | `compute_ef_vs_baseline()`, `ef_auc()` | EF metric helpers shared across strategies |

The scoring-option enum (1..6) is shared by A/B/C/D — see
`prepare_scored_pool()`.

This module is dependency-light: pandas + numpy + (optional) hypothesis
for the tests. No plotting; the notebook's matplotlib parts stay in
the notebook.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Scoring-option enum (shared by A/B/C/D)
# ----------------------------------------------------------------------

STRATEGY_OPTION_LABELS = {
    1: "RTCNN only",
    2: "Hard filters + RTCNN",
    3: "Strain+RTCNN combined",
    4: "Hard filters + Strain+RTCNN combined",
    5: "Strain+RMSD+RTCNN combined",
    6: "Hard filters + Strain+RMSD+RTCNN combined",
}


def hard_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop synthons with Strain > 15 or CoreRmsd > 2.0 (Wenjin's hard filter).
    Used by scoring options 2, 4, 6."""
    return df[(df["Strain"] <= 15) & (df["CoreRmsd"] <= 2.0)].copy()


def global_minmax_combined(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max-normalize each column to [0, 1], then average. Used by options
    3-6. Result column name is `_score`; lower = better (because the input
    Strain/CoreRmsd/RTCNN columns all use the lower-is-better convention,
    and the normalized mean preserves that)."""
    d = df.copy()
    norm_cols = []
    for c in cols:
        mn, mx = d[c].min(), d[c].max()
        nc = f"_n_{c}"
        d[nc] = (d[c] - mn) / (mx - mn + 1e-9)
        norm_cols.append(nc)
    d["_score"] = d[norm_cols].mean(axis=1)
    return d


def prepare_scored_pool(synthon_df: pd.DataFrame, option: int) -> pd.DataFrame:
    """Apply the scoring-option transformation. Returns a DataFrame with a
    `_score` column where lower = better. Strategy A/B/C/D all consume
    a `_score`-equipped DataFrame."""
    if option not in STRATEGY_OPTION_LABELS:
        raise ValueError(f"unknown scoring option {option!r}; "
                         f"valid: {sorted(STRATEGY_OPTION_LABELS)}")
    df = hard_filter(synthon_df) if option in (2, 4, 6) else synthon_df.copy()
    if option in (1, 2):
        df["_score"] = df["RTCNN_Score"]
    elif option in (3, 4):
        df = global_minmax_combined(df, ["Strain", "RTCNN_Score"])
    else:
        df = global_minmax_combined(df, ["Strain", "CoreRmsd", "RTCNN_Score"])
    return df


# ----------------------------------------------------------------------
# VS baseline rank walk
# ----------------------------------------------------------------------

def vs_baseline_rank_walk(
    mel_ranked: pd.DataFrame, synthon_ground_truth: pd.DataFrame,
    budget: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk MELs in docking-rank order, take ALL synthons per MEL until
    budget is exhausted. The V-SYNTHES default.

    `mel_ranked` MUST already be sorted by Stage-1 Score ascending and
    carry a `key_norm` column (hyphen-normalized icm_inchikey) and a
    `mel_rank` column (1..N integer).

    `synthon_ground_truth` MUST carry a `key_norm` column matching
    `mel_ranked.key_norm`.

    Returns (baseline_df, baseline_ligands):
      baseline_df       — one row per included MEL: (mel_rank, key_norm,
                           Score, n_synthons, cumulative).
      baseline_ligands  — the synthon rows of the included MELs (subset
                          of synthon_ground_truth).
    """
    synthon_counts = synthon_ground_truth.groupby("key_norm").size()
    available_keys = set(synthon_counts.index)

    cumulative = 0
    records: list[dict] = []
    for _, row in mel_ranked.iterrows():
        key = row["key_norm"]
        if key not in available_keys:
            continue
        n = int(synthon_counts[key])
        cumulative += n
        records.append({
            "mel_rank": int(row["mel_rank"]),
            "key_norm": key,
            "Score": float(row["Score"]),
            "n_synthons": n,
            "cumulative": cumulative,
        })
        if cumulative >= budget:
            break

    baseline_df = pd.DataFrame(records)
    baseline_keys = set(baseline_df["key_norm"])
    baseline_ligands = synthon_ground_truth[
        synthon_ground_truth["key_norm"].isin(baseline_keys)
    ].copy()
    return baseline_df, baseline_ligands


# ----------------------------------------------------------------------
# Strategy A — global hard cutoff
# ----------------------------------------------------------------------

@dataclass
class StrategyResult:
    """Common output shape for all four strategies. Keeps the cross-strategy
    comparison cell consumable."""
    selected: pd.DataFrame
    n_mels: int
    n_ligands: int
    rank_min: int | None
    rank_max: int | None
    # Strategy-specific extras (S_min for A, n_capped for D, etc.) — stored
    # in `extras` to avoid bloating the dataclass.
    extras: dict


def strategy_a_global_hard_cutoff(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    top_frac: float = 0.20, budget: int = 1_000_000,
) -> StrategyResult:
    """Global hard cutoff: keep the globally-best `top_frac` of synthons by
    `_score`, then walk MELs in rank order and take all survivors per MEL.

    `scored_df` must come from `prepare_scored_pool()`."""
    S_min = scored_df["_score"].quantile(top_frac)
    pool = scored_df[scored_df["_score"] <= S_min].copy()

    pool_counts = pool.groupby("key_norm").size()
    keys_in_pool = set(pool_counts.index)

    cumulative = 0
    selected_keys: list[str] = []
    rank_min = rank_max = None
    for _, row in mel_ranked.iterrows():
        key = row["key_norm"]
        if key not in keys_in_pool:
            continue
        cumulative += int(pool_counts[key])
        selected_keys.append(key)
        if rank_min is None:
            rank_min = int(row["mel_rank"])
        rank_max = int(row["mel_rank"])
        if cumulative >= budget:
            break

    selected = pool[pool["key_norm"].isin(set(selected_keys))]
    return StrategyResult(
        selected=selected,
        n_mels=len(selected_keys),
        n_ligands=len(selected),
        rank_min=rank_min,
        rank_max=rank_max,
        extras={"S_min": float(S_min)},
    )


# ----------------------------------------------------------------------
# Strategy B — greedy top-X% per MEL
# ----------------------------------------------------------------------

def strategy_b_greedy_per_mel(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    fraction: float = 0.20, budget: int = 1_000_000,
) -> StrategyResult:
    """Walk MELs in rank order; for each MEL take its top `fraction` of
    synthons by `_score` (at least 1). Stop when total ≥ budget."""
    mel_groups = {k: g for k, g in scored_df.groupby("key_norm")}
    available_keys = set(mel_groups.keys())

    selected_parts: list[pd.DataFrame] = []
    cumulative = 0
    n_mels = 0
    rank_min = rank_max = None

    for _, row in mel_ranked.iterrows():
        if cumulative >= budget:
            break
        key = row["key_norm"]
        if key not in available_keys:
            continue
        mel_df = mel_groups[key]
        n_total = len(mel_df)
        n_take = max(1, int(math.ceil(n_total * fraction)))
        n_take = min(n_take, budget - cumulative)
        top_synthons = mel_df.nsmallest(n_take, "_score")
        selected_parts.append(top_synthons)
        cumulative += len(top_synthons)
        n_mels += 1
        if rank_min is None:
            rank_min = int(row["mel_rank"])
        rank_max = int(row["mel_rank"])

    selected = (pd.concat(selected_parts, ignore_index=True)
                if selected_parts else pd.DataFrame())
    return StrategyResult(
        selected=selected, n_mels=n_mels, n_ligands=len(selected),
        rank_min=rank_min, rank_max=rank_max, extras={"fraction": fraction},
    )


# ----------------------------------------------------------------------
# Strategy C — softmax sampling per MEL  (Wenjin's declared winner)
# ----------------------------------------------------------------------
#
# Notebook's `run_strategy_d` (cell 18) is the Strategy C implementation
# despite the function name — confirmed by the call site populating
# `results_c`. We rename it correctly here.


def _softmax_sample_df(mel_df: pd.DataFrame, n_samples: int,
                       T: float, rng: np.random.Generator) -> pd.DataFrame:
    scores = mel_df["_score"].values.astype(float)
    logits = -scores / T            # negate: lower _score → higher prob
    logits -= logits.max()           # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen = rng.choice(len(mel_df), size=n_samples, replace=False, p=probs)
    return mel_df.iloc[chosen]


def strategy_c_softmax_per_mel(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    T: float = 1.0, per_mel_cap: int = 5_000,
    budget: int = 1_000_000, seed: int = 42,
) -> StrategyResult:
    """Walk MELs in rank order. For each MEL sample
    min(per_mel_cap, n_synthons, remaining_budget) synthons WITHOUT
    replacement using softmax(-_score / T) probabilities.

    Second pass: any unfilled budget gets sampled from the per-MEL
    leftover pools (synthons not chosen in the first pass).

    This is the strategy Wenjin selected as the winner in her notebook
    (T=1.0, scoring option 1)."""
    rng = np.random.default_rng(seed)
    available_keys = set(scored_df["key_norm"].unique())
    ranked_mels = [row for _, row in mel_ranked.iterrows()
                   if row["key_norm"] in available_keys]
    if not ranked_mels:
        return StrategyResult(pd.DataFrame(), 0, 0, None, None,
                              extras={"T": T, "per_mel_cap": per_mel_cap})
    mel_groups = {k: g for k, g in scored_df.groupby("key_norm")}

    first_sel: list[pd.DataFrame] = []
    remainders: dict[str, pd.DataFrame] = {}
    total = 0
    for row in ranked_mels:
        if total >= budget:
            break
        key = row["key_norm"]
        mel_df = mel_groups[key]
        n = len(mel_df)
        take = min(per_mel_cap, n, budget - total)
        if take == n:
            first_sel.append(mel_df)
        else:
            sampled = _softmax_sample_df(mel_df, take, T, rng)
            first_sel.append(sampled)
            sampled_idx = set(sampled.index)
            remainders[key] = mel_df[~mel_df.index.isin(sampled_idx)]
        total += take

    remaining = budget - total
    second_sel: list[pd.DataFrame] = []
    for row in ranked_mels:
        if remaining <= 0:
            break
        key = row["key_norm"]
        if key not in remainders:
            continue
        leftover = remainders[key]
        n_take = min(remaining, len(leftover))
        extra = (_softmax_sample_df(leftover, n_take, T, rng)
                 if n_take < len(leftover) else leftover)
        second_sel.append(extra)
        remaining -= n_take

    all_parts = first_sel + second_sel
    selected = (pd.concat(all_parts, ignore_index=True)
                if all_parts else pd.DataFrame())

    sel_keys = set(selected["key_norm"].unique())
    contrib = [r for r in ranked_mels if r["key_norm"] in sel_keys]
    rank_min = min(int(r["mel_rank"]) for r in contrib) if contrib else None
    rank_max = max(int(r["mel_rank"]) for r in contrib) if contrib else None

    return StrategyResult(
        selected=selected, n_mels=len(contrib), n_ligands=len(selected),
        rank_min=rank_min, rank_max=rank_max,
        extras={"T": T, "per_mel_cap": per_mel_cap, "seed": seed},
    )


# ----------------------------------------------------------------------
# Strategy D — per-MEL cap then global rank
# ----------------------------------------------------------------------

def strategy_d_global_rank_per_mel_cap(
    scored_df: pd.DataFrame, mel_ranked: pd.DataFrame,
    per_mel_cap: int = 10_000, budget: int = 1_000_000,
) -> StrategyResult:
    """For each MEL keep only the top per_mel_cap synthons by _score;
    pool everything, sort globally, take top `budget`."""
    pool = (scored_df
            .sort_values("_score", ascending=True)
            .groupby("key_norm", group_keys=False)
            .head(per_mel_cap))
    selected = pool.sort_values("_score", ascending=True).head(budget)

    mel_rank_lookup = dict(zip(mel_ranked["key_norm"], mel_ranked["mel_rank"]))
    contributing_keys = selected["key_norm"].unique()
    mel_ranks = [mel_rank_lookup[k] for k in contributing_keys
                 if k in mel_rank_lookup]

    n_mels = len(contributing_keys)
    n_capped = int((pool.groupby("key_norm").size() >= per_mel_cap).sum())
    rank_min = int(min(mel_ranks)) if mel_ranks else None
    rank_max = int(max(mel_ranks)) if mel_ranks else None

    return StrategyResult(
        selected=selected, n_mels=n_mels, n_ligands=len(selected),
        rank_min=rank_min, rank_max=rank_max,
        extras={"per_mel_cap": per_mel_cap, "n_capped": n_capped},
    )


# ----------------------------------------------------------------------
# Enrichment-Factor metric (against the VS baseline)
# ----------------------------------------------------------------------

def compute_ef_vs_baseline(
    selected_df: pd.DataFrame, baseline_ligands: pd.DataFrame,
    thresholds: Iterable[float],
) -> np.ndarray:
    """EF(t) = (rate of FullLigand_Score ≤ t in selected) /
              (rate of FullLigand_Score ≤ t in baseline).

    Lower threshold (more negative) = stricter hit definition; bigger
    EF at a given t means the selection is more enriched for hits.

    Returns an array aligned with `thresholds`. NaN where the baseline
    rate is zero at that threshold."""
    thresholds = np.asarray(list(thresholds))
    n_sel = len(selected_df)
    n_base = len(baseline_ligands)
    if n_sel == 0 or n_base == 0:
        return np.full(thresholds.shape, np.nan)
    sel_scores = selected_df["FullLigand_Score"].dropna().values
    base_scores = baseline_ligands["FullLigand_Score"].dropna().values
    ef = np.empty(thresholds.shape, dtype=float)
    for i, t in enumerate(thresholds):
        r_sel = (sel_scores <= t).sum() / n_sel
        r_base = (base_scores <= t).sum() / n_base
        ef[i] = r_sel / r_base if r_base > 0 else np.nan
    return ef


def ef_auc(ef: np.ndarray) -> float:
    """Mean EF across the threshold sweep — Wenjin's single-aggregate rank
    metric in cell 26. NaN-aware."""
    finite = ef[np.isfinite(ef)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmean(finite))


# ----------------------------------------------------------------------
# Convenience: run a strategy by name with default parameters
# ----------------------------------------------------------------------

def strategy_label(letter: str, option: int) -> str:
    return f"{letter}-S{option}: {STRATEGY_OPTION_LABELS[option]}"
