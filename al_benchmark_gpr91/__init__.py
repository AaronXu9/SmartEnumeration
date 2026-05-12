"""Reproduction of Wenjin Liu's GPR91-6RNK Enrichment-Factor benchmark,
plus our AL policy arms as additional comparisons.

See [`../docs/AL_Pilot.md`](../docs/AL_Pilot.md) (pilot retrospective)
and [`../docs/AL_Benchmark.md`](../docs/AL_Benchmark.md) (reference doc)
for the original CB2-SRG pilot this builds on.

The current module ports Wenjin's notebook
[`../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb`](../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb)
into testable Python. See `wenjin_strategies.py` for the four strategies
(A/B/C/D), the VS baseline rank-walk, and the EF metric.
"""
from al_benchmark_gpr91.wenjin_strategies import (
    STRATEGY_OPTION_LABELS,
    StrategyResult,
    compute_ef_vs_baseline,
    ef_auc,
    hard_filter,
    global_minmax_combined,
    prepare_scored_pool,
    strategy_a_global_hard_cutoff,
    strategy_b_greedy_per_mel,
    strategy_c_softmax_per_mel,
    strategy_d_global_rank_per_mel_cap,
    strategy_label,
    vs_baseline_rank_walk,
)

__all__ = [
    "STRATEGY_OPTION_LABELS",
    "StrategyResult",
    "compute_ef_vs_baseline",
    "ef_auc",
    "hard_filter",
    "global_minmax_combined",
    "prepare_scored_pool",
    "strategy_a_global_hard_cutoff",
    "strategy_b_greedy_per_mel",
    "strategy_c_softmax_per_mel",
    "strategy_d_global_rank_per_mel_cap",
    "strategy_label",
    "vs_baseline_rank_walk",
]
