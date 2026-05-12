#!/usr/bin/env python3
"""Reproduce Wenjin Liu's GPR91-6RNK EF benchmark from her notebook.

Loads the three CSVs from `csv/`, runs all four strategies (A/B/C/D)
across all six scoring options, computes EF AUC and per-threshold EF
against the VS baseline, and dumps a ranked comparison table.

Expected to recover her declared winner: Strategy C, T=1.0, option 1
(RTCNN softmax only). If the winner shifts on our infra, that's a
reproduction failure worth investigating.

Run with the OpenVsynthes008 mamba env (needs pandas + numpy):
  /home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
      al_benchmark_gpr91/run_reproduction.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import via the project root (works under the local→NAS symlink).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from al_benchmark_gpr91 import (  # noqa: E402
    STRATEGY_OPTION_LABELS,
    compute_ef_vs_baseline,
    ef_auc,
    prepare_scored_pool,
    strategy_a_global_hard_cutoff,
    strategy_b_greedy_per_mel,
    strategy_c_softmax_per_mel,
    strategy_d_global_rank_per_mel_cap,
    strategy_label,
    vs_baseline_rank_walk,
)
from al_benchmark_gpr91.al_ext_strategies import (  # noqa: E402
    strategy_e_ucb_alloc_softmax_pick,
    strategy_f_ts_alloc_softmax_pick,
    strategy_g_baseline_alloc_softmax_pick,
    strategy_h_greedy_alloc_softmax_pick,
)
from paths import PROJECT_ROOT  # noqa: E402

CSV_DIR = PROJECT_ROOT / "csv"
OUT_DIR = PROJECT_ROOT / "al_benchmark_gpr91"

# Wenjin's notebook defaults.
BUDGET = 1_000_000
EVAL_THRESHOLDS = [-55, -53, -51, -49, -47]
SCORE_SWEEP = np.linspace(-60, -40, 200)
PER_MEL_CAP_C = 5_000           # Strategy C cap
PER_MEL_CAP_D = 10_000          # Strategy D cap (her notebook also tried 15K)
TEMPS = [0.5, 1.0, 2.0]
GLOBAL_TOP = 0.20               # Strategy A top-X% global cutoff
FRACTION = 0.10                 # Strategy B top-X% per MEL
SEED = 42


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_inputs(csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mel_ranking_csv = csv_dir / "Top1K_2Comp_MEL_Frags_With_VS_OpenVS_Mapping.csv"
    oracle_csv = csv_dir / "all_mels_combined_core.csv"
    random_csv = csv_dir / "GPR91_6RNK_Random1M_2CompLigands_ICM3.9.3_Docked.csv"
    for f in (mel_ranking_csv, oracle_csv, random_csv):
        if not f.is_file():
            print(f"missing input: {f}", file=sys.stderr)
            sys.exit(2)

    print(f"loading {mel_ranking_csv.name}", file=sys.stderr)
    mel_frag_scores = pd.read_csv(mel_ranking_csv)
    print(f"  shape: {mel_frag_scores.shape}", file=sys.stderr)

    print(f"loading {oracle_csv.name}", file=sys.stderr)
    synthon_ground_truth = pd.read_csv(oracle_csv)
    print(f"  shape: {synthon_ground_truth.shape}", file=sys.stderr)

    print(f"loading {random_csv.name}", file=sys.stderr)
    random1m = pd.read_csv(random_csv)
    print(f"  shape: {random1m.shape}", file=sys.stderr)

    # Notebook normalization: underscore → dash in MEL inchikey columns.
    mel_frag_scores["key_norm"] = mel_frag_scores["icm_inchikey"].str.replace("_", "-")
    synthon_ground_truth["key_norm"] = (
        synthon_ground_truth["mel_inchikey"].str.replace("_", "-")
    )

    # Rank MELs by Stage-1 Score ascending (most negative = best).
    mel_ranked = mel_frag_scores.sort_values("Score", ascending=True).reset_index(drop=True)
    mel_ranked["mel_rank"] = mel_ranked.index + 1

    return mel_ranked, synthon_ground_truth, random1m


# ----------------------------------------------------------------------
# Strategy sweep
# ----------------------------------------------------------------------

def run_all_strategies(
    mel_ranked: pd.DataFrame, synthon_ground_truth: pd.DataFrame,
    baseline_ligands: pd.DataFrame,
    options: list[int],
) -> list[dict]:
    """Returns one row per (strategy_letter, scoring_option, hyperparameter)
    combination, with EF AUC and per-threshold EF."""
    records: list[dict] = []

    # Cache the scored pools per option (expensive on real data).
    print("preparing scored pools", file=sys.stderr)
    pools: dict[int, pd.DataFrame] = {}
    for opt in options:
        pools[opt] = prepare_scored_pool(synthon_ground_truth, opt)
        print(f"  option {opt} ({STRATEGY_OPTION_LABELS[opt]}): "
              f"{len(pools[opt]):,} synthons", file=sys.stderr)

    # ---- Strategy A: global hard cutoff ----
    print("\n=== Strategy A (global hard cutoff) ===", file=sys.stderr)
    for opt in options:
        result = strategy_a_global_hard_cutoff(
            pools[opt], mel_ranked, top_frac=GLOBAL_TOP, budget=BUDGET,
        )
        records.append(_record("A", opt, f"top={GLOBAL_TOP:.0%} global", result, baseline_ligands))
        _print_progress("A", opt, result, records[-1])

    # ---- Strategy B: greedy top-X% per MEL ----
    print("\n=== Strategy B (greedy top-X% per MEL) ===", file=sys.stderr)
    for opt in options:
        result = strategy_b_greedy_per_mel(
            pools[opt], mel_ranked, fraction=FRACTION, budget=BUDGET,
        )
        records.append(_record("B", opt, f"frac={FRACTION:.0%} per MEL", result, baseline_ligands))
        _print_progress("B", opt, result, records[-1])

    # ---- Strategy C: softmax sampling per MEL  (Wenjin's winner) ----
    print("\n=== Strategy C (softmax sampling per MEL) ===", file=sys.stderr)
    for opt in options:
        for T in TEMPS:
            result = strategy_c_softmax_per_mel(
                pools[opt], mel_ranked, T=T,
                per_mel_cap=PER_MEL_CAP_C, budget=BUDGET, seed=SEED,
            )
            records.append(_record("C", opt, f"T={T}", result, baseline_ligands))
            _print_progress("C", opt, result, records[-1], suffix=f" T={T}")

    # ---- Strategy D: per-MEL cap then global rank ----
    print("\n=== Strategy D (per-MEL cap + global rank) ===", file=sys.stderr)
    for opt in options:
        result = strategy_d_global_rank_per_mel_cap(
            pools[opt], mel_ranked, per_mel_cap=PER_MEL_CAP_D, budget=BUDGET,
        )
        records.append(_record("D", opt, f"cap={PER_MEL_CAP_D:,}", result, baseline_ligands))
        _print_progress("D", opt, result, records[-1])

    # ---- AL-extension strategies E/F/G/H: probe → allocate → softmax-pick ---
    # These run only on option 1 (RTCNN-only) for the first cut. The
    # MEL-level allocators in al_policies/ don't see Strain/CoreRmsd, so
    # mixing them with the multi-feature options doesn't add information.
    al_ext_funcs = [
        ("E", "ucb",      strategy_e_ucb_alloc_softmax_pick),
        ("F", "ts",       strategy_f_ts_alloc_softmax_pick),
        ("G", "baseline", strategy_g_baseline_alloc_softmax_pick),
        ("H", "greedy",   strategy_h_greedy_alloc_softmax_pick),
    ]
    print("\n=== AL-extension strategies (E/F/G/H: probe + MEL-alloc + softmax-pick) ===",
          file=sys.stderr)
    for letter, alloc_name, fn in al_ext_funcs:
        result = fn(pools[1], mel_ranked, budget=BUDGET, seed=SEED)
        cfg = f"alloc={alloc_name} | T={1.0} | n_probe=50"
        records.append(_record(letter, 1, cfg, result, baseline_ligands))
        _print_progress(letter, 1, result, records[-1])

    return records


def _record(letter: str, opt: int, config: str, result, baseline_ligands: pd.DataFrame) -> dict:
    ef_sweep = compute_ef_vs_baseline(result.selected, baseline_ligands, SCORE_SWEEP)
    auc = ef_auc(ef_sweep)
    per_threshold = compute_ef_vs_baseline(result.selected, baseline_ligands, EVAL_THRESHOLDS)
    row = {
        "strategy": f"{letter}-S{opt}",
        "strategy_name": strategy_label(letter, opt),
        "config": config,
        "n_mels": result.n_mels,
        "n_ligands": result.n_ligands,
        "rank_min": result.rank_min,
        "rank_max": result.rank_max,
        "ef_auc": auc,
        **{f"ef@{t}": v for t, v in zip(EVAL_THRESHOLDS, per_threshold)},
        **{f"extras.{k}": v for k, v in result.extras.items()},
    }
    return row


def _print_progress(letter: str, opt: int, result, row: dict, suffix: str = "") -> None:
    print(
        f"  {letter}-S{opt}{suffix:>8s}  "
        f"n_mels={row['n_mels']:>4}  "
        f"rank {row['rank_min']}–{row['rank_max']}  "
        f"n_ligs={row['n_ligands']:>8}  "
        f"EF AUC={row['ef_auc']:>6.3f}",
        file=sys.stderr,
    )


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def write_outputs(records: list[dict], baseline_ligands: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df = df.sort_values("ef_auc", ascending=False, na_position="last").reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    # Add the baseline row (EF = 1.0 by definition).
    base = {
        "strategy": "VS-baseline",
        "strategy_name": "Baseline (VS) — full enum, MEL rank-walk",
        "config": "—",
        "n_mels": len(baseline_ligands["key_norm"].unique()),
        "n_ligands": len(baseline_ligands),
        "rank_min": 1,
        "rank_max": None,
        "ef_auc": 1.0,
        **{f"ef@{t}": 1.0 for t in EVAL_THRESHOLDS},
    }

    results_csv = out_dir / "wenjin_reproduction_results.csv"
    df.to_csv(results_csv)
    print(f"\nwrote {results_csv}", file=sys.stderr)

    print(f"\n=== Top 10 strategies by EF AUC ===")
    cols = ["strategy", "config", "n_mels", "n_ligands", "ef_auc"] + [
        f"ef@{t}" for t in EVAL_THRESHOLDS
    ]
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df.head(10)[cols].to_string())

    print(f"\n=== Baseline (VS) for reference ===")
    print(f"  n_mels={base['n_mels']}  n_ligands={base['n_ligands']:,}  EF AUC = 1.000 by definition")

    winner = df.iloc[0]
    print(f"\n=== Winner ===")
    print(f"  {winner['strategy_name']}  [{winner['config']}]")
    print(f"  EF AUC = {winner['ef_auc']:.3f}")
    for t in EVAL_THRESHOLDS:
        print(f"  EF@{t} = {winner[f'ef@{t}']:.3f}")
    expected = "C-S1: RTCNN softmax only"
    if winner["strategy_name"].startswith("C-S1"):
        print(f"\n  ✓ Reproduces Wenjin's declared winner ({expected})")
    else:
        print(f"\n  ⚠ Winner does NOT match Wenjin's declared winner ({expected}).")
        print(f"     This warrants investigation — see notebook for context.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--options", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                    help="Scoring-option subset to run (1..6). Default: all.")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: option 1 only.")
    args = ap.parse_args(argv)

    if args.quick:
        args.options = [1]

    mel_ranked, synthon_ground_truth, _random1m = load_inputs(args.csv_dir)

    # VS baseline = walk MELs in rank order, take all synthons per MEL until BUDGET.
    print(f"\nrunning VS baseline (budget = {BUDGET:,})", file=sys.stderr)
    baseline_df, baseline_ligands = vs_baseline_rank_walk(
        mel_ranked, synthon_ground_truth, budget=BUDGET,
    )
    print(f"  baseline MELs: {len(baseline_df)} "
          f"(rank {baseline_df['mel_rank'].min()}–{baseline_df['mel_rank'].max()}) "
          f"ligands: {len(baseline_ligands):,}", file=sys.stderr)

    records = run_all_strategies(
        mel_ranked, synthon_ground_truth, baseline_ligands,
        options=args.options,
    )
    write_outputs(records, baseline_ligands, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
