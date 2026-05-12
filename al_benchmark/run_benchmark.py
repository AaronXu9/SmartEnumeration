#!/usr/bin/env python3
"""Offline AL benchmark: replay each policy against the SRG score oracle.

Loads `oracle/srg_scores.csv` (built by `oracle/build_srg_oracle.py`),
simulates the probe → allocate → commit flow for each combination of
(policy, budget, seed), and emits a results table. No ICM at run time —
the "score" of each (MEL, synthon) pair is a direct lookup in the
oracle CSV.

Replay loop per (policy, budget B, seed s):
  1. Load oracle, group by MEL InChIKey.
  2. For each MEL with ≥ N₀ synthons in the oracle:
       synthon_pool[mel] = oracle entries shuffled with seed s.
  3. Probe phase: draw the first N₀ synthons from each pool, record
     scores into the per-MEL history.
  4. Pass / abort each MEL using the same `evaluate_probe`-style rule
     the live runner uses (default: top-score criterion).
  5. policy.allocate(passing, budget=B_remaining, history=...) → dict.
  6. Draw `commit_n` more synthons from each passing MEL's pool. Record
     all scores. Decrement the budget for each draw.
  7. Compute metrics from the union of probe + commit observations.

Outputs:
  al_benchmark/results.csv   one row per (policy, budget, seed)
  al_benchmark/summary.tsv   mean ± stderr per (policy, budget)

Run as:
  python3 al_benchmark/run_benchmark.py            # full sweep, ~1 minute
  python3 al_benchmark/run_benchmark.py --quick    # 1 seed, smoke test

The harness is deliberately pure-stdlib + simple math — no numpy, no
pandas. Output CSVs are small enough that the downstream notebook can
load and plot them in pandas. Mirrors `edit_mel_cap.py` and
`oracle/build_srg_oracle.py` for the project's "no heavy deps at
runtime" convention.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Import via the project root (works under the local→NAS symlink).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from al_policies import DictHistory, POLICY_REGISTRY, get  # noqa: E402
from paths import PROJECT_ROOT  # noqa: E402

ORACLE_CSV = PROJECT_ROOT / "oracle" / "srg_scores.csv"
OUT_DIR = PROJECT_ROOT / "al_benchmark"

# Top-K depths reported in the metrics column.
TOPK_DEPTHS = (10, 50, 100)

# Default hit threshold (matches docs/MELSelection.md guidance for CB2_5ZTY).
HIT_THRESHOLD_DEFAULT = -25.0

# Probe sample size per MEL (matches docs/MELSelection.md §"Phase A"
# recommendation: N₀ ≈ 50).
N_PROBE_DEFAULT = 50

# Floor budget per MEL (matches the existing run_srg_batch.py default).
MIN_COMMIT_DEFAULT = 50


# ----------------------------------------------------------------------
# Oracle loading
# ----------------------------------------------------------------------

def load_oracle(oracle_csv: Path) -> dict[str, list[tuple[str, float]]]:
    """Return {mel_inchikey: [(synthon_inchikey, rtcnn_score), ...]}.

    Sorted by score (most-negative first) before being returned — the
    per-MEL "true ranking" is then index 0..K-1 for top-K computations."""
    pool: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with oracle_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pool[row["mel_inchikey"]].append(
                    (row["synthon_inchikey"], float(row["rtcnn_score"]))
                )
            except (KeyError, ValueError):
                continue
    # Sort each MEL by score ascending (best first) for downstream top-K.
    for ik in pool:
        pool[ik].sort(key=lambda x: x[1])
    return dict(pool)


# ----------------------------------------------------------------------
# Probe-style decision (matches evaluate_probe semantics)
# ----------------------------------------------------------------------

def _decide_pass(scores: list[float], hit_threshold: float,
                 stop_criterion: str, top_threshold: float) -> bool:
    """Replicates evaluate_probe's pass logic for the top-score and
    expected-hits criteria. Default = top-score: pass iff min(scores)
    <= top_threshold."""
    if not scores:
        return False
    if stop_criterion == "top-score":
        return min(scores) <= top_threshold
    if stop_criterion == "expected-hits":
        # Use min_expected_hits = 1 as the default — i.e. "at least one
        # hit projected over the full library" passes.
        return sum(1 for s in scores if s <= hit_threshold) > 0
    raise ValueError(f"unknown stop_criterion: {stop_criterion!r}")


# ----------------------------------------------------------------------
# Probe-result stand-in
# ----------------------------------------------------------------------

class FakeProbeResult:
    """Duck-typed stand-in for ProbeResult — exposes .row, .remainder,
    .expected_hits to the policy. The benchmark doesn't need the full
    ProbeResult plumbing (file paths, status enums, etc.)."""

    __slots__ = ("row", "remainder", "expected_hits")

    def __init__(self, row: int, remainder: int, expected_hits: float) -> None:
        self.row = row
        self.remainder = remainder
        self.expected_hits = expected_hits


# ----------------------------------------------------------------------
# Single replay
# ----------------------------------------------------------------------

def run_one(
    policy_name: str, budget: int, seed: int,
    pool: dict[str, list[tuple[str, float]]],
    n_probe: int, hit_threshold: float, stop_criterion: str,
    top_threshold: float, alpha: float, min_commit: int,
) -> dict:
    """Replay one (policy, budget, seed) combination. Returns a dict of
    metrics suitable for writing as a CSV row."""
    rng = random.Random(seed)
    policy = get(policy_name)

    # Per-MEL pool shuffled deterministically with seed.
    shuffled: dict[str, list[tuple[str, float]]] = {}
    true_best_per_mel: dict[str, float] = {}
    pool_size_per_mel: dict[str, int] = {}
    for ik, lst in pool.items():
        if len(lst) < n_probe:
            continue  # MEL too small to even probe — skip
        copy = list(lst)
        rng.shuffle(copy)
        shuffled[ik] = copy
        true_best_per_mel[ik] = lst[0][1]   # lst is sorted ascending
        pool_size_per_mel[ik] = len(lst)
    # Assign integer rows for the policy's row-keyed allocation dict.
    mel_ik_by_row: dict[int, str] = {i: ik for i, ik in enumerate(sorted(shuffled))}
    row_by_ik: dict[str, int] = {ik: i for i, ik in mel_ik_by_row.items()}

    history = DictHistory()
    drawn: dict[str, int] = {ik: 0 for ik in shuffled}
    remaining_budget = budget

    # Phase 1: probe each MEL with the first n_probe synthons.
    for ik, lst in shuffled.items():
        if remaining_budget <= 0:
            break
        n = min(n_probe, len(lst), remaining_budget)
        for i in range(n):
            history.observe(row_by_ik[ik], lst[i][1])
        drawn[ik] += n
        remaining_budget -= n

    # Phase 2: decide pass/abort per MEL, build the policy's `passing` list.
    passing: list[FakeProbeResult] = []
    for ik in shuffled:
        probe_scores = history.scores_for(row_by_ik[ik])
        if not _decide_pass(probe_scores, hit_threshold, stop_criterion,
                            top_threshold):
            continue
        n_total = pool_size_per_mel[ik]
        n_drawn = drawn[ik]
        remainder = n_total - n_drawn
        hits = sum(1 for s in probe_scores if s <= hit_threshold)
        expected_hits = hits * n_total / max(1, len(probe_scores))
        passing.append(FakeProbeResult(row_by_ik[ik], remainder, expected_hits))

    # Phase 3: allocate the remaining budget across passing MELs.
    if remaining_budget > 0 and passing:
        allocations = policy.allocate(
            passing, budget=remaining_budget, history=history,
            alpha=alpha, min_commit=min_commit,
        )
        # Phase 4: draw the allocated synthons from each MEL's pool.
        for row, commit_n in allocations.items():
            if commit_n <= 0:
                continue
            ik = mel_ik_by_row[row]
            lst = shuffled[ik]
            start = drawn[ik]
            n = min(commit_n, len(lst) - start)
            for i in range(start, start + n):
                history.observe(row, lst[i][1])
            drawn[ik] += n
            remaining_budget -= n

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    all_observed: list[tuple[str, str, float]] = []
    for ik in shuffled:
        for s in history.scores_for(row_by_ik[ik]):
            all_observed.append((ik, "", s))  # synthon-id not tracked here
    n_observed = len(all_observed)
    best_score = min((s for _, _, s in all_observed), default=float("inf"))
    n_hits = sum(1 for _, _, s in all_observed if s <= hit_threshold)

    # Top-K recall: for each K, fraction of the global top-K (across all
    # oracle entries in the in-scope MEL set) that we observed.
    global_scores: list[float] = sorted(
        s for ik in shuffled for _, s in pool[ik]
    )
    observed_scores_sorted = sorted(s for _, _, s in all_observed)
    recall: dict[int, float] = {}
    for K in TOPK_DEPTHS:
        if K >= len(global_scores):
            recall[K] = float("nan")
            continue
        true_topK = set(global_scores[:K])    # set of score values
        # Find how many of those scores appear in our observed set.
        hits = 0
        obs_counts: dict[float, int] = defaultdict(int)
        for s in observed_scores_sorted:
            obs_counts[s] += 1
        for s in global_scores[:K]:
            if obs_counts[s] > 0:
                hits += 1
                obs_counts[s] -= 1
        recall[K] = hits / K

    # Regret vs the oracle-optimal: oracle-optimal at budget B is the
    # B-best scores summed. Our regret = sum(true top-B) - sum(our best B).
    B_used = budget - remaining_budget
    oracle_best_B = sum(global_scores[:B_used])
    our_best_B = sum(observed_scores_sorted[:B_used])
    # Lower = better, so positive regret = ours is worse.
    regret = our_best_B - oracle_best_B

    return {
        "policy": policy_name,
        "budget_total": budget,
        "budget_used": B_used,
        "seed": seed,
        "n_mels_in_scope": len(shuffled),
        "n_mels_passing": len(passing),
        "n_observed": n_observed,
        "best_score": best_score,
        "n_hits": n_hits,
        "regret_sum_topB": regret,
        **{f"recall_top{K}": v for K, v in recall.items()},
    }


# ----------------------------------------------------------------------
# Sweep + write
# ----------------------------------------------------------------------

def run_sweep(
    pool: dict[str, list[tuple[str, float]]],
    policies: Iterable[str], budgets: Iterable[int], seeds: Iterable[int],
    n_probe: int, hit_threshold: float, stop_criterion: str,
    top_threshold: float, alpha: float, min_commit: int,
) -> list[dict]:
    """Run the cartesian product (policy × budget × seed)."""
    out: list[dict] = []
    for p in policies:
        for b in budgets:
            for s in seeds:
                t0 = time.time()
                row = run_one(p, b, s, pool, n_probe, hit_threshold,
                              stop_criterion, top_threshold, alpha, min_commit)
                row["elapsed_s"] = round(time.time() - t0, 3)
                out.append(row)
                print(
                    f"  {p:>8s}  B={b:<6d}  seed={s:<2d}  "
                    f"best={row['best_score']:8.2f}  hits={row['n_hits']:5d}  "
                    f"recall@50={row['recall_top50']:.3f}  "
                    f"regret={row['regret_sum_topB']:+10.1f}  "
                    f"({row['elapsed_s']}s)",
                    file=sys.stderr,
                )
    return out


def write_results(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(rows: list[dict], path: Path) -> None:
    """One row per (policy, budget). Means and stderrs over seeds."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["policy"], r["budget_total"])].append(r)
    num_keys = ("best_score", "n_hits", "regret_sum_topB",
                "recall_top10", "recall_top50", "recall_top100")
    fieldnames = ["policy", "budget", "n_seeds"]
    for k in num_keys:
        fieldnames.append(f"{k}_mean")
        fieldnames.append(f"{k}_stderr")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for (policy, budget), group in sorted(grouped.items()):
            row = {"policy": policy, "budget": budget, "n_seeds": len(group)}
            for k in num_keys:
                vals = [g[k] for g in group if g[k] == g[k]]   # filter NaN
                if not vals:
                    row[f"{k}_mean"] = ""
                    row[f"{k}_stderr"] = ""
                    continue
                mu = statistics.fmean(vals)
                row[f"{k}_mean"] = round(mu, 4)
                if len(vals) > 1:
                    sd = statistics.stdev(vals)
                    row[f"{k}_stderr"] = round(sd / math.sqrt(len(vals)), 4)
                else:
                    row[f"{k}_stderr"] = 0.0
            w.writerow(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", type=Path, default=ORACLE_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--policies", nargs="+",
                    default=sorted(POLICY_REGISTRY),
                    help="Subset of policies to run. Default: all registered.")
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[1000, 5000, 25000],
                    help="Per-run total budget in synthon-equivalent units.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-probe", type=int, default=N_PROBE_DEFAULT)
    ap.add_argument("--hit-threshold", type=float, default=HIT_THRESHOLD_DEFAULT)
    ap.add_argument("--stop-criterion", default="top-score",
                    choices=("top-score", "expected-hits"))
    ap.add_argument("--top-threshold", type=float, default=-15.0,
                    help="Pass MEL iff min(probe scores) <= this.")
    ap.add_argument("--alloc-alpha", type=float, default=1.0)
    ap.add_argument("--min-commit", type=int, default=MIN_COMMIT_DEFAULT)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 1 seed, small budgets, fast exit.")
    args = ap.parse_args(argv)

    if args.quick:
        args.budgets = [1000, 5000]
        args.seeds = [0]

    if not args.oracle.is_file():
        print(f"oracle not found: {args.oracle}", file=sys.stderr)
        print("Run `python3 oracle/build_srg_oracle.py` first.", file=sys.stderr)
        return 1
    pool = load_oracle(args.oracle)
    if not pool:
        print("oracle is empty — no rows to replay", file=sys.stderr)
        return 1
    print(
        f"loaded oracle: {len(pool)} MELs, "
        f"{sum(len(v) for v in pool.values())} (MEL, synthon) pairs",
        file=sys.stderr,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"running sweep: policies={args.policies} "
          f"budgets={args.budgets} seeds={args.seeds}", file=sys.stderr)
    rows = run_sweep(
        pool, args.policies, args.budgets, args.seeds,
        n_probe=args.n_probe, hit_threshold=args.hit_threshold,
        stop_criterion=args.stop_criterion, top_threshold=args.top_threshold,
        alpha=args.alloc_alpha, min_commit=args.min_commit,
    )

    results_path = args.out_dir / "results.csv"
    summary_path = args.out_dir / "summary.tsv"
    write_results(rows, results_path)
    write_summary(rows, summary_path)
    print(f"wrote {results_path}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
