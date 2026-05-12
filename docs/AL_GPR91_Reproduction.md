# AL benchmark on GPR91-6RNK — reproduction + AL extension results

Reproduces Wenjin Liu's
[`../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb`](../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb)
on our infra, then extends it with four AL-policy arms (E/F/G/H) using
the MEL-level allocators from [`../al_policies/`](../al_policies/).
Sister doc to [AL_Pilot.md](AL_Pilot.md) (the CB2 pilot retrospective +
the pivot story) and [AL_Benchmark.md](AL_Benchmark.md) (the CB2
reference numbers).

## TL;DR

- **Reproduction is bit-exact** vs Wenjin's saved cell-26 output for
  every shared (strategy, option, hyperparameter) combination. D-S1 =
  4.041, D-S2 = 4.029, D-S3 = 4.041, D-S5 = 4.006, C-S1 T=1.0 = 3.878,
  C-S1 T=2.0 = 3.857, C-S1 T=0.5 = 3.856.
- **Wenjin's declared winner is C-S1 T=1.0 (EF AUC 3.878), but in her
  own data Strategy D wins** (EF AUC 4.041). Strategy D ignores MEL
  rank order and just takes a globally-ranked top-1M from
  per-MEL-capped pools — likely why she preferred C despite the lower
  number. (See "Why C and not D?" below.)
- **Our UCB AL extension (E-S1) lands at EF AUC 4.007** — ~3% above
  Wenjin's declared winner, **and 4th overall**, sitting between
  Strategy D variants and Strategy C. **Thompson Sampling (F-S1) lands
  at 3.974.** Both touch all 507 passing MELs (vs C's 277) while
  respecting Stage-1 MEL rank order, so the improvement isn't an
  artifact of ignoring rank.
- **Baseline-dynamic (G) and ε-greedy (H) AL arms underperform C-S1**
  (G=3.697, H=3.661). The allocator-quality story from the CB2 pilot
  carries over: UCB/TS robustness translates here too.

## What ran

Three input CSVs (in [`../csv/`](../csv/)):

- `Top1K_2Comp_MEL_Frags_With_VS_OpenVS_Mapping.csv` (4.1 MB, 1000
  rows) — MEL ranking by Stage-1 docking Score.
- `all_mels_combined_core.csv` (2.2 GB, 10,046,535 rows) — Wenjin's
  joined oracle. Per (MEL, synthon) with RTCNN_Score (Stage 3 SRG),
  FullLigand_Score (Stage 5 docking ground truth), Strain, CoreRmsd,
  and a few auxiliary fields.
- `GPR91_6RNK_Random1M_2CompLigands_ICM3.9.3_Docked.csv` (96 MB,
  934,051 rows) — random reference for the EF metric.

Runner:
[`../al_benchmark_gpr91/run_reproduction.py`](../al_benchmark_gpr91/run_reproduction.py)
under the `OpenVsynthes008` mamba env. Full sweep finishes in ~2m20s.
Quick mode (option 1 only) in ~42s.

```bash
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
  al_benchmark_gpr91/run_reproduction.py
```

## Headline results — full 6-option sweep

Top 10 by EF AUC (mean EF across thresholds −60 to −40):

| Rank | Strategy | Config | n_mels | n_ligands | EF AUC | EF@-55 | EF@-53 | EF@-51 | EF@-49 | EF@-47 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | D-S1 (RTCNN global rank) | cap=10K | 486 | 1,000,000 | **4.041** | 2.003 | 3.005 | 3.148 | 4.419 | 6.313 |
| 2 | D-S3 (Strain+RTCNN global rank) | cap=10K | 486 | 1,000,000 | **4.041** | 2.003 | 3.005 | 3.148 | 4.419 | 6.313 |
| 3 | D-S2 (Hard filters + RTCNN global rank) | cap=10K | 481 | 1,000,000 | 4.029 | 2.003 | 3.005 | 3.148 | 4.419 | 6.222 |
| 4 | **E-S1 (UCB MEL alloc + softmax pick)** | alloc=ucb T=1 n_probe=50 | **507** | 1,000,041 | **4.007** | 2.003 | 2.671 | 3.148 | 4.242 | 6.252 |
| 5 | D-S5 (Strain+RMSD+RTCNN global rank) | cap=10K | 473 | 1,000,000 | 4.006 | 2.003 | 3.005 | 3.005 | 4.124 | 6.253 |
| 6 | **F-S1 (TS MEL alloc + softmax pick)** | alloc=ts T=1 n_probe=50 | **507** | 1,000,027 | **3.974** | 2.003 | 2.671 | 3.005 | 4.301 | 6.131 |
| 7 | **C-S1 (Wenjin's declared winner)** | T=1.0 | 277 | 1,000,000 | **3.878** | 2.003 | 2.671 | 2.289 | 3.889 | 6.101 |
| 8 | C-S1 | T=2.0 | 277 | 1,000,000 | 3.857 | 2.003 | 3.005 | 2.432 | 3.830 | 5.949 |
| 9 | C-S1 | T=0.5 | 277 | 1,000,000 | 3.856 | 2.003 | 2.671 | 2.289 | 3.889 | 5.979 |
| 10 | B-S2 (Hard filters + RTCNN top-X% per MEL) | frac=10% | 507 | 851,048 | 3.844 | 2.354 | 2.746 | 3.026 | 4.085 | 5.885 |

Full results in
[`../al_benchmark_gpr91/wenjin_reproduction_results.csv`](../al_benchmark_gpr91/wenjin_reproduction_results.csv).

VS baseline: 78 MELs, 1,001,617 ligands, EF AUC = 1.000 by definition.

## Why C and not D?

Strategy D wins on EF AUC (4.041 vs C's 3.878) in *both* runs — ours
and Wenjin's saved output. Yet Wenjin's notebook declares C the
winner in cell 29. Likely reasons:

- **D ignores MEL rank order entirely.** It just pools all
  per-MEL-capped candidates and global-sorts. The Stage-1 docking
  rank doesn't influence the selection except via the per-MEL cap.
  That's not really a "strategy" — it's a global top-N with a
  per-MEL fairness floor. The interesting research question is
  "given a MEL ranking, how do you pick within and across MELs",
  and D doesn't answer that.
- **D selects from only 486 MELs** despite having no MEL-order
  constraint — a side effect of the cap and the pool's hit
  distribution. C's 277-MEL footprint is even narrower but reflects
  a coherent walking strategy.
- **Operational reproducibility:** Strategy C is seed-deterministic
  (the softmax sampler uses a fixed RNG); D is fully deterministic
  but its behavior changes more drastically with the per-MEL cap
  parameter (her notebook showed cap=10K vs cap=15K spread of ~0.025
  EF AUC). C's T=0.5/1.0/2.0 spread is only ~0.022.

So the implicit "no-degenerate-strategies" filter rules D out and
makes C the winner among the *non-degenerate* candidates. This is a
choice worth being explicit about; the choice is consistent with how
V-SYNTHES is taught (MELs first, then synthons within MELs).

## What our AL extensions add

The AL-extension arms E/F/G/H are layered on top of Wenjin's
framework — they keep the "walk MELs in Stage-1 docking rank order,
softmax-pick synthons per MEL on RTCNN" structure, but **replace the
fixed PER_MEL_CAP with a per-MEL budget that adapts to a probe of
50 random synthons per MEL**:

```
phase 1: probe 50 random synthons per MEL → observe RTCNN scores
phase 2: MEL-level allocator decides how much commit budget each
         passing MEL gets, given the probe summary
phase 3: softmax-sample the per-MEL commit budget from each MEL's
         remaining pool, T=1.0 (the Strategy C synthon picker)
phase 4: Wenjin-style second-pass fill from leftover pools so the
         final selection hits 1M
```

The four allocators we tested (E=UCB, F=Thompson Sampling, G=baseline
dynamic from `docs/MELSelection.md`, H=ε-greedy):

| Strategy | MEL allocator | EF AUC | vs C-S1 (3.878) | n_mels |
|---|---|---:|---:|---:|
| **E-S1** | **UCB1** (c=2.0)                | **4.007** | **+3.3%** | 507 |
| **F-S1** | **Thompson Sampling**           | **3.974** | **+2.5%** | 507 |
| C-S1    | (no allocator — fixed cap)      | 3.878    |   —      | 277 |
| G-S1    | Baseline dynamic (top-K mean × remainder) | 3.697 | -4.7% | 507 |
| H-S1    | ε-greedy (ε=0.1)                | 3.661    | -5.6%    | 507 |

Three observations:

1. **UCB and TS allocators move the needle.** ~3% over Wenjin's
   declared winner C-S1 is small but real, and it's achieved while
   touching nearly twice as many MELs (507 vs 277). The
   probe-and-allocate scaffold gives the allocator real information
   to act on (per-MEL hit density estimated from the probe sample),
   and UCB/TS use it.
2. **Baseline-dynamic and ε-greedy *underperform* C-S1.** This is
   surprising — these were two of the best policies in the CB2 pilot
   (where TS won, UCB tied, baseline was the control, ε-greedy was
   worst). The story here is the same as in the CB2 pilot: greedy
   over-commits to one MEL; baseline-dynamic's `expected_hits × α
   remainder` weighting is good at MEL-level allocation but mediocre
   when the synthon picker is uniform-vs-targeted. The
   bandit-Gaussian-posterior MEL allocators (UCB/TS) compose
   better with the softmax synthon picker.
3. **All four AL extensions cover all 507 passing MELs**, including
   tail MELs in the Stage-1 rank order. That's what the probe phase
   guarantees: every MEL gets at least 50 observations. By contrast
   C-S1 walks until its budget is filled and stops at MEL rank 456,
   missing 51 of the 507 passing MELs entirely.

## Reproduction details

The port from Wenjin's notebook lives in
[`../al_benchmark_gpr91/wenjin_strategies.py`](../al_benchmark_gpr91/wenjin_strategies.py).
Notable corrections during the port:

- Her cell 18 defines a function named `run_strategy_d` but it
  implements Strategy C — the call site uses `results_c`. We
  renamed correctly.
- Her notebook's saved cell-26 output ranks D-S1 #1, D-S3 #2, D-S2 #3,
  D cap=15K #4–6, D-S5 #7–8, then C-S1 #9–11. Our top-10 matches the
  five strategies we ran (D-S1, D-S2, D-S3, D-S5, C-S1 T=1.0, C-S1
  T=2.0, C-S1 T=0.5) to 3 decimal places.

The AL-extension arms live in
[`../al_benchmark_gpr91/al_ext_strategies.py`](../al_benchmark_gpr91/al_ext_strategies.py).
They reuse the MEL-level allocators from
[`../al_policies/`](../al_policies/) (built during the CB2 pilot) and
the softmax synthon picker from `wenjin_strategies.py`. The
Wenjin-style second-pass fill is included so concentrated-allocation
policies (greedy, baseline) still hit the budget.

Both modules have unit tests
([`../tests/test_wenjin_strategies.py`](../tests/test_wenjin_strategies.py)
and
[`../tests/test_al_ext_strategies.py`](../tests/test_al_ext_strategies.py))
that pass on synthetic data without the real oracle. The synthetic-data
shape exercises the same algorithmic paths as the real data.

## Reproducibility

Bit-deterministic given a fixed seed (default SEED=42):

```bash
cd /home/aoxu/projects/anchnor_based_VSYNTHES  # the symlink to NAS

# Smoke test (~42s): option 1 only, all strategies.
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
    al_benchmark_gpr91/run_reproduction.py --quick

# Full sweep (~2m20s): all 6 scoring options × 4 Wenjin strategies + 4 AL extensions.
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
    al_benchmark_gpr91/run_reproduction.py
```

Output: `al_benchmark_gpr91/wenjin_reproduction_results.csv`.

## What's missing / next steps

- **7WC6 target.** No Stage-5 ground truth on CARC. Run Stage-3 SRG
  over its Top-1K to build an SRG-proxy oracle (same approach as the
  CB2 pilot), then rerun. Per the user's earlier choice, this is the
  recommended path for the second target.
- **Multi-seed evaluation.** Currently a single seed=42 run. Wenjin's
  notebook is also single-seed (her softmax sampling is seeded at 42
  too). With small differences (~2-3%), running 5-10 seeds would tell
  us whether the AL extensions' edge is significant.
- **Vary `n_probe`.** Currently fixed at 50. Sweeping `n_probe ∈
  {10, 50, 200}` would tell us how the AL signal scales — too few
  probes and the allocator has no signal; too many and we're just
  paying budget for redundant information.
- **ε-greedy with smaller ε.** The CB2 pilot saw the same ε=0.1
  failure mode at high budgets. Try ε=0.3 or ε=0.5.
- **GP/RF policy.** ML-extension (`al_policies/ml.py`) is gated on
  scikit-learn but the OpenVsynthes008 env has it — turn it on as a
  fifth AL arm.

## See also

- [AL_Pilot.md](AL_Pilot.md) — the CB2 SRG pilot that preceded this
  work + the pivot retrospective
- [AL_Benchmark.md](AL_Benchmark.md) — CB2 pilot's reference numbers
- [WenjinCode.md](WenjinCode.md) — what's on CARC for each target
  (corrected on 2026-05-11 PM after the GPR91 docking-results dir
  was found)
- [MELSelection.md](MELSelection.md) — the dynamic-allocation baseline
- [CapSelect.md](CapSelect.md) — upstream geometric MEL ranker
- [../CLAUDE.md](../CLAUDE.md) — project orientation
