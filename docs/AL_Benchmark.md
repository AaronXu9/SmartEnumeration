# AL Benchmark — CB2 SRG-proxy pilot (HISTORICAL)

> **⚠ Numerically invalid in the GPR91 EF framework.** This document
> records an early pilot on CB2_5ZTY using **Stage-3 SRG RTCNN as both
> the selection signal and the oracle**. The downstream-validated
> pilot uses Stage-5 full-compound docking ground truth + Wenjin Liu's
> Enrichment Factor framework on GPR91-6RNK; see
> [AL_GPR91_Reproduction.md](AL_GPR91_Reproduction.md) for the actual
> findings.
>
> The CB2 pilot is preserved here for the methodology + the
> `AllocationPolicy` Protocol definition (which the GPR91 work reuses)
> + the `tests/test_al_*.py` contract tests (which still pass). Its
> *numerical* conclusions ("TS wins by ~7%") do not transfer — different
> oracle, different metric, different scope (7 MELs not 1000+).

Offline benchmark for active-learning MEL+synthon-budget allocation
policies. Sister doc to [MELSelection.md](MELSelection.md) (the baseline
allocator's design) and [CapSelect.md](CapSelect.md) (the upstream
geometric MEL ranker that MELSelection.md replaces).

## TL;DR

On a 7-MEL CB2_5ZTY oracle scraped from existing SRG runs (108,589
unique (MEL, synthon) score observations, RTCNN_Score range
−45 to +114, hit density ranging 0.05% to 55% per MEL), four
allocation policies were benchmarked at three budgets and five seeds
each. Result:

| Budget | baseline regret | TS regret | UCB regret | greedy regret |
|---|---:|---:|---:|---:|
| 1K   | 22.2K ± 0.2 | **20.4K ± 0.1** | 20.4K ± 0.1 | 20.1K ± 0.7 |
| 5K   | 57.0K ± 1.2 | **55.7K ± 0.8** | 56.8K ± 0.7 | 86.7K ± 8.1 |
| 25K  | 160K ± 9 | **149K ± 1** | 150K ± 0.3 | 288K ± 30 |

(Regret = sum of the B best true scores minus sum of the B best
observed scores. Lower is better. ± is one stderr over 5 seeds.)

**Recommendation:** Thompson Sampling (`ts`) is the most-robust
small-improvement over the baseline (~7% lower regret at B=25K with
near-zero variance across seeds). UCB matches it. ε-greedy at ε=0.1 is
not safe at this scale — its single-MEL exploit collapses recall@50
from 100% to ~46% at B=25K because it abandons the lower-density MELs
whose tails actually hold half the hits. **Do not productionize until
the oracle covers ≥50 MELs**; the 7-MEL conclusion is qualitatively
suggestive, not quantitatively decisive.

## What we tested

### Policies (all in [`../al_policies/`](../al_policies/))

| Policy | One-line semantics |
|---|---|
| `baseline` (control) | `weight_i = expected_hits_i ** α × remaining_synthons_i` — wraps the existing `allocate_budget()` per [MELSelection.md](MELSelection.md) |
| `greedy` (ε=0.1) | Exploit-best-MEL: (1-ε)·budget to the MEL with the best observed top-K mean; ε·budget split uniformly across the rest |
| `ucb` (c=2.0) | weight = exp(-(μ̂ - c·SE)/σ_prior). UCB1-style: rewards low observed mean AND high posterior uncertainty |
| `ts` (seed=0) | Per-MEL Gaussian posterior on mean; sample one score per MEL each call; weight = exp(-sample/σ_prior) |
| `ml` (gated) | GradientBoostingRegressor on (per-MEL features → observed hit rate). Requires scikit-learn; gated at import. **Not run in this benchmark** — see "What we don't know yet". |

All five policies implement the same `AllocationPolicy` Protocol
([al_policies/base.py](../al_policies/base.py)) and are drop-in
compatible with `run_srg_batch.py` via a future `--al-policy` flag
(not yet wired up in the live runner — see "Next steps").

### Harness

[`../al_benchmark/run_benchmark.py`](../al_benchmark/run_benchmark.py):

1. Load oracle CSV (built by
   [`oracle/build_srg_oracle.py`](../oracle/build_srg_oracle.py) from
   the existing per-MEL `results_*/MEL_<row>_<rank>/enumerated.sdf`
   files).
2. For each MEL in scope, shuffle its synthon pool with the seed
   (deterministic per (policy, budget, seed) triple).
3. Probe phase: draw the first N₀=50 synthons from each MEL, record
   RTCNN scores into the per-MEL history.
4. Pass/abort per MEL (default: top-score criterion with threshold
   −15.0 — i.e. pass iff `min(probe scores) ≤ −15`).
5. Single allocate call: `policy.allocate(passing, budget=B - probe_used,
   history=..., alpha=1.0, min_commit=50)`.
6. Draw the allocated synthons from each MEL's shuffled pool.
7. Compute metrics from the union of probe + commit observations:
   `best_score`, `n_hits` (at threshold −25), `regret_sum_topB`,
   `recall_top{10,50,100}`.

Sweep: 4 policies × 3 budgets × 5 seeds = 60 runs. Total time: ~1.5
seconds end-to-end (pure CSV lookup, no ICM at run time).

### Data scope

Oracle ingested 22 `enumerated.sdf` files across `results/`,
`results_katlab/`, `results_local_macos/`, `results_carc/`. Per-MEL
coverage (from [`oracle/coverage.tsv`](../oracle/coverage.tsv)):

| MEL InChIKey | Rank | Synthons in oracle | Range | Hits ≤ −25 |
|---|---|---:|---|---:|
| BACFEABMJVTTBB-QSPOBJNRSA-N | Rank3 | 500    | −41.4 .. −10.3 | 397 |
| DYJMAJZUVGLFFG-JDTDCXQMSA-N | Rank8 | 27,247 | −32.6 .. +95.0 | 84 |
| KBHFKLHIPFYHSY-ROWHFZDMSA-N | Rank9 | 500    | −42.2 .. −3.8  | 342 |
| NBYAKZFGIATWSZ-LYCJGSLVSA-N | Rank6 | 29,696 | −27.3 .. +88.5 | 31 |
| PKVSJQCKBTWHOT-FYRBZERYSA-N | Rank5 | 33,109 | −37.4 .. +113.7 | 386 |
| SRUQYMWCUHUFST-CAEBOLNTSA-N | Rank7 | 12,544 | −44.9 .. +18.4 | 6,843 |
| ZMPONTLLTHXELC-DBJRSSSESA-N | Rank2 | 4,993  | −38.4 .. +16.9 | 2,450 |

**Hit-density skew is large.** Rank7 has 55% hits in its full pool;
Rank6 has 0.1%. The benchmark therefore tests whether each policy can
spot the high-density MELs from a 50-synthon probe and steer budget
toward them — exactly the question MELSelection.md was designed for.

## Headline numbers

Full results in [`../al_benchmark/results.csv`](../al_benchmark/results.csv);
mean ± stderr in [`../al_benchmark/summary.tsv`](../al_benchmark/summary.tsv);
plots in [`../notebooks/AL_benchmark_results.ipynb`](../notebooks/AL_benchmark_results.ipynb).

**Regret (sum-top-B) — lower is better**:

| Budget | baseline | greedy | ts | ucb |
|---|---:|---:|---:|---:|
| 1K   | 22,224 ± 208  | 20,062 ± 711   | 20,397 ± 139  | 20,413 ± 108  |
| 5K   | 56,976 ± 1,187| 86,652 ± 8,074 | 55,711 ± 797  | 56,822 ± 670  |
| 25K  | 160,238 ± 8,681 | 287,865 ± 29,991 | 149,077 ± 1,426 | 149,597 ± 319 |

**Top-50 recall — higher is better**:

| Budget | baseline | greedy | ts | ucb |
|---|---:|---:|---:|---:|
| 1K   | 0.024 ± 0.012 | 0.020 ± 0.013 | 0.028 ± 0.010 | 0.032 ± 0.012 |
| 5K   | 0.236 ± 0.034 | 0.124 ± 0.050 | 0.188 ± 0.045 | 0.188 ± 0.036 |
| 25K  | 1.000 ± 0.000 | 0.464 ± 0.135 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**Total hits found (at score ≤ −25) — higher is better**:

| Budget | baseline | greedy | ts | ucb |
|---|---:|---:|---:|---:|
| 1K   | 518 ± 7   | 554 ± 9    | 544 ± 6   | 539 ± 5   |
| 5K   | 2,593 ± 37 | 1,727 ± 201 | 2,586 ± 19 | 2,560 ± 18 |
| 25K  | 10,073 ± 7 | 5,738 ± 732 | 10,073 ± 5 | 10,073 ± 4 |

## What we learned

**1. The baseline rule is solid.** At all three budgets, the existing
`expected_hits ** α × remainder` rule recovers all of the global top-50
at B=25K and finds 10K hits — within statistical noise of the bandit
policies. The MELSelection.md design works.

**2. UCB and Thompson Sampling improve marginally.** ~7% lower regret
than baseline at B=25K, with tighter seed-to-seed variance (TS stderr =
1,426 vs baseline 8,681 at B=25K — TS is more stable across seeds).
The bandit posteriors smooth out the probe-phase noise that makes the
baseline's `expected_hits ** α` weights swing seed-to-seed.

**3. ε-greedy is unsafe at scale.** At B=5K and 25K, the lone-exploit
MEL captures 90% of the budget but only sees the top of its own pool;
the 10% explore share is too thin to find the second- and third-best
MELs. Top-50 recall collapses to 46% at B=25K, vs 100% for the other
three policies. Don't ship this.

**4. The probe-phase passes too aggressively at our default
`--top-threshold -15`.** Every MEL in scope passed in every seed
(`n_mels_passing = 7` consistently). The pass/abort decision didn't
discriminate. Consider tightening to `-20` once the oracle has MELs
whose probes legitimately don't see a hit.

## What we don't know yet

**Statistical significance is weak with 7 MELs.** The 7% TS-vs-baseline
regret edge is on the order of the seed-to-seed noise at B=5K. We
need ≥50 MELs in the oracle before this conclusion can be acted on.

**The ML policy isn't tested here.** scikit-learn isn't in the system
Python and the project's primary env (`/usr/bin/python3`) doesn't have
pip. To run the ML arm: activate the OpenVsynthes mamba env and rerun.
The ML policy is implemented and tested under skip — see
[tests/test_al_ml.py](../tests/test_al_ml.py).

**Multi-round allocation isn't tested.** The current harness does ONE
allocate call per (policy, budget, seed). Bandit policies in
particular have a structural advantage when they can update their
posterior mid-budget and re-allocate. This is "Phase 3b" in
[the original plan](../../../home/aoxu/.claude/plans/ok-first-read-about-inherited-origami.md);
defer until the single-round numbers show enough signal to justify
the engineering.

**The oracle is biased toward MELs that have been run.** Of the 7 MELs
in scope, 4 are from the small-budget probe-only adaptive runs (500
synthons each); only 3 have ≥10K observations. A larger expanded
oracle (next bullet) would let us re-run with synthon-pool sizes
matching the production Top-1K distribution.

## Next steps (in priority order)

1. **Expand the oracle to ≥50 MELs.** Two paths:
   - Workstation SRG over a stratified MEL sample (50 MELs from the
     Top-1K spanning the synthon-count distribution, ~12 hours
     wall-clock with `nProc=0`).
   - CARC batch run via Wenjin's SLURM infrastructure (adopt the
     `run_ICM_ScreenReplacement_NodeWorker.py` pattern from
     [WenjinCode.md](WenjinCode.md), few wall-clock days for full
     Top-1K).
2. **Wire `--al-policy {baseline,greedy,ucb,ts,ml}` into `run_srg_batch.py`.**
   The policies are drop-in compatible — see
   [al_policies/base.py](../al_policies/base.py) for the interface —
   but no live-runner CLI exposure yet. Add the flag and the registry
   lookup at the call site (`allocate_budget(...)` at
   `run_srg_batch.py:481`); regression-test that `--al-policy baseline`
   produces bit-identical output to the legacy default.
3. **Enable the ML policy.** Install scikit-learn into the
   OpenVsynthes mamba env (`mamba install -n OpenVsynthes scikit-learn`);
   re-run the benchmark with `--policies baseline greedy ucb ts ml`.
4. **Multi-round allocation (Phase 3b).** Extend the harness so the
   bandit and ML policies can re-allocate after each commit batch.
   This is where AL has its strongest theoretical advantage; expect
   the regret gap to widen.
5. **Compare against CapSelect** — once the oracle expands to ≥50 MELs,
   re-rank them by CapSelect's `MergedScore` and ask: do the AL
   policies recover the same top-K MELs the geometric heuristic would
   have picked? This is the `MELSelection.md` Phase A → Phase B
   validation the original design doc anticipated.

## Reproducing the benchmark

```bash
cd /home/aoxu/projects/anchnor_based_VSYNTHES   # symlink to NAS

# 1. (Re-)build the oracle from existing SRG runs:
python3 oracle/build_srg_oracle.py

# 2. Run the sweep:
python3 al_benchmark/run_benchmark.py            # full
python3 al_benchmark/run_benchmark.py --quick    # 1 seed, fast

# 3. Inspect results:
cat al_benchmark/summary.tsv | column -t -s $'\t'
jupyter notebook notebooks/AL_benchmark_results.ipynb
```

All four policies are deterministic given a seed (TS uses
`random.Random(seed)`; UCB/baseline/greedy have no randomness beyond
the synthon-pool shuffle). Results are bit-reproducible.

## See also

- [MELSelection.md](MELSelection.md) — the baseline rule the AL
  policies extend
- [CapSelect.md](CapSelect.md) — the upstream MEL ranking the baseline
  replaces
- [WenjinCode.md](WenjinCode.md) — the data state on CARC, including
  why the AL oracle uses SRG scores rather than Stage 5 full-compound
  docking scores
- [../CLAUDE.md](../CLAUDE.md) — pipeline placement; cap convention;
  the ICM async-callback caveat that motivates the wait-and-read-binary
  block in our templates
- [../al_policies/](../al_policies/) — the four policies + the Protocol
  interface
- [../al_benchmark/run_benchmark.py](../al_benchmark/run_benchmark.py)
  — the replay harness
- [../oracle/build_srg_oracle.py](../oracle/build_srg_oracle.py) — the
  oracle builder
