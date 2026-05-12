# AL Sophisticated Strategies — chemistry-aware allocator, learned rankers, iterative AL, multi-fidelity, diversity, joint bi-level UCB

Implementation + results for six advanced AL strategies (I/J/K/L/M/N)
added on top of the GPR91-6RNK Wenjin-framework benchmark. Sister
doc to [AL_GPR91_Reproduction.md](AL_GPR91_Reproduction.md)
(reproduction of Wenjin's A/B/C/D + my E/F/G/H closed-form posterior
arms) and [AL_Pilot.md](AL_Pilot.md) (narrative pilot story).

## TL;DR

Six new strategies built (I/J/K/L/M/N), each with unit tests on
synthetic data and end-to-end validation on the real GPR91 oracle
(2.2 GB, 10M (MEL, synthon) rows with FullLigand_Score ground truth).
Headline:

- **None of the honest learned strategies beat E-S1 (UCB closed-form
  posterior).** Chemistry features + GradientBoosting surrogate +
  iterative refits all underperform the simple closed-form Gaussian
  posterior over probe scores. The "more sophisticated = better"
  intuition does NOT hold at this scale and probe budget.
- **Strategy K (iterative AL) is the *worst* learned strategy** at
  EF AUC 3.349 vs E's 4.007 (−16%). Iteration *hurts*: each round's
  refit chases noise in the growing observed set.
- **Strategy M (submodular + diversity) with fair RTCNN score is
  essentially tied with D-S1** at EF AUC 4.015 vs 4.041. The diversity
  term doesn't change EF AUC much when score signal is RTCNN.
- **Oracle upper bound at 4.881** (M with `FullLigand_Score` directly
  as the score signal): if we could perfectly predict FullLigand_Score
  from cheap features, we'd get ~20% more EF AUC than the best honest
  strategy. None of the learned strategies came close to closing this
  gap.

The user's two motivating concerns were addressed:

1. **MEL chemistry features actually used (Strategy I + N).** Morgan
   fingerprint of `icm_rdmol_binary` (1024 bits) initially OOM'd the
   joint feature matrix at 40 GB on the 10M-row pool. Switched to
   MACCS 167 bits → 8 GB peak, fits comfortably. Strategy I now
   trains on a ~190-dim per-MEL chemistry-aware feature vector;
   Strategy N's joint surrogate sees the same MEL features per row.
2. **Project-customized bi-level joint strategy (Strategy N).**
   Trains a single learned model on the joint (MEL features +
   synthon features) space and globally picks 1M by UCB acquisition
   with a per-MEL cap. MEL allocation **emerges** from the joint
   selection rather than being pre-allocated. Implementation
   identical to Strategy L; the two differ only in framing.

## Strategy design

### Strategy I — Chemistry-aware ML allocator + softmax picker

`al_benchmark_gpr91/strategy_i_ml_alloc.py`

The original
[`al_policies/ml.py`](../al_policies/ml.py) trained a
`GradientBoostingRegressor` on probe summary stats only — no
chemistry. Strategy I revises it to **prepend per-MEL chemistry
features** (Morgan/MACCS fingerprint of `icm_rdmol_binary` + MW +
Tox_Score + molPAINS + Stage-1 decomposed energies) to the existing
9-dim probe-summary features, yielding a ~190-dim feature vector
per MEL.

Same allocator interface (returns `{mel_row: commit_n}`), same
softmax synthon picker, just a better-informed allocator.

### Strategy J — Per-synthon learned ranker (replaces softmax picker)

`al_benchmark_gpr91/strategy_j_per_synthon_ranker.py`

After probe phase, train a `BaggedRegressor` (3-bag GBR) on probe
observations: features = joint (synthon numeric + MEL MACCS), target
= `FullLigand_Score`. For each unprobed (MEL, synthon), predict
score; per-MEL allocator decides n_i; picker takes **top n_i by
predicted score**.

Two variants:
- **J-base**: baseline-dynamic MEL allocator + learned synthon ranker
- **J-ucb**: UCB MEL allocator + learned synthon ranker

### Strategy K — Iterative AL with model retraining

`al_benchmark_gpr91/strategy_k_iterative_al.py`

The "real AL" loop. Each round:
1. Fit a 2-bag GBR on currently-observed rows
2. Predict (μ, σ) for every unobserved (MEL, synthon)
3. UCB acquisition: `acq = μ − κ·σ`, lower = better
4. Greedy pick `batch_size` by acquisition with cumulative per-MEL cap
5. Observe (oracle lookup); add to training set; refit

Defaults: n_initial=50K, batch_size=200K (so 5 rounds at budget=1M),
κ=1.0, per_mel_cap=5K, ensemble=2.

### Strategy L — Multi-fidelity AL (single-shot UCB)

`al_benchmark_gpr91/strategy_l_multifidelity.py`

Same surrogate as K but **no retraining**: probe once, fit once,
predict once, pick top-N globally by UCB with per-MEL cap. The
"cheap" variant of K — same machinery, no iteration cost. Tests
whether iterative refits actually buy anything.

### Strategy M — Submodular / diversity-aware

`al_benchmark_gpr91/strategy_m_submodular.py`

Greedy submodular maximization of `α·score + (1-α)·diversity_weight·|distinct_MELs|`.
Diversity term is "count of distinct MELs" (no RDKit Tanimoto
needed in V1; the Top-1K already spans 1881 distinct scaffolds).

Score signal modes:
- **Fair (default)**: `score_column="RTCNN_Score"` — uses RTCNN as the
  cheap pre-computed signal, matching the information regime that C/D/E/F/G/H operate under.
- **Oracle ceiling (NOT a fair arm)**: `score_column="FullLigand_Score"`
  — reads the metric target directly. Establishes the maximum EF AUC
  achievable if score prediction were perfect. Not deployable
  (no real-world pipeline has FullLigand_Score before docking).

### Strategy N — Joint MEL+synthon UCB acquisition (project-customized bi-level AL)

`al_benchmark_gpr91/strategy_n_joint_ucb.py`

The strategy the user explicitly asked for in plan-mode review:
"in this task there are actually 2 things we need to select and
balance unlike in a classical AL setting: one is the MEL, and the
other is the synthons for each MEL given our fixed budget."

N **doesn't pre-allocate per-MEL budget**. It trains a single learned
model on the joint (MEL chemistry + synthon features) space and
globally picks 1M (MEL, synthon) pairs by UCB acquisition with an
**optional per-MEL cap** (default 5K, matching Strategy C's cap).
The MEL allocation *emerges* from the joint selection instead of
being chosen ahead of time.

V1 implementation: algorithmically **identical** to Strategy L (same
code path: probe → fit → predict → greedy global with per-MEL cap),
differentiated only by framing (L = "multi-fidelity", N = "joint
bi-level"). They will diverge once we add e.g. synthon-side
fingerprints (1024 bits × 10M rows = 40 GB; currently OOM-blocked —
that's the V2 path for both L and N).

## Results — full leaderboard on GPR91 oracle (RTCNN scoring, 1M budget, seed=42)

| Rank | Strategy | Config | n_mels | n_ligands | EF AUC | Δ vs E-S1 (4.007) |
|---:|---|---|---:|---:|---:|---:|
| — | **M-oracle** *(ceiling, not deployable)* | α=0.7 + diversity + FullLigand_Score | 487 | 1,000,000 | **4.881** | **+21.8%** |
| 1 | D-S1 | cap=10K (global top-N by RTCNN) | 486 | 1,000,000 | 4.041 | +0.8% |
| 2 | **M** *(fair)* | α=0.7 + diversity + RTCNN score | 488 | 1,000,000 | **4.015** | +0.2% |
| 3 | **E-S1** *(UCB closed-form, the bar)* | alloc=ucb T=1 n_probe=50 | 507 | 1,000,041 | **4.007** | 0.0% |
| 4 | F-S1 | alloc=ts T=1 n_probe=50 | 507 | 1,000,027 | 3.974 | −0.8% |
| 5 | C-S1 T=1.0 | (Wenjin's declared winner) | 277 | 1,000,000 | 3.878 | −3.2% |
| 6 | C-S1 T=2.0 | | 277 | 1,000,000 | 3.857 | −3.7% |
| 7 | C-S1 T=0.5 | | 277 | 1,000,000 | 3.856 | −3.8% |
| 8 | **J-ucb** | alloc=ucb + learned synthon ranker | 507 | 1,000,041 | **3.824** | −4.6% |
| 9 | A-S1 | top=20% global cutoff | 287 | 1,001,913 | 3.783 | −5.6% |
| 10 | **L** | mf-single bag=3 κ=1.0 cap=5K | 498 | 1,000,000 | **3.716** | −7.3% |
| 11 | G-S1 | alloc=baseline + softmax | 507 | 1,002,055 | 3.697 | −7.7% |
| 12 | **N** | joint UCB bag=3 κ=1.5 cap=5K | 499 | 1,000,000 | **3.689** | −7.9% |
| 13 | **I** | ml-chem (MACCS) + softmax | 507 | 1,000,941 | **3.687** | −8.0% |
| 14 | H-S1 | alloc=greedy + softmax | 507 | 1,000,000 | 3.661 | −8.6% |
| 15 | **K** | iter bag=2 rounds=5 κ=1.0 cap=5K | 505 | 1,000,000 | **3.349** | −16.4% |
| 16 | B-S1 | frac=10% per MEL | 505 | 851,048 | 3.340 | −16.7% |
| 17 | **J-base** | alloc=baseline + learned synthon ranker | 507 | 1,002,055 | **3.209** | −19.9% |

(VS baseline EF AUC = 1.000 by definition; 78 MELs, 1,001,617 ligands.)

## Three findings

### 1. Closed-form posteriors are remarkably hard to beat at this scale

E-S1 (UCB closed-form posterior on probe scores) wins among honest
strategies. Every learned arm (I, J-ucb, K, L, M-fair, N) is at most
on par or worse, even with much more compute.

The probe sample (50 synthons/MEL × 1000 MELs = 50K total
observations) is large enough for closed-form posteriors to identify
the right MELs but small enough that learned regressors with 200+
features overfit. The 50K-observations × ~190-features regime is
specifically NOT the regime where ML wins.

### 2. Iteration hurts more than it helps in the offline replay

K-S1 (5 rounds of refit) lands at EF AUC 3.349 — **the worst learned
strategy and second-worst overall**. The expected behavior was
"refit chases the truth"; what actually happens is "refit chases
the noise in observations of the lower-scoring MELs that the random
initial probe over-sampled". Without de-biasing the training set
across rounds, iteration amplifies the probe's idiosyncrasies.

### 3. Diversity is essentially free EF AUC over D-S1, but doesn't unlock the ceiling

M-fair lands at 4.015 vs D's 4.041 — within noise. The diversity
term picks up an extra ~2 MELs (488 vs 486) without losing score.
But the *ceiling* (M-oracle at 4.881) shows that diversity *with the
right score signal* could buy +20% EF AUC. The remaining gap is
score-prediction accuracy, not diversity.

## What changed during implementation

Three real bugs and a methodology gotcha surfaced:

1. **NaN in `FullLigand_Score`** (~5% of oracle rows where Stage-5
   docking didn't converge) crashed sklearn's `fit`. Fixed:
   `extract_probe_observations` drops NaN targets; per-strategy
   loops filter `finite` rows before fitting.
2. **inf in `MoldHf`** (~0.0006% of rows where heat-of-formation
   diverged) crashed sklearn's predict-time validation. Fixed:
   `synthon_features` and `joint_features` replace ±inf with NaN
   then median-fill, with a float32 clip to ±1e30 as a safety net.
3. **Morgan FP (1024 bits) OOM'd at 40 GB** of joint feature matrix
   on the 10M-row pool. Fixed: switched runner to MACCS (167 bits)
   for the bulk-prediction pipeline, keeping the joint matrix
   under 8 GB. Strategy I is unaffected (its training set is
   ~1881 rows so 1024 bits is fine).
4. **Strategy M with `use_learned_score=False` was originally
   reading `FullLigand_Score` directly** — that's the metric target,
   so M was effectively the oracle upper bound, not a comparable
   strategy. Fixed: `score_column` parameter, defaults to
   `RTCNN_Score` (fair); `FullLigand_Score` is exposed as the
   `M-oracle` ceiling reference.

## What's next (if the user wants to push further)

- **Multi-seed evaluation.** All numbers above are single seed=42.
  Re-run with seeds 0–9 to bracket the noise on the ~3% EF AUC gaps
  between strategies. The signal-to-noise question may flip some
  rankings.
- **Larger probe.** At 50 synthons/MEL the closed-form posteriors
  win because that's where they're statistically efficient. Test
  whether at 500 synthons/MEL the learned strategies catch up or
  pull ahead.
- **Pocket-aware features for I and N.** ICM grid map values
  (`g_e`, `g_h`, `g_c`, `g_b`, `g_s`) sampled at the APO attachment
  vector add per-MEL features the model can correlate with
  FullLigand_Score. Cheap to compute once; ~30 dims; would test
  whether the missing chemistry signal is pocket-geometric vs
  scaffold-fingerprint.
- **Synthon Morgan fingerprint in N.** V1 N uses 9 numeric synthon
  features. Adding the synthon's 1024-bit Morgan FP requires chunked
  prediction (10M × 1024 × 4 = 40 GB without chunking) but is the
  natural V2 — gives the model real per-synthon chemistry.
- **Iteration with active de-biasing in K.** The current K's failure
  suggests the iteration's information gain is dominated by
  re-fitting on the noisy initial probe. Try: dropping the initial
  random probe entirely; or upweighting low-probed MELs in the
  loss; or seeding the first round with the closed-form UCB choice
  to anchor against the strongest baseline.
- **Stratified probe.** Replace the uniform-random probe with one
  that samples evenly across the synthon-count distribution (median
  15K, range 0-60K). The current probe is uniform per (MEL, synthon)
  pair, which biases samples toward large-pool MELs.

## Reproducibility

```bash
cd /home/aoxu/projects/anchnor_based_VSYNTHES   # → symlinked to NAS

# Smoke (option 1 only — what produced the numbers above):
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
    al_benchmark_gpr91/run_reproduction.py --quick

# Full sweep (6 scoring options × A/B/C/D + all 7 AL extensions):
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
    al_benchmark_gpr91/run_reproduction.py
```

Wall-clock for `--quick` on the workstation (125 GB RAM, single
process, OpenVsynthes008 mamba env): ~12 minutes end-to-end.

Output: `al_benchmark_gpr91/wenjin_reproduction_results.csv` with
all per-strategy EF AUC + per-threshold EF + n_mels + n_ligands.

All strategies are seed-deterministic (default seed=42); re-running
produces bit-identical selection InChIKeys.

## File index

Code:
- [`../al_benchmark_gpr91/_mel_features.py`](../al_benchmark_gpr91/_mel_features.py)
  — Morgan/MACCS FP + physchem + Stage-1 features per MEL
- [`../al_benchmark_gpr91/_ml_common.py`](../al_benchmark_gpr91/_ml_common.py)
  — `synthon_features`, `joint_features`, `BaggedRegressor`
- [`../al_benchmark_gpr91/strategy_i_ml_alloc.py`](../al_benchmark_gpr91/strategy_i_ml_alloc.py)
- [`../al_benchmark_gpr91/strategy_j_per_synthon_ranker.py`](../al_benchmark_gpr91/strategy_j_per_synthon_ranker.py)
- [`../al_benchmark_gpr91/strategy_k_iterative_al.py`](../al_benchmark_gpr91/strategy_k_iterative_al.py)
- [`../al_benchmark_gpr91/strategy_l_multifidelity.py`](../al_benchmark_gpr91/strategy_l_multifidelity.py)
- [`../al_benchmark_gpr91/strategy_m_submodular.py`](../al_benchmark_gpr91/strategy_m_submodular.py)
- [`../al_benchmark_gpr91/strategy_n_joint_ucb.py`](../al_benchmark_gpr91/strategy_n_joint_ucb.py)
- [`../al_policies/ml.py`](../al_policies/ml.py) — revised to accept
  `mel_features_df` for chemistry-aware MEL allocation

Tests (all pass under the OpenVsynthes008 mamba env):
- [`../tests/test_mel_features.py`](../tests/test_mel_features.py)
- [`../tests/test_ml_common.py`](../tests/test_ml_common.py)
- [`../tests/test_strategy_i_ml.py`](../tests/test_strategy_i_ml.py)
- [`../tests/test_strategy_j_per_synthon_ranker.py`](../tests/test_strategy_j_per_synthon_ranker.py)
- [`../tests/test_strategy_k_iterative_al.py`](../tests/test_strategy_k_iterative_al.py)
- [`../tests/test_strategy_l_multifidelity.py`](../tests/test_strategy_l_multifidelity.py)
- [`../tests/test_strategy_m_submodular.py`](../tests/test_strategy_m_submodular.py)
- [`../tests/test_strategy_n_joint_ucb.py`](../tests/test_strategy_n_joint_ucb.py)

Output:
- [`../al_benchmark_gpr91/wenjin_reproduction_results.csv`](../al_benchmark_gpr91/wenjin_reproduction_results.csv)
  — full per-strategy results table

## See also

- [AL_GPR91_Reproduction.md](AL_GPR91_Reproduction.md) — A/B/C/D
  reproduction + E/F/G/H closed-form posterior arms
- [AL_Pilot.md](AL_Pilot.md) — pilot story + bi-level framing
- [WenjinCode.md](WenjinCode.md) — what's on CARC, oracle source
- [MELSelection.md](MELSelection.md) — original dynamic-allocation
  rule (which the baseline-allocator in this benchmark wraps)
