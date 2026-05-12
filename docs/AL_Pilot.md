# AL benchmark — idea, implementation, results

Project memory of the offline active-learning benchmark for the
MEL+synthon selection step of V-SYNTHES Stage 3. This doc is the
narrative summary; for the full per-strategy numbers, methodology,
and reproduction commands see [AL_GPR91_Reproduction.md](AL_GPR91_Reproduction.md).

## The idea

V-SYNTHES Stage 3 docks synthons at a MEL's cap attachment vector and
ranks the resulting full ligands by docking score. The pipeline has
a fixed compute budget — practically, a fixed total number of
synthon-dock evaluations — and that budget has to be allocated across
MELs. The question is **which MELs get how much budget, and which
synthons per MEL get docked**.

[MELSelection.md](MELSelection.md) proposed a dynamic-allocation rule
(probe each MEL with N₀≈50 synthons, then weight remaining budget by
top-K mean × remaining synthons). Wenjin Liu's GPR91-6RNK notebook
[`../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb`](../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb)
takes a different angle: walk MELs in Stage-1 docking-rank order and
**softmax-sample synthons within each MEL** by RTCNN_Score (her
"Strategy C"). She validates with Enrichment Factor against a random
1M reference.

This benchmark answers: **can an active-learning policy beat both
the rank-walk baseline (V-SYNTHES default) and Wenjin's Strategy C
in her own EF framework, using probe-and-allocate?**

## Implementation

Three artifacts live in [`../al_benchmark_gpr91/`](../al_benchmark_gpr91/):

- **`wenjin_strategies.py`** — ports Wenjin's four strategies (A/B/C/D)
  + her VS rank-walk baseline + her EF metric into testable Python.
  Naming corrected: her cell 18 calls the Strategy C implementation
  `run_strategy_d`; we renamed.
- **`al_ext_strategies.py`** — four AL-extension arms (E/F/G/H), each
  layering one of my MEL-level allocators from
  [`../al_policies/`](../al_policies/) on top of Wenjin's softmax
  synthon picker. Two-phase shape: probe N₀=50 random synthons per
  MEL → MEL-level allocator decides per-MEL commit budget → softmax
  picks synthons within each MEL → Wenjin-style second-pass fill if
  short of total budget.
- **`run_reproduction.py`** — loads
  [`../csv/all_mels_combined_core.csv`](../csv/) (Wenjin's 2.2 GB
  joined oracle) + the MEL ranking + Random1M reference; runs all
  4×6 + 4 strategy combinations; dumps a ranked EF AUC table.

Tests in
[`../tests/test_wenjin_strategies.py`](../tests/test_wenjin_strategies.py)
and
[`../tests/test_al_ext_strategies.py`](../tests/test_al_ext_strategies.py)
(16 passing on synthetic data, no oracle needed).

## Results — full 6-option sweep on GPR91-6RNK

Bit-exact reproduction of Wenjin's saved cell-26 numbers
(D-S1=4.041, D-S2=4.029, D-S3=4.041, D-S5=4.006, C-S1 T=1.0=3.878,
T=2.0=3.857, T=0.5=3.856 — all match to 3 decimal places). Top 10
by EF AUC:

| Rank | Strategy | Config | n_mels | n_ligands | EF AUC |
|---:|---|---|---:|---:|---:|
| 1 | D-S1 RTCNN global rank | cap=10K | 486 | 1,000,000 | **4.041** |
| 2 | D-S3 Strain+RTCNN global rank | cap=10K | 486 | 1,000,000 | **4.041** |
| 3 | D-S2 Hard filter + RTCNN global rank | cap=10K | 481 | 1,000,000 | 4.029 |
| **4** | **E-S1 UCB + softmax (AL ext)** | alloc=ucb T=1 N₀=50 | **507** | 1,000,041 | **4.007** |
| 5 | D-S5 Strain+CoreRMSD+RTCNN global rank | cap=10K | 473 | 1,000,000 | 4.006 |
| **6** | **F-S1 TS + softmax (AL ext)** | alloc=ts T=1 N₀=50 | **507** | 1,000,027 | **3.974** |
| 7 | **C-S1 (Wenjin's declared winner)** | T=1.0 | 277 | 1,000,000 | **3.878** |
| 8 | C-S1 | T=2.0 | 277 | 1,000,000 | 3.857 |
| 9 | C-S1 | T=0.5 | 277 | 1,000,000 | 3.856 |
| 10 | B-S2 top-X% per MEL + hard filter | frac=10% | 507 | 851K | 3.844 |

VS rank-walk baseline EF AUC = 1.000 by definition.

## Three findings

**1. Wenjin's declared winner C is not the EF AUC winner.** D is.
Strategy D wins in our run AND in her saved data, but she chose C in
cell 29. The reason: D ignores MEL rank order entirely (it's "global
top-N from per-MEL-capped pools"). It's not really a selection
strategy — it's a degenerate global sort with a fairness floor. C
wins among the **non-degenerate** candidates (the strategies that
walk MELs in some order), and that's the relevant comparison for
V-SYNTHES.

**2. UCB and TS AL extensions beat C-S1 by 2–3%.** E-S1 (UCB +
softmax) at 4.007 and F-S1 (TS + softmax) at 3.974 both clear C-S1
at 3.878 — while *respecting* MEL rank order and touching all 507
passing MELs (vs C's 277). The probe phase gives the allocator real
per-MEL hit-density signal; UCB/TS turn that signal into a
non-uniform per-MEL budget that the softmax picker then concentrates
on the strong synthons within each MEL.

**3. Baseline-dynamic and ε-greedy AL extensions underperform C.**
G-S1 (baseline_dynamic + softmax) at 3.697 and H-S1 (greedy +
softmax) at 3.661 both lose to C-S1. The allocator-quality story
holds: bandit-Gaussian-posterior policies (UCB/TS) compose well with
the softmax synthon picker; the simpler rules don't.

## Important methodology gotcha

The first run had H-S1 (ε-greedy) "winning" with EF AUC = 4.085 —
but it had only 884K of 1M ligands. The greedy's concentrated
allocation hit per-MEL caps and lost budget; with fewer total
ligands selected and a similar absolute hit count, its hit-rate-ratio
(= EF) was artificially inflated.

The fix is Wenjin's **second-pass fill** (from her Strategy C cell 18):
if the first-pass selection is short of the budget, sample more from
each MEL's leftover pool in MEL-rank order. With that added to
`_pick_synthons_softmax`, H-S1 hit 1M ligands and dropped to EF AUC
3.661. **Any new AL strategy here needs the second-pass fill before
its EF AUC is comparable.**

## Earlier CB2 SRG-proxy exploration (historical note)

Before the GPR91 pivot, an offline AL benchmark was built on CB2_5ZTY
using **Stage-3 SRG RTCNN as both the selection signal and the
oracle** (we hadn't yet found Wenjin's GPR91 Stage-5 docking results
on CARC). That pilot defined the MEL-level `AllocationPolicy` Protocol
+ four policies (baseline / UCB / TS / greedy) under
[`../al_policies/`](../al_policies/), with unit tests under
[`../tests/test_al_baseline.py`](../tests/test_al_baseline.py),
[`../tests/test_al_greedy.py`](../tests/test_al_greedy.py),
[`../tests/test_al_bandit.py`](../tests/test_al_bandit.py),
[`../tests/test_al_ml.py`](../tests/test_al_ml.py). The policies are
reused by E/F/G/H above; the unit tests still pass.

**The CB2 pilot's *numerical results* are not valid evidence for the
GPR91-style claim** — different oracle (SRG proxy not Stage-5
docking), different metric (regret/recall not EF), different scope
(7 MELs not 1000+). They're still on disk at
[`../al_benchmark/`](../al_benchmark/) +
[`../oracle/`](../oracle/) +
[`../docs/AL_Benchmark.md`](AL_Benchmark.md) for historical reference,
but the conclusion "AL works on GPR91" rests on the GPR91 numbers
above, not the CB2 ones.

## Reproducibility

```bash
cd /home/aoxu/projects/anchnor_based_VSYNTHES   # symlink to NAS

# Smoke (~42s, option 1 only — RTCNN softmax pathway):
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
    al_benchmark_gpr91/run_reproduction.py --quick

# Full sweep (~2m20s, 6 scoring options × 4 Wenjin strategies + 4 AL extensions):
/home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python \
    al_benchmark_gpr91/run_reproduction.py
```

Output: `al_benchmark_gpr91/wenjin_reproduction_results.csv`. Single
seed (=42), deterministic given seed.

## What's missing / next steps

- **7WC6 target.** No Stage-5 ground truth on CARC. Run Stage-3 SRG
  over its Top-1K to build an SRG-proxy oracle (workstation, ~1 day),
  then rerun the same sweep. Per the user's earlier choice, this is
  the second target on the to-do list.
- **Multi-seed evaluation.** Currently a single seed=42 run. With
  small differences (~2-3% E/F vs C), running 5–10 seeds would tell
  us whether the AL edge is significant or seed-noise.
- **Sweep `n_probe`.** Currently fixed at 50. `n_probe ∈ {10, 50, 200,
  500}` would show how the AL signal scales with probe size — too
  few and the allocator has no signal; too many and budget is wasted
  on redundant info.
- **Sophisticated AL techniques.** UCB and TS are closed-form
  posteriors. Worth considering: GP/RF regression (already gated in
  [`../al_policies/ml.py`](../al_policies/ml.py) — turn it on as a
  fifth AL arm), per-synthon neural ranker on (RTCNN, Strain,
  CoreRMSD, MEL features), iterative AL with model retraining, and
  multi-fidelity AL treating RTCNN and FullLigand_Score as cheap/
  expensive fidelities. See "Next AL techniques to consider" below.

## Next AL techniques to consider

Listed roughly in increasing engineering cost. None implemented yet
— this section is for the next iteration of discussion.

1. **Enable the gated GP/RF allocator** ([`al_policies/ml.py`](../al_policies/ml.py)).
   Existing scaffold: per-MEL features = pool size, log pool size,
   probe summary stats, expected hits. Target = observed per-MEL hit
   rate at the probe. With `OpenVsynthes008` mamba env (which has
   sklearn), this becomes a 1-line activation. Adds Strategy I to
   the sweep.
2. **Per-synthon learned ranker.** Train a `GradientBoostingRegressor`
   (or small MLP) on (synthon RTCNN, Strain, CoreRmsd, MolLogP,
   MolPSA, MolVolume, MoldHf, MEL pool size, MEL Stage-1 Score) →
   predicted FullLigand_Score. Replace the softmax picker with
   "top-N by predicted FullLigand_Score". Tests whether a learned
   surrogate beats Wenjin's RTCNN-only softmax. Strategy J.
3. **Iterative AL with model retraining.** Currently the AL extensions
   are single-step: probe → allocate → commit. True AL does multiple
   rounds: probe a few MELs → fit a model → allocate the next batch
   → observe → refit. The oracle CSV is full-information, so we can
   simulate this offline with a controlled "what would the model
   have predicted at round k" replay. Strategy K.
4. **Multi-fidelity AL** (RTCNN cheap, FullLigand_Score expensive).
   The bandit picks not just **which** (MEL, synthon) but also at
   **what fidelity**. Useful if we ever want to validate the AL
   extension story against a smaller real Stage-5 docking budget.
   Strategy L.
5. **Submodular / diverse selection.** Bake in scaffold diversity or
   pharmacophore clustering. Important if downstream users care about
   chemical diversity of hits, not just hit count. Strategy M.

## See also

- [AL_GPR91_Reproduction.md](AL_GPR91_Reproduction.md) — full reproduction
  doc with all 6 scoring options, per-strategy interpretation, "why C
  and not D" analysis
- [MELSelection.md](MELSelection.md) — the dynamic-allocation rule's
  design doc (the baseline policy for the CB2 pilot)
- [WenjinCode.md](WenjinCode.md) — what's on CARC for each target
  (corrected 2026-05-11 PM after GPR91's docking-results dir was found)
- [AL_Benchmark.md](AL_Benchmark.md) — CB2 SRG-proxy pilot reference
  numbers (numerically invalid in the GPR91 EF framework — see disclaimer
  at top of that doc)
- [CapSelect.md](CapSelect.md) — upstream geometric MEL ranker
- [../CLAUDE.md](../CLAUDE.md) — project orientation
