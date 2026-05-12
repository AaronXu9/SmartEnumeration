# MEL filtering and synthon-budget allocation

Design proposal for replacing CapSelect's geometric proxy with a
score-distribution-based MEL filter that builds on the existing
Screen Replacement Group (SRG) infrastructure. Sister doc to
[CapSelect.md](CapSelect.md).

## TL;DR

CapSelect uses a hand-tuned geometric heuristic (sphere-chain depth
into the pocket) to predict whether a MEL is worth enumerating. That
prediction is a *proxy* for the real question: "if we enumerate this
MEL with synthons, will we find good binders?" The SRG step in this
project already answers that question directly — we have actual scores
for synthons at the cap attachment vector. **Use those.**

The proposal:

1. **Small-sample SRG pre-screen** on every top-30K MEL: dock a small
   uniform sample of N₀ synthons (e.g. N₀ = 50) and look at the score
   distribution.
2. **Drop MELs with empty left tails** (no synthon scored below a
   threshold T₀ in the sample).
3. **Redistribute the saved synthon budget** to the surviving MELs
   using a tail-sensitive metric (top-K mean × remaining-synthon
   factor, recommended).
4. **Re-dock the allocated synthon counts** per MEL.

This replaces "CapSelect picks top-N MELs uniformly enumerated" with
"SRG-prescreen picks survivors and gives each one a budget proportional
to its score-distribution evidence." The total compute is comparable
(both spend roughly one-MEL-dock-equivalent of budget per MEL pruned);
the signal is much better.

## Why CapSelect is a geometric proxy

CapSelect ranks MELs by `5·log₂|Score| + 0.5·log₂|CapScore|` where
`CapScore` is the count of pocket-fitting 2-Å spheres extending from
the cap. That's an answer to "how much room does the cap have to
grow?" Two implicit assumptions:

1. More room → more chance of finding a productive full-compound
2. Cap geometry is independent of what synthon goes there

Both are true on average, but neither is the actual signal we want.
What we want: *which MELs, after enumeration, produce binders?* SRG
scoring synthons at the attachment vector is a much closer-to-target
proxy than counting empty spheres. Empirically, on GPR91 we already
saw CapSelect re-orders only ~10–15% of the top-N selection vs
docking-Score alone — meaning it adds limited value beyond the
information already in the docking score.

The case for replacing CapSelect entirely is strong:

- We have SRG → we have *actual* score distributions per MEL
- The geometric heuristic ignores chemistry (e.g. polar synthons in
  hydrophobic pockets get the same CapScore as nonpolar)
- The signal-to-cost ratio of small-sample SRG is favorable: ~50
  synthon-docks ≈ 1.5× the cost of one MEL-dock, and gives a real
  distribution rather than a single scalar geometric proxy

## The budget framework

Frame the workflow as **fixed total compute, allocated to MELs**:

```
Total budget B = N_MELs_explored × n_synthons_each      (uniform baseline)
              = sum over MELs i of  n_i (allocated)     (informed)
```

Two sub-decisions:

1. **Pruning** — which MELs survive to receive any further budget?
2. **Allocation** — how is the surviving budget split among them?

### Phase A: small-sample SRG pre-screen (pruning)

Run SRG on every top-30K MEL with a small synthon sample size N₀
(typical 30–100). Choose synthons uniformly from each MEL's
compatible-synthon library (the library is already built upstream by
`Find_Compatible_And_Surviving_Syntons_*.py` in this project).

Collect for each MEL the small-sample distribution
`{s_i^(1), s_i^(2), ..., s_i^(N₀)}` of `<FullLigand_Score>` from the
SRG output (`modifiersTab` SDF).

Pruning rule (recommended): **drop MEL i if min(s_i) > T₀** where T₀
is set near the production hit threshold (e.g. −15 or −20 kcal/mol
depending on target). MELs with no left-tail signal in the small
sample are unlikely to produce binders even if their full library is
explored.

Why min, not mean: the goal is finding strong binders, not optimizing
the average. A MEL whose distribution averages −10 but spikes to −18
on one synthon is more interesting than a MEL whose distribution
averages −12 with no extremes.

Expected survivors: 30–60% of the top-30K MELs (varies by target /
score distribution). Dropping ~50% of the MELs doubles the per-MEL
budget for survivors.

### Phase B: allocation among survivors

You enumerated the candidate strategies in the prompt; here are the
tradeoffs distilled, with a recommendation.

**Central-tendency strategies (rejected):**

- *Mean score, median score* — both wash out the left tail. A MEL
  with one extreme hit looks "average" by these. Wrong objective
  for hit-finding.

**Tail-sensitive strategies (good choices):**

- *Hit rate (hits / N₀)* — direct, interpretable. Noisy at small N₀
  (1/50 vs 2/50 has 2× ratio difference but both are statistically
  weak signals; the variance of a binomial proportion at N₀=50 with
  p=0.04 has std error √(0.04·0.96/50) ≈ 0.028, so two MELs with
  rates 0.02 and 0.06 are within 1σ of each other).

- *Absolute hit count* — favors larger MEL libraries even if rate is
  lower, which is *correct* if the goal is total hits found from
  the budget. Doesn't disadvantage small MEL libraries unfairly
  because the small-sample size N₀ is uniform across MELs.

- *Best score (min)* — rewards extreme outliers. Maximally
  tail-sensitive but maximally variance-prone. A MEL gets all its
  budget because of one lucky draw. Use only with a robust prior.

- *Top-K mean (e.g. mean of top 3)* — best compromise for K=3 to 5.
  Robust to single-draw flukes, still tail-sensitive. **This is the
  recommended primary signal.**

**Synthon-space-aware strategies (apply on top):**

- *Remaining synthons* — Some MEL libraries have 1k synthons, some
  have 10k. Allocating budget proportional to remaining-synthon
  count alone is "just spread the budget" — neutral to score signal.

- *Hit rate × remaining synthons* — multiplies efficiency-per-draw by
  unexplored space. A high-hit-rate MEL with a large library gets
  the largest allocation. **This is the recommended composite.**

- *Top-K mean × remaining synthons* — same intent but with the
  more-robust top-K signal instead of hit rate. Equivalent to
  recommended primary above when budget-allocation knob is
  "expected number of new hits per unit budget."

**Recommended allocation rule:**

```
weight_i = (top-K mean of small-sample scores)_i  ×  (remaining synthons)_i
budget_i = B_remaining × weight_i / sum_j(weight_j)
```

with K=3 unless you have a reason to tune it. If the top-K mean is
negative (more-negative = better), use `(-top_K_mean_i)` so weights
are positive. Floor budget_i at some minimum (e.g. n_min = 50) to
maintain coverage of the distribution tails — pure greedy can
under-explore.

### Phase C: optional ML / pocket-aware refinement

The SRG-prescreen approach is a strong baseline. To go further:

**Idea: predict hit rate from cheap pocket+MEL features.**

Train a regressor:

```
features:  pocket fingerprint at attachment vector  +  MEL descriptors
target:    hit rate observed in small-sample SRG (= hits / N₀)
```

Once trained, use the regressor to predict hit rate **before**
running the small-sample SRG, and use the prediction to:

- Skip the small-sample step entirely for MELs with predicted rate
  near zero (saves compute)
- Or warm-start the budget allocation (use predicted rate as a prior,
  refine with observed sample)

Pocket fingerprint candidates:

- Atomic environment vector at the attachment vector (radial-shell
  histogram of element types within 6 Å of the APO point)
- ICM grid map values (g_e, g_h, g_c, g_b, g_s) sampled in a small
  cube around the attachment vector — already computed for docking
- Simple shape descriptors (pocket volume reachable from attachment,
  pocket polarity ratio, max bottleneck radius)

MEL descriptors: scaffold MW, polarity, attachment vector orientation
relative to pocket axis, etc. (cheap, computed once per MEL)

Bootstrap data: the project's existing SRG runs already produce
per-MEL score distributions on real targets (CB2 5ZTY, GPR91 6RNK).
That's enough labelled data to train a target-agnostic model with
modest complexity (e.g. gradient boosting on ~30 features).

Caveat: this adds engineering. Worth it only after the simpler
SRG-prescreen approach has been benchmarked and a clear signal is
visible in the residuals (i.e. the prescreen-derived budget allocation
leaves predictable structure on the table).

## Concrete pilot proposal

Run on one target where we already have full top-30K MEL data
(GPR91 6RNK is a candidate, since we have docked MELs and the SRG
infrastructure runs on it).

**Pilot setup (1 target, single afternoon of compute):**

1. **Baseline**: take the existing CapSelect-ranked top-1000 MELs
   from `capselect/gpr91/frags_for_enum_mol2_ranking.tsv`. For each,
   record CapScore, MergedScore, docking Score.

2. **SRG small-sample**: run SRG on each of the top-1000 MELs with
   N₀=50 synthons each (50K total synthon-docks). Use the existing
   `run_ICM_ScreenReplacement_*.icm` driver, just sample 50 synthons
   from each MEL's compatible-synthon SDF before running.

3. **Distribution metrics per MEL**:
   - min score
   - top-3 mean score
   - hit count below threshold T₀ (set T₀ = the empirical 1st-percentile
     of the pooled score distribution)

4. **Compare rankings**:
   - CapSelect MergedScore-based top-K
   - Small-sample top-3-mean × remaining-synthons-based top-K
   - Score-only (docking) top-K

5. **Validate**: enumerate the union of the top-K sets fully (all
   compatible synthons per MEL). For each MEL, compute the *true*
   hit rate (hits in full enumeration). Then evaluate which top-K
   strategy has the highest recall on the truly-hit-rich MELs.

6. **Decide**: if the SRG-prescreen ranking out-recalls CapSelect on
   true-hit-rich MELs by ≥10%, it's worth productionizing.

**Estimated cost**: 50K synthon-docks for the pilot (≈ 5–10 hours on
CARC depending on parallelism); subsequent full enumeration of top-K
union (~3–5K MELs × ~1k synthons each = ~3M docks, multi-day on CARC).

## What NOT to do

- **Don't blend CapScore into the SRG-prescreen weight.** They're
  measuring different things; combining adds noise. CapSelect's value
  is as a comparison baseline, not as a signal to mix into the new
  approach. (If a hybrid is needed later, run CapSelect FIRST as a
  cheap pre-prune to drop the bottom 10% by CapScore = 0, then SRG-
  prescreen the remaining 90%.)

- **Don't use a large N₀.** The whole point of small-sample SRG is
  cheap signal acquisition. N₀ = 50 is the sweet spot — large enough
  to detect 1-in-50 hits, small enough that pruning is worth the
  prescreen cost. Going to N₀ = 200 inflates the prescreen cost 4×
  for marginal gain in hit-rate variance.

- **Don't skip the floor budget.** Pure proportional allocation will
  starve MELs whose small-sample didn't happen to draw a hit by
  chance. n_min = 50 (= N₀) ensures the prescreen sample alone is
  retained as a safety net.

- **Don't bias the synthon sample.** The small-sample SRG must use
  uniform random synthons from each MEL's compatible-synthon library.
  If you cherry-pick "diverse" synthons for the prescreen, you bias
  the score distribution and the budget allocation downstream.

## Pointers

- Existing SRG driver: [run_ICM_ScreenReplacement_SingleMEL_GUI_Parallel.icm](../run_ICM_ScreenReplacement_SingleMEL_GUI_Parallel.icm)
- Cap removal preprocessor: [edit_mel_cap.py](../edit_mel_cap.py)
- Compatible-synthon filter: [Find_Compatible_And_Surviving_Syntons_TopN_MELs.py](../Find_Compatible_And_Surviving_Syntons_TopN_MELs.py)
- Batch runner template: [run_srg_batch.py](../run_srg_batch.py) (where `--only-row` could be adapted to `--sample-n`)
- CapSelect comparison baseline: [capselect/gpr91/frags_for_enum_mol2_ranking.tsv](../capselect/gpr91/frags_for_enum_mol2_ranking.tsv)
- Project context and pipeline placement: [../CLAUDE.md](../CLAUDE.md)
