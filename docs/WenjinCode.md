# WenjinCode

Audit of Wenjin Liu's ("Selina") SmartEnum codebase as mirrored at
[../selina/](../selina/). For mirror layout and what's not pulled, see
[../selina/README.md](../selina/README.md). For the bidirectional sync
mechanism that brings it to this NAS from CARC, see
[../.sync/README.md](../.sync/README.md).

This doc exists so a fresh reader can decide, **per script**, whether to
reuse, adapt, supersede, or skip Wenjin's work. It's the input the
follow-on active-learning plan leans on — without it we'd duplicate
infrastructure that already exists.

## What Wenjin built

Two automation tracks:

1. **`selina/AutomateICMScreenReplacement/`** (16 files, ~180 KB) — the
   SLURM-driven Stage 3 Screen Replacement Group pipeline: cap-edit a
   raw MEL SDF, fan out per-MEL ICM jobs across CARC compute nodes,
   collect modifiersTab results, run a post-hoc consistency check
   against the enumeration PKLs.
2. **`selina/FullLigandICMDocking/`** (8 files, ~28 KB) — the Stage 5
   full-compound docking pipeline: load enumerated products into a
   primed ICB, dock, post-process the `.ob` outputs to CSV, merge with
   the enumeration metadata, pick top-N.

Both tracks are dispatched as SLURM array jobs on USC CARC. They share a
naming convention (`Rank{N}_ICMInChiKey_{HASH}_OpenVSInChiKey_{HASH}_*`)
and a per-target directory layout under `/project2/katritch_223/selina/SmartEnum/<TARGET>/`.

## Pipeline placement

```
V-SYNTHES:
  Stage 1   MEL docking (~1.7M fragments)                     upstream
  Stage 2   MEL hitlist + CapSelect (top-1K)                  upstream
  Stage 2.5 Compatible-synthon pre-filter + enumeration       selina/AutomateICMScreenReplacement (TopN_MEL_Prepare_*)
  Stage 3   Screen Replacement Group scoring                  THIS REPO   +   selina/AutomateICMScreenReplacement (run_ICM_*)
  Stage 4   Validity-weighted MEL+synthon sampling            future (AL plan)
  Stage 5   Full-compound docking on the chosen pairs         selina/FullLigandICMDocking
```

Wenjin and this repo overlap **exactly at Stage 3**: cap-edit a MEL,
read receptor+maps, fan synthons through `processLigandICM` at the cap
attachment vector, write the result SDF. The other stages are
non-overlapping — Wenjin owns 2.5 (enumeration) and 5 (full docking);
we own 4 (sampling policy) once the AL plan is written.

## The Stage 3 overlap, in detail

These three pairs are the only places where Wenjin's code and this
repo's code do the same thing:

### 1. MEL cap-removal preprocessor

| File | Lines | Logic |
|---|---|---|
| `selina/AutomateICMScreenReplacement/Preprocess_MEL_Frags.py` | 470 | strips iso-labeled cap atoms, injects `M APO` at junction |
| [`../edit_mel_cap.py`](../edit_mel_cap.py) | 482 | identical algorithm |

These are **semantically identical** (not byte-identical — `diff` shows
28 lines of blank-line spacing only). Same cap-detection rule: every
atom whose `(symbol, isotope_mass)` matches the CAP_ISO set is deleted,
including `N(iso=16)`. Same junction selection (first non-H scaffold
heavy atom bonded to any deleted cap). Same connected-component
warning after deletion. Same stdlib-only Python (no RDKit at runtime).

**Reuse decision: skip.** Keep [`../edit_mel_cap.py`](../edit_mel_cap.py)
as canonical. It's documented in [`../CLAUDE.md`](../CLAUDE.md) and
validated against the 10-MEL CB2_5ZTY benchmark; Wenjin's adds nothing
beyond formatting drift.

### 2. The four `run_ICM_ScreenReplacement_SingleMEL_*.icm` templates

Both sides have files with **the same four names**:

| Variant | What it is |
|---|---|
| `GUI_Parallel` | `openFile` + `processLigandICM ... yes nProc = 0` (l_bg=yes, async `_ligedit_bg`); needs Qt-linked icm64 (macOS, KatLab) |
| `GUI_NoParallel` | `openFile` + foreground (`l_bg=no`); slower but synchronous |
| `NoGUI_Parallel` | `read object/map/table` + `l_bg=yes nProc=0`; required for icmng on CARC |
| `NoGUI_NoParallel` | `read object/map/table` + foreground; single-core debug path on icmng |

The filenames match but the **contents differ substantially** (56-377
line diffs). The deltas are:

- **Hardcoded paths.** Wenjin's point at her `/Users/liuwenjin/Desktop/...`
  laptop layout and her CARC `/project2/katritch_223/...` paths;
  ours point at `/Users/aoxu/...` and `/mnt/katritch_lab2/...`. Cosmetic.
- **Output filename convention.** Wenjin's emit
  `MEL_ICM_InChiKey_<KEY>_ICM_Screen_Replacement_Results.sdf`; ours emit
  `MEL_<row>_enumerated.sdf`. The InChIKey-named files are easier to
  match against her cache; ours are easier to spot-check against
  `final_table.sdf` rows.
- **Async wait-and-read block (local-only).** Our `GUI_Parallel` and
  `NoGUI_Parallel` add a post-`processLigandICM` block that polls
  `lig*_out*.icb` in `$s_tempDir`, reads it with `read binary`, and
  deletes it. This is the workaround for the
  `make background ... command=` callback not firing under
  `icm64 -g` batch mode, documented in [`../CLAUDE.md`](../CLAUDE.md)
  ("Critical caveat" under "Verified on first ICM run"). Wenjin's
  templates don't have this — they assume the callback path fires, which
  only works in interactive GUI.
- **Local `GUI_NoParallel` is mis-named.** Despite the suffix, our
  local copy uses `processLigandICM ... yes nProc = 0` (i.e., it's
  parallel). Wenjin's matches its name. Looks like an editing accident
  in the Apr 29 4-template-rename refactor.

**Reuse decision per variant:**

| Variant | Verdict | Notes |
|---|---|---|
| `GUI_Parallel` | **keep local** (with caveat) | Our async-block fix is necessary for batch-mode operation. Caveat: re-derive Wenjin's NoGUI_Parallel async behavior — does she rely on GUI callback for icmng too? See below. |
| `GUI_NoParallel` | **fix local** to actually be no-parallel | Restore `l_bg=no` (currently shows `yes nProc=0` despite the name). Use Wenjin's variant as the reference for the correct shape. |
| `NoGUI_Parallel` | **keep local** | Same async fix as `GUI_Parallel`; needed for CARC headless. |
| `NoGUI_NoParallel` | **keep local** | Single-core debug path is fine on either side. |

### 3. Per-MEL Stage 3 orchestrator

| File | Pipeline | Pattern |
|---|---|---|
| `selina/AutomateICMScreenReplacement/run_ICM_ScreenReplacement_NodeWorker.py` | SLURM array node worker — picks MEL rows for this node, patches a chosen ICM template per row, launches `icm64 ...` | string-based template patching, per-MEL result dir |
| `selina/AutomateICMScreenReplacement/run_screen_replacement_workstation.py` | Same flow but single-machine sequential, score-ordered | same template patching pattern |
| [`../run_srg_batch.py`](../run_srg_batch.py) (NAS, the canonical one after Phase 0) | Workstation-side orchestrator using the 3-template choice (`default/headless/converge`) | Python `Template`-based rendering, manifest TSV emission |
| [`../scripts/`](../scripts/) {srg_array.sbatch, submit.sh, run_one_mel.py} | SLURM-side alternative wired to our `paths.py` | slimmer than Wenjin's but covers the same fan-out + per-MEL run pattern |

**Reuse decision: adapt.** Wenjin's node worker has a few patterns that
are more battle-tested than ours and worth porting back:

- **Per-MEL result directory layout** with the InChIKey-named output —
  makes re-running a subset and joining against her enumeration PKLs
  trivial.
- **Score-ordered MEL iteration** in
  `run_screen_replacement_workstation.py:35-78` — saves wall-clock if
  early stopping fires.
- **stderr regex parsing** of `[skip]/[warn]` lines from
  `Preprocess_MEL_Frags.py` (her version of edit_mel_cap.py). Wire the
  same up to our preprocessor.

Patterns NOT to adopt:

- **Hardcoded conda env name** in `sbatch_screen_replacement_node_job_template.sbatch`.
- **String-based ICM template patching** (`re.sub` on `s_var = "..."`
  lines). Fragile if comments contain the variable name. Our Python
  `string.Template` rendering in
  [`../srg_core.py`](../srg_core.py) is safer.

## Wenjin-only Stage 2.5 infrastructure (no local analog)

These four scripts are new territory — there is nothing in this repo
that does the same job. They were added by Wenjin in early May 2026 to
build the **Top-1K MEL fully-enumerated** dataset that the AL benchmark
needs:

| File | Stage | What it does |
|---|---|---|
| `TopN_MEL_Prepare_Main.py` | 2.5→3 | Single-MEL entry point: runs LoToT enumeration, applies filters, extracts surviving synthons, converts to ICM-ready APO format, caches all intermediates by OpenVS InChIKey |
| `TopN_MEL_Prepare_FullLigandEnumeration.py` | 2.5 | The core enumeration: takes a `Synthon` object + full synthon ID list + property/substructure filters, returns `(inchi_list, rdmol_list)` and writes surviving synthons SDF |
| `TopN_MEL_Prepare_ICMScreenReplacePreprocess.py` | 2.5→3 bridge | RDKit-driven conversion of OpenVS `[101Si]/[102Si]` dummies to ICM `M APO` (the synthon-side counterpart to `edit_mel_cap.py`) |
| `TopN_MEL_Prepare_Cache_Logic.py` | 2.5 | Cache-hit detection by filename pattern (`Rank{N}_ICMInChiKey_{HASH}_OpenVSInChiKey_{HASH}_*`); skips re-enumeration of MELs whose synthon set was already enumerated |

**Reuse decision: use as-is, with one caveat.** All four depend on
Wenjin's `Synthon` class API + LoToT enumeration library, which is
not in `selina/` — it lives somewhere else in her tree on CARC. If
the AL benchmark needs to re-enumerate at scale, vendor the Synthon
library into the project; if it needs only the pre-enumerated outputs,
just consume the PKL/SDF files via `selina-pull.sh` (size budget
permitting — see Data state below).

**Caching is by filename only.** If you suspect a stale or corrupt
cache hit, the only way to invalidate is to `rm` the offending file
manually. No content checksum.

## Stage 3→4 result aggregation

| File | Role |
|---|---|
| `selina/AutomateICMScreenReplacement/collect_icm_screen_replace_results.py` | Single-MEL post-SRG analysis: dedup InChIKeys, find missing synthons (in input but not in result), three-way join with enumeration PKL metadata + (optionally) the full-ligand docking CSV |
| `selina/AutomateICMScreenReplacement/batch_collect_icm_screen_replace_results.py` | Multiprocessing wrapper around the above; scans a directory of result SDFs and matches against per-MEL companion files by ICM InChIKey |

**Reuse decision: use as-is.** Nothing in this repo does the missing-
synthon recovery + three-way join. The current
[`../run_srg_batch.py`](../run_srg_batch.py) is only the "fan out and
run" half; this is the "collect and reconcile" half we're missing.

Caveats:

- **Pure-string SDF parsing** (no RDKit), so a malformed SDF row will
  break parsing rather than be skipped. Bulletproof at typical scale.
- **Multiprocessing pool aborts on first worker exception** — at
  CARC scale with 1000+ MELs, a single bad input crashes the batch.
  Worth wrapping in a try/except per worker.

## Stage 5: full-compound docking

Out of scope for the active-learning offline benchmark (Phase 0 of
this plan, not Phase 2/3 of the upcoming AL plan), but documented
here because the AL plan will eventually need real docking scores as
ground truth.

| File | Role |
|---|---|
| `selina/FullLigandICMDocking/ICM_Docked_LoadHits_ToICB_InputParams.py` | Generate ICM input-params JSON (pdb_id, receptor path, docking box) for the loader template |
| `selina/FullLigandICMDocking/ICM_Docked_LoadHits_ToICB.template.icm` | ICM script: load enumerated SDF into primed ICB, run dock, write final_table |
| `selina/FullLigandICMDocking/batch_preprocess_fully_enumerated_ligands_PKL_to_SDF.py` | Convert the 119 GB of `*_enumerated_products.pkl` into per-MEL SDFs for docking |
| `selina/FullLigandICMDocking/run_ICM_Docking_Result_PreProcess_ob_To_csv.py` (+ Batch_Parallel) | Convert ICM `.ob` result-binary to CSV (per-MEL and parallel-batch versions) |
| `selina/FullLigandICMDocking/run_ICM_Docking_Result_Merge_With_InputPKL_SelectTopN.py` | Merge dock-result CSV with the enumeration PKLs (scaffold dedup), pick top-N |
| `selina/FullLigandICMDocking/sbatch_epyc_*` | CARC epyc-node SLURM templates |

**Reuse decision: use as-is** if/when we run Stage 5 for AL ground
truth; **skip** if the AL benchmark can be carried out on simpler
proxies (e.g., the modifiersTab scores from Stage 3 alone — the
Sadybekov 2023 review suggests they correlate well enough with full-
compound scores for the purposes of seed selection).

## Data state on CARC (for the AL plan)

Verified 2026-05-11 by `ssh aoxu@discovery.usc.edu du -sh ...`. **NOTE**:
this section was originally written from a CB2-only audit and missed
Wenjin's GPR91 Stage-5 docking data — corrected below.

### CB2_5ZTY (`selina/SmartEnum/CB2/`)

| Path | Size | What it has | Stage |
|---|---|---|---|
| `CB2-5ZTY-TopN-MELFrags/` | 25 MB | Top-N MEL hits (ICB + SDF + CSV per top-N cut) | 2 |
| `CB2-5ZTY-Comaptiable-And-Surviving-Syntons/` | 77 GB / 1473 files | Per-MEL surviving synthons in OpenVS format | 2.5 |
| `CB2-5ZTY-ICM3.9.3-Docked-2Comp-MEL-Fully-Enumerated-Top1KMEL/` | 119 GB / 1881 PKLs | Pre-docking enumerated products — `(list[InChIKey_str], list[RDKit_binary])`, **no docking scores** | 5 input |
| `.../5ZTY-PKL-Entry-Count-CARC.csv` | 224 KB | Sidecar metadata: entry count per MEL PKL | 5 input |
| (no docking-results dir for CB2) | — | Stage 5 docking score CSVs | **missing on CARC** |

The user has **seven** per-MEL CB2 Stage-5 docking CSVs on the
workstation (in `csv/Dock_5ZTY_Rank{1,2,5,6,7,8,9}_*_instructions_products1.csv`,
~570 KB to 3.6 MB each) — origin unknown but pre-NAS-merge. Schema
matches Wenjin's GPR91 CSVs: `Name, Nat, Nva, RTCNNscore, Score, dEel,
dEgrid, dEhb, dEhp, dEin, dEsurf, dTSsc, mfScore`. These could be
treated as a partial CB2 Stage-5 oracle (7 MELs).

### GPR91_6RNK (`selina/SmartEnum/GPR91/`)  ← real Stage-5 ground truth

| Path | Size | What it has | Stage |
|---|---|---|---|
| `GPR91-6RNK-TopN-MELFrags/` | ~25 MB | Top-N MEL hits ICB/SDF/CSV | 2 |
| `GPR91-6RNK-Comaptiable-And-Surviving-Syntons/` | — | Per-MEL surviving synthon SDFs in OpenVS format | 2.5 |
| `GPR91-6RNK-ICM3.9.3-Docked-2Comp-MEL-Fully-Enumerated-Top1KMEL/` | — | Pre-docking enumerated products PKLs | 5 input |
| **`GPR91-6RNK-Fully-Enumerated-ICM3.9.3-Docking-Results/`** | **858 MB** | **Real per-MEL Stage-5 docking CSVs** (one per MEL). Schema: `Name, Nat, Nva, RTCNNscore, Score, dEel, dEgrid, dEhb, dEhp, dEin, dEsurf, dTSsc, mfScore` | **5 output** |

The `Docking-Results/` dir is the missing Stage-5 ground truth the
original audit incorrectly claimed didn't exist. Wenjin's notebook
[`../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb`](../BenchMark-GPR91-6RNK-ICMScreenReplaceEnrichFactorSimulation.ipynb)
joins these CSVs with Stage-3 SRG RTCNN scores into a single
`all_mels_combined_core.csv` (her oracle, currently on her laptop) and
runs an EF-vs-VS-baseline benchmark over four selection strategies
(A/B/C/D). See [`AL_Pilot.md`](AL_Pilot.md) and the upcoming
[`AL_GPR91_Reproduction.md`](AL_GPR91_Reproduction.md) for the pivot
this triggers in our AL plan.

### 5TH2A_7WC6 (`selina/SmartEnum/5TH2A/`)

| Path | Has | Stage |
|---|---|---|
| `5TH2A-7WC6-TopN-MELFrags/` | Top-N MEL hits | 2 |
| `5TH2A-7WC6-Comaptiable-And-Surviving-Syntons/` | Surviving synthon SDFs | 2.5 |
| `5TH2A-7WC6-ICM3.9.3-Docked-2Comp-MEL-Fully-Enumerated-Top1KMEL/` | Pre-docking enumerated PKLs + `7WC6-PKL-Entry-Count-CARC.csv` | 5 input |
| (no docking-results dir for 7WC6) | — | **missing on CARC** |

7WC6 has 3 of GPR91's 4 subdirs but lacks the `*-Fully-Enumerated-ICM3.9.3-Docking-Results/` dir. **No Stage-5 ground truth for 7WC6.** To benchmark on 7WC6, run Stage 3 SRG (workstation, ~1 day for Top-1K) and use SRG RTCNN as a proxy oracle.

### Implications for the AL benchmark

- **6RNK is the priority target.** Real Stage-5 docking scores exist
  (Wenjin's 858 MB CSVs) and her joined `all_mels_combined_core.csv`
  is the canonical oracle.
- **7WC6 needs a proxy oracle.** Stage 3 SRG over the Top-1K is the
  cheapest path (matches the SRG-proxy choice used in the original
  CB2 pilot — see [AL_Pilot.md](AL_Pilot.md)).
- **The CB2 pilot's oracle is small but real.** 7 MELs of Stage-5
  docking CSVs in `csv/` are usable as a partial CB2 oracle if needed
  for cross-target comparison.

## Notable gaps and dangers across the audit

These cut across multiple scripts; bringing them up here so they don't
get lost in the per-file tables:

1. **String-based ICM template patching** in `run_ICM_ScreenReplacement_NodeWorker.py`,
   `submit_screen_replacement_from_docked_*.py`, and
   `run_screen_replacement_workstation.py` — regex/`re.sub` on
   `s_var = "..."` lines. Will silently fail if the template grows a
   comment containing the variable name. Replace with Python
   `string.Template` (already used in [`../srg_core.py`](../srg_core.py))
   when porting.
2. **Hardcoded ICM 3.9-3b binary path** in some of Wenjin's scripts
   (`/project2/katritch_223/icm-3.9-3b/icmng`). The project uses
   3.9-4. Verify behavior parity before swapping.
3. **No per-MEL checkpointing** in batch runs. If a node dies mid-array,
   surviving result dirs are kept but in-flight MELs restart from
   scratch.
4. **Cache validation by filename only** in `TopN_MEL_Prepare_Cache_Logic.py`.
   No content hash. Corrupted PKL is silently treated as cache hit.
5. **Multiprocessing fan-out without per-worker error isolation** in
   `batch_collect_icm_screen_replace_results.py` and
   `run_ICM_Docking_Result_PreProcess_ob_To_csv_Batch_Parallel.py`. One
   bad input → entire batch aborts.

## Summary: reuse decisions

| File | Stage | Decision |
|---|---|---|
| `Preprocess_MEL_Frags.py` | 2.5 | **skip** — use local `edit_mel_cap.py` |
| `run_ICM_ScreenReplacement_SingleMEL_GUI_Parallel.icm` | 3 | **keep local** (async wait block is necessary) |
| `run_ICM_ScreenReplacement_SingleMEL_GUI_NoParallel.icm` | 3 | **fix local** (currently mis-named: uses parallel mode) |
| `run_ICM_ScreenReplacement_SingleMEL_NoGUI_Parallel.icm` | 3 | **keep local** |
| `run_ICM_ScreenReplacement_SingleMEL_NoGUI_NoParallel.icm` | 3 | **keep local** |
| `run_ICM_ScreenReplacement_NodeWorker.py` | 3 | **adapt** — port InChIKey-named result layout + score-ordered iteration |
| `run_screen_replacement_workstation.py` | 3 | **adapt** — SDF parsing + score-ordered iteration patterns |
| `submit_screen_replacement_from_docked_mel_hits_icb.py` | 3 | **adapt** — adopt the ICB→SDF export step |
| `submit_screen_replacement_from_docked_preprocessed_mel_sdf.py` | 3 | **adapt** — lightweight SLURM submit for pre-edited input |
| `sbatch_screen_replacement_node_job_template.sbatch` | 3 | **adapt** — update conda env + module loads for our CARC config |
| `collect_icm_screen_replace_results.py` | 3→4 | **use as-is** — no local analog |
| `batch_collect_icm_screen_replace_results.py` | 3→4 | **use as-is** — add per-worker error isolation |
| `TopN_MEL_Prepare_Main.py` | 2.5→3 | **use as-is** — depends on Wenjin's Synthon library (vendor if needed) |
| `TopN_MEL_Prepare_FullLigandEnumeration.py` | 2.5 | **use as-is** |
| `TopN_MEL_Prepare_ICMScreenReplacePreprocess.py` | 2.5→3 | **use as-is** |
| `TopN_MEL_Prepare_Cache_Logic.py` | 2.5 | **adapt** — naming convention only; skip the filename-only validation |
| `FullLigandICMDocking/*` | 5 | **adapt** if running real docking for AL ground truth; **skip** otherwise |

## See also

- [`../CLAUDE.md`](../CLAUDE.md) — pipeline placement, cap convention, the
  async `make background` issue that motivates the local-only wait
  block in the ICM templates.
- [`../docs/MELSelection.md`](MELSelection.md) — the dynamic-allocation
  baseline the next plan extends.
- [`../docs/CapSelect.md`](CapSelect.md) — sister doc for the upstream
  CapSelect step (different from cap *removal*).
- [`../selina/README.md`](../selina/README.md) — what's mirrored locally,
  what isn't, how to update.
