# CLAUDE.md

Orientation file for Claude. Read this first. Search the project knowledge
(the three PDFs — Sadybekov 2022 V-SYNTHES, Pocket-Informed Synton Selection
notes, Sadybekov 2023 computational approaches review) for anything deeper.

---

## Project in one paragraph

This is the ICM-side automation for the **Pocket-Informed Synton Selection**
pilot study — a proposed upgrade to Stage 3 of V-SYNTHES that replaces uniform
synthon enumeration with validity-weighted probabilistic selection guided by
ICM Screen Replacement Group scores. The target of the automation here is
specifically the **Screen Replacement Group run itself**: for each top-N
docked MEL seed, score all compatible synthons at the cap's attachment vector
and dump a ranked SDF. The current workflow is all GUI (MolSoft ICM on macOS)
and does not scale past a handful of MELs. The goal is a fully scripted
pipeline that can process the full top-N batch non-interactively.

## Pipeline location

```
V-SYNTHES:
  Stage 1  MEL docking (~1.7M fragments)                  DONE upstream
  Stage 2  MEL selection (top K seeds)                    DONE upstream
  Stage 2.5 Compatible-synthon pre-filter (Python/OpenVS) DONE upstream
  Stage 3  Screen Replacement Group scoring       <------ THIS PROJECT
  Stage 4  Validity-weighted sampling             downstream
  Stage 5  Full compound docking                  downstream
```

The upstream produces `final_table.sdf` (top-N docked MELs with isotope-
labeled caps) and an `ICMReady_APO.sdf` synthon library per MEL. The
downstream consumes one `modifiersTab` SDF per MEL.

---

## The automation gap (what we fixed)

The original GUI workflow had one step that left no trace in ICM's script
log: the ligand-editor "Delete Cap Group, replace with double-bond O" manual
edit. Everything else in the GUI pipeline is already scriptable — we just
didn't have a programmatic substitute for the cap edit.

**Our fix:** do the cap edit in Python before ICM ever sees the MEL. Output
is an SDF where the cap is already stripped and the attachment atom carries
an `M APO` annotation. ICM then loads the pre-edited MEL and proceeds
identically to the rest of the GUI workflow.

This mirrors how the **synthon side** is already handled: your
`Find_Compatible_And_Surviving_Syntons_*.py` scripts strip the `[102Si]`
dummy from OpenVS synthons and inject `M APO`. Our MEL preprocessor is the
symmetric operation on the scaffold side. Same convention, same ICM
annotation.

## ICM's isotope-based cap convention (decoded)

ICM marks cap atoms with non-natural isotope masses. The rule for the
preprocessor is uniform: **every atom carrying one of these isotope labels
is a cap atom and gets deleted.** No context-dependent exceptions.

| Symbol | Isotope | Role                                          |
|--------|---------|-----------------------------------------------|
| C      | 13      | alternate aromatic ring placeholder           |
| C      | 14      | structural junction placeholder (sp2/sp3)     |
| C      | 15      | phenyl-ring placeholder carbons               |
| N      | 14, 15  | nitrogen cap placeholders                     |
| N      | 16      | linking nitrogen of the cap                   |
| O      | 17, 18  | carbonyl/sulfonyl oxygen placeholders         |
| S      | 33, 34  | sulfonyl sulfur placeholders                  |

The junction atom (where `M APO` is placed) is always the non-H heavy
atom that is bonded to a cap atom but is itself NOT iso-labeled. In a
correctly built 2-component MEL there is exactly one such atom per cap.

**Worked examples from the 10-MEL benchmark** (patterns derived from the
M ISO lines in `final_table.sdf`, not from guesses):

- **MELs 2/3/5 — N-aryl (aniline) cap on a scaffold carbonyl:**
  `[scaffold-C(=O)] - N(iso=16)H - C(iso=14)<phenyl(iso=15)>`.
  Cap atoms: `{N(iso=16), C(iso=14), 5x C(iso=15)}` (the whole NH-phenyl group).
  Junction: the scaffold-side **carbonyl carbon** (the C of the C=O,
  which is unlabeled). The C=O is preserved; APO sits on the carbonyl C.
  Synthons attach via the C=O like an amide/ester/ketone bond — visible
  in the verified GUI output as `aryl-CH=CH-C(=O)-CH3` for MEL 2.

- **MEL 10 — non-phenyl aromatic cap:**
  `scaffold-N(iso=16)-[5×C(iso=13) ring]`.
  Same structural pattern as 2/3/5 (cap hangs off via N(iso=16)), but the
  cap ring uses iso=13 carbons instead of iso=15 — this is how ICM marks
  non-phenyl aromatic caps. No C(iso=14) bridge atom and no carbonyl O.
  Junction is the scaffold-side carbonyl carbon (C#43 in the new numbering).

- **MELs 4/7 — dimethyl-amide (urea-style) cap:**
  `[scaffold-N] - C(iso=14)(=O iso=18) - N(iso=16)H - C(iso=13)H3`.
  Cap atoms: `{C(iso=13), N(iso=16), C(iso=14), O(iso=18)}` (the whole
  CH3-NH-C(=O)- group). Junction is the unlabeled scaffold N. APO on
  that N — synthons attach as N-substituents.
  This was the case that broke the very first version of the preprocessor
  (it left N(iso=16) as a disconnected fragment).

- **MEL 6 — aryl-sulfonamide cap:**
  `[scaffold-N] - S(iso=34)(=O iso=18)(=O iso=18) - C(iso=14)<phenyl(iso=15)>`.
  Cap atoms: `{S(iso=34), 2x O(iso=18), C(iso=14), 5x C(iso=15)}` (the
  whole SO2-aryl group). Junction is the unlabeled scaffold N (no
  iso=16 anywhere in this MEL).

- **MEL 9 — methanesulfonamide cap:**
  `[scaffold-N] - S(iso=34)(=O iso=18)(=O iso=18) - C(iso=13)H3`.
  Like MEL 6 but the S carries a methyl instead of an aryl ring.
  Junction is the unlabeled scaffold N.

- **MEL 8 — two-C14 aryl cap with no iso=16:**
  `[scaffold-X] - C(iso=14) - C(iso=14)<phenyl(iso=15)>`.
  Junction is the non-iso scaffold atom bonded to the first C(iso=14).

**Why earlier versions of this code were wrong:**

Two prior implementations treated `N(iso=16)` as a "scaffold marker" or
"context-dependent" label that should sometimes be kept as the APO atom.
Both produced visibly wrong outputs in different cases:

1. **First version always kept `N(iso=16)` as the junction.** This created
   disconnected fragments in MEL 4 (where the iso=16 N was internal to a
   dimethyl-amide cap) because deleting its neighbors stranded the N.

2. **Second version kept iso=16 N only when it had a non-iso heavy
   neighbor.** This avoided the disconnection bug but produced wrong
   attachment in MELs 2/3/5/10 — synthons would attach to the cap-side
   NH instead of the correct scaffold-side carbonyl carbon. Caught by
   the screenshot evidence: the correct GUI output for MEL 2 has
   `aryl-CH=CH-C(=O)-CH3` whereas the wrong preprocessor output had
   `aryl-NH-synthon`.

3. **Current (correct) version: every iso atom is a cap atom**, including
   `N(iso=16)`. The junction is whatever non-iso heavy atom was bonded
   to the cap. Validated against MEL 2 in the GUI.

**Safety net:** the preprocessor runs a connected-components check after
cap deletion and warns if any heavy atom ends up disconnected from the
junction. This caught the MEL 4 bug in the very first version and is kept
in case future MELs have unusual cap topologies we have not seen.

Reference: the live mapping was derived from all 10 MELs in the Rank2
`CB2_5ZTY` `final_table.sdf` (benchmark dataset for the pilot study).

## Design decisions (do not re-litigate)

- **APO-only, no `=O` variant.** We briefly considered reproducing the
  GUI edit (add a terminal =O to the junction) as a fallback. Rejected:
  the user's GUI edit is only chemically sensible at C junctions, and
  when the junction is an N it produces valence-5 nitrogen (nitroso/
  nitrate). APO is cleaner, matches the synthon side, and avoids this.
- **Docked coordinates preserved bit-for-bit.** The preprocessor does
  NOT re-minimize or perturb any atom that survives the edit. The
  docked pose is load-bearing for Stage 3 — any coordinate drift
  invalidates the anchor assumption.
- **Pure-stdlib Python.** No RDKit at runtime. RDKit is fine for
  validation but we avoid the dependency in the production preprocessor
  so it runs anywhere.
- **One APO atom per MEL.** 2-component reactions have exactly one
  attachment point after cap removal. If the preprocessor finds more
  than one, it warns and emits all — but the ICM driver currently
  uses only the first.

---

## Files in this directory

```
CLAUDE.md                            <-- you are here
edit_mel_cap.py                      <-- MEL preprocessor (Python, stdlib)
run_srg_single.icm                   <-- ICM driver, single MEL at a time
final_table.sdf                      <-- raw docked MELs (from upstream)
final_table_edited.sdf               <-- output of edit_mel_cap.py
final_table_edited_apo_index.tsv     <-- sidecar: which atom is APO per MEL
Rank2_ICMReady_APO.sdf               <-- synthon library for Rank-2 MEL
                                         (per-MEL; filename pattern is
                                         Rank{N}_ICMInChiKey_{...}_
                                         OpenVSInChiKey_{...}_surviving_
                                         synthons_ICMReady_APO.sdf)
CB2_5ZTY_..._Top10_Hits.icb          <-- ICB with docked receptor + MEL
                                         table (from upstream)
logs/                                <-- ICM run outputs (modifiersTab_v*.sdf,
                                         console captures, etc.)
```

**The preprocessor + TSV + ICM driver are the complete handoff:**
preprocess once, then the ICM driver reads the TSV to know which atom
is the attachment point for each entry.

---

## Running things

**Preprocess a MEL file:**
```bash
python3 edit_mel_cap.py final_table.sdf final_table_edited.sdf
```
Emits the edited SDF and `final_table_edited_apo_index.tsv` alongside it.
Logs edited/unchanged/skipped counts to stderr.

**Validate the edit with RDKit** (optional, for confidence):
```python
from rdkit import Chem
for m in Chem.SDMolSupplier("final_table_edited.sdf", removeHs=False, sanitize=True):
    if m is None:
        print("FAIL")
    else:
        print(Chem.MolToSmiles(Chem.RemoveHs(m)))
```
All 10 MELs in the current `final_table.sdf` sanitize with zero problems.

**Run Screen Replacement Group (single MEL):**
Edit the six variables at the top of `run_srg_single.icm`:
```
s_receptor_icb  = "..../CB2_5ZTY_..._Top10_Hits.icb"
s_edited_mel    = "..../final_table_edited.sdf"
i_mel_row       = 2
i_apo_atom      = 3     # from final_table_edited_apo_index.tsv
s_synthon_sdf   = "..../Rank2_..._ICMReady_APO.sdf"
s_out_sdf       = "..../srg_result_mel2.sdf"
```
Then invoke ICM as `icm64 -g run_srg_single.icm`.

---

## Current status and known unknowns

**Working (Python preprocessor):**
- `edit_mel_cap.py` validated end-to-end on the 10-MEL `final_table.sdf`.
  All MELs parse, edit, and pass RDKit sanitization. Edited MEL 2 matches
  the structure produced by the GUI cap-edit step (verified by visual
  comparison of the resulting full compound — `aryl-CH=CH-C(=O)-synthon`).

**Verified on first ICM run (2026-04-16):**
- **MEL table name** is the SDF filename stem (`final_table_edited` for
  `final_table_edited.sdf`). Works as expected.
- **Synthon table name** is also filename stem, but with `-` → `_`
  sanitization applied (dashes re-parse as subtraction in `$var`-expansion
  inside ICM). For `combiDock_R1.sdf` no sanitization is needed; for the
  real `Rank{N}_..._OpenVSInChiKey_{HASH}_..._ICMReady_APO.sdf` files the
  hyphens in the inchikey component need to become underscores.
- **`M APO` on the scaffold works** — it is a drop-in replacement for the
  GUI cap-edit step. ICM injects a virtual attachment atom at the
  junction. On the pristine loaded object it appears as a pseudo-atom
  named `'a'` (`Warning> [427] added extra atom 'a' for the attachment
  point at a_m.m/1/n1`). After `e3dSetLigand` renames the object to
  `m_lig`, the attachment atom appears as a virtual element-X atom named
  `x1` (workspace formula like `C24H18N3OX`; rendered as the mustard /
  olive-colored atom in the 3D view).
- **Correct `as_graph` syntax** is `as_graph = a_LIG.m/1/x1` — select the
  virtual element-X attachment atom by name. Three things that do NOT
  work and produced empty results:
  - `a_LIG.I & a_LIG.` resolves to BOTH the junction heavy atom AND the
    virtual `x1` (two atoms), and `_ligedit_bg` silently returns empty
    when handed more than one attachment atom.
  - `a_LIG.//a` returns Nof = 0 (the `'a'` name only applies to the
    pristine `m` object before `e3dSetLigand`).
  - `a_m_lig.m/1/a` also returns Nof = 0 for the same reason.
  See `logs/modifiersTab_v2.sdf` (success after fix) vs the empty-result
  failure it replaced.
- **`processLigandICM ... l_bg = yes nProc = 0` is the fast path on APO**
  (verified 2026-04-19). `_ligedit_bg` auto-chunks into parallel workers;
  Rank2 completes in ~3 min vs. ~15 h at `nProc = 1`. The full 7-MEL
  batch runs in ~54 min.
  - The earlier "nProc = 1 is mandatory" claim was scoped wrong: the
    `unsupported action 'findgroupTab1'` failure is specific to R6
    wildcard SDFs (see the R6 memory note), not to all large tables.
    The APO path handles `nProc = 0` chunks correctly.
  - Critical caveat: in `icm64 -g script.icm` batch mode the
    `make background ... command=s_out` callback never fires (no GUI
    event-loop drain), so `modifiersTab` stays unpopulated if you just
    wait. `wait background` waits only for the subprocess exit. Fix:
    after `wait background`, `unix "ls -t $s_tempDir/lig*_out.icb |
    head -1"` to find the output, then `read binary` it manually. See
    [run_srg_single_apo_export_diskmaps.icm:208-223](run_srg_single_apo_export_diskmaps.icm#L208).

**Batch runner (wired up 2026-04-19):**
- [run_srg_batch.py](run_srg_batch.py) loops over all MELs with a matching
  APO + synthon library, templates [run_srg_single_apo_export_diskmaps.icm](run_srg_single_apo_export_diskmaps.icm)
  per row, and runs each ICM job to completion sequentially. Writes
  `results/MEL_<row>_<rank>/enumerated.sdf` and a `batch_manifest.tsv`
  summary. `--only-row N` for single-MEL, `--dry-run` for plan preview.

---

## How to work with me (Claude) on this project

- **Search the project PDFs first.** The three PDFs in the project are
  authoritative — the V-SYNTHES paper (Sadybekov 2022), your Pocket-
  Informed Synton Selection design doc, and the Sadybekov 2023 review.
  Do not re-derive anything that is already written down there.
- **Prefer verified commands over speculation.** When editing ICM
  scripts specifically, every line should trace back to your original
  GUI script log or to confirmed ICM behavior captured in the
  "Verified on first ICM run" section above. If I propose a line I'm
  guessing at, I mark it as a guess. ICM's scripting language has
  low-feedback failure modes — things silently fall through to defaults
  rather than erroring out (see the `_ligedit_bg` empty-result and the
  `nProc=0` chunking issue above for examples that bit us).
- **Show me failure output, not symptoms.** When something breaks in
  ICM, paste the full console output from the run. Vague symptoms
  ("the synthons didn't dock right") lead to speculation. Error lines
  + warning counts + the first failure trace let me debug with real
  evidence.
- **Bit-for-bit reproducibility.** Any transformation of the docked
  pose needs explicit justification. Default is "do not touch the
  coordinates of atoms that survive the edit."
- **When I write ICM code I am guessing unless otherwise stated.** My
  reading of ICM's scripting language is inferred from your script log
  plus general ICM conventions. I do not have access to the ICM
  language reference and cannot test scripts end-to-end. Treat my
  ICM output as a starting draft, not a verified artifact.
- **Ask before expanding scope.** If I think a refactor or extension
  is a good idea, I will propose it and wait. I will not silently
  extend the preprocessor to cover 3-component MELs, add logging
  frameworks, or restructure the CLI unless you ask.
