# CapSelect

Reference doc for the CapSelect algorithm — what it does, how it works,
where to find the source, and how it relates to this project.

This is a project-internal distillation of the investigation captured
in [../capselect/TEST_RESULTS.md](../capselect/TEST_RESULTS.md) plus
session-time exploration. For the algorithm's published description
see Nazarova et al. 2025 (DOI 10.21203/rs.3.rs-7782723/v1, PMC12633183);
for the authoritative manual see
[../capselect/sources/Manual_Cap_Selection_Enhancement_ALNazarova.pdf](../capselect/sources/Manual_Cap_Selection_Enhancement_ALNazarova.pdf).

## What CapSelect is

CapSelect is a **re-ranking step** that scores each docked MEL fragment
by how much room a synthon has to grow out from the cap into the
binding pocket, then sorts by a combined score so the top-N fragments
are likely to enumerate into productive full molecules. It is **not**
a cap-removal step (this project's [edit_mel_cap.py](../edit_mel_cap.py)
is the cap-removal tool — they are orthogonal).

Inputs:

- `fragments.sdf` — docked MEL hitlist (top-30K from the upstream
  docking pass), caps still attached, with at minimum a `<Score>` SD
  tag.
- `protein.sdf` — pocket atoms within ~5 Å of the docking box, in the
  **same coordinate frame as the docked MELs**, with hydrogens.
- `protein2.sdf` (optional) — second receptor conformation for the 4D
  variant (ensemble docking).

Output: an SDF with new SD tags per fragment:

| Tag           | Meaning |
|---------------|---------|
| `<CapScore>`  | 0..10 score derived from sphere-chain geometry (10 = ideal) |
| `<Spheres>`   | number of spheres placed before the chain stopped (1..10) |
| `<Max(min)>`  | per-sphere min-distance from sphere centroid to nearest pocket atom |
| `<Distance>`  | per-sphere chain-length distance from cap centroid |
| `<MergedScore>` | weighted log sum of `\|Score\|` and `CapScore` (the ranking metric) |

The downstream `icm_CapSelect_to_frags_for_enum.icm` step sorts by
`MergedScore` desc and writes `frags_for_enum.sdf` — the input to the
next pipeline stage (enumeration).

## Pipeline placement

```
V-SYNTHES2:
  Stage 1   MEL docking (~1.7M fragments)                       upstream
  Stage 2a  load hits → top-30K (icm_load_hits_enamine_*.icm)   upstream
  Stage 2b  CapSelect re-ranking on the 30K  ← THIS ALGORITHM   upstream
  Stage 2c  compatible-synthon pre-filter (Find_Compatible_*)   upstream
  Stage 3   Screen Replacement Group scoring                    THIS PROJECT
  Stage 4   validity-weighted sampling                          downstream
  Stage 5   full-compound docking                               downstream
```

CapSelect picks **which** MELs are worth enumerating; this project's
preprocessor (`edit_mel_cap.py`) and ICM driver
(`run_ICM_ScreenReplacement_*.icm`) then enumerate synthons on the
selected MELs. The two steps don't conflict — they run in different
stages on different inputs.

## Algorithm

The published description is in Nazarova et al. 2025 §Methods. What
follows is the authoritative read of the C++ source
[../capselect/sources/CAPBS_4D.cpp](../capselect/sources/CAPBS_4D.cpp),
cross-referenced to the published prose where useful.

### Cap atom identification (CAPBS.cpp:384-518)

The C++ scans each ligand atom and reads column-35 (0-indexed) of its
SDF V2000 atom line — the last digit of the V2000 mass-difference
field. Atoms tagged with:

- `'3'` → aromatic cap atoms (counted in groups of 5 — one phenyl ring)
- `'1'` → non-aromatic cap atoms (single methyl carbon per cap)

Other isotope labels in the V-SYNTHES_2.2 cap convention (C(iso=14)
junction, N(iso=14/15/16), O(iso=17/18), S(iso=33/34) — see project
[CLAUDE.md](../CLAUDE.md)) get column-35 ≠ '1','3' and are
**ignored** by CapSelect — including the C(iso=14) junction. CapSelect
was designed for the simpler v1 cap convention; the extra isotope
labels in v2.2 don't cause errors because the centroid calculation only
uses the labeled-and-recognized atoms.

For each cap, the **starting sphere position is the centroid (mean
coordinate) of the cap's labeled atoms**:

- Aromatic cap (5 phenyl carbons) → centroid of the ring (= ring center)
- Non-aromatic cap (1 methyl C) → that single atom

The fragment classification (`num_lab_l`):

| `num_lab_l` | Type | Caps |
|-------------|------|------|
| 1 | SINGLE_CAP non-aromatic   | 1 |
| 5 | SINGLE_CAP aromatic       | 1 |
| 2 | 2 non-aromatic caps       | 2 |
| 6 | aromatic + non-aromatic   | 2 |
| 10 | 2 aromatic caps          | 2 |
| 0 | reject (no labeled cap)   | 0 — `CapScore = -100` sentinel |

For 2-cap fragments (3-component reactions), CapSelect runs the chain
algorithm on each cap independently and takes the **max** of the two
CapScores ("best of routes" — paraphrasing the paper, "fragments remain
productive if at least one R-group supports continued growth").

### Sphere chain placement (CAPBS.cpp:526-822)

The chain algorithm uses a precomputed grid of 2,592 candidate
directions on the unit sphere (72 azimuth steps × 36 longitude steps
at 5° resolution; longitude covers −90° to +85°, missing only the
top 5° polar cap). For each candidate direction the centroid is
evaluated; clash filters reject it; the surviving candidate that
maximizes the min-distance to any protein atom wins.

**Step 1 — first sphere on shell of radius `r` around cap centroid:**

```
Inputs:  cap centroid (x_in, y_in, z_in)
         r      = 3.5 Å (aromatic cap)  |  3.0 Å (non-aromatic cap)
         l_check = 1.1 (aromatic)        |  1.3 (non-aromatic)
         p_check = 2.0 (aromatic)        |  3.0 (non-aromatic)

For each of 2592 candidates (x, y, z) = cap_centroid + r * unit_dir:
    Reject if dist(candidate, any non-cap heavy ligand atom) < r
    Reject if dist(cap_centroid, any non-cap heavy ligand atom) < l_check
    Reject if dist(candidate, any protein atom)              < p_check
    Score: min over protein atoms P of dist(candidate, P)
    Track best (= max of scores).

If no valid candidate:  Spheres = 0, reject fragment
Else: sphere 1 placed at best position; record Max(min)[0], Distance[0].
```

**Step 2 — extend chain along 2-Å spheres up to length 10:**

```
After sphere k at (xl, yl, zl):
    Search radius = 2.0 Å (every chain sphere has fixed 2-Å radius)
    For each of 2592 candidates (x, y, z) = (xl, yl, zl) + 2 * unit_dir:
        Reject if dist(candidate, any non-cap heavy ligand atom) < 3.0
        Reject if dist(prev_sphere, any non-cap heavy ligand atom) < l_check
        Reject if dist(candidate, any protein atom) < 2.0
        Reject if dist(prev_sphere, any protein atom) < 3.0
        Reject if dist(candidate, any earlier chain sphere) < 2.0
        At sphere 2 only: reject if dist(candidate, cap_centroid) < {3.5,3.0}
        At sphere ≥ 3:    reject if dist(candidate, sphere{k-1}) < 3.46  ← 120° cone
        Score: min over protein atoms of dist(candidate, P)
        Track best.
    If no valid candidate: chain stops, final length = k
    Else: place sphere k+1, record Max(min)[k], Distance[k], continue (cap at 10)
```

**The 120° cone** is implemented as the chord-distance check
`dist(candidate, sphere_{k-1}) ≥ 3.46 Å`. With sphere radius 2 Å,
adjacent centroids are 4 Å apart; via the law of cosines, the chord
between the candidate (at distance 2 from sphere k) and sphere k−1 (at
distance 4 from sphere k) has length `sqrt(20 + 16·cos θ)` where θ is
the internal angle at sphere k. `θ = 120°` → chord = `2√3 ≈ 3.46`. So
the rule enforces "internal angle ≥ 120°" = "next direction within 60°
cone of straight ahead" = "within a 120°-aperture cone around the prior
direction." (The source's `// 60degree` comment names the maximum bend;
the paper's "120-degree cone" names the aperture — same constraint.)

**Implicit "10 Å past pocket" stop**: as the chain wanders out of the
pocket, candidates start failing the `dist < 2.0` protein check (not
because they hit the protein, but because no protein atom is close
enough to anchor the maximize-min-distance criterion against). The
chain terminates organically, no explicit boundary check.

### CapScore formula (CAPBS.cpp:869-995)

```
penalty_s   = 0.4 · (5 − Spheres)²    if Spheres ≤ 5    else 0
penalty_max = (10/169) · (7 − Max(min)[9])²   if Spheres > 9 and 7 ≤ Max(min)[9] ≤ 20
            = 10                              if Spheres > 9 and Max(min)[9] > 20
            = 0                               otherwise

CapScore    = 10 − penalty_s − penalty_max
```

**Intent of the two penalties**:

- `penalty_s` rewards reaching at least 5 productive spheres into the
  pocket. The first 5 spheres are worth a quadratic-decreasing weight
  (sphere 1 = 6.4 penalty if missing, sphere 5 = 0); after 5, no
  additional reward.
- `penalty_max` (only fires when the chain reached the full 10
  spheres) penalizes the 10th sphere being far from the pocket — i.e.
  the chain has wandered into bulk solvent. Ideal `Max(min)[9]` ≈ 7 Å
  (still touching pocket geometry); at 20+ Å the penalty saturates.

**Spheres → CapScore lookup** (for short, terminated chains):

| Spheres | penalty_s | CapScore |
|---------|-----------|----------|
| 0       | 10.0      | 0.0      |
| 1       | 6.4       | 3.6      |
| 2       | 3.6       | 6.4      |
| 3       | 1.6       | 8.4      |
| 4       | 0.4       | 9.6      |
| ≥ 5     | 0.0       | 10.0 (less penalty_max if sp > 9) |

For 2-cap fragments: CapScore = max(CapScore_cap_1, CapScore_cap_2).

### MergedScore formula

The 2021 manual specified:

```
MergedScore = 5 · log₂|Score| + 1.0 · log₂|CapScore|
```

with the user adding the column via ICM after the binary ran. The
**v2.5 binary** (current V-SYNTHES_2_2 production) writes MergedScore
directly into the SDF and **halved** the CapScore weight:

```
MergedScore_v2.5 = 5 · log₂|Score| + 0.5 · log₂|CapScore|       (CapScore > 0)
MergedScore      = 5 · log₂|Score|                              (CapScore = 0, Spheres ≥ 1)
MergedScore      = -1000                                        (CapScore < 0, or Spheres = 0)
```

The halving makes CapScore a tiebreaker rather than a primary signal:
docking score contributes ~25–28 units, CapScore contributes ≤ 1.66
units. Empirically, sorting by `MergedScore` swaps ~10–15% of the
top-N selection vs sorting by docking score alone.

## Source code and binaries

| File | Location | Notes |
|------|----------|-------|
| Compiled binary (production) | `CARC:/project2/katritch_223/VSYNTHES_2_2__012024/VSYNTHES_2_2_CARC_example_project_012024/CapSelect_3D_4D/CapSelect` | 75 KB ELF x86-64. Source filename embedded: `GitHub_CapSelect_v2_5.cpp` (not shipped). Replicated byte-identically across many lab member project trees on CARC. |
| Compiled binary (2021) | `KatLab:/mnt/katritch_lab/Antonina/CapSelection/Example_project_input/CapSelection` | 53 KB ELF, June 2021. Predates `ip2_1` cap-clearance constraint. |
| C++ source (2021, 3D) | `KatLab:/mnt/katritch_lab/Antonina/GHSR_4D_MEL/CapSelect_cor_0917/CapSelect.cpp` | 30 KB, Oct 2021. Local copy at [../capselect/sources/CapSelect_2021.cpp](../capselect/sources/CapSelect_2021.cpp). |
| C++ source (2022, 4D) | `KatLab:/mnt/katritch_lab/Antonina/For_Caroline/For_Caroline/CapSelect4D/CAPBS.cpp` | 33 KB, Jan 2022. Local copy at [../capselect/sources/CAPBS_4D.cpp](../capselect/sources/CAPBS_4D.cpp). |
| Algorithm manual | `KatLab:/mnt/katritch_lab/Antonina/CapSelection/Manual_Cap_Selection_Enhancement_ALNazarova.pdf` | Antonina's 2021 algorithm manual. Local copy at [../capselect/sources/Manual_Cap_Selection_Enhancement_ALNazarova.pdf](../capselect/sources/Manual_Cap_Selection_Enhancement_ALNazarova.pdf). |
| Driver script | `CARC:.../CapSelectMP_full.sh` | Bash chunker — splits fragments.sdf into N chunks, fans the binary out across cores, concatenates. |
| Pre-step ICM | `CARC:.../icm_generate_CapSelect_files.icm` | Generates `protein1.sdf` (and optionally `protein2.sdf`) from the docking project. |
| Post-step ICM | `CARC:.../icm_CapSelect_to_frags_for_enum.icm` | Sorts by `MergedScore` desc, writes `frags_for_enum.sdf`. |

The paper advertises [github.com/KatritchLab/V-SYNTHES2_pipeline](https://github.com/KatritchLab/V-SYNTHES2_pipeline)
with a `CapSelect/` subdirectory containing "Python and C++ versions";
the repo URL returns 404 as of 2026-05-01 (likely held private pending
peer review). The 2021/2022 source files above plus the v2.5 production
binary on CARC are what we have.

### Notable changes between source versions

- **2021 → v2.5**: `MergedScore` writing moved from ICM post-step into
  the binary itself; CapScore weight halved (1.0 → 0.5).
- **June 2021 binary → Sep 2021 source**: an `ip2_1` constraint was
  added (sphere 2 must be ≥ 3.0/3.5 Å from cap centroid). Visible as
  `// added on 09/17/21` comments in the source. The example output
  shipped with the 2021 binary (`Example_project_output/CapSelect.sdf`)
  predates this constraint and produces sphere-2 placements that the
  source-as-documented would reject — see molecule 5 in the verification.

## This project's Python port

[../capselect/capselect_py.py](../capselect/capselect_py.py) — a
faithful NumPy port of `CAPBS.cpp`. ~390 lines, vectorized over the
2,592-candidate grid. Supports both the 2021 and v2.5 MergedScore
formulas via `--formula`.

Verification status (full details in
[../capselect/TEST_RESULTS.md](../capselect/TEST_RESULTS.md)):

| Reference | What it verifies | Result |
|-----------|------------------|--------|
| Antonina's 5-fragment example (2021 binary output) | end-to-end algorithm: cap detection, sphere placement, CapScore | **5/5 CapScore match**; 4/5 chain-placement match (mol 5 differs because the 2021 binary predates the `ip2_1` constraint) |
| Antonina's 5-fragment example (Mac-compiled-from-2021-source binary) | port matches the source-as-documented exactly | **5/5 OVERALL PASS** (strict drift threshold) |
| 29,999 GPR119 production rows (v2.5 binary) | only the `MergedScore` formula, given pre-computed `(Score, CapScore, Spheres)` from the binary | **0 mismatches** — bit-for-bit `MergedScore` parity across all four sentinel categories |
| GPR91 6RNK 1000 MELs vs **v2.5 production binary** on CARC | end-to-end: sphere placement + CapScore + ranking against the actual production target | ~91% chain placement bit-for-bit (657/725 productive pairs); 86% top-100 ranking overlap on MergedScore. Differences explained by v2.5 schema (anchor entry, 9 vs 10 spheres) + one undocumented step-1 filter tightening in v2.5. |

The four checks span complementary layers:

- Geometry vs the 2021 source: ✅ verified bit-for-bit
- Scoring formula vs v2.5: ✅ verified bit-for-bit on 30K rows
- End-to-end vs v2.5 binary: ⚠️ ~91% match — the gap is explainable
  but to close it bit-for-bit we'd need either the
  `GitHub_CapSelect_v2_5.cpp` source (still private) or to
  reverse-engineer the additional step-1 filter change.

For practical MEL-selection use the 86% top-100 ranking overlap is
adequate; the swaps are judgment calls between near-tied fragments,
not systematic errors.

Driver scripts in the same directory:

- [../capselect/extract_protein_sdf.py](../capselect/extract_protein_sdf.py)
  — build a CapSelect-compatible `protein.sdf` from a PDB or mol2
  receptor. Supports `--chains`, `--keep-hetatm`, `--drop-residues` for
  filtering. The `AAAA9999` counts-line format is generated correctly.
- [../capselect/sort_by_mergedscore.py](../capselect/sort_by_mergedscore.py)
  — sort an SDF by `MergedScore` descending (equivalent to
  `icm_CapSelect_to_frags_for_enum.icm`). Emits an optional ranking
  TSV.
- [../capselect/verify.py](../capselect/verify.py) — pass/fail diff of
  Python output vs reference SDF.

GPR91 6RNK Top-1000 results in
[../capselect/gpr91/](../capselect/gpr91/), with the canonical run
using the docking-frame mol2 receptor:

- [../capselect/gpr91/protein_mol2.sdf](../capselect/gpr91/protein_mol2.sdf)
- [../capselect/gpr91/CapSelect_gpr91_mol2.sdf](../capselect/gpr91/CapSelect_gpr91_mol2.sdf)
- [../capselect/gpr91/frags_for_enum_mol2.sdf](../capselect/gpr91/frags_for_enum_mol2.sdf)
  (sorted by MergedScore)
- [../capselect/gpr91/frags_for_enum_mol2_ranking.tsv](../capselect/gpr91/frags_for_enum_mol2_ranking.tsv)

## How to run

Four ways to invoke CapSelect, choose by use case:

| Scenario | Tool |
|----------|------|
| Quick local experimentation, modifying the algorithm | Python port (`capselect_py.py`) |
| Reproducible binary comparison on Mac | Mac-compiled `CapSelect_local` from `build_local_binary.sh` |
| Production runs (canonical V-SYNTHES2 pipeline) | v2.5 binary on CARC via `CapSelectMP_full.sh` |
| Verifying port changes | Mac-compiled binary on Antonina's example; spot-check against v2.5 on a small CARC run |

All paths in the snippets below are relative to the project root
`/Users/aoxu/projects/anchnor_based_VSYNTHES/`.

### Option 1 — Python port (local, portable)

Best for quick runs, debugging, modifications.

```bash
# Build protein.sdf from a docking-frame receptor (mol2 from ICM is the gold
# standard; PDB works too but needs --chains/--keep-hetatm flags to drop
# nanobody / bound ligand).
python3 capselect/extract_protein_sdf.py \
    GPR91/dock_6rnk_new_rec.mol2 \
    GPR91/GPR91_6RNK_ICM393_Eff2_2Comp_MEL_Top1000_Hits.sdf \
    capselect/gpr91/protein_mol2.sdf --margin 5.0

# Run the port. Last arg = MergedScore formula: 'v2_5' (production) or '2021'.
python3 capselect/capselect_py.py \
    GPR91/GPR91_6RNK_ICM393_Eff2_2Comp_MEL_Top1000_Hits.sdf \
    capselect/gpr91/protein_mol2.sdf \
    capselect/gpr91/out.sdf \
    v2_5

# Produce frags_for_enum.sdf (sorted by MergedScore desc) + ranking TSV
python3 capselect/sort_by_mergedscore.py \
    capselect/gpr91/out.sdf \
    capselect/gpr91/frags_for_enum.sdf \
    capselect/gpr91/ranking.tsv

# Sanity check against the reference example
cd capselect/test_data && python3 ../capselect_py.py fragments.sdf protein.sdf out.sdf 2021 \
    && python3 ../verify.py CapSelect.sdf out.sdf
```

**Speed:** single-threaded NumPy, ~9 min for 1000 MELs at ~1400 protein
atoms. ~4.5 hr for 30K MELs.

### Option 2 — Mac-compiled 2021 binary (local ground truth)

Faster than the port (~18×, ~29 s for 1000 MELs). The compiled binary
reproduces the 2021 source exactly, including the `ip2_1` constraint.

```bash
# Build once
bash capselect/build_local_binary.sh
# → capselect/test_data/CapSelect_local

# Run (binary hard-codes "fragments.sdf" / "protein.sdf" in cwd)
mkdir -p run_dir && cd run_dir
cp /path/to/fragments.sdf .
cp /path/to/protein.sdf .
cp ../capselect/test_data/CapSelect_local CapSelect   # rename so log says "CapSelect"
./CapSelect > run.log 2>&1
# → CapSelect.sdf with <CapScore> <Spheres> <Max(min)> <Distance>
# (NO MergedScore; add via sort_by_mergedscore.py after, or compute manually)
```

The 2021 binary doesn't write `<MergedScore>` — Antonina's original
workflow expected you to add it in ICM with
`add column ... 5.*Log(Abs(Score),2)+Log(Abs(CapScore),2) ...`.

### Option 3 — v2.5 production binary on CARC

For matching production exactly. The v2.5 binary is at
`/project2/katritch_223/VSYNTHES_2_2__012024/VSYNTHES_2_2_CARC_example_project_012024/CapSelect_3D_4D/CapSelect`.

```bash
# Push inputs to CARC
ssh CARC 'mkdir -p /tmp/$USER/capselect_run'
scp protein1.sdf CARC:/tmp/$USER/capselect_run/protein1.sdf
scp fragments.sdf CARC:/tmp/$USER/capselect_run/fragments.sdf

# Small batch (≤ ~5k MELs, < 5 min): login node is fine, no SLURM
ssh CARC '
    cd /tmp/$USER/capselect_run && \
    cp /project2/katritch_223/VSYNTHES_2_2__012024/VSYNTHES_2_2_CARC_example_project_012024/CapSelect_3D_4D/CapSelect . && \
    ./CapSelect > run.log 2>&1
'
scp CARC:/tmp/$USER/capselect_run/CapSelect.sdf .
```

For large batches (30K+ MELs) use Antonina's MP chunker. It generates
`protein1.sdf` from an ICM docking project, fans the binary across
cores, and writes the final `frags_for_enum.sdf` sorted by
`MergedScore`:

```bash
ssh CARC '
    mkdir -p /tmp/$USER/capselect_prod && cd /tmp/$USER/capselect_prod && \
    cp /project2/katritch_223/VSYNTHES_2_2__012024/VSYNTHES_2_2_CARC_example_project_012024/{CapSelectMP_full.sh,icm_CapSelect_to_frags_for_enum.icm,icm_generate_CapSelect_files.icm} . && \
    cp -r /project2/katritch_223/VSYNTHES_2_2__012024/VSYNTHES_2_2_CARC_example_project_012024/CapSelect_3D_4D . && \
    # Place your run/ subdir (with _rec.ob, .dtb, hitlist.icb in run/processing_files/) here, then:
    ./CapSelectMP_full.sh -c 30
'
```

Wrap in an sbatch script with appropriate `-c` and time limits for
SLURM submission.

### Option 4 — Re-running the verification suite

```bash
# Front 1: port vs Mac-compiled binary on Antonina's 5-fragment example
cd capselect/test_data
./CapSelect_local > /dev/null && \
python3 ../capselect_py.py fragments.sdf protein.sdf python_out.sdf 2021 && \
python3 ../verify.py CapSelect.sdf python_out.sdf
# Expected: 5/5 PASS

# Front 3: port vs v2.5 production binary on GPR91 — outputs already saved at
#   capselect/gpr91/CapSelect_gpr91_mol2_v25binary.sdf
#   capselect/gpr91/CapSelect_gpr91_mol2_port_2021.sdf
# Element-wise comparison logic is in TEST_RESULTS.md (see "Front 3").
```

## Caveats and pitfalls

1. **Receptor coordinate frame**. CapSelect's clash checks must be
   against receptor atoms in the **same frame as the docked MELs**.
   The right input is the receptor as exported from the ICM docking
   project (typically a mol2 with hydrogens), not the bare crystal
   PDB. Using a misaligned PDB will silently produce wrong results.
2. **Hydrogens matter**. With H's, the protein clash surface is
   ~2× denser, and chains terminate earlier. The GPR91 6RNK comparison
   showed saturated-CapScore fraction drop from 27% → 2% when H's
   were added — the relative ranking was largely preserved (88/100
   top-100 overlap) but absolute CapScore distributions are not
   directly comparable across receptor preparations.
3. **Pocket box**. The receptor must include atoms within ~5 Å of the
   docking box. Production uses ICM's `$ProjectName.R_boxDim` directly
   from the docking project; for ad-hoc runs we approximate it by
   inflating the MEL bounding box by 5 Å (negligible difference for
   top-1000+).
4. **Cap convention drift**. The V-SYNTHES_2.2 MEL convention (per this
   project's [../CLAUDE.md](../CLAUDE.md)) uses more isotope labels
   than CapSelect was designed for. Atoms with mass-diff `'2'`
   (C iso=14 junction, N iso=16, etc.) are **silently ignored** by
   CapSelect. This is fine because CapSelect uses cap *centroids*
   anyway, and the recognized atoms (5 phenyl carbons, or 1 methyl C)
   are sufficient to define the centroid.
5. **2021 binary vs v2.5 SDF indexing differs**. The 2021 binary's
   `<Max(min)>` and `<Distance>` lists start at index 0 = first
   placed sphere. The v2.5 binary prepends a synthetic `(0, 0)` anchor
   at index 0, with the placed chain starting at index 1. Our port
   matches the 2021 schema; for v2.5 parity, prepend the anchor.
6. **Bound ligand in PDB**. Crystal PDBs often include the bound
   antagonist in the pocket as a HETATM. If you extract from a PDB,
   exclude it (and water/glycerol/lipids) — see
   `extract_protein_sdf.py --keep-hetatm`. The mol2 from ICM's
   dock-prep is receptor-only by construction.

## Open / unresolved

- **v2.5 step-1 filter divergence**: ~90 cases on GPR91 where the v2.5
  binary rejects the first sphere placement but the 2021 source / our
  port accept it. Suggests v2.5 has a tighter clash threshold or
  additional filter at step 1 that's not in the 2021 source. Without
  `GitHub_CapSelect_v2_5.cpp` (held private at
  [github.com/KatritchLab/V-SYNTHES2_pipeline](https://github.com/KatritchLab/V-SYNTHES2_pipeline))
  we can't pin down the exact change. Empirical reverse-engineering
  on more MEL/receptor pairs could narrow it down.
- **v2.5 chain-length cap**: v2.5 caps at 10 entries (= anchor + 9
  placed spheres) while the 2021 source caps at 10 placed spheres.
  Easy fix in our port (subtract 1 from the chain-cap loop in
  `place_chain`), but requires deciding whether to match v2.5
  schema (anchor entry) or 2021 schema. Currently port follows 2021.
- The 4D variant (two receptor conformations) is implemented in
  `CAPBS.cpp` but our port currently only handles 3D. For Stage 2b
  with ensemble-docked MELs, the 4D path needs porting.
- Single-threaded NumPy gives ~9 min for 1000 MELs at 1428 protein
  atoms. Production scale (30K MELs) is ~4.5 hours — tolerable for
  one-off runs but worth parallelizing via `multiprocessing.Pool`
  before routine use.
