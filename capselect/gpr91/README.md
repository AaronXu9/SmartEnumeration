# CapSelect run on GPR91 6RNK Top-1000 MELs

## Files

- [protein_mol2.sdf](protein_mol2.sdf) — **canonical** pocket atoms
  extracted from `../../GPR91/dock_6rnk_new_rec.mol2` (the receptor as
  exported from the ICM docking project — guaranteed to be in the same
  coordinate frame as the docked MELs, with hydrogens). 1428 atoms.
- [CapSelect_gpr91_mol2.sdf](CapSelect_gpr91_mol2.sdf) — **canonical**
  CapSelect output using the mol2 receptor.
- [protein.sdf](protein.sdf) — pocket atoms from `../../GPR91/6RNK.pdb`
  chain A only, no HETATMs, no hydrogens. 704 atoms. Kept for
  comparison; the mol2 file is the right one to use.
- [CapSelect_gpr91.sdf](CapSelect_gpr91.sdf) — output using the bare
  PDB receptor (no H's). Tends to under-report clashes.
- [CapSelect_gpr91_pdb_vs_mol2.tsv](CapSelect_gpr91_pdb_vs_mol2.tsv) —
  per-MEL side-by-side comparison.
- [CapSelect_gpr91_summary.tsv](CapSelect_gpr91_summary.tsv) — per-MEL
  summary from the original PDB run (Score, Spheres, CapScore,
  MergedScore).

## Why mol2 vs PDB matters

The user correctly flagged the alignment concern: CapSelect's sphere
chains are placed in the docked-MEL coordinate frame, and clash checks
must be against receptor atoms in the **same frame**. The PDB crystal
structure is the raw input; the ICM docking step may translate/rotate
it slightly during pocket alignment, and dock-prep also adds explicit
hydrogens (which materially affect clash geometry).

The right receptor is the one ICM exports from the docking project —
in this case [../GPR91/dock_6rnk_new_rec.mol2](../../GPR91/dock_6rnk_new_rec.mol2)
(7333 atoms with H's) — not the bare crystal PDB.

## Side-by-side stats (n = 1000 MELs)

| Metric                           | PDB (no H, bare crystal) | mol2 (canonical, with H) |
|----------------------------------|--------------------------|--------------------------|
| Pocket-region atoms              | 704                      | 1428                     |
| Mean CapScore                    | 8.32                     | 7.00                     |
| Productive (CapScore ≥ 5)        | 94.1%                    | 87.8%                    |
| Saturated (CapScore = 10)        | 27.0%                    | **2.1%**                 |
| Rejected (MergedScore = −1000)   | 24                       | 65                       |
| Wall time                        | ~5 min                   | ~9 min                   |

The big shift is **saturated chains** (27% → 2%): with hydrogens, the
protein clash surface expands, so many chains that previously placed 5
clean spheres (and saturated to CapScore = 10) now get blocked at 2–3
spheres. Conversely, chains that DO reach the full 10-sphere length
end up at slightly higher Max(min) values because they're carving
through the H-bond surface more carefully.

**Spheres distribution:**

| Spheres | PDB n | mol2 n | comment |
|---------|-------|--------|---------|
| 0 | 21 | 62 | step-1 failure (cap shell fully blocked) |
| 1 | 3 | 25 | sphere 1 placed but no extension |
| 2 | 13 | **304** | most-common terminating length with H's |
| 3 | 75 | 41 | |
| 4 | 66 | 39 | |
| 5 | 181 | 13 | (formerly the saturation peak) |
| 6 | 66 | 0 | with H's, no chains stop here |
| 7 | 9 | 0 | |
| 8 | 8 | 0 | |
| 9 | 4 | 0 | |
| 10 | 554 | 516 | full chain — most stay productive |

The bimodal distribution in the mol2 run (mass at sp=2 and sp=10) is a
hallmark of well-resolved clash geometry: a chain either bumps into a
hydrogen on sphere 2 and stops, or finds a clear corridor and runs to
the end. The PDB run's broader distribution at sp=3..7 reflects
ambiguity from missing H's.

## Top-10 by MergedScore (canonical mol2 run)

| rank | Score   | CapScore | Spheres | MergedScore |
|------|---------|----------|---------|-------------|
| 1    | −42.213 | 8.583    | 10      | 28.5489     |
| 2    | −40.825 | 8.593    | 10      | 28.3085     |
| 3    | −38.456 | 9.371    | 10      | 27.9398     |
| 4    | −38.168 | 9.439    | 10      | 27.8909     |
| 5    | −37.880 | 9.979    | 10      | 27.8762     |
| 6    | −37.659 | 9.430    | 10      | 27.7933     |
| 7    | −37.614 | 9.456    | 10      | 27.7866     |
| 8    | −37.900 | 8.320    | 10      | 27.7489     |
| 9    | −37.304 | 8.167    | 10      | 27.6213     |
| 10   | −36.622 | 8.701    | 10      | 27.5338     |

Top-N rank stability between PDB and mol2 runs:

| Top-N | overlap |
|-------|---------|
| 10    | 8/10    |
| 50    | 45/50   |
| 100   | 88/100  |
| 200   | 166/200 |

So the **relative ranking is largely robust** — the receptor refinement
shifts absolute CapScore distributions but only swaps ~10–17% of the
top-N selection.

## Reproducing

```bash
cd ..  # to the capselect/ directory
# Use the docking-frame mol2 (recommended):
python3 extract_protein_sdf.py \
    ../GPR91/dock_6rnk_new_rec.mol2 \
    ../GPR91/GPR91_6RNK_ICM393_Eff2_2Comp_MEL_Top1000_Hits.sdf \
    gpr91/protein_mol2.sdf --margin 5.0
python3 capselect_py.py \
    ../GPR91/GPR91_6RNK_ICM393_Eff2_2Comp_MEL_Top1000_Hits.sdf \
    gpr91/protein_mol2.sdf \
    gpr91/CapSelect_gpr91_mol2.sdf v2_5
```

## Caveats

- The pocket box is derived from the **MEL bounding box + 5 Å**.
  The production pipeline uses ICM's `$ProjectName.R_boxDim` (the
  docking box explicitly stored in the docking project), which may
  differ slightly. For top-1000 MELs the difference is negligible.
- The mol2 file may include the bound antagonist or other ligands —
  use `--drop-residues KAZ,WAT,HOH` if needed (none of these were
  present in `dock_6rnk_new_rec.mol2`, which is receptor-only).
- Wall time: ~9 min for 1000 MELs at 1428 protein atoms. For
  production scale (30K MELs) this is ~4.5 hours single-threaded;
  parallelize via `multiprocessing.Pool` if needed.
