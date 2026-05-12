"""extract_protein_sdf.py — Build a CapSelect-compatible protein.sdf
from a receptor file (PDB or mol2) and a fragments.sdf bounding box.

Antonina's manual specifies that protein.sdf must:
- Contain protein heavy atoms within ~5 Å of the docking pocket box
- Have hydrogens included (her manual emphasizes "display hydrogens"
  and "Toggle Wire Representation" before exporting from ICM)
- Use the AAAA9999 counts-line format on line 4 (so the C++ parser
  can read substr(0, 4) as the atom count)

For best results, use the **receptor as exported from the ICM docking
project** (typically a .mol2 file with hydrogens). This guarantees the
coordinates are in the same frame as the docked MELs. A raw PDB
crystal structure may be in a slightly different frame after
dock-prep (translation/rotation during pocket alignment).

Strategy:
1. Read the docked MEL fragments to determine the pocket bounding box
   (= bounding box of all fragment atoms + a margin in each direction).
2. Read the receptor file (PDB or mol2); keep only atoms within the box.
3. Write as a single-molecule SDF with the AAAA9999 counts line.

Usage:
    python3 extract_protein_sdf.py receptor.{pdb,mol2} fragments.sdf protein.sdf [--margin 5.0]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_fragment_box(sdf_path: str, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) corners of the bounding box covering all fragment atoms,
    inflated by `margin` Å in each direction.
    """
    coords: list[tuple[float, float, float]] = []
    with open(sdf_path) as f:
        text = f.read()
    for blob in text.split('$$$$\n'):
        if not blob.strip():
            continue
        lines = blob.splitlines()
        if len(lines) < 4:
            continue
        try:
            n_atoms = int(lines[3][:3])
        except ValueError:
            continue
        for k in range(n_atoms):
            ln = lines[4 + k]
            if len(ln) < 30:
                continue
            try:
                x = float(ln[0:10]); y = float(ln[10:20]); z = float(ln[20:30])
            except ValueError:
                continue
            coords.append((x, y, z))
    if not coords:
        raise SystemExit(f'no atoms parsed from {sdf_path}')
    arr = np.array(coords)
    lo = arr.min(axis=0) - margin
    hi = arr.max(axis=0) + margin
    return lo, hi


def parse_mol2_atoms(mol2_path: str,
                     drop_residues: set[str] | None = None,
                     ) -> list[tuple[float, float, float, str]]:
    """Read mol2 ATOM block; return (x, y, z, element) per atom.

    Uses Tripos atom-type prefix to derive element (e.g. 'C.ar' -> 'C').
    Optionally drops atoms whose subst_name (column 8) is in drop_residues
    (e.g. {'WAT', 'HOH'} to exclude waters; {'KAZ'} to exclude bound ligand).
    """
    out = []
    drop = drop_residues or set()
    with open(mol2_path) as f:
        in_atom = False
        for ln in f:
            if ln.startswith('@<TRIPOS>ATOM'):
                in_atom = True
                continue
            if ln.startswith('@<TRIPOS>'):
                in_atom = False
                continue
            if not in_atom or not ln.strip():
                continue
            parts = ln.split()
            # mol2 ATOM line: id name x y z atom_type [subst_id [subst_name [charge]]]
            if len(parts) < 6:
                continue
            try:
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
            except ValueError:
                continue
            atom_type = parts[5]
            elem = atom_type.split('.')[0]   # 'C.ar' -> 'C', 'N.am' -> 'N'
            subst_name = parts[7] if len(parts) >= 8 else ''
            # Strip trailing residue numbers from subst_name (e.g. "ARG123" -> "ARG")
            res_code = ''.join(c for c in subst_name if c.isalpha())[:3]
            if res_code and res_code in drop:
                continue
            out.append((x, y, z, elem))
    return out


def parse_pdb_atoms(pdb_path: str,
                    keep_chains: set[str] | None = None,
                    keep_hetatm: bool = False,
                    drop_hetatm_residues: set[str] | None = None,
                    ) -> list[tuple[float, float, float, str]]:
    """Read ATOM/HETATM lines, return list of (x, y, z, element).

    keep_chains: only keep atoms with chain id in this set (None = all).
    keep_hetatm: include HETATM records (default False — drops bound ligands,
                 waters, lipids, ions).
    drop_hetatm_residues: if keep_hetatm=True, still drop these specific
                          residue codes (e.g. {'KAZ'}).
    """
    out = []
    with open(pdb_path) as f:
        for ln in f:
            rec = ln[:6]
            if rec not in ('ATOM  ', 'HETATM'):
                continue
            if rec == 'HETATM' and not keep_hetatm:
                continue
            altloc = ln[16] if len(ln) > 16 else ' '
            if altloc not in (' ', 'A'):
                continue
            chain = ln[21] if len(ln) > 21 else ' '
            if keep_chains is not None and chain not in keep_chains:
                continue
            res = ln[17:20].strip()
            if drop_hetatm_residues and rec == 'HETATM' and res in drop_hetatm_residues:
                continue
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except ValueError:
                continue
            elem = ln[76:78].strip() if len(ln) >= 78 else ''
            if not elem:
                name = ln[12:16].strip()
                elem = name[0] if name else 'C'
            out.append((x, y, z, elem))
    return out


def write_protein_sdf(out_path: str, atoms: list[tuple[float, float, float, str]],
                       title: str = 'protein') -> None:
    """Write a CapSelect-compatible SDF (counts line uses AAAA9999 format)."""
    n = len(atoms)
    if n == 0:
        raise SystemExit('no atoms to write')
    if n > 9998:
        # AAAA9999 format implies a 4-digit count (max 9999); above that the C++
        # parser would still substr(0, 4) which works for any digit count up to 9999.
        # If exceeded, we'd need the binary's parser tested at higher counts; for now warn.
        print(f'warning: {n} atoms exceeds AAAA9999 format range', file=sys.stderr)
    counts_line = f'{n:>4d}9999  0  0  1  0  0  0  0  0999 V2000'
    lines = [
        title,
        '  CapSelect 05012600003D',
        '',
        counts_line,
    ]
    for x, y, z, elem in atoms:
        # Standard V2000 atom line: 10.4f x 3, then space, then 3-char element
        # left-justified, then mass-diff '  0' (which gives col-35 = '0')
        elem_padded = f'{elem:<3s}'[:3]
        lines.append(f'{x:>10.4f}{y:>10.4f}{z:>10.4f} {elem_padded}  0  0  0  0  0  0  0  0  0  0  0  0')
    lines.append('M  END')
    lines.append('$$$$')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('receptor', help='input receptor file: .pdb or .mol2 (auto-detected by extension)')
    ap.add_argument('fragments_sdf', help='docked MEL fragments SDF (defines pocket box)')
    ap.add_argument('out_sdf', help='output protein.sdf')
    ap.add_argument('--margin', type=float, default=5.0,
                    help='angstroms to inflate the fragment bounding box (default 5.0)')
    ap.add_argument('--chains', default='A',
                    help='[PDB only] comma-separated list of chain IDs to keep (default A)')
    ap.add_argument('--keep-hetatm', action='store_true',
                    help='[PDB only] include HETATM records (default: drop ligands/waters/lipids)')
    ap.add_argument('--drop-residues', default='',
                    help='[mol2] comma-separated 3-letter residue codes to drop (e.g. WAT,HOH,KAZ)')
    args = ap.parse_args()

    lo, hi = parse_fragment_box(args.fragments_sdf, args.margin)
    print(f'fragment bounding box (with {args.margin} Å margin):', file=sys.stderr)
    print(f'  x: [{lo[0]:.2f}, {hi[0]:.2f}]', file=sys.stderr)
    print(f'  y: [{lo[1]:.2f}, {hi[1]:.2f}]', file=sys.stderr)
    print(f'  z: [{lo[2]:.2f}, {hi[2]:.2f}]', file=sys.stderr)

    if args.receptor.lower().endswith('.mol2'):
        drop = set(args.drop_residues.split(',')) if args.drop_residues else None
        atoms = parse_mol2_atoms(args.receptor, drop_residues=drop)
        print(f'parsed {len(atoms)} atoms from {args.receptor} (mol2; drop={drop})',
              file=sys.stderr)
    else:
        chains = set(args.chains.split(',')) if args.chains else None
        atoms = parse_pdb_atoms(args.receptor, keep_chains=chains,
                                keep_hetatm=args.keep_hetatm)
        print(f'parsed {len(atoms)} atoms from {args.receptor} '
              f'(pdb; chains={chains}, hetatm={args.keep_hetatm})', file=sys.stderr)
    kept = [(x, y, z, e) for (x, y, z, e) in atoms
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and lo[2] <= z <= hi[2]]
    print(f'{len(kept)} atoms within pocket box', file=sys.stderr)

    write_protein_sdf(args.out_sdf, kept)
    print(f'wrote {args.out_sdf}', file=sys.stderr)


if __name__ == '__main__':
    main()
