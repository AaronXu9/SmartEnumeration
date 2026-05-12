#!/usr/bin/env python3
"""
edit_mel_cap.py

Preprocess an ICM docked-MEL SDF (with isotope-labeled cap atoms) into an
"attachment-ready" SDF suitable for ICM Screen Replacement Group. This
replaces the manual GUI edit step:

    Right-click m (Docked MEL Frag) -> Edit -> Edit Compound -> Delete Cap Group

with a deterministic, scriptable transformation.

INPUT  : SDF containing one or more docked MEL fragments, with ICM's
         isotope-based cap labels (M ISO lines). The cap atoms are marked
         with non-natural isotope masses:
             - iso=14 on C  : structural-placeholder atom of the cap
             - iso=15 on C  : phenyl-ring placeholder carbons
             - iso=13 on C  : alternate ring placeholders
             - iso=18 on O  : carbonyl/sulfonyl oxygen placeholders
             - iso=34 on S  : sulfonyl sulfur placeholders
             - iso=16 on N  : SCAFFOLD-side N bonded to the cap
                              (this atom is KEPT -- it's the growth-vector origin)

OUTPUT : SDF where each MEL has had its cap atoms removed and an M APO
         annotation injected on the junction atom (the scaffold atom that
         lost a bond when the cap was deleted). This is the annotation
         ICM Screen Replacement Group uses to know where to attach each
         candidate synthon.

The script does NOT re-minimize coordinates -- the retained atoms keep
their docked 3D positions exactly as in the input. This matters because
the docked pose is what we are anchoring to.

Usage:
    python3 edit_mel_cap.py <input.sdf> <output.sdf>
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Cap-atom classification                                                      #
# --------------------------------------------------------------------------- #

# Isotope masses ICM uses to mark atoms that are part of the cap.
# Every atom in the molecule that carries one of these (symbol, isotope_mass)
# labels is a cap atom and gets deleted. The attachment point (where M APO
# is placed) is the non-iso heavy neighbor of a cap atom -- always a
# scaffold-body atom, never a cap-body atom.
CAP_ISO = {
    ("C", 13), ("C", 14), ("C", 15),
    ("N", 14), ("N", 15), ("N", 16),
    ("O", 17), ("O", 18),
    ("S", 33), ("S", 34),
}


# --------------------------------------------------------------------------- #
# Minimal SDF/MOL parser and writer (V2000)                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Atom:
    idx: int          # 1-based index in the input
    x: float
    y: float
    z: float
    sym: str
    iso: Optional[int] = None          # isotope mass if set, else None
    charge: int = 0                    # from V2000 atom line (legacy)


@dataclass
class Bond:
    a1: int           # 1-based atom index
    a2: int           # 1-based atom index
    order: int


@dataclass
class Mol:
    title: str
    software: str
    comment: str
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    # Copy-through property lines we should preserve or rewrite
    charge_entries: list[tuple[int, int]] = field(default_factory=list)  # (atom, charge)
    # SDF data block (everything between M END and $$$$)
    sdf_data_block: str = ""


def _read_int(s: str) -> int:
    s = s.strip()
    return int(s) if s else 0


def parse_mol_block(text: str) -> Mol:
    lines = text.splitlines()
    title = lines[0] if len(lines) > 0 else ""
    software = lines[1] if len(lines) > 1 else ""
    comment = lines[2] if len(lines) > 2 else ""
    counts = lines[3]
    n_atoms = int(counts[0:3])
    n_bonds = int(counts[3:6])

    atoms: list[Atom] = []
    for i in range(n_atoms):
        line = lines[4 + i]
        x = float(line[0:10]); y = float(line[10:20]); z = float(line[20:30])
        sym = line[31:34].strip()
        # Legacy charge field (positions 36-39)
        charge_code = 0
        if len(line) >= 39:
            try:
                charge_code = int(line[36:39])
            except ValueError:
                charge_code = 0
        atoms.append(Atom(idx=i+1, x=x, y=y, z=z, sym=sym, charge=charge_code))

    bonds: list[Bond] = []
    for i in range(n_bonds):
        line = lines[4 + n_atoms + i]
        a1 = int(line[0:3]); a2 = int(line[3:6]); order = int(line[6:9])
        bonds.append(Bond(a1=a1, a2=a2, order=order))

    # Property lines after the bond block
    # We care about: M ISO (isotope assignments), M CHG (charges), M END (terminator)
    charge_entries: list[tuple[int, int]] = []
    prop_start = 4 + n_atoms + n_bonds
    for line in lines[prop_start:]:
        if line.startswith("M  END"):
            break
        if line.startswith("M  ISO"):
            parts = line.split()
            count = int(parts[2])
            for j in range(count):
                a = int(parts[3 + 2*j])
                iso = int(parts[4 + 2*j])
                if 1 <= a <= len(atoms):
                    atoms[a-1].iso = iso
        elif line.startswith("M  CHG"):
            parts = line.split()
            count = int(parts[2])
            for j in range(count):
                a = int(parts[3 + 2*j])
                chg = int(parts[4 + 2*j])
                charge_entries.append((a, chg))

    return Mol(title=title, software=software, comment=comment,
               atoms=atoms, bonds=bonds, charge_entries=charge_entries)


def split_sdf_entries(text: str) -> list[tuple[str, str]]:
    """Split an SDF file into (mol_block, data_block) pairs.

    mol_block ends at the line "M  END" (inclusive)
    data_block is everything after M END up to the $$$$ terminator (exclusive)
    """
    out: list[tuple[str, str]] = []
    # Normalize line endings and split on $$$$ record separator
    raw_entries = text.replace("\r\n", "\n").split("$$$$\n")
    for raw in raw_entries:
        if not raw.strip():
            continue
        # Find "M  END" terminator
        end_idx = raw.find("M  END")
        if end_idx < 0:
            continue
        # Include the M END line fully
        nl = raw.find("\n", end_idx)
        mol_block = raw[:nl+1] if nl != -1 else raw
        data_block = raw[nl+1:] if nl != -1 else ""
        out.append((mol_block, data_block))
    return out


def write_mol_block(mol: Mol, kept_apo_atoms: list[tuple[int, int]]) -> str:
    """Serialize a Mol back to V2000 text (mol block through 'M  END\n').

    kept_apo_atoms : list of (new_atom_idx_1based, apo_value) pairs to emit
                     as M APO property lines.
    """
    lines = [mol.title, mol.software, mol.comment]
    n_atoms = len(mol.atoms); n_bonds = len(mol.bonds)
    # Counts line: nnnbbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    lines.append(f"{n_atoms:>3d}{n_bonds:>3d}  0  0  0  0  0  0  0  0999 V2000")
    # Atom lines
    for a in mol.atoms:
        # V2000 atom line format (fixed columns):
        #   xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz aaa ddcccsssHHHbbbvvvHHH  mmmnnneee
        # We only fill the coordinate/symbol/charge fields.
        line = (f"{a.x:10.4f}{a.y:10.4f}{a.z:10.4f} "
                f"{a.sym:<3s}"
                f"{a.charge:>3d}"   # 3-char charge code slot (legacy)
                f"  0  0  0  0  0  0  0  0  0  0  0")
        lines.append(line)
    # Bond lines
    for b in mol.bonds:
        line = f"{b.a1:>3d}{b.a2:>3d}{b.order:>3d}  0  0  0  0"
        lines.append(line)
    # Property lines
    # Emit M CHG entries preserved from the input (re-indexed by caller)
    if mol.charge_entries:
        # Batch into rows of up to 8 entries per line per V2000 convention
        chunks = [mol.charge_entries[i:i+8] for i in range(0, len(mol.charge_entries), 8)]
        for chunk in chunks:
            count = len(chunk)
            parts = f"M  CHG{count:>3d}"
            for a, c in chunk:
                parts += f"{a:>4d}{c:>4d}"
            lines.append(parts)
    # Emit M APO entries
    if kept_apo_atoms:
        chunks = [kept_apo_atoms[i:i+8] for i in range(0, len(kept_apo_atoms), 8)]
        for chunk in chunks:
            count = len(chunk)
            parts = f"M  APO{count:>3d}"
            for a, v in chunk:
                parts += f"{a:>4d}{v:>4d}"
            lines.append(parts)
    lines.append("M  END")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Cap-editing logic                                                            #
# --------------------------------------------------------------------------- #

def find_cap_and_junction(mol: Mol) -> tuple[set[int], list[int]]:
    """Identify (cap_atom_indices, junction_atom_indices) for a MEL.

    Rule: every iso-labeled atom (any atom with an isotope mass in CAP_ISO)
    is part of the cap and gets deleted. The junction atom is the non-H
    heavy atom that is bonded to a cap atom and is itself NOT iso-labeled.

    This single rule covers all observed cap topologies. In particular it
    correctly handles N(iso=16): although that label sometimes appears at
    the cap/scaffold boundary (in benzylamine-style caps) and sometimes
    deeper inside the cap (in dimethyl-amide-style caps), in BOTH cases
    the manual GUI workflow deletes the iso=16 N along with the rest of
    the cap and places the attachment point on the next atom out -- the
    non-iso heavy atom that the cap was hanging off.

    Worked example -- MEL 2 (benzylamine cap):
        Original:  ...[vinyl-CH]---N(iso=16)H---C(iso=14)<phenyl(iso=15)>
        Cap atoms: {N(iso=16), C(iso=14), 5x C(iso=15)}
        Junction:  the vinyl-CH carbon (non-iso, bonded to N(iso=16))
        After:     ...[vinyl-CH]*  with M APO on the vinyl-CH

    Worked example -- MEL 4 (dimethyl-amide cap):
        Original:  [scaffold-N]---C(iso=14)(=O iso=18)---N(iso=16)H---C(iso=13)H3
        Cap atoms: {C(iso=13), N(iso=16), C(iso=14), O(iso=18)}
        Junction:  the scaffold N (non-iso, bonded to C(iso=14))
        After:     [scaffold-N]*  with M APO on the scaffold N

    Worked example -- MEL 6 (sulfonamide cap):
        Original:  [scaffold-N]---S(iso=34)(=O iso=18)(=O iso=18)---<phenyl(iso=15) with C(iso=14) ipso>
        Cap atoms: {S(iso=34), 2x O(iso=18), C(iso=14), 5x C(iso=15)}
        Junction:  the scaffold N (non-iso, bonded to S(iso=34))
        After:     [scaffold-N]*  with M APO on the scaffold N
    """
    cap_set: set[int] = {a.idx for a in mol.atoms
                         if a.iso is not None
                         and (a.sym, a.iso) in CAP_ISO}

    if not cap_set:
        return set(), []

    junction_set: set[int] = set()
    for b in mol.bonds:
        cross = (b.a1 in cap_set) ^ (b.a2 in cap_set)
        if not cross:
            continue
        nonc = b.a2 if b.a1 in cap_set else b.a1
        if mol.atoms[nonc - 1].sym == "H":
            continue
        junction_set.add(nonc)

    return cap_set, sorted(junction_set)


def dangling_hydrogens(mol: Mol, to_delete: set[int]) -> set[int]:
    """H atoms whose only bond is to an atom in `to_delete`. These are orphaned
    by the cap deletion and must also be removed.
    """
    orphan_hs: set[int] = set()
    # Build adjacency: atom -> list of neighbors
    adj: dict[int, list[int]] = {a.idx: [] for a in mol.atoms}
    for b in mol.bonds:
        adj[b.a1].append(b.a2)
        adj[b.a2].append(b.a1)
    for a in mol.atoms:
        if a.sym != "H":
            continue
        neighbors = adj[a.idx]
        if neighbors and all(n in to_delete for n in neighbors):
            orphan_hs.add(a.idx)
    return orphan_hs


def edit_mel(mol: Mol) -> tuple[Mol, list[tuple[int, int]]]:
    """Return (edited_mol, apo_entries).

    apo_entries are (new_atom_idx_1based, apo_value=1) for M APO lines.
    """
    cap_set, junction_list = find_cap_and_junction(mol)

    if not cap_set:
        # Nothing isotope-labeled -- this MEL has no cap to strip.
        return mol, []

    if not junction_list:
        raise RuntimeError(
            f"MEL '{mol.title}': found {len(cap_set)} cap atoms but no "
            f"junction atom (no bond crosses cap<->non-cap boundary)."
        )

    if len(junction_list) > 1:
        # Multiple junction atoms => this MEL has multiple attachment points.
        # Screen Replacement Group handles one at a time; emit all APOs and
        # let the downstream script iterate.
        sys.stderr.write(
            f"[warn] MEL '{mol.title}': {len(junction_list)} junction atoms "
            f"detected; emitting APO on all of them.\n"
        )

    # Orphaned hydrogens that only bonded to cap atoms
    orphan_hs = dangling_hydrogens(mol, cap_set)
    to_delete = cap_set | orphan_hs

    # -----------------------------------------------------------------
    # Connected-component check: after removing cap atoms, some heavy
    # atoms (like a terminal NH2 on a urea-style cap) can end up
    # disconnected from the main scaffold. Find the component that
    # contains the junction(s) and delete any heavy atom outside it.
    # -----------------------------------------------------------------
    surviving = {a.idx for a in mol.atoms} - to_delete
    adj: dict[int, list[int]] = {i: [] for i in surviving}
    for b in mol.bonds:
        if b.a1 in surviving and b.a2 in surviving:
            adj[b.a1].append(b.a2)
            adj[b.a2].append(b.a1)
    # BFS from the first junction atom
    seen: set[int] = set()
    stack = [junction_list[0]]
    while stack:
        u = stack.pop()
        if u in seen: continue
        seen.add(u)
        stack.extend(adj[u])
    # Any surviving atom not in `seen` is in a disconnected fragment.
    disconnected = surviving - seen
    if disconnected:
        disconnected_heavy = {i for i in disconnected
                              if mol.atoms[i - 1].sym != "H"}
        if disconnected_heavy:
            sys.stderr.write(
                f"[warn] MEL '{mol.title}': {len(disconnected_heavy)} heavy "
                f"atom(s) disconnected from scaffold after cap removal -- "
                f"also deleting: "
                f"{sorted((i, mol.atoms[i-1].sym) for i in disconnected_heavy)}\n"
            )
        to_delete |= disconnected

    # Build index remapping: old 1-based -> new 1-based
    old_to_new: dict[int, int] = {}
    new_atoms: list[Atom] = []
    new_idx = 0
    for a in mol.atoms:
        if a.idx in to_delete:
            continue
        new_idx += 1
        old_to_new[a.idx] = new_idx
        # Strip iso labels on junction atoms (they are no longer cap-marked
        # placeholders once the cap is gone; keep regular chemistry).
        new_iso = None if a.idx in junction_list else a.iso
        new_atoms.append(Atom(idx=new_idx, x=a.x, y=a.y, z=a.z,
                              sym=a.sym, iso=new_iso, charge=a.charge))

    # Rebuild bonds, skipping any that touch a deleted atom
    new_bonds: list[Bond] = []
    for b in mol.bonds:
        if b.a1 in to_delete or b.a2 in to_delete:
            continue
        new_bonds.append(Bond(a1=old_to_new[b.a1], a2=old_to_new[b.a2],
                              order=b.order))

    # Remap preserved charges
    new_charges: list[tuple[int, int]] = []
    for old_a, c in mol.charge_entries:
        if old_a in old_to_new:
            new_charges.append((old_to_new[old_a], c))

    # APO entries -> the junction atoms in their new indexing
    apo_entries = [(old_to_new[j], 1) for j in junction_list]

    edited = Mol(
        title=mol.title, software=mol.software, comment=mol.comment,
        atoms=new_atoms, bonds=new_bonds, charge_entries=new_charges,
    )
    return edited, apo_entries


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        sys.stderr.write("\nUsage: edit_mel_cap.py <input.sdf> <output.sdf>\n")
        return 2
    in_path, out_path = argv[1], argv[2]

    with open(in_path) as f:
        sdf_text = f.read()

    entries = split_sdf_entries(sdf_text)
    sys.stderr.write(f"[info] {len(entries)} MEL entries in {in_path}\n")

    out_chunks: list[str] = []
    # Map output position -> (title, list of APO atom indices in the WRITTEN
    # mol block). We emit a sidecar TSV alongside the SDF so the downstream
    # ICM driver knows which atom to point processLigandICM at.
    apo_index: list[tuple[int, str, list[int]]] = []

    n_edited = n_unchanged = n_skipped = 0
    for i, (mol_block, data_block) in enumerate(entries, 1):
        try:
            mol = parse_mol_block(mol_block)
        except Exception as e:
            sys.stderr.write(f"[skip] entry #{i}: parse failed: {e}\n")
            n_skipped += 1
            continue

        try:
            edited_mol, apo_entries = edit_mel(mol)
        except Exception as e:
            sys.stderr.write(f"[skip] entry #{i} ({mol.title!r}): {e}\n")
            n_skipped += 1
            continue

        if not apo_entries:
            sys.stderr.write(
                f"[note] entry #{i} ({mol.title!r}): no isotope-labeled cap, "
                f"passing through unchanged\n"
            )
            n_unchanged += 1
            out_chunks.append(mol_block + data_block + "$$$$\n")
            apo_index.append((len(apo_index) + 1, mol.title, []))
            continue

        n_edited += 1
        new_mol_block = write_mol_block(edited_mol, apo_entries)
        out_chunks.append(new_mol_block + data_block + "$$$$\n")
        apo_index.append(
            (len(apo_index) + 1, mol.title, [a for a, _ in apo_entries])
        )

    with open(out_path, "w") as f:
        f.write("".join(out_chunks))

    # Write a sidecar TSV with the APO atom index(es) for each output entry
    tsv_path = out_path.rsplit(".", 1)[0] + "_apo_index.tsv"
    with open(tsv_path, "w") as f:
        f.write("entry_idx\ttitle\tapo_atom_indices\n")
        for eidx, title, apos in apo_index:
            f.write(f"{eidx}\t{title}\t{','.join(str(a) for a in apos)}\n")

    sys.stderr.write(
        f"[done] wrote {out_path} (and {tsv_path}): "
        f"edited={n_edited}, unchanged={n_unchanged}, skipped={n_skipped}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
