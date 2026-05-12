#!/usr/bin/env python3
"""
For the Top-N ranked MEL fragments in a mapping CSV:
generate enumeration instruction rules, enumerate and filter full ligands,
then extract compatible synthons whose full compounds survived all filters.
"""

import sys
import os
import pickle
import argparse
from typing import Optional, List, Tuple, Iterable, Dict
from collections import defaultdict
import pandas as pd
from rdkit import Chem


# ── Lookup: ICM InChIKey → all openvs_full_synthon_ids ────────────────────────

def lookup_openvs_full_synthon_ids_from_row(row) -> List[str]:
    """Extract openvs_full_synthon_id(s) directly from a DataFrame row."""
    val = row['full_synthon_ids']
    if not val:
        return []
    return val


# ── Enumeration core ───────────────────────────────────────────────────────────

def extract_inchikey_from_full_synthon_id(full_synthon_id: str) -> Optional[str]:
    """Extract the real synthon OpenVS InChIKey from a full_synthon_id string."""
    parts = full_synthon_id.split('_____')
    if len(parts) != 3:
        return None
    k = [x for x in parts[1:] if not x.startswith('sssss')]
    if len(k) == 1:
        return k[0]
    raise ValueError(f"expected 1 InChIKey, found {len(k)}")


def build_DoSoT_SynthinInChiKey_Rxnid_Synthonplace(synth) -> Dict[str, set]:
    """Build reverse lookup: synthon InChIKey → set of (rxnid, slot) pairs."""
    rxn_slot_pairs = defaultdict(set)
    for rxnid, pairs in synth.DoLoT_Rxnid__SynthonInchikey_Synthonplace.items():
        for key, slot in pairs:
            rxn_slot_pairs[key].add((rxnid, slot))
    return rxn_slot_pairs


def get_allowed_rxn_slots_for_inchikey(synth, inchikey: str) -> List[Tuple[str, int]]:
    """Return all (rxnid, slot) pairs for a synthon InChIKey, 2-component reactions only."""
    rev_attr = "DoSoT_SynthinInChiKey_Rxnid_Synthonplace"
    rev = getattr(synth, rev_attr, None)
    if rev is None:
        rev = build_DoSoT_SynthinInChiKey_Rxnid_Synthonplace(synth)
        setattr(synth, rev_attr, rev)

    pairs = list(rev.get(inchikey, []))
    dos = synth.DoS_Rxnid_Synthonplace
    pairs = [(r, s) for (r, s) in pairs if len(dos.get(r, set())) == 2]

    return sorted(set(pairs), key=lambda x: (x[0], x[1]))


def get_candidates_for_remaining_slots(synth, rxnid: str, occupied_slot: int) -> Dict[int, List[str]]:
    """Return all candidate synthon InChIKeys for each unoccupied slot in a reaction."""
    slot_set = synth.DoS_Rxnid_Synthonplace.get(rxnid)
    remaining = sorted(s for s in slot_set if s != occupied_slot)
    slot_map = synth.DoDoL_Rxnid___Synthonplace_SynthonInchi[rxnid]
    return {slot: slot_map.get(slot, []) for slot in remaining}


def make_lotot_keys_for_2comp_rxn(
    rxnid: str,
    occupied_slot: int,
    action_inchikey: str,
    empty_slot: int,
    candidate_inchikeys: Iterable[str],
) -> List[Tuple[str, Tuple]]:
    """Build LoToT instruction tuples for one reaction slot pair."""
    lotot = []
    a, b = occupied_slot - 1, empty_slot - 1
    for k in candidate_inchikeys:
        pair = [None, None]
        pair[a] = action_inchikey
        pair[b] = k
        lotot.append((rxnid, (pair[0], pair[1])))
    return lotot


def generate_instruction_rules_for_full_synthon_id(synth, full_synthon_id: str) -> List[Tuple]:
    """Generate all LoToT enumeration instruction rules for a single full_synthon_id."""
    inchikey = extract_inchikey_from_full_synthon_id(full_synthon_id)
    if inchikey is None:
        raise ValueError(f"Failed to extract InChIKey from: {full_synthon_id}")

    rxn_slot_pairs = get_allowed_rxn_slots_for_inchikey(synth, inchikey)

    lotot = []
    for rxnid, occupied_slot in rxn_slot_pairs:
        slot_candidates = get_candidates_for_remaining_slots(synth, rxnid, occupied_slot)
        for empty_slot, candidates in slot_candidates.items():
            lotot.extend(
                make_lotot_keys_for_2comp_rxn(
                    rxnid, occupied_slot, inchikey, empty_slot, candidates
                )
            )

    return list(set(lotot))


def generate_instruction_rules_for_full_synthon_id_list(synth, full_synthon_ids: List[str]) -> List[Tuple]:
    """Merge instruction rules across all full_synthon_ids for one MEL."""
    all_lotot = set()
    for fid in full_synthon_ids:
        rules = generate_instruction_rules_for_full_synthon_id(synth, fid)
        print(f"        {fid}: {len(rules)} instructions")
        all_lotot.update(rules)
    return list(all_lotot)


# ── Save instruction rules ─────────────────────────────────────────────────────

def save_instruction_rules(lotot: list, rank_prefix: str, output_dir: str) -> str:
    """Save instruction rules PKL using rank-based filename, return save path."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(lotot)
    filename = f"{rank_prefix}_{n}_instruction_rules.pkl"
    save_path = os.path.join(output_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(lotot, f)
    return save_path


# ── Enumerate and filter full ligands ─────────────────────────────────────────

def enumerate_and_filter(
    synth,
    lotot: list,
    rank_prefix: str,
    products_output_dir: str,
    IncrementalPropertyFilter,
    SubstructureCatalogFilter,
) -> Tuple[List, List]:
    """Enumerate full ligands from instruction rules, apply filters, save and return products."""
    os.makedirs(products_output_dir, exist_ok=True)
    logs_dir = os.path.join(products_output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    result_path = os.path.join(products_output_dir, f"{rank_prefix}_enumerated_products.pkl")
    log_path    = os.path.join(logs_dir, f"{rank_prefix}.log")

    # Enumerate all full products from instruction rules
    inchi_raw, rdmol_raw = synth.Parallel_GetFullProduct(
        lotot,
        User_ParallelizationInput="LoToT",
        User_EnforceIsotope=False,
        User_ShowProgress=True,
        User_SCap=False,
    )

    # Apply property and substructure filters, then deduplicate
    prop_filter      = IncrementalPropertyFilter()
    substruct_filter = SubstructureCatalogFilter(max_workers=8, chunk_size=512)

    with open(log_path, "w") as log:
        log.write(f"Initial products: {len(inchi_raw)}\n")

        inchi_prop, rdmol_prop = prop_filter.Get_FilteredInchiRdmol(rdmol_raw)
        log.write(f"After property filter: {len(inchi_prop)}\n")

        inchi_final, rdmol_final = substruct_filter.Get_FilteredInchiRdmol(rdmol_prop)
        log.write(f"After substructure filter: {len(inchi_final)}\n")

        # Deduplicate by InChIKey
        seen = set()
        inchi_dedup, rdmol_dedup = [], []
        for ik, mol in zip(inchi_final, rdmol_final):
            if ik in seen:
                continue
            seen.add(ik)
            inchi_dedup.append(ik)
            rdmol_dedup.append(mol)
        log.write(f"After InChIKey dedup: {len(inchi_dedup)}\n")

    with open(result_path, "wb") as f:
        pickle.dump((inchi_dedup, rdmol_dedup), f)

    print(f"      {len(inchi_dedup)} filtered products saved → {result_path}")
    print(f"      Log → {log_path}")

    return inchi_dedup, rdmol_dedup


# ── Extract surviving compatible synthons from filtered products ───────────────

def extract_surviving_synthon_inchikeys(rdmol_list: list, mel_openvs_inchikeys: set) -> set:
    """Parse full_synthon_id from each surviving product to find non-MEL synthon InChIKeys."""
    surviving = set()
    for mol_bin in rdmol_list:
        mol = Chem.Mol(mol_bin)
        full_id = mol.GetProp("full_synthon_id")
        parts = full_id.split("_____")
        for part in parts[1:]:
            if part.startswith("sssss"):
                continue
            if part not in mel_openvs_inchikeys:
                surviving.add(part)
    return surviving


# ── Write surviving synthons to SDF ───────────────────────────────────────────

def write_surviving_synthons_sdf(surviving_inchikeys: set, synthon_dict: dict, output_path: str):
    """Write surviving compatible synthons to SDF using the synthon RDKit mol dict."""
    writer = Chem.SDWriter(output_path)
    n_written = n_missing = n_failed = 0

    for ik in surviving_inchikeys:
        mol_binary = synthon_dict.get(ik)
        if mol_binary is None:
            n_missing += 1
            continue
        mol = Chem.Mol(mol_binary)
        mol.SetProp("_Name", ik)
        mol.SetProp("InChIKey", ik)
        try:
            writer.write(mol)
            n_written += 1
        except Exception as e:
            print(f"  WARNING: failed to write {ik}: {e}")
            n_failed += 1

    writer.close()
    print(f"      Raw SDF — written: {n_written}, missing: {n_missing}, failed: {n_failed}")
    print(f"      Saved → {output_path}")


# ── ICM APO conversion ────────────────────────────────────────────────────────

def find_si_and_anchor(mol):
    """Find [101Si] or [102Si] atom index and its heavy neighbor index (0-based)."""
    si_idx = next(
        (a.GetIdx() for a in mol.GetAtoms()
         if a.GetAtomicNum() == 14 and a.GetIsotope() in (101, 102)),
        None
    )
    if si_idx is None:
        return None, None
    anchor_idx = next(
        (nbr.GetIdx() for nbr in mol.GetAtomWithIdx(si_idx).GetNeighbors()
         if nbr.GetAtomicNum() != 1),
        None
    )
    return si_idx, anchor_idx


def extract_mol_block(raw_entry):
    """Pull the mol block up to and including M  END from a raw SDF entry string."""
    lines = raw_entry.strip().split("\n")
    mol_block_lines = []
    for line in lines:
        mol_block_lines.append(line)
        if line.strip() == "M  END":
            break
    return "\n".join(mol_block_lines) if mol_block_lines else None


def convert_synthon_raw(raw_mol_block, props_dict):
    """Remove [101/102Si] + its explicit Hs, remap indices, inject M APO, return SDF entry string."""
    mol = Chem.MolFromMolBlock(raw_mol_block, removeHs=False, sanitize=False)
    if mol is None:
        return None

    si_idx, anchor_idx_0based = find_si_and_anchor(mol)
    if si_idx is None or anchor_idx_0based is None:
        return None

    lines        = raw_mol_block.split("\n")
    counts_line  = lines[3]
    n_atoms      = int(counts_line[0:3])
    n_bonds      = int(counts_line[3:6])
    header_lines = lines[0:3]
    atom_lines   = lines[4: 4 + n_atoms]
    bond_lines   = lines[4 + n_atoms: 4 + n_atoms + n_bonds]
    footer_lines = lines[4 + n_atoms + n_bonds:]

    remove_set = {si_idx}
    for nbr in mol.GetAtomWithIdx(si_idx).GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            remove_set.add(nbr.GetIdx())

    kept_indices    = [i for i in range(n_atoms) if i not in remove_set]
    old_to_new      = {old: new + 1 for new, old in enumerate(kept_indices)}
    kept_atom_lines = [atom_lines[i] for i in kept_indices]

    kept_bond_lines = []
    for bline in bond_lines:
        if len(bline) < 6:
            continue
        a1 = int(bline[0:3]) - 1
        a2 = int(bline[3:6]) - 1
        if a1 in remove_set or a2 in remove_set:
            continue
        kept_bond_lines.append(f"{old_to_new[a1]:3d}{old_to_new[a2]:3d}" + bline[6:])

    new_counts   = f"{len(kept_atom_lines):3d}{len(kept_bond_lines):3d}" + counts_line[6:]
    apo_atom_num = old_to_new[anchor_idx_0based]

    clean_footer = []
    for line in footer_lines:
        if line.startswith("M  ISO"):
            continue
        if line.strip() == "M  END":
            clean_footer.append(f"M  APO  1 {apo_atom_num:3d}   1")
            clean_footer.append("M  END")
        else:
            clean_footer.append(line)

    mol_block  = "\n".join(header_lines + [new_counts] + kept_atom_lines + kept_bond_lines + clean_footer)
    prop_block = "".join(f">  <{k}>\n{v}\n\n" for k, v in props_dict.items() if not k.startswith("_"))
    return mol_block + "\n" + prop_block + "$$$$\n"


def write_icm_apo_sdf(input_path, output_path):
    """Convert all synthons in a raw SDF from [101/102Si] format to ICM M APO format."""
    with open(input_path, "r") as f:
        raw_text = f.read()
    raw_entries = [e for e in raw_text.split("$$$$") if e.strip()]

    suppl  = Chem.SDMolSupplier(str(input_path), removeHs=False)
    total  = written = skipped = 0

    with open(output_path, "w") as out_f:
        for raw_entry, mol in zip(raw_entries, suppl):
            total += 1
            if mol is None:
                print(f"  [WARN] RDKit could not parse entry {total}, skipping.")
                skipped += 1
                continue

            props         = {k: v for k, v in mol.GetPropsAsDict().items() if not k.startswith("_")}
            raw_mol_block = extract_mol_block(raw_entry)
            if raw_mol_block is None:
                print(f"  [WARN] Could not extract mol block for entry {total}, skipping.")
                skipped += 1
                continue

            result = convert_synthon_raw(raw_mol_block, props)
            if result is None:
                print(f"  [WARN] Conversion failed for entry {total}, skipping.")
                skipped += 1
                continue

            out_f.write(result)
            written += 1

    print(f"  ICM APO SDF — written: {written}/{total}, skipped: {skipped}")
    print(f"  Saved to: {output_path}")


# ── Per-MEL processing ────────────────────────────────────────────────────────

def process_single_mel(
    rank: int,
    icm_inchikey: str,
    full_synthon_ids: List[str],
    synth,
    synthon_dict: dict,
    instructions_output_dir: str,
    products_output_dir: str,
    surviving_synthons_output_dir: str,
    IncrementalPropertyFilter,
    SubstructureCatalogFilter,
):
    """Run the full pipeline for one MEL fragment: instructions → enumerate → filter → synthons."""

    # Build rank prefix for all output filenames
    mel_openvs_inchikeys = set()
    for fid in full_synthon_ids:
        ik = extract_inchikey_from_full_synthon_id(fid)
        if ik:
            mel_openvs_inchikeys.add(ik)
    openvs_ik_str = "_".join(sorted(mel_openvs_inchikeys))
    rank_prefix = f"Rank{rank}_ICMInChiKey_{icm_inchikey}_OpenVSInChiKey_{openvs_ik_str}"

    print(f"\n{'='*70}")
    print(f"  Rank{rank} | ICM: {icm_inchikey} | OpenVS: {openvs_ik_str}")
    print(f"{'='*70}")

    # Generate instruction rules
    print(f"  [1/3] Generating instruction rules...")
    lotot = generate_instruction_rules_for_full_synthon_id_list(synth, full_synthon_ids)
    instr_path = save_instruction_rules(lotot, rank_prefix, instructions_output_dir)
    print(f"        {len(lotot)} instructions → {instr_path}")

    # Enumerate and filter
    print(f"  [2/3] Enumerating and filtering full ligands...")
    inchi_list, rdmol_list = enumerate_and_filter(
        synth, lotot, rank_prefix, products_output_dir,
        IncrementalPropertyFilter, SubstructureCatalogFilter,
    )

    # Extract and write surviving synthons
    print(f"  [3/3] Extracting surviving compatible synthons...")
    if len(rdmol_list) == 0:
        print(f"        WARNING: 0 products survived filtering — skipping synthon SDF.")
        return

    surviving_inchikeys = extract_surviving_synthon_inchikeys(rdmol_list, mel_openvs_inchikeys)
    print(f"        {len(surviving_inchikeys)} unique surviving synthons found.")

    os.makedirs(surviving_synthons_output_dir, exist_ok=True)

    raw_sdf_path = os.path.join(surviving_synthons_output_dir, f"{rank_prefix}_surviving_synthons_raw.sdf")
    write_surviving_synthons_sdf(surviving_inchikeys, synthon_dict, raw_sdf_path)

    apo_sdf_path = os.path.join(surviving_synthons_output_dir, f"{rank_prefix}_surviving_synthons_ICMReady_APO.sdf")
    print(f"        Converting to ICM M APO format...")
    write_icm_apo_sdf(raw_sdf_path, apo_sdf_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For the Top-N ranked MEL fragments in a mapping CSV: "
            "generate enumeration instruction rules, enumerate and filter full ligands, "
            "then extract compatible synthons whose full compounds survived all filters."
        )
    )
    parser.add_argument(
        "--mapping_csv", required=True,
        help=(
            "CSV mapping ICM InChIKeys (column 'Name') to openvs_full_synthon_id. "
            "Example: Top20_2Comp_MEL_Frags_With_VS_OpenVS_Mapping.csv"
        )
    )
    parser.add_argument(
        "--score_col", required=True,
        help="Column in mapping_csv to rank MEL fragments by (ascending, lower = better)."
    )
    parser.add_argument(
        "--top_n", type=int, required=True,
        help="Number of top-ranked MELs to process (1-based, e.g. 10 = Rank1–Rank10)."
    )
    parser.add_argument(
        "--instructions_output_dir", required=True,
        help="Directory to save generated instruction rules PKL files."
    )
    parser.add_argument(
        "--products_output_dir", required=True,
        help="Directory to save enumerated and filtered full ligand products PKL files."
    )
    parser.add_argument(
        "--surviving_synthons_output_dir", required=True,
        help="Directory to save raw and ICM APO SDF files of surviving compatible synthons."
    )
    parser.add_argument(
        "--synthon_dict_path", required=True,
        help=(
            "Path to Synthon_Dict_InchiFull_RdMol.pkl. "
            "Example: /home/yourname/OpenVsynthes/MEL-Enumeration/Synthon_Dict_InchiFull_RdMol.pkl"
        )
    )
    parser.add_argument(
        "--openvs_repo",
        default="/home/wenjinl/Desktop/OpenVsynthes",
        help=(
            "Path to the root of the OpenVsynthes repository. "
            "Added to sys.path so the Synthesizer and filters can be imported."
        )
    )
    parser.add_argument(
        "--openvs_data_dir",
        default="/home/wenjinl/Desktop/OpenVsynthes/MEL-Enumeration",
        help=(
            "Path to the OpenVS MEL-Enumeration data directory. "
            "Contains reaction/synthon dictionaries used by the Synthesizer."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sys.path.append(args.openvs_repo)
    from OpenVsynthes.Synthesizer.Synthesizer import Synthesizer
    from OpenVsynthes.Filter.IncrementalProperty import IncrementalPropertyFilter
    from OpenVsynthes.Filter.SubstructureCatalog import SubstructureCatalogFilter

    # ── Load CSV and assign ranks on the FULL table before any dropping ────────
    print(f"[INIT] Loading mapping CSV: {args.mapping_csv}")
    df = pd.read_csv(args.mapping_csv)

    # Sort full table and assign rank based on position in full ICM docking results
    df = df.sort_values(by=args.score_col, ascending=True).reset_index(drop=True)
    df['_icm_rank'] = df.index + 1

    # Group by ICM InChIKey: take best rank (lowest number), collect all full_synthon_ids
    grouped = df.groupby('Name').agg(
        best_rank=('_icm_rank', 'min'),
        best_score=(args.score_col, 'min'),
        full_synthon_ids=('openvs_full_synthon_id', lambda x: list(x.dropna().unique()))
    ).reset_index()

    # Drop MELs with no OpenVS mapping after grouping
    grouped = grouped[grouped['full_synthon_ids'].map(len) > 0]

    # Sort by best_rank to preserve full-table ordering
    grouped = grouped.sort_values(by='best_rank', ascending=True).reset_index(drop=True)

    # Take top_n and report
    grouped = grouped.head(args.top_n)
    print(f"[INIT] Processing Top {len(grouped)} MEL fragments ranked by full ICM docking table")
    print(grouped[['best_rank', 'Name', 'best_score', 'full_synthon_ids']].to_string())

    # Load Synthesizer once
    print(f"\n[INIT] Loading Synthesizer from: {args.openvs_data_dir}")
    synth = Synthesizer(DIR_DataFormatted=args.openvs_data_dir)

    # Load synthon dict once
    print(f"[INIT] Loading synthon dict from: {args.synthon_dict_path}")
    with open(args.synthon_dict_path, "rb") as f:
        synthon_dict = pickle.load(f)
    print(f"[INIT] Synthon dict loaded: {len(synthon_dict)} entries")

    # Process each MEL fragment using its true ICM rank
    for _, row in grouped.iterrows():
        rank   = int(row['best_rank'])
        icm_ik = str(row['Name']).replace('-', '_')
        fids   = row['full_synthon_ids']

        if not fids:
            print(f"\n[SKIP] Rank{rank} {icm_ik} — no openvs_full_synthon_id found.")
            continue

        try:
            process_single_mel(
                rank=rank,
                icm_inchikey=icm_ik,
                full_synthon_ids=fids,
                synth=synth,
                synthon_dict=synthon_dict,
                instructions_output_dir=args.instructions_output_dir,
                products_output_dir=args.products_output_dir,
                surviving_synthons_output_dir=args.surviving_synthons_output_dir,
                IncrementalPropertyFilter=IncrementalPropertyFilter,
                SubstructureCatalogFilter=SubstructureCatalogFilter,
            )
        except Exception as e:
            print(f"\n[ERROR] Rank{rank} {icm_ik} failed: {e}")
            continue

    print(f"\n[DONE] Processed {len(grouped)} MEL fragments.")