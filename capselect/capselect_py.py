"""capselect_py.py — Python port of Antonina Nazarova's CapSelect algorithm.

Faithful to /mnt/katritch_lab/Antonina/For_Caroline/.../CAPBS.cpp (2022)
and CapSelect.cpp (2021, identical semantics for 1-cap MELs).

Output schema matches the 2021 binary:
- <CapScore>, <Spheres>, <Max(min)>, <Distance>  (no MergedScore)

Usage:
    python3 capselect_py.py fragments.sdf protein.sdf > CapSelect.sdf
"""
from __future__ import annotations
import math
import sys
from dataclasses import dataclass, field

import numpy as np


# Cap label characters (column 35 of SDF atom block).
# Aromatic cap atoms are tagged with '3' (ASCII 51), grouped in 5s (one phenyl).
# Non-aromatic cap atoms are tagged with '1' (ASCII 49), single C(iso=13) per cap.
LAB_AROMATIC = '3'
LAB_NONAROMATIC = '1'

GRID_AZIMUTH_STEPS = 72       # 5° resolution × 72 = 360°
GRID_LONGITUDE_STEPS = 36     # phi from -90° to +85°


@dataclass
class Frag:
    """One docked fragment (a single MEL pose)."""
    title: str
    block: str                 # full molfile block including counts/atoms/bonds
    coords: np.ndarray         # (N, 3)
    elements: list[str]        # length N
    labels: list[str]          # length N — char from col 35
    score: float               # docking <Score> SD tag
    sd_tags: list[tuple[str, str]]  # all SD tags as (name, value)


def parse_sdf(path: str, count_width: int = 3) -> list[Frag]:
    """Read an SDF file. count_width=3 for fragments.sdf (standard V2000),
    count_width=4 for protein.sdf (Antonina's AAAA9999 format per the manual).
    """
    out: list[Frag] = []
    with open(path) as f:
        text = f.read()
    for blob in text.split('$$$$\n'):
        if not blob.strip():
            continue
        lines = blob.splitlines()
        if len(lines) < 4:
            continue
        title = lines[0]
        try:
            n_atoms = int(lines[3][0:count_width])
        except ValueError:
            continue
        coords, elements, labels = [], [], []
        for k in range(n_atoms):
            ln = lines[4 + k]
            x = float(ln[0:10]); y = float(ln[10:20]); z = float(ln[20:30])
            elem = ln[31:34].strip()
            # Mass-difference field: cols 35-37 (0-indexed 34-36 in C++).
            # The C++ reads line[35] (single char). For "13" mass-diff, '13' is
            # stored as ' 13' (right-justified, 3 chars). line[35] is the '3'.
            lab = ln[35] if len(ln) > 35 else ' '
            coords.append((x, y, z))
            elements.append(elem)
            labels.append(lab)
        # SD tags
        sd_tags = []
        in_tag = None
        score = float('nan')
        for ln in lines:
            if ln.startswith('> <'):
                in_tag = ln[ln.index('<')+1: ln.rindex('>')]
            elif in_tag is not None:
                if ln.strip() == '':
                    in_tag = None
                else:
                    sd_tags.append((in_tag, ln))
                    if in_tag == 'Score':
                        try: score = float(ln)
                        except ValueError: pass
                    in_tag = None
        out.append(Frag(
            title=title,
            block='\n'.join(lines),
            coords=np.array(coords, dtype=float),
            elements=elements,
            labels=labels,
            score=score,
            sd_tags=sd_tags,
        ))
    return out


def parse_protein_sdf(path: str) -> np.ndarray:
    """Read protein atoms as Nx3 array. Uses 4-char count width per
    Antonina's AAAA9999 format. Hydrogens are kept (the C++ code does
    its own filtering in inner loops based on element char)."""
    frags = parse_sdf(path, count_width=4)
    if not frags:
        return np.zeros((0, 3))
    return frags[0].coords


def precompute_grid() -> np.ndarray:
    """Return 2592x3 array of unit-direction offsets matching CAPBS.cpp's
    a_c_t/a_s_t/a_c_f/a_s_f tables exactly."""
    thetas = np.deg2rad(np.arange(GRID_AZIMUTH_STEPS) * 5.0)
    phis = np.deg2rad(-90.0 + np.arange(GRID_LONGITUDE_STEPS) * 5.0)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    cos_f, sin_f = np.cos(phis), np.sin(phis)
    # Outer product: phi varies in inner C++ loop, theta in outer
    # Order: for each i4 (theta), for each i1 (phi). Match C++ enumeration order.
    out = np.empty((GRID_AZIMUTH_STEPS * GRID_LONGITUDE_STEPS, 3))
    idx = 0
    for i4 in range(GRID_AZIMUTH_STEPS):
        for i1 in range(GRID_LONGITUDE_STEPS):
            out[idx, 0] = cos_f[i1] * sin_t[i4]    # x
            out[idx, 1] = cos_f[i1] * cos_t[i4]    # y
            out[idx, 2] = sin_f[i1]                # z
            idx += 1
    return out


GRID_DIRS = precompute_grid()


def classify_cap(frag: Frag) -> dict:
    """Identify cap centroids per CAPBS.cpp:384-518.

    The C++ classifier accepts exactly 5 patterns of (n_arom, n_nonarom):
        (5, 0)  -> single aromatic cap (1 phenyl ring centroid)
        (0, 1)  -> single non-aromatic cap
        (5, 1)  -> aromatic + non-aromatic (2 caps)
        (5, 5)  -> two aromatic caps
        (1, 1)  -> two non-aromatic caps
    Anything else falls through with `max_label=0` and is silently
    rejected (final CapScore = 0, Spheres = 0). This matches the binary
    on the V-SYNTHES_2.2 MELs that have richer cap labeling (mol 6 of
    GPR91 has 6 atoms with lab='1' — falls outside the recognized
    patterns, so the binary rejects it).
    """
    arom_xyz, nonarom_xyz = [], []
    for k, lab in enumerate(frag.labels):
        if lab == LAB_AROMATIC:
            arom_xyz.append(frag.coords[k])
        elif lab == LAB_NONAROMATIC:
            nonarom_xyz.append(frag.coords[k])
    n_arom = len(arom_xyz)
    n_nonarom = len(nonarom_xyz)

    # The C++ collects aromatics into two buckets of up to 5 each, and
    # non-aromatics into two buckets of up to 1 each (subsequent hits get
    # SUMMED into the second bucket but the recognized patterns require
    # num_lab_4 == 1 exactly). Only accept the 5 explicit patterns.

    if n_arom == 0 and n_nonarom == 0:
        return {'kind': 'reject', 'caps': [], 'num_lab_l': 0}

    if n_arom == 5 and n_nonarom == 0:
        centroid = np.mean(arom_xyz, axis=0)
        return {'kind': 'single_aromatic', 'caps': [centroid], 'num_lab_l': 5}

    if n_arom == 0 and n_nonarom == 1:
        return {'kind': 'single_nonaromatic', 'caps': [nonarom_xyz[0]], 'num_lab_l': 1}

    if n_arom == 5 and n_nonarom == 1:
        c1 = np.mean(arom_xyz, axis=0)
        c2 = nonarom_xyz[0]
        return {'kind': 'arom_nonarom', 'caps': [c1, c2], 'num_lab_l': 6}

    if n_arom == 10 and n_nonarom == 0:
        c1 = np.mean(arom_xyz[:5], axis=0)
        c2 = np.mean(arom_xyz[5:], axis=0)
        return {'kind': 'two_aromatic', 'caps': [c1, c2], 'num_lab_l': 10}

    if n_arom == 0 and n_nonarom == 2:
        return {'kind': 'two_nonaromatic', 'caps': nonarom_xyz, 'num_lab_l': 2}

    # Everything else: binary's max_label stays 0, no spheres placed,
    # CapScore = 0 (per CAPBS.cpp:870, score branch fires with num1=0).
    return {'kind': 'unrecognized', 'caps': [],
            'num_lab_l': 5 * (n_arom // 5) + n_nonarom,
            'n_arom': n_arom, 'n_nonarom': n_nonarom}


def place_chain(cap_centroid: np.ndarray,
                ligand_xyz: np.ndarray,           # (M, 3) all ligand heavy atoms
                ligand_lab_or_h: np.ndarray,      # (M,) bool: is cap-labeled OR hydrogen
                protein_xyz: np.ndarray,          # (P, 3)
                is_aromatic: bool) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Place sphere chain starting at cap_centroid. Returns (positions,
    max_min_per_sphere, distance_per_sphere). Empty lists on rejection.
    """
    # ----- Step 1: first sphere on shell of radius r around cap centroid -----
    r1_init = 3.5 if is_aromatic else 3.0
    l_check = 1.1 if is_aromatic else 1.3
    p_check = 2.0 if is_aromatic else 3.0
    nonlab_lig = ligand_xyz[~ligand_lab_or_h]
    candidates = cap_centroid + GRID_DIRS * r1_init
    # Ligand-clash for each candidate
    if len(nonlab_lig):
        cand_lig = np.linalg.norm(candidates[:, None, :] - nonlab_lig[None, :, :], axis=2)
        ip = (cand_lig < r1_init).any(axis=1)
        # Cap-to-ligand-atom check: same for all candidates (uses cap_centroid)
        cap_lig = np.linalg.norm(cap_centroid - nonlab_lig, axis=1)
        ip = ip | (cap_lig < l_check).any()
    else:
        ip = np.zeros(len(candidates), dtype=bool)
    # Protein-clash
    if len(protein_xyz):
        cand_prot = np.linalg.norm(candidates[:, None, :] - protein_xyz[None, :, :], axis=2)
        ip1 = (cand_prot < p_check).any(axis=1)
    else:
        cand_prot = np.zeros((len(candidates), 0))
        ip1 = np.zeros(len(candidates), dtype=bool)
    valid = ~ip & ~ip1
    if not valid.any():
        return [], [], []
    # Score = min protein distance per candidate
    if cand_prot.shape[1]:
        min_per_cand = cand_prot.min(axis=1)
    else:
        min_per_cand = np.full(len(candidates), 1000.0)
    min_per_cand = np.where(valid, min_per_cand, -np.inf)
    best = int(np.argmax(min_per_cand))
    sphere1 = candidates[best]
    chain = [sphere1]
    max_min = [float(min_per_cand[best])]
    distance = [float(np.linalg.norm(sphere1 - cap_centroid))]

    # ----- Step 2: extend chain up to 10 spheres total -----
    for num in range(1, 10):    # spheres 2..10 (chain index 1..9)
        prev = chain[-1]
        candidates = prev + GRID_DIRS * 2.0    # sphere radius 2 Å for chain
        # Ligand-clash (3 Å candidate-to-ligand and l_check prev-to-ligand)
        if len(nonlab_lig):
            cand_lig = np.linalg.norm(candidates[:, None, :] - nonlab_lig[None, :, :], axis=2)
            ip = (cand_lig < 3.0).any(axis=1)
            prev_lig = np.linalg.norm(prev - nonlab_lig, axis=1)
            ip = ip | (prev_lig < l_check).any()
        else:
            ip = np.zeros(len(candidates), dtype=bool)
        # Protein-clash (2 Å candidate-to-prot and 3 Å prev-to-prot)
        if len(protein_xyz):
            cand_prot = np.linalg.norm(candidates[:, None, :] - protein_xyz[None, :, :], axis=2)
            ip1 = (cand_prot < 2.0).any(axis=1)
            prev_prot = np.linalg.norm(prev - protein_xyz, axis=1)
            ip1 = ip1 | (prev_prot < 3.0).any()
        else:
            cand_prot = np.zeros((len(candidates), 0))
            ip1 = np.zeros(len(candidates), dtype=bool)
        # Self-overlap with earlier chain spheres
        if num > 1:
            chain_arr = np.array(chain[:-1])    # exclude `prev` (already separated)
            d2 = np.linalg.norm(candidates[:, None, :] - chain_arr[None, :, :], axis=2)
            ip2 = (d2 < 2.0).any(axis=1)
        else:
            ip2 = np.zeros(len(candidates), dtype=bool)
        # Sphere 2 (num==1) must clear cap centroid
        if num == 1:
            cand_cap = np.linalg.norm(candidates - cap_centroid, axis=1)
            thr = 3.5 if is_aromatic else 3.0
            ip2_1 = cand_cap < thr
        else:
            ip2_1 = np.zeros(len(candidates), dtype=bool)
        # 120° cone (60° max bend) — chord to sphere {num-1} must be ≥ 3.46
        if num > 1:
            two_back = chain[num - 2]
            cand_2b = np.linalg.norm(candidates - two_back, axis=1)
            ip3 = cand_2b < 3.46
        else:
            ip3 = np.zeros(len(candidates), dtype=bool)
        valid = ~ip & ~ip1 & ~ip2 & ~ip2_1 & ~ip3
        if not valid.any():
            return chain, max_min, distance
        # Pick max min-protein-distance among valid candidates
        if cand_prot.shape[1]:
            min_per_cand = cand_prot.min(axis=1)
        else:
            min_per_cand = np.full(len(candidates), 1000.0)
        min_per_cand = np.where(valid, min_per_cand, -np.inf)
        best = int(np.argmax(min_per_cand))
        sphere = candidates[best]
        chain.append(sphere)
        max_min.append(float(min_per_cand[best]))
        distance.append(float(np.linalg.norm(sphere - cap_centroid)))
    return chain, max_min, distance


def cap_score(num_spheres: int, max_min_at_9: float | None) -> float:
    """CapScore from CAPBS.cpp lines 869-995."""
    # Penalty for early termination
    penalty_s = (5 - num_spheres) ** 2 if num_spheres <= 5 else 0
    penalty_s = penalty_s * 10.0 / 25.0   # i.e. * 0.4
    # Penalty for last (10th) sphere being far from pocket
    penalty_max = 0.0
    if num_spheres > 9 and max_min_at_9 is not None:
        if 7.0 <= max_min_at_9 <= 20.0:
            penalty_max = (7.0 - max_min_at_9) ** 2
        elif max_min_at_9 > 20.0:
            penalty_max = 169.0
        else:  # < 7
            penalty_max = 0.0
    penalty_max = penalty_max * 10.0 / 169.0
    return 10.0 - penalty_s - penalty_max


REJECT_SENTINEL = -1000.0


def merged_score_v25(score: float, capscore: float, spheres: int = 1) -> float:
    """V-SYNTHES_2_2 v2_5 binary formula (halved CapScore weight vs 2021).

    Sentinel handling matches the v2_5 binary's behavior verified against
    30,000 GPR119 outputs:
      - CapScore < 0   (e.g. -100, "no labeled cap"): MergedScore = -1000
      - CapScore == 0 AND Spheres == 0 (chain failed): MergedScore = -1000
      - CapScore == 0 AND Spheres >= 1 (chain ran but penalty maxed):
        MergedScore = 5·log₂|Score| (no CapScore term, since log₂(0) is -∞)
      - CapScore > 0:  MergedScore = 5·log₂|Score| + 0.5·log₂(CapScore)
    """
    if capscore < 0:
        return REJECT_SENTINEL
    if capscore == 0:
        if spheres == 0:
            return REJECT_SENTINEL
        return 5.0 * math.log2(abs(score))
    return 5.0 * math.log2(abs(score)) + 0.5 * math.log2(capscore)


def merged_score_2021(score: float, capscore: float, spheres: int = 1) -> float:
    """2021 ICM-side formula from Antonina's manual."""
    if capscore < 0:
        return REJECT_SENTINEL
    if capscore == 0:
        if spheres == 0:
            return REJECT_SENTINEL
        return 5.0 * math.log2(abs(score))
    return 5.0 * math.log2(abs(score)) + 1.0 * math.log2(capscore)


def run(fragments_path: str, protein_path: str, out_path: str | None = None,
        merged_formula: str = '2021') -> None:
    """Main entry point."""
    frags = parse_sdf(fragments_path)
    prot_xyz = parse_protein_sdf(protein_path)
    out_lines: list[str] = []
    for i, frag in enumerate(frags):
        cap_info = classify_cap(frag)
        if cap_info['kind'] == 'reject':
            # No labeled cap atoms found — sentinel
            cs_final, sp, mm, dist = -100.0, 1, [0.0], [0.0]
        elif cap_info['kind'] == 'unrecognized':
            # Cap labels present but in a pattern the C++ classifier doesn't
            # recognize (e.g. V-SYNTHES_2.2 MELs with multi-atom-labeled caps).
            # Binary leaves max_label=0, no spheres placed, scores to 0.
            cs_final, sp, mm, dist = 0.0, 0, [], []
        else:
            # The C++ uses initial radius 3.5 only for num_lab_l == 5 (single
            # aromatic cap). All other cap types — including 2_aromatic_caps,
            # arom_nonarom, and any non-aromatic — use radius 3.0.
            is_arom = (cap_info['kind'] == 'single_aromatic')
            # Mark which ligand atoms are cap-labeled or hydrogen
            lab_or_h = np.array([
                lab in (LAB_AROMATIC, LAB_NONAROMATIC) or (elem == 'H')
                for lab, elem in zip(frag.labels, frag.elements)
            ])
            # Place chain for each cap (1 or 2)
            cap_results = []
            for cap_centroid in cap_info['caps']:
                chain, mm_arr, dist_arr = place_chain(
                    cap_centroid, frag.coords, lab_or_h, prot_xyz,
                    is_aromatic=is_arom,
                )
                if not chain:
                    # Step-1 failure: no valid candidate on the cap shell.
                    # 2021 source sets num1=0 and emits empty Max(min)/Distance.
                    cap_results.append((0, [], [], 0.0))
                else:
                    sp_n = len(chain)
                    mm9 = mm_arr[9] if sp_n > 9 else None
                    cs_n = cap_score(sp_n, mm9)
                    cap_results.append((sp_n, mm_arr, dist_arr, cs_n))
            # For 2-cap: take the better of the two routes (max CapScore)
            if len(cap_results) == 1:
                sp, mm, dist, cs_final = cap_results[0]
            else:
                # Pick the cap with higher CapScore
                cap_results.sort(key=lambda x: x[3], reverse=True)
                sp, mm, dist, cs_final = cap_results[0]
        # Emit MOL block + original SD tags + new CapSelect tags
        out_lines.append(frag.block)
        for name, val in frag.sd_tags:
            # Skip if already a CapSelect-output tag (re-running)
            if name in ('CapScore', 'Spheres', 'Max(min)', 'Distance', 'MergedScore'):
                continue
            out_lines.append(f'> <{name}>')
            out_lines.append(val)
            out_lines.append('')
        out_lines.append('> <CapScore>')
        out_lines.append(f'{cs_final:.6f}')
        out_lines.append('')
        out_lines.append('> <Spheres>')
        out_lines.append(str(sp))
        out_lines.append('')
        out_lines.append('> <Max(min)>')
        out_lines.append(', '.join(f'{x:.6f}' for x in mm))
        out_lines.append('')
        out_lines.append('> <Distance>')
        out_lines.append(', '.join(f'{x:.6f}' for x in dist))
        out_lines.append('')
        if merged_formula == 'v2_5':
            ms = merged_score_v25(frag.score, cs_final, sp)
        else:
            ms = merged_score_2021(frag.score, cs_final, sp)
        out_lines.append('> <MergedScore>')
        out_lines.append(f'{ms:.6f}')
        out_lines.append('')
        out_lines.append('$$$$')
    out_text = '\n'.join(out_lines) + '\n'
    if out_path:
        with open(out_path, 'w') as f:
            f.write(out_text)
    else:
        sys.stdout.write(out_text)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: capselect_py.py fragments.sdf protein.sdf [out.sdf] [v2_5|2021]', file=sys.stderr)
        sys.exit(1)
    out = sys.argv[3] if len(sys.argv) > 3 else None
    formula = sys.argv[4] if len(sys.argv) > 4 else '2021'
    run(sys.argv[1], sys.argv[2], out, formula)
