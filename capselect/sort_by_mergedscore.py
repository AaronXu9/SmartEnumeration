"""sort_by_mergedscore.py — Sort a CapSelect output SDF by MergedScore
descending. Equivalent to what icm_CapSelect_to_frags_for_enum.icm does
in the production V-SYNTHES_2 pipeline.

Usage:
    python3 sort_by_mergedscore.py CapSelect.sdf frags_for_enum.sdf [tsv_summary]
"""
from __future__ import annotations

import re
import sys


def split_molecules(text: str) -> list[str]:
    """Split SDF on $$$$ separator. Each entry includes a trailing $$$$.

    Returns molecule blocks each ending with the terminator line.
    """
    blocks = []
    parts = text.split('$$$$')
    for i in range(len(parts) - 1):
        blocks.append(parts[i].lstrip('\n') + '$$$$\n')
    return blocks


def get_tag(blob: str, name: str) -> str | None:
    m = re.search(rf'> <{re.escape(name)}>\s*\n\s*([^\n]+)', blob)
    return m.group(1).strip() if m else None


def main():
    if len(sys.argv) < 3:
        sys.exit('usage: sort_by_mergedscore.py in.sdf out.sdf [summary.tsv]')
    in_path, out_path = sys.argv[1], sys.argv[2]
    tsv_path = sys.argv[3] if len(sys.argv) > 3 else None

    text = open(in_path).read()
    mols = split_molecules(text)

    indexed = []
    for i, blob in enumerate(mols):
        ms = get_tag(blob, 'MergedScore')
        sc = get_tag(blob, 'Score')
        cs = get_tag(blob, 'CapScore')
        sp = get_tag(blob, 'Spheres')
        ms_v = float(ms) if ms is not None else float('-inf')
        indexed.append({
            'orig_idx': i + 1,
            'block': blob,
            'Score': sc,
            'CapScore': cs,
            'Spheres': sp,
            'MergedScore': ms,
            'ms_sort': ms_v,
        })

    indexed.sort(key=lambda d: d['ms_sort'], reverse=True)

    with open(out_path, 'w') as f:
        for d in indexed:
            f.write(d['block'])
    print(f'wrote {out_path} ({len(indexed)} molecules sorted by MergedScore desc)',
          file=sys.stderr)

    if tsv_path:
        with open(tsv_path, 'w') as f:
            f.write('rank\torig_idx\tScore\tCapScore\tSpheres\tMergedScore\n')
            for r, d in enumerate(indexed, 1):
                f.write(f'{r}\t{d["orig_idx"]}\t{d["Score"]}\t{d["CapScore"]}\t'
                        f'{d["Spheres"]}\t{d["MergedScore"]}\n')
        print(f'wrote {tsv_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
