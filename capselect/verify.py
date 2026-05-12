"""Compare Python port output against the reference CapSelect.sdf.

Acceptance criteria:
- Spheres count: exact match
- CapScore:      |Δ| < 0.05 (1 part in 200)
- Max(min)[k]:   |Δ| < 0.5 Å for the first 4 spheres (early chain;
                 grid-search ties may diverge after that)
- Distance[k]:   |Δ| < 0.5 Å for the first 4 spheres
"""
import re
import sys


def extract(path):
    out = []
    for blob in open(path).read().split('$$$$'):
        if 'CapScore' not in blob:
            continue
        d = {}
        for tag in ('CapScore', 'Spheres', 'Max(min)', 'Distance', 'Score'):
            m = re.search(rf'> <{re.escape(tag)}>\s*\n\s*([^\n]+)', blob)
            if m:
                d[tag] = m.group(1).strip()
        out.append(d)
    return out


def parse_list(s):
    return [float(x.strip()) for x in s.split(',')]


def check(ref_path, py_path):
    ref = extract(ref_path)
    py = extract(py_path)
    if len(ref) != len(py):
        print(f'FAIL: molecule count differs (ref={len(ref)} py={len(py)})')
        return 1
    fails = 0
    for i, (r, p) in enumerate(zip(ref, py)):
        # Spheres exact
        if r['Spheres'] != p['Spheres']:
            print(f'mol {i+1}: SPHERES MISMATCH ref={r["Spheres"]} py={p["Spheres"]}')
            fails += 1
            continue
        # CapScore within 0.05
        cs_r, cs_p = float(r['CapScore']), float(p['CapScore'])
        cs_d = abs(cs_r - cs_p)
        # Early-chain Max(min) and Distance within 0.5
        mm_r, mm_p = parse_list(r['Max(min)']), parse_list(p['Max(min)'])
        ds_r, ds_p = parse_list(r['Distance']), parse_list(p['Distance'])
        early = min(4, len(mm_r), len(mm_p))
        mm_drift = max(abs(mm_r[k] - mm_p[k]) for k in range(early))
        ds_drift = max(abs(ds_r[k] - ds_p[k]) for k in range(early))
        ok = (cs_d < 0.05) and (mm_drift < 0.5) and (ds_drift < 0.5)
        marker = 'PASS' if ok else 'FAIL'
        print(f'mol {i+1}: {marker}  ΔCapScore={cs_d:.5f}  '
              f'ΔMaxMin(spheres 1-{early})_max={mm_drift:.4f}  '
              f'ΔDistance(spheres 1-{early})_max={ds_drift:.4f}')
        if not ok:
            fails += 1
    print()
    if fails == 0:
        print(f'OVERALL: PASS  ({len(ref)} molecules)')
        return 0
    print(f'OVERALL: FAIL  ({fails}/{len(ref)} molecules failed)')
    return 1


if __name__ == '__main__':
    ref = sys.argv[1] if len(sys.argv) > 1 else 'CapSelect.sdf'
    py = sys.argv[2] if len(sys.argv) > 2 else 'python_output.sdf'
    sys.exit(check(ref, py))
