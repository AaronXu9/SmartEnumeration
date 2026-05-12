#!/usr/bin/env python3
"""Per-MEL SRG driver for CARC SLURM array tasks. One task = one MEL row.

Reads MEL row from --row (normally `$SLURM_ARRAY_TASK_ID + 1`), resolves
the matching synthon SDF from SYNTHONS_DIR by inchikey, renders the ICM
template, invokes icmng, writes status.json for the outer scheduler loop.

Idempotent: if `results/MEL_<row>_<rank>/status.json` already shows a
success status (`ok*`), exits 0 without re-running — so submitting the
full array on top of a partial result is safe.

Usage:
  VSYNTHES_ENV=carc python3 run_one_mel.py --row 2 --template headless
  VSYNTHES_ENV=carc python3 run_one_mel.py --row 2 --template converge --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Scripts live at PROJECT_ROOT/scripts/; add PROJECT_ROOT to path so we can
# import srg_core and paths without installing the project.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from paths import (  # noqa: E402
    APO_TSV,
    ENV_NAME,
    ICM_BIN,
    MEL_SDF,
    PROJECT_ROOT,
    RESULTS_DIR,
    SYNTHONS_DIR,
)
from srg_core import (  # noqa: E402
    parse_apo_tsv,
    parse_mel_sdf,
    run_one_classic,
    scan_synthons,
    select_template_and_renderer,
)

# ok*, warn_nn_zero, dry_run all count as "don't re-run"; fail_* re-run.
_DONE_STATUSES = frozenset({"ok", "ok_dirty_exit", "ok_committed",
                            "ok_probe_only", "ok_aborted", "dry_run"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", type=int, required=True,
                    help="1-based MEL row index into final_table_edited.sdf. "
                         "In an sbatch array, pass `$((SLURM_ARRAY_TASK_ID + 1))` "
                         "so array task 0 maps to row 1.")
    # CARC defaults to headless; --template default would fail with
    # `openFile` under icmng. Keep explicit choice so KatLab can reuse this
    # same driver if needed.
    ap.add_argument("--template", choices=("default", "headless", "converge"),
                    default=None,
                    help="ICM template variant. Default: 'headless' on carc env, "
                         "'default' elsewhere. 'default' will fail on carc "
                         "(openFile requires a GUI).")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if status.json already shows success.")
    ap.add_argument("--dry-run", action="store_true",
                    help="render the .icm + status.json plan, don't invoke ICM.")
    ap.add_argument("--icm", default=str(ICM_BIN),
                    help=f"ICM executable path (default: {ICM_BIN})")
    args = ap.parse_args()

    if args.template is None:
        args.template = "headless" if ENV_NAME == "carc" else "default"
    if args.template == "default" and ENV_NAME == "carc":
        print("ERROR: --template default uses openFile which is GUI-only; "
              "icmng on CARC cannot run it. Use --template headless or converge.",
              file=sys.stderr)
        return 2

    template_path, renderer = select_template_and_renderer(args.template)
    template = template_path.read_text()

    entries = parse_mel_sdf(MEL_SDF)
    parse_apo_tsv(APO_TSV, entries)
    mel = next((e for e in entries if e.row == args.row), None)
    if mel is None:
        print(f"ERROR: MEL row {args.row} not found in {MEL_SDF} "
              f"(have {len(entries)} rows)", file=sys.stderr)
        return 2
    if not mel.apo_idx:
        print(f"SKIP: row {args.row} has no APO index in {APO_TSV}", file=sys.stderr)
        return 0

    synthons = scan_synthons(SYNTHONS_DIR)
    synth = synthons.get(mel.icm_inchikey)
    if synth is None:
        print(f"SKIP: row {args.row} inchikey {mel.icm_inchikey} has no "
              f"matching synthon SDF in {SYNTHONS_DIR}", file=sys.stderr)
        return 0

    out_dir = RESULTS_DIR / f"MEL_{mel.row}_{synth.rank_label}"
    status_path = out_dir / "status.json"

    if status_path.exists() and not args.force:
        try:
            prev = json.loads(status_path.read_text())
            if prev.get("status") in _DONE_STATUSES:
                print(f"SKIP: row {mel.row} already {prev.get('status')} "
                      f"(--force to re-run)")
                return 0
        except (json.JSONDecodeError, OSError):
            pass  # malformed status → proceed with run

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MEL row {mel.row} -> {out_dir.relative_to(PROJECT_ROOT)}  "
          f"template={args.template}  env={ENV_NAME}")

    t0 = time.time()
    result = run_one_classic(mel, synth, out_dir, template, args.icm,
                             args.dry_run, renderer=renderer)
    wall_elapsed = time.time() - t0

    status_payload = {
        "row": mel.row,
        "inchikey": mel.icm_inchikey,
        "rank": synth.rank_label,
        "apo_idx": mel.apo_idx,
        "template": args.template,
        "env": ENV_NAME,
        "status": result["status"],
        "exit_code": result.get("exit_code"),
        "n_records": result.get("n_records", result.get("final_n", 0)),
        "n_nonzero_nn": result.get("n_nonzero_nn"),
        "elapsed_s": result.get("elapsed_s", 0.0),
        "wall_elapsed_s": wall_elapsed,
    }
    status_path.write_text(json.dumps(status_payload, indent=2) + "\n")
    print(f"status: {result['status']}  records={status_payload['n_records']}  "
          f"wall={wall_elapsed:.1f}s")

    # Exit non-zero only for hard failures so SLURM's sacct marks the task
    # FAILED. `ok_dirty_exit` (ICM SIGABRT on quit but valid output) counts
    # as success for SLURM's purposes.
    if result["status"].startswith("ok") or result["status"] == "dry_run":
        return 0
    return result.get("exit_code", 1) or 1


if __name__ == "__main__":
    sys.exit(main())
