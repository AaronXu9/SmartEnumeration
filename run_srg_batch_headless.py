#!/usr/bin/env python3
"""Headless SRG batch runner.

Auto-selects ICM's NoGraphics binary (`icmng`) on Linux — the same pattern
used by USC CARC's V-SYNTHES pipeline (see
/mnt/katritch_lab2/VSYNTHES_2_2__012024/.../sbatch_epyc_template.sbatch).
icmng runs scripts with no DISPLAY/XAUTHORITY/Xvfb dependency; every
_ligedit_bg subprocess it spawns inherits the binary via ICM's `macro`
built-in, so the no-X property propagates down the entire subprocess tree.

On macOS no icmng binary exists (Mac ICM is already headless-capable via
`icm64 -g`), so this driver passes through to `run_srg_batch.main()` with
the default ICM_BIN.

Usage is identical to run_srg_batch.py:
    python3 run_srg_batch_headless.py --dry-run --only-row 2
    python3 run_srg_batch_headless.py --only-row 2 --synthon-path combiDock_R1.sdf
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import run_srg_batch
from paths import ICM_BIN


def resolve_headless_icm(icm_bin: Path) -> Path:
    """Return the icmng sibling of icm_bin on Linux; icm_bin unchanged on Mac.

    Raises FileNotFoundError on Linux if the sibling is missing — better to
    fail loudly than silently fall back to the display-dependent icm64 and
    segfault inside a cluster job.
    """
    if platform.system() == "Darwin":
        return icm_bin
    candidate = icm_bin.parent / "icmng"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Expected ICM NoGraphics binary next to icm_bin: {candidate}. "
            "Install ICM's 'icmng' or symlink it alongside icm64."
        )
    return candidate


HEADLESS_TEMPLATE = (
    Path(__file__).resolve().parent
    / "run_ICM_ScreenReplacement_SingleMEL_NoGUI_Parallel.icm"
)


def main() -> int:
    if platform.system() != "Darwin":
        if not HEADLESS_TEMPLATE.is_file():
            raise FileNotFoundError(
                f"Expected headless ICM template: {HEADLESS_TEMPLATE}. "
                "icmng does not support the GUI `openFile` calls in the "
                "default template; this variant uses `read object` / "
                "`read map` / `read table mol` instead."
            )
        if "--template" not in sys.argv:
            sys.argv.extend(["--template", "nogui_parallel"])
    if "--icm" not in sys.argv:
        sys.argv.extend(["--icm", str(resolve_headless_icm(ICM_BIN))])
    return run_srg_batch.main()


if __name__ == "__main__":
    sys.exit(main())
