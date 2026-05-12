"""Platform-aware paths for the Pocket-Informed Synton Selection pipeline.

Detects whether we're running on the local macOS laptop or the KatLab Linux
workstation (br-443), and exports the appropriate project root + ICM binary
path. Downstream scripts (run_srg_batch.py, tests) import from here so no
host-specific string lives elsewhere.

Override via env var `VSYNTHES_ENV=local_macos|katlab` for tests / CI.

Why a `results_<env>` suffix: the project dir is auto-synced between the two
hosts. If both hosts wrote to plain `results/` the sync would cause them to
step on each other. Per-env subtree keeps them isolated.
"""
from __future__ import annotations

import os
import platform
import socket
from pathlib import Path

_KATLAB_HOST_SUBSTRINGS = ("br-443", "katlab")
_CARC_HOST_SUBSTRINGS = (".hpc.usc.edu", "discovery.usc.edu")


def _detect_env() -> str:
    override = os.environ.get("VSYNTHES_ENV")
    if override:
        if override not in _CONFIGS:
            raise RuntimeError(
                f"VSYNTHES_ENV={override!r} is not a known env. "
                f"Known: {sorted(_CONFIGS)}"
            )
        return override
    system = platform.system()
    host = socket.gethostname().lower()
    if system == "Darwin":
        return "local_macos"
    if system == "Linux" and any(h in host for h in _KATLAB_HOST_SUBSTRINGS):
        return "katlab"
    if system == "Linux" and any(h in host for h in _CARC_HOST_SUBSTRINGS):
        return "carc"
    raise RuntimeError(
        f"Unsupported environment: system={system!r} host={host!r}. "
        "Set VSYNTHES_ENV=local_macos|katlab|carc to override."
    )


_CONFIGS: dict[str, dict] = {
    "local_macos": {
        "project_root": Path("/Users/aoxu/projects/anchnor_based_VSYNTHES"),
        "icm_bin":      Path("/Applications/MolsoftICM64.app/Contents/MacOS/icm64"),
        # Mac ICM under `-g` runs headlessly and routes `print` to stdout.
        "icm_flags":    ["-g"],
    },
    "katlab": {
        # Canonical workspace lives on the lab NAS (auto-syncs to CARC).
        # The old /home/aoxu/projects/anchnor_based_VSYNTHES location is now
        # a downstream code-only replica — the pipeline reads from here.
        "project_root": Path("/mnt/katritch_lab2/pocketInformedV-SYNTHES"),
        "icm_bin":      Path("/home/aoxu/icm-3.9-4/icm64"),
        # On Linux `-g` opens a real GUI window (when DISPLAY is set) and
        # sends `print` output to the GUI console — not stdout. `-s` (silent/
        # scriptable) keeps prints on stdout and still runs the full script.
        # Verified 2026-04-20 with DISPLAY=:1 XAUTHORITY=/run/user/.../gdm/Xauthority.
        "icm_flags":    ["-s"],
    },
    "carc": {
        # USC CARC SLURM cluster. Project root is the user's own dir under the
        # shared katritch_223 project space. ICM on CARC is `icmng` (headless,
        # no GUI, no DISPLAY). `-s` keeps prints on stdout for log capture.
        # openFile is GUI-only on icmng, so the default diskmaps template
        # won't run here — use --template headless or --template converge.
        #
        # `icmhome` must point at the icmng install dir so `call _startup`
        # can find its resources. Unlike the Mac/Linux icm64 installs, icmng
        # on CARC does not auto-detect $ICMHOME from the binary location,
        # so we export it explicitly at import time (see bottom of file).
        "project_root": Path("/project2/katritch_223/aoxu/pocketInformedV-SYNTHES"),
        "icm_bin":      Path("/project2/katritch_223/icm-3.9-4a/icmng"),
        "icm_flags":    ["-s"],
        "icmhome":      Path("/project2/katritch_223/icm-3.9-4a"),
    },
}


ENV_NAME     = _detect_env()
_cfg         = _CONFIGS[ENV_NAME]
PROJECT_ROOT = _cfg["project_root"]
ICM_BIN      = _cfg["icm_bin"]
ICM_FLAGS    = list(_cfg["icm_flags"])

# Per-target dataset switch. Each target dir contains its own
# pocket_maps/, mel_hits/, compatible_syntons/, and (optionally)
# enumerated_products/. Defaulting to the debug fixture preserves
# behavior of existing scripts; flip to a production target via
#   VSYNTHES_TARGET=CB2_5ZTY  python ...
TARGET       = os.environ.get("VSYNTHES_TARGET", "CB2_5ZTY_debug")
TARGET_ROOT  = PROJECT_ROOT / TARGET

# Per-target MEL filename. Debug uses the bootstrapped table; production
# targets use the collaborator's ICM-prepared Top-N hits SDF.
_MEL_FILENAME_BY_TARGET = {
    "CB2_5ZTY_debug": "final_table_edited.sdf",
    "CB2_5ZTY":       "CB2-5ZTY-Top1K-MEL-ICMPrepared.sdf",
}
_MEL_FILENAME = _MEL_FILENAME_BY_TARGET.get(TARGET, "final_table_edited.sdf")

MEL_SDF          = TARGET_ROOT / "mel_hits" / _MEL_FILENAME
APO_TSV          = TARGET_ROOT / "mel_hits" / "final_table_edited_apo_index.tsv"
SYNTHONS_DIR     = TARGET_ROOT / "compatible_syntons"
MAPS_DIR         = TARGET_ROOT / "pocket_maps"
ENUMERATED_DIR   = TARGET_ROOT / "enumerated_products"

TEMPLATE_ICM = PROJECT_ROOT / "run_srg_single_apo_export_diskmaps.icm"
TEMPLATE_ICM_HEADLESS = PROJECT_ROOT / "run_srg_single_apo_export_diskmaps_headless.icm"
TEMPLATE_CONVERGE_NOGUI = PROJECT_ROOT / "run_srg_single_apo_export_diskmaps_converge_noGUI.icm"
RESULTS_DIR  = PROJECT_ROOT / f"results_{ENV_NAME}"

if "icmhome" in _cfg:
    os.environ["ICMHOME"] = str(_cfg["icmhome"])


if __name__ == "__main__":
    for name in ("ENV_NAME", "PROJECT_ROOT", "ICM_BIN", "ICM_FLAGS",
                 "TARGET", "TARGET_ROOT",
                 "MEL_SDF", "APO_TSV", "SYNTHONS_DIR", "MAPS_DIR",
                 "ENUMERATED_DIR",
                 "TEMPLATE_ICM", "TEMPLATE_ICM_HEADLESS",
                 "TEMPLATE_CONVERGE_NOGUI", "RESULTS_DIR"):
        print(f"{name:14s} = {globals()[name]}")
