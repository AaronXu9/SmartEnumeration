"""Shared SRG helpers used by both the local batch driver and the CARC
per-MEL driver. This is a pure extraction from run_srg_batch.py — all
functions below retain their original semantics and callers.

Two entry points import this module:
  run_srg_batch.py      — local / KatLab sequential batch runner
  scripts/run_one_mel.py — CARC SLURM per-task driver (one MEL per array job)

Keeping them on a single source of truth for SDF parsing, template
rendering, and ICM invocation ensures the two hosts can't silently diverge.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paths import (
    MAPS_DIR,
    MEL_SDF,
    ICM_FLAGS,
)

Renderer = Callable[..., str]

SYNTHON_FILENAME_RE = re.compile(
    r"^(Rank\d+)_ICMInChiKey_(?P<icm>[A-Z]+_[A-Z]+_[A-Z])_"
    r"OpenVSInChiKey_[A-Z]+-[A-Z]+-[A-Z]_surviving_synthons_ICMReady_APO\.sdf$"
)


@dataclass
class MelEntry:
    row: int
    icm_inchikey: str
    apo_idx: str


@dataclass
class SynthonFile:
    path: Path
    rank_label: str
    icm_inchikey: str


def parse_mel_sdf(path: Path) -> list[MelEntry]:
    text = path.read_text()
    records = [r for r in text.split("$$$$") if r.strip()]
    entries: list[MelEntry] = []
    for idx, record in enumerate(records, start=1):
        lines = record.splitlines()
        name = ""
        for i, line in enumerate(lines):
            if line.strip() == "> <NAME>":
                name = lines[i + 1].strip() if i + 1 < len(lines) else ""
                break
        entries.append(MelEntry(row=idx, icm_inchikey=name, apo_idx=""))
    return entries


def parse_apo_tsv(path: Path, entries: list[MelEntry]) -> None:
    with path.open() as f:
        header = f.readline()
        assert header.startswith("entry_idx"), f"unexpected header: {header!r}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            row = int(parts[0])
            apo = parts[2]
            for e in entries:
                if e.row == row:
                    e.apo_idx = apo
                    break


def scan_synthons(dir_: Path) -> dict[str, SynthonFile]:
    out: dict[str, SynthonFile] = {}
    for p in sorted(dir_.iterdir()):
        if not p.is_file():
            continue
        m = SYNTHON_FILENAME_RE.match(p.name)
        if not m:
            continue
        out[m["icm"]] = SynthonFile(
            path=p, rank_label=m.group(1), icm_inchikey=m["icm"]
        )
    return out


def render_icm(template: str, mel_row: int, synthon_path: Path,
               out_dir: Path, synthon_table: str,
               out_sdf_name: str = "enumerated.sdf") -> str:
    """Substitute per-MEL and per-host parameters into the default template.

    The template ships with `/Users/aoxu/...` Mac paths as sentinels; the
    real paths come from paths.py (env-detected) so the same template works
    on macOS, KatLab Linux, and CARC.
    """
    subs = [
        ("i_mel_row       = 2",
         f"i_mel_row       = {mel_row}"),
        ('s_map_dir       = "/Users/aoxu/projects/anchnor_based_VSYNTHES/maps"',
         f's_map_dir       = "{MAPS_DIR}"'),
        ('s_edited_mel    = "/Users/aoxu/projects/anchnor_based_VSYNTHES/final_table_edited.sdf"',
         f's_edited_mel    = "{MEL_SDF}"'),
        ('s_synthon_sdf   = "/Users/aoxu/projects/anchnor_based_VSYNTHES/combiDock_R1.sdf"',
         f's_synthon_sdf   = "{synthon_path}"'),
        ('s_results_dir   = "/Users/aoxu/projects/anchnor_based_VSYNTHES/results"',
         f's_results_dir   = "{out_dir}"'),
        ('s_synthon_table = "combiDock_R1"',
         f's_synthon_table = "{synthon_table}"'),
        ('s_out_sdf = s_results_dir + "/MEL_" + String(i_mel_row) + "_" + s_synthon_table + "_enumerated_diskmaps.sdf"',
         f's_out_sdf = s_results_dir + "/{out_sdf_name}"'),
    ]
    out = template
    for old, new in subs:
        if old not in out:
            raise RuntimeError(
                f"template substitution failed; missing line:\n  {old}\n"
                "The template file may have drifted from what srg_core expects."
            )
        out = out.replace(old, new, 1)
    return out


def _replace_assignment(source: str, lhs: str, new_rhs: str) -> str:
    pattern = re.compile(
        rf'^(\s*{re.escape(lhs)}\s*=\s*).*$', re.MULTILINE)
    new_source, n = pattern.subn(lambda m: m.group(1) + new_rhs, source, count=1)
    if n == 0:
        raise RuntimeError(
            f"Assignment line not found for {lhs!r} — template "
            "may have drifted from what srg_core expects."
        )
    return new_source


def _rewrite_read_table_name(source: str, var_name: str, new_name: str) -> str:
    pattern = re.compile(
        rf'^(\s*read\s+table\s+mol\s+{re.escape(var_name)}\s+name\s*=\s*)"[^"]*"',
        re.MULTILINE)
    new_source, n = pattern.subn(rf'\g<1>"{new_name}"', source, count=1)
    if n == 0:
        raise RuntimeError(
            f"`read table mol {var_name} name=\"...\"` line not found in template."
        )
    return new_source


def render_icm_headless(template: str, mel_row: int, synthon_path: Path,
                        out_dir: Path, synthon_table: str,
                        out_sdf_name: str = "enumerated.sdf") -> str:
    out = render_icm(template, mel_row, synthon_path, out_dir, synthon_table,
                     out_sdf_name)
    out = _rewrite_read_table_name(out, "s_synthon_sdf", synthon_table)
    return out


def render_icm_converge(template: str, mel_row: int, synthon_path: Path,
                        out_dir: Path, synthon_table: str,
                        out_sdf_name: str = "enumerated.sdf") -> str:
    out = template
    out = _replace_assignment(out, "i_mel_row", str(mel_row))
    out = _replace_assignment(out, "s_map_dir", f'"{MAPS_DIR}"')
    out = _replace_assignment(out, "s_edited_mel", f'"{MEL_SDF}"')
    out = _replace_assignment(out, "s_synthon_sdf", f'"{synthon_path}"')
    out = _replace_assignment(out, "s_synthon_table", f'"{synthon_table}"')
    out = _replace_assignment(out, "s_results_dir", f'"{out_dir}"')
    out = _replace_assignment(out, "s_out_sdf",
                              f's_results_dir + "/{out_sdf_name}"')
    out = _rewrite_read_table_name(out, "s_synthon_sdf", synthon_table)
    return out


def count_sdf_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.startswith("$$$$"))


def check_nn_score(sdf_path: Path) -> tuple[int, int]:
    """Return (n_records, n_nonzero_RTCNN_Score). Post-run sanity check for
    the RTCNN-unbound bug: all-zero RTCNN_Score == receptor not bound during
    scoring. The field is `RTCNN_Score` in the exported SDF even though
    CLAUDE.md / the template's comments call it "NN_Score".
    """
    if not sdf_path.exists():
        return 0, 0
    n_records, n_nonzero = 0, 0
    expect_value = False
    with sdf_path.open() as f:
        for line in f:
            if line.startswith("$$$$"):
                n_records += 1
                continue
            if expect_value:
                try:
                    if float(line.strip()) != 0.0:
                        n_nonzero += 1
                except ValueError:
                    pass
                expect_value = False
                continue
            if line.strip() == "> <RTCNN_Score>":
                expect_value = True
    return n_records, n_nonzero


def invoke_icm(icm_bin: str, run_icm: Path, log_path: Path,
               cwd: Path) -> tuple[int, float]:
    """Run `icm_bin <ICM_FLAGS> run_icm` with combined stdout+stderr going to
    log_path. Returns (exit_code, elapsed_seconds). ICM_FLAGS come from
    paths.py (e.g. `-g` on Mac, `-s` on Linux/CARC)."""
    t0 = time.time()
    with log_path.open("wb") as log:
        proc = subprocess.run(
            [icm_bin, *ICM_FLAGS, str(run_icm)],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd),
        )
    return proc.returncode, time.time() - t0


def run_one_classic(mel: MelEntry, synth: SynthonFile, out_dir: Path,
                    template: str, icm_bin: str, dry_run: bool,
                    renderer: Renderer = render_icm) -> dict:
    """Single-pass flow: dock the entire synthon library. Returns manifest fields."""
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_link = out_dir / f"synthons_{synth.rank_label}.sdf"
    if synth_link.is_symlink() or synth_link.exists():
        synth_link.unlink()
    synth_link.symlink_to(synth.path)
    synthon_table = synth_link.stem

    run_icm = out_dir / "run.icm"
    run_icm.write_text(renderer(template, mel.row, synth_link, out_dir,
                                synthon_table))

    if dry_run:
        return {"status": "dry_run", "elapsed_s": 0.0, "final_n": 0}

    log_path = out_dir / "icm.log"
    exit_code, elapsed = invoke_icm(icm_bin, run_icm, log_path, out_dir)

    enum_sdf = out_dir / "enumerated.sdf"
    n_rec, n_nn = check_nn_score(enum_sdf)
    print(f"  [row {mel.row}] exit={exit_code}  elapsed={elapsed:.1f}s  "
          f"records={n_rec}  nonzero_NN={n_nn}")

    if n_rec == 0:
        status = "fail_empty" if exit_code == 0 else "fail_exit"
    elif n_nn == 0 and n_rec > 10:
        print(f"    WARN: all NN_Score=0 — suggests RTCNN unbound (see CLAUDE.md)")
        status = "warn_nn_zero"
    elif exit_code != 0:
        print(f"    WARN: ICM exit={exit_code} but {n_rec} records with {n_nn} "
              f"nonzero scores — likely benign Linux ICM shutdown crash")
        status = "ok_dirty_exit"
    else:
        status = "ok"
    return {"status": status, "elapsed_s": elapsed, "final_n": n_rec,
            "exit_code": exit_code, "n_records": n_rec, "n_nonzero_nn": n_nn}


def select_template_and_renderer(name: str) -> tuple[Path, Renderer]:
    """Resolve `--template` name to (template_path, renderer) using paths.py.
    Used by both run_srg_batch.py and run_one_mel.py so the two CLIs stay in
    sync on which template each alias points to."""
    from paths import (
        TEMPLATE_CONVERGE_NOGUI,
        TEMPLATE_ICM,
        TEMPLATE_ICM_HEADLESS,
    )
    if name == "converge":
        return TEMPLATE_CONVERGE_NOGUI, render_icm_converge
    if name == "headless":
        return TEMPLATE_ICM_HEADLESS, render_icm_headless
    if name == "default":
        return TEMPLATE_ICM, render_icm
    raise ValueError(f"unknown template alias: {name!r}")
