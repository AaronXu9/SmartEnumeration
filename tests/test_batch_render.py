"""Tests for run_srg_batch.render_icm.

Verifies that per-host sentinels in the ICM template get rewritten to the
env-selected paths, and that no foreign-env paths leak through.

Each env case runs in a subprocess because paths.py caches ENV_NAME at
import (and run_srg_batch imports paths at module load).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _strip_comments(icm_source: str) -> str:
    return "\n".join(ln for ln in icm_source.splitlines()
                     if not ln.lstrip().startswith("#"))


def _render(env_name: str, mel_row: int = 2,
            synthon_path: str = "/SOMEWHERE/synth.sdf",
            out_dir: str = "/SOMEWHERE/out",
            synthon_table: str = "synth_table") -> subprocess.CompletedProcess:
    # Read the template from the actual checkout (REPO_ROOT), not from
    # paths.TEMPLATE_GUI_PARALLEL — the latter resolves to the KatLab path
    # under VSYNTHES_ENV=katlab and doesn't exist when this test runs on a Mac.
    template_path = REPO_ROOT / "run_ICM_ScreenReplacement_SingleMEL_GUI_Parallel.icm"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from pathlib import Path
        import run_srg_batch
        template = Path({str(template_path)!r}).read_text()
        out = run_srg_batch.render_icm(
            template,
            {mel_row},
            Path({synthon_path!r}),
            Path({out_dir!r}),
            {synthon_table!r},
        )
        sys.stdout.write(out)
    """)
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "VSYNTHES_ENV": env_name},
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


class RenderIcmTests(unittest.TestCase):
    def test_katlab_render_rewrites_all_paths(self) -> None:
        r = _render("katlab")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        out = r.stdout
        # Rewritten host-specific paths:
        self.assertIn('s_map_dir       = "/home/aoxu/projects/anchnor_based_VSYNTHES/maps"', out)
        self.assertIn('s_edited_mel    = "/home/aoxu/projects/anchnor_based_VSYNTHES/final_table_edited.sdf"', out)
        # Per-run substitutions:
        self.assertIn('s_synthon_sdf   = "/SOMEWHERE/synth.sdf"', out)
        self.assertIn('s_results_dir   = "/SOMEWHERE/out"', out)
        self.assertIn('s_synthon_table = "synth_table"', out)
        self.assertIn('i_mel_row       = 2', out)
        # No Mac paths should leak through executable lines. (The template
        # carries one comment line showing an example `icm64 -g /Users/...`
        # invocation; comments aren't executed by ICM, so we strip them.)
        non_comment = _strip_comments(out)
        self.assertNotIn("/Users/aoxu/", non_comment)

    def test_local_macos_render_keeps_mac_paths(self) -> None:
        r = _render("local_macos")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        out = r.stdout
        self.assertIn('s_map_dir       = "/Users/aoxu/projects/anchnor_based_VSYNTHES/maps"', out)
        self.assertIn('s_edited_mel    = "/Users/aoxu/projects/anchnor_based_VSYNTHES/final_table_edited.sdf"', out)
        self.assertIn('s_synthon_sdf   = "/SOMEWHERE/synth.sdf"', out)
        self.assertIn('s_results_dir   = "/SOMEWHERE/out"', out)
        # No Linux paths should leak into executable lines.
        non_comment = _strip_comments(out)
        self.assertNotIn("/home/aoxu/", non_comment)

    def test_missing_sentinel_raises(self) -> None:
        """If the template drifts and a sentinel disappears, render_icm should
        fail loudly so we don't silently emit an unpatched script."""
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import run_srg_batch  # noqa: PLC0415
        finally:
            sys.path.pop(0)
        bogus_template = "nothing matching any sentinel here\n"
        with self.assertRaises(RuntimeError):
            run_srg_batch.render_icm(bogus_template, 2, Path("/x"),
                                     Path("/y"), "tbl")


if __name__ == "__main__":
    unittest.main()
