"""Integration smoke test: python3 run_srg_batch.py --dry-run --only-row 2.

Runs the batch driver in dry-run mode (no ICM invocation) and verifies the
rendered run.icm contains host-appropriate paths. Uses the current host's
environment (no override), so on macOS we expect Mac paths and on KatLab we
expect Linux paths.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class DryRunSmokeTest(unittest.TestCase):
    def test_dry_run_only_row_2(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import paths  # noqa: PLC0415
        finally:
            sys.path.pop(0)

        expected_out_dir = paths.RESULTS_DIR / "MEL_2_Rank2"
        rendered = expected_out_dir / "run.icm"
        if rendered.exists():
            rendered.unlink()

        r = subprocess.run(
            [sys.executable, "run_srg_batch.py", "--dry-run", "--only-row", "2"],
            cwd=str(REPO_ROOT),
            env={k: v for k, v in os.environ.items() if k != "VSYNTHES_ENV"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, msg=f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}")

        self.assertTrue(rendered.is_file(),
                        f"Rendered ICM not found: {rendered}")
        text = rendered.read_text()

        # s_map_dir and s_edited_mel must point at the current host's tree.
        expected_map = f's_map_dir       = "{paths.MAPS_DIR}"'
        expected_mel = f's_edited_mel    = "{paths.MEL_SDF}"'
        self.assertIn(expected_map, text)
        self.assertIn(expected_mel, text)

        # Per-run substitutions:
        self.assertIn("i_mel_row       = 2", text)
        self.assertIn(f's_results_dir   = "{expected_out_dir}"', text)

        # No foreign-env paths should appear.
        foreign = "/home/aoxu/" if paths.ENV_NAME == "local_macos" else "/Users/aoxu/"
        # template has an example RUN command in comments that references the
        # canonical invocation path -- strip lines starting with '#' before
        # the foreign-path check (comments are not executed by ICM).
        non_comment = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        self.assertNotIn(foreign, non_comment,
                         f"foreign path {foreign!r} appeared in non-comment lines")

        # Manifest was written.
        manifest = paths.RESULTS_DIR / "batch_manifest.tsv"
        self.assertTrue(manifest.is_file(), f"manifest missing: {manifest}")


if __name__ == "__main__":
    unittest.main()
