"""Integration smoke test for run_srg_batch_headless.py --dry-run.

Mirrors tests/test_dry_run.py: invokes the headless driver with --dry-run
--only-row 2 and asserts the rendered run.icm contains host-appropriate
paths. The headless wrapper should produce byte-identical templated output
to the classic path — the only difference is the ICM binary we'd invoke at
run time, which dry-run never does.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class HeadlessDryRunSmokeTest(unittest.TestCase):
    def test_headless_dry_run_only_row_2(self) -> None:
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
            [sys.executable, "run_srg_batch_headless.py",
             "--dry-run", "--only-row", "2"],
            cwd=str(REPO_ROOT),
            env={k: v for k, v in os.environ.items() if k != "VSYNTHES_ENV"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            r.returncode, 0,
            msg=f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}")

        self.assertTrue(rendered.is_file(),
                        f"Rendered ICM not found: {rendered}")
        text = rendered.read_text()

        expected_map = f's_map_dir       = "{paths.MAPS_DIR}"'
        expected_mel = f's_edited_mel    = "{paths.MEL_SDF}"'
        self.assertIn(expected_map, text)
        self.assertIn(expected_mel, text)
        self.assertIn("i_mel_row       = 2", text)
        self.assertIn(f's_results_dir   = "{expected_out_dir}"', text)

        foreign = "/home/aoxu/" if paths.ENV_NAME == "local_macos" else "/Users/aoxu/"
        non_comment = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        self.assertNotIn(foreign, non_comment)

        manifest = paths.RESULTS_DIR / "batch_manifest.tsv"
        self.assertTrue(manifest.is_file(), f"manifest missing: {manifest}")

        # Per-OS template selection: Mac keeps the GUI template (with
        # openFile); Linux must swap to the icmng-compatible variant.
        non_comment = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        if platform.system() == "Darwin":
            self.assertIn("openFile", non_comment,
                          "Mac must retain the GUI template unchanged")
        else:
            self.assertNotIn("openFile", non_comment,
                             "icmng does not support openFile")
            self.assertIn("read object", non_comment)
            self.assertIn("read map", non_comment)
            self.assertIn("read table mol", non_comment)

    def test_headless_respects_user_icm_flag(self) -> None:
        """User-supplied --icm wins; the wrapper should not inject its own
        headless binary path on top of the user's explicit choice."""
        r = subprocess.run(
            [sys.executable, "run_srg_batch_headless.py",
             "--dry-run", "--only-row", "2",
             "--icm", "/explicit/path/to/icm64"],
            cwd=str(REPO_ROOT),
            env={k: v for k, v in os.environ.items() if k != "VSYNTHES_ENV"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            r.returncode, 0,
            msg=f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
