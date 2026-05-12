"""Tests for paths.py env detection.

paths.py caches ENV_NAME at import time, so env-override tests are run in
subprocesses (so each has a fresh import).
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_paths_module(env: dict[str, str]) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "paths.py")],
        cwd=str(REPO_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
    )


class EnvOverrideTests(unittest.TestCase):
    def test_override_local_macos(self) -> None:
        r = _run_paths_module({"VSYNTHES_ENV": "local_macos"})
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("ENV_NAME       = local_macos", r.stdout)
        self.assertIn("/Users/aoxu/projects/anchnor_based_VSYNTHES", r.stdout)
        self.assertIn("/Applications/MolsoftICM64.app/Contents/MacOS/icm64", r.stdout)
        self.assertIn("results_local_macos", r.stdout)
        self.assertNotIn("/home/aoxu/", r.stdout)

    def test_override_katlab(self) -> None:
        r = _run_paths_module({"VSYNTHES_ENV": "katlab"})
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("ENV_NAME       = katlab", r.stdout)
        self.assertIn("/home/aoxu/projects/anchnor_based_VSYNTHES", r.stdout)
        self.assertIn("/home/aoxu/icm-3.9-4/icm64", r.stdout)
        self.assertIn("results_katlab", r.stdout)
        self.assertNotIn("/Users/aoxu/", r.stdout)

    def test_unknown_override_raises(self) -> None:
        r = _run_paths_module({"VSYNTHES_ENV": "bogus_env"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("bogus_env", r.stderr)


class CurrentEnvSanityTests(unittest.TestCase):
    """Host-introspection: whichever env we're on, required files must exist."""

    def test_paths_are_consistent(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        try:
            import paths  # noqa: PLC0415
        finally:
            sys.path.pop(0)
        self.assertIn(paths.ENV_NAME, ("local_macos", "katlab"))
        self.assertTrue(paths.PROJECT_ROOT.is_dir(),
                        f"PROJECT_ROOT missing: {paths.PROJECT_ROOT}")
        self.assertTrue(paths.MAPS_DIR.is_dir(),
                        f"MAPS_DIR missing: {paths.MAPS_DIR}")
        self.assertTrue(paths.SYNTHONS_DIR.is_dir(),
                        f"SYNTHONS_DIR missing: {paths.SYNTHONS_DIR}")
        self.assertTrue(paths.MEL_SDF.is_file(),
                        f"MEL_SDF missing: {paths.MEL_SDF}")
        self.assertTrue(paths.APO_TSV.is_file(),
                        f"APO_TSV missing: {paths.APO_TSV}")
        for name in ("TEMPLATE_GUI_PARALLEL", "TEMPLATE_GUI_NOPARALLEL",
                     "TEMPLATE_NOGUI_PARALLEL", "TEMPLATE_NOGUI_NOPARALLEL"):
            tpl = getattr(paths, name)
            self.assertTrue(tpl.is_file(), f"{name} missing: {tpl}")
        self.assertTrue(paths.ICM_BIN.is_file(),
                        f"ICM_BIN missing: {paths.ICM_BIN}")


if __name__ == "__main__":
    unittest.main()
