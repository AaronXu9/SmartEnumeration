"""Unit tests for run_srg_batch_headless.resolve_headless_icm.

Pure stdlib, Mac-runnable (no ICM binary, no Linux needed). We patch
platform.system() to exercise both branches.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_srg_batch_headless  # noqa: E402


class ResolveHeadlessIcmTests(unittest.TestCase):
    def test_darwin_returns_input_unchanged(self) -> None:
        """On macOS, the resolver is a no-op — Mac ICM has no icmng sibling
        and icm64 -g already runs headlessly there."""
        with patch("platform.system", return_value="Darwin"):
            out = run_srg_batch_headless.resolve_headless_icm(
                Path("/does/not/exist/icm64"))
        self.assertEqual(out, Path("/does/not/exist/icm64"))

    def test_linux_returns_icmng_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            icm64 = Path(td) / "icm64"
            icmng = Path(td) / "icmng"
            icm64.touch()
            icmng.touch()
            with patch("platform.system", return_value="Linux"):
                out = run_srg_batch_headless.resolve_headless_icm(icm64)
            self.assertEqual(out, icmng)

    def test_linux_missing_icmng_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            icm64 = Path(td) / "icm64"
            icm64.touch()
            with patch("platform.system", return_value="Linux"):
                with self.assertRaises(FileNotFoundError) as cm:
                    run_srg_batch_headless.resolve_headless_icm(icm64)
            self.assertIn("icmng", str(cm.exception))
            self.assertIn(str(Path(td) / "icmng"), str(cm.exception))


class HeadlessTemplateTests(unittest.TestCase):
    def test_headless_template_file_exists(self) -> None:
        """The headless template must ship alongside the driver — it is not
        optional on Linux. Guards against accidental rename/delete."""
        self.assertTrue(
            run_srg_batch_headless.HEADLESS_TEMPLATE.is_file(),
            f"missing {run_srg_batch_headless.HEADLESS_TEMPLATE}",
        )

    def test_headless_template_has_no_openfile_calls(self) -> None:
        """`openFile` is a GUI menu command that icmng does not link in.
        The headless template must use `read object` / `read map` /
        `read table mol` instead."""
        text = run_srg_batch_headless.HEADLESS_TEMPLATE.read_text()
        non_comment = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        self.assertNotIn("openFile", non_comment)
        self.assertIn("read object", non_comment)
        self.assertIn("read map", non_comment)
        self.assertIn("read table mol", non_comment)


if __name__ == "__main__":
    unittest.main()
