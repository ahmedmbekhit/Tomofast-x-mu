"""Tests for tomofast_x_mu.runner module."""

import io
import math
import sys
import unittest.mock as mock

import pytest

from tomofast_x_mu.runner import to_wsl_path, _print_progress, _run_live


class TestToWslPath:
    def test_basic_conversion(self):
        assert to_wsl_path(r"C:\Users\test\file.txt") == "/mnt/c/Users/test/file.txt"

    def test_lowercase_drive(self):
        result = to_wsl_path(r"D:\Data\project")
        assert result.startswith("/mnt/d/")

    def test_forward_slashes(self):
        result = to_wsl_path("E:/some/path")
        assert result == "/mnt/e/some/path"

    def test_nested_path(self):
        result = to_wsl_path(r"C:\Users\user\Documents\project\data")
        assert result == "/mnt/c/Users/user/Documents/project/data"


class TestPrintProgress:
    def _capture(self, *args, **kwargs):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _print_progress(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_normal_output(self):
        out = self._capture(3, 10, "Blob_001", "grain_1", "mesh_001.txt", 1.234e-3)
        assert "\r" in out
        assert "3/10" in out
        assert "1.234000e-03" in out

    def test_nan_cost(self):
        out = self._capture(1, 5, "Blob_001", "grain_1", "mesh_001.txt", float("nan"))
        assert "NaN" in out


class TestRunLiveBashTime:
    def test_bash_time_wraps_command(self):
        captured_args = []

        def fake_popen(args, **kwargs):
            captured_args.extend(args)
            m = mock.MagicMock()
            m.stdout.read.return_value = ""
            m.wait.return_value = None
            m.returncode = 0
            return m

        with mock.patch("tomofast_x_mu.runner.subprocess.Popen", side_effect=fake_popen):
            _run_live("mpirun -np 4 ./tomofastx -p parfile.txt", bash_time=True)

        # The bash -lc argument should contain "{ time"
        cmd_arg = captured_args[-1]
        assert "{ time" in cmd_arg
