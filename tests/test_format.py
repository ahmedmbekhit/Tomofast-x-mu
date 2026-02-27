"""Tests for tomofast_x_mu.format module."""

import numpy as np
import pytest

from tomofast_x_mu.format import format_tomofast_obs, write_obs, write_meshgrid_tomofast


class TestFormatTomofastObs:
    def test_output_file_created(self, synthetic_csv, tmp_path):
        out = tmp_path / "subdir" / "out.obs"
        result = format_tomofast_obs(str(synthetic_csv["path"]), str(out))
        assert out.exists()

    def test_returns_correct_grid_dims(self, synthetic_csv, tmp_path):
        out = tmp_path / "out.obs"
        result = format_tomofast_obs(str(synthetic_csv["path"]), str(out))
        assert result["nx"] == synthetic_csv["nx"]
        assert result["ny"] == synthetic_csv["ny"]

    def test_flip_bz(self, synthetic_csv, tmp_path):
        out1 = tmp_path / "flipped.obs"
        out2 = tmp_path / "not_flipped.obs"
        r1 = format_tomofast_obs(str(synthetic_csv["path"]), str(out1), flip_bz=True)
        r2 = format_tomofast_obs(str(synthetic_csv["path"]), str(out2), flip_bz=False)
        np.testing.assert_allclose(r1["bz"], -r2["bz"])

    def test_clipping(self, synthetic_csv, tmp_path):
        out = tmp_path / "clipped.obs"
        result = format_tomofast_obs(
            str(synthetic_csv["path"]),
            str(out),
            x_start=5,
            x_end=15,
            y_start=2,
            y_end=10,
        )
        assert result["nx"] == 10
        assert result["ny"] == 8

    def test_header_has_ndata(self, synthetic_csv, tmp_path):
        out = tmp_path / "check_header.obs"
        format_tomofast_obs(str(synthetic_csv["path"]), str(out))
        with open(out) as f:
            header = f.readline().strip()
        ndata = int(header)
        assert ndata == synthetic_csv["nx"] * synthetic_csv["ny"]


class TestWriteObs:
    def test_writes_with_updated_header(self, tmp_path):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = tmp_path / "obs.txt"
        write_obs(str(out), "999", arr)
        with open(out) as f:
            header = f.readline().strip()
        assert header == "2"

    def test_non_integer_header_unchanged(self, tmp_path):
        arr = np.ones((3, 2))
        out = tmp_path / "obs2.txt"
        write_obs(str(out), "# comment line", arr)
        with open(out) as f:
            header = f.readline().strip()
        assert header == "# comment line"


class TestWriteMeshgridTomofast:
    def test_writes_file_with_header(self, tmp_path):
        line_data = np.array(
            [[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1], [1, 2, 0, 1, 0, 1, 0, 0, 0, 2, 1, 1]],
            dtype=float,
        )
        out = tmp_path / "subdir" / "meshgrid.txt"
        write_meshgrid_tomofast(str(out), line_data)
        assert out.exists()
        with open(out) as f:
            header = f.readline().strip()
        assert header == "2"
