"""Tests for tomofast_x_mu.parfile module."""

import os

import numpy as np
import pytest

from tomofast_x_mu.parfile import (
    create_inversion_parfiles,
    create_source_parfiles,
    create_fitting_parfile,
)


@pytest.fixture
def summary_file(tmp_path):
    """Create a blob_window_summary.txt for testing."""
    p = tmp_path / "summary.txt"
    p.write_text("# blob_id nx ny nz ndata\n1 60 40 120 2400\n2 80 50 120 4000\n")
    return str(p)


class TestCreateInversionParfiles:
    def test_creates_parfiles(self, summary_file, tmp_path):
        par_dir = tmp_path / "parfiles"
        paths = create_inversion_parfiles(summary_file, str(par_dir))
        assert len(paths) == 2
        for p in paths:
            assert os.path.exists(p)

    def test_parfile_contains_blob_id(self, summary_file, tmp_path):
        par_dir = tmp_path / "parfiles"
        paths = create_inversion_parfiles(summary_file, str(par_dir))
        with open(paths[0]) as f:
            text = f.read()
        assert "Blob_001" in text
        assert "60 40 120" in text

    def test_compression_rate(self, summary_file, tmp_path):
        par_dir = tmp_path / "parfiles"
        paths = create_inversion_parfiles(summary_file, str(par_dir), compression_rate=0.05)
        with open(paths[0]) as f:
            text = f.read()
        assert "0.05" in text


class TestCreateSourceParfiles:
    def test_creates_source_parfiles(self, summary_file, tmp_path):
        par_dir = tmp_path / "source_par"
        paths = create_source_parfiles(summary_file, str(par_dir))
        assert len(paths) == 2

    def test_source_parfile_has_1x1x1_grid(self, summary_file, tmp_path):
        par_dir = tmp_path / "source_par"
        paths = create_source_parfiles(summary_file, str(par_dir))
        with open(paths[0]) as f:
            text = f.read()
        assert "1 1 1" in text


class TestCreateFittingParfile:
    def test_creates_fitting_parfile(self, tmp_path):
        out = tmp_path / "subdir" / "parfile.txt"
        result = create_fitting_parfile(str(out), nx=5, ndata=1000)
        assert os.path.exists(result)

    def test_fitting_parfile_content(self, tmp_path):
        out = tmp_path / "fitting.txt"
        create_fitting_parfile(str(out), nx=10, ndata=5000)
        with open(out) as f:
            text = f.read()
        assert "modelGrid.size" in text
        assert "10 1 1" in text
        assert "5000" in text
