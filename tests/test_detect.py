"""Tests for tomofast_x_mu.detect module."""

import numpy as np
import pytest

from tomofast_x_mu.detect import detect_anomalies


class TestDetectAnomalies:
    def test_detects_blob(self, synthetic_obs, tmp_path):
        results = detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=2.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=10,
            max_blob_cells=100000,
            nz_fixed=10,
        )
        # Should detect at least 1 blob from the central anomaly
        assert len(results) >= 1

    def test_blob_keys(self, synthetic_obs):
        results = detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=2.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=10,
        )
        if len(results) > 0:
            expected_keys = {
                "blob_id", "x_min", "x_max", "y_min", "y_max",
                "nx_win", "ny_win", "nz_win", "ndata", "obs_out",
            }
            assert set(results[0].keys()) == expected_keys

    def test_writes_obs_files(self, synthetic_obs, tmp_path):
        obs_dir = tmp_path / "blob_obs"
        results = detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=2.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=10,
            out_obs_dir=str(obs_dir),
        )
        if len(results) > 0:
            assert obs_dir.exists()
            assert results[0]["obs_out"] is not None
            import os
            assert os.path.exists(results[0]["obs_out"])

    def test_writes_windows_file(self, synthetic_obs, tmp_path):
        win_path = tmp_path / "blob_windows.txt"
        detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=2.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=10,
            windows_path=str(win_path),
        )
        assert win_path.exists()

    def test_writes_summary_file(self, synthetic_obs, tmp_path):
        sum_path = tmp_path / "summary.txt"
        detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=2.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=10,
            summary_path=str(sum_path),
        )
        assert sum_path.exists()

    def test_min_blob_filter(self, synthetic_obs):
        # With very high min_blob_cells, should filter out everything
        results = detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=2.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=999999,
        )
        assert len(results) == 0

    def test_high_threshold_no_blobs(self, synthetic_obs):
        # Very high threshold should find nothing
        results = detect_anomalies(
            synthetic_obs["path"],
            threshold_factor=100.0,
            pad_x=5,
            pad_y=5,
            min_blob_cells=1,
        )
        assert len(results) == 0
