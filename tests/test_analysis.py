"""Tests for tomofast_x_mu.analysis module."""

import os

import numpy as np
import pandas as pd
import pytest

from tomofast_x_mu.analysis import (
    grain_statistics,
    vector_analysis,
    combine_best_locations,
    extract_blob_data_costs,
    compute_field_data_cost,
)


@pytest.fixture
def grain_tree(tmp_path):
    """Create a synthetic blob directory tree with grain CSVs."""
    blob_name = "Blob_001"
    grains_dir = tmp_path / blob_name / blob_name / "Grains_detection"
    grains_dir.mkdir(parents=True)

    # Create a grain CSV for threshold 0.5
    df = pd.DataFrame(
        {
            "X": [10.0, 14.4, 10.0],
            "Y": [20.0, 20.0, 24.4],
            "Z": [-5.0, -5.0, -5.0],
            "GrainID": [1, 1, 2],
            "Mx": [100.0, 120.0, -50.0],
            "My": [0.0, 10.0, 30.0],
            "Mz": [0.0, 0.0, 80.0],
        }
    )
    df.to_csv(grains_dir / "grain_labels_0.5.csv", index=False)

    return str(tmp_path)


class TestGrainStatistics:
    def test_returns_dataframe(self, grain_tree):
        df = grain_statistics(grain_tree, threshold_list=[0.5])
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, grain_tree):
        df = grain_statistics(grain_tree, threshold_list=[0.5])
        expected = {
            "Blob", "Threshold", "GrainID", "Num_Voxels",
            "Mx_mean", "My_mean", "Mz_mean", "Mean_Magnitude",
            "Mean_Moment_Amplitude", "Resultant_R", "Angular_STD_deg",
            "Dec", "Inc", "Dipole_X", "Dipole_Y", "Dipole_Z",
        }
        assert expected.issubset(set(df.columns))

    def test_grain_count(self, grain_tree):
        df = grain_statistics(grain_tree, threshold_list=[0.5])
        # 2 grains at threshold 0.5
        assert len(df) == 2
        assert set(df["GrainID"].values) == {1, 2}

    def test_voxel_counts(self, grain_tree):
        df = grain_statistics(grain_tree, threshold_list=[0.5])
        grain1 = df[df["GrainID"] == 1].iloc[0]
        grain2 = df[df["GrainID"] == 2].iloc[0]
        assert grain1["Num_Voxels"] == 2
        assert grain2["Num_Voxels"] == 1

    def test_empty_blob_root(self, tmp_path):
        df = grain_statistics(str(tmp_path), threshold_list=[0.5])
        assert len(df) == 0


class TestVectorAnalysis:
    def test_returns_dataframe(self, tmp_path):
        # Create minimal Tomofast output
        model_dir = tmp_path / "combined_grains" / "model"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "mag_final_model_full.txt"
        model_file.write_text("2\n100.0 50.0 -30.0\n200.0 10.0 -80.0\n")

        df = vector_analysis(str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_mz_flipped(self, tmp_path):
        model_dir = tmp_path / "combined_grains" / "model"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "mag_final_model_full.txt"
        model_file.write_text("1\n10.0 20.0 -30.0\n")

        df = vector_analysis(str(tmp_path))
        assert df.iloc[0]["Mz_raw"] == pytest.approx(-30.0)
        assert df.iloc[0]["Mz_flipped"] == pytest.approx(30.0)

    def test_missing_model_file(self, tmp_path):
        df = vector_analysis(str(tmp_path), grain_folders=["nonexistent"])
        assert len(df) == 0


class TestCombineBestLocations:
    def test_picks_best_per_blob(self, tmp_path):
        summary = tmp_path / "costs.txt"
        summary.write_text(
            "# blob grain meshfile xmin xmax ymin ymax zmin zmax cost\n"
            "Blob_001 grain_1 mesh_001.txt 0.0 4.4 0.0 4.4 0.0 4.4 100.0\n"
            "Blob_001 grain_1 mesh_002.txt 4.4 8.8 0.0 4.4 0.0 4.4 50.0\n"
            "Blob_002 grain_1 mesh_001.txt 10.0 14.4 10.0 14.4 0.0 4.4 200.0\n"
        )

        out = tmp_path / "combined.txt"
        best = combine_best_locations(str(summary), str(out))

        assert len(best) == 2
        assert out.exists()
        # Blob_001 should pick the row with cost=50
        blob1 = best[best["blob"] == "Blob_001"].iloc[0]
        assert blob1["final_data_cost"] == pytest.approx(50.0)

    def test_output_file_format(self, tmp_path):
        summary = tmp_path / "costs.txt"
        summary.write_text(
            "# header\n"
            "Blob_001 grain_1 mesh_001.txt 1.0 2.0 3.0 4.0 5.0 6.0 10.0\n"
        )

        out = tmp_path / "out_mesh.txt"
        combine_best_locations(str(summary), str(out))

        with open(out) as f:
            header = f.readline().strip()
            assert header == "1"
            line = f.readline().strip()
            parts = line.split()
            # Should have: xmin xmax ymin ymax zmin zmax 0 0 0 i 1 1
            assert len(parts) == 12


class TestExtractBlobDataCosts:
    def test_basic(self, tmp_path):
        blob_dir = tmp_path / "Blob_001"
        blob_dir.mkdir()
        log = blob_dir / "QDM_log_Blob_001.txt"
        log.write_text(
            "iteration 1: data cost (new) = 5.000E-01\n"
            "iteration 2: data cost (new) = 2.500E-01\n"
            "iteration 3: data cost (new) = 1.234E-03\n"
        )

        df = extract_blob_data_costs(str(tmp_path))
        assert len(df) == 1
        assert df.iloc[0]["blob"] == "Blob_001"
        assert df.iloc[0]["final_data_cost"] == pytest.approx(1.234e-3)

    def test_empty_dir(self, tmp_path):
        df = extract_blob_data_costs(str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["blob", "final_data_cost"]


class TestComputeFieldDataCost:
    def _write_obs(self, path, rows):
        """Write a minimal Tomofast .obs file."""
        with open(path, "w") as f:
            f.write(f"{len(rows)}\n")
            for x, y, z, bz in rows:
                f.write(f"{x} {y} {z} {bz}\n")

    def test_zero_misfit(self, tmp_path):
        rows = [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 2.0)]
        obs = tmp_path / "obs.obs"
        calc = tmp_path / "calc.obs"
        self._write_obs(str(obs), rows)
        self._write_obs(str(calc), rows)

        result = compute_field_data_cost(str(obs), str(calc))
        assert result == pytest.approx(0.0)

    def test_known_misfit(self, tmp_path):
        # obs = [1, 0], calc = [0, 0]; misfit = 1 / 1 = 1.0
        obs_rows = [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)]
        calc_rows = [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        obs = tmp_path / "obs.obs"
        calc = tmp_path / "calc.obs"
        self._write_obs(str(obs), obs_rows)
        self._write_obs(str(calc), calc_rows)

        result = compute_field_data_cost(str(obs), str(calc))
        assert result == pytest.approx(1.0)
