"""Tests for tomofast_x_mu.io module."""

import warnings

import numpy as np
import pytest

from tomofast_x_mu.io import load_qdm_mat, load_field_table, read_mesh_bounds


class TestLoadQdmMat:
    def test_no_numpy_deprecation_warning_for_scalar_h(self, synthetic_mat):
        # NumPy deprecates float(ndarray) even for 1x1 arrays; ensure we extract scalars safely.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            _ = load_qdm_mat(synthetic_mat["path"], save_csv=False, h_units="m")
        # Only fail on the specific NumPy scalar-conversion deprecation we intend to avoid.
        dep_w = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "Conversion of an array" in str(x.message)
        ]
        assert dep_w == []

    def test_h_units_um_keeps_scalar_h_as_um(self, tmp_path):
        import scipy.io

        ny, nx = 5, 7
        step = 4.4e-6  # meters
        h_um = 5.0  # micrometers
        Bz = np.zeros((ny, nx), dtype=float)

        mat_path = tmp_path / "h_um.mat"
        scipy.io.savemat(str(mat_path), {"Bz": Bz, "step": np.array([[step]]), "h": np.array([[h_um]])})

        ds = load_qdm_mat(mat_path, save_csv=False, h_units="um")
        # z coordinate should be 5 um everywhere
        assert float(ds.z.values[0, 0]) == pytest.approx(5.0)

    def test_h_units_auto_infers_um_for_typical_qdm_export(self, tmp_path):
        import scipy.io

        ny, nx = 5, 7
        step = 4.4e-6  # meters
        h_um = 5.0  # typical micrometers
        Bz = np.zeros((ny, nx), dtype=float)

        mat_path = tmp_path / "h_auto.mat"
        scipy.io.savemat(str(mat_path), {"Bz": Bz, "step": np.array([[step]]), "h": np.array([[h_um]])})

        ds = load_qdm_mat(mat_path, save_csv=False, h_units="auto")
        assert float(ds.z.values[0, 0]) == pytest.approx(5.0)

    def test_returns_xarray_dataset(self, synthetic_mat):
        ds = load_qdm_mat(synthetic_mat["path"], save_csv=False)
        assert hasattr(ds, "bz")

    def test_shape_matches(self, synthetic_mat):
        ds = load_qdm_mat(synthetic_mat["path"], save_csv=False)
        assert ds.bz.shape == (synthetic_mat["ny"], synthetic_mat["nx"])

    def test_bz_values_converted_to_nT(self, synthetic_mat):
        ds = load_qdm_mat(synthetic_mat["path"], save_csv=False)
        np.testing.assert_allclose(
            ds.bz.values, synthetic_mat["Bz_nT"], rtol=1e-10
        )

    def test_coord_units_um(self, synthetic_mat):
        ds = load_qdm_mat(synthetic_mat["path"], save_csv=False)
        spacing = synthetic_mat["step_um"]
        # x should start at 0, step by spacing
        assert ds.x.values[0] == pytest.approx(0.0)
        assert ds.x.values[1] == pytest.approx(spacing)

    def test_csv_saved(self, synthetic_mat, tmp_path):
        csv_path = tmp_path / "output.csv"
        load_qdm_mat(synthetic_mat["path"], save_csv=True, csv_path=csv_path)
        assert csv_path.exists()
        # CSV should have header + ny*nx data rows
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 1 + synthetic_mat["ny"] * synthetic_mat["nx"]

    def test_csv_default_path(self, synthetic_mat):
        ds = load_qdm_mat(synthetic_mat["path"], save_csv=True)
        default_csv = synthetic_mat["path"].with_suffix(".csv")
        assert default_csv.exists()


class TestLoadFieldTable:
    def test_reads_whitespace_delimited(self, tmp_path):
        p = tmp_path / "field.txt"
        p.write_text("1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n")
        df = load_field_table(p)
        assert list(df.columns) == ["x", "y", "z", "value"]
        assert len(df) == 2
        assert df.iloc[0]["x"] == pytest.approx(1.0)

    def test_custom_column_names(self, tmp_path):
        p = tmp_path / "field2.txt"
        p.write_text("10 20\n30 40\n")
        df = load_field_table(p, names=("a", "b"))
        assert list(df.columns) == ["a", "b"]


class TestReadMeshBounds:
    def test_valid_mesh(self, tmp_path):
        # 2 cells: header + 2 data rows
        p = tmp_path / "mesh.txt"
        lines = [
            "2",
            "0.0 4.4 0.0 4.4 0.0 4.4 0 0 0 1 1 1",
            "4.4 8.8 0.0 4.4 0.0 4.4 0 0 0 2 1 1",
        ]
        p.write_text("\n".join(lines) + "\n")
        result = read_mesh_bounds(str(p))
        assert result is not None
        assert result["nx"] == 2
        assert result["ny"] == 1
        assert result["nz"] == 1

    def test_empty_mesh_returns_none(self, tmp_path):
        p = tmp_path / "empty_mesh.txt"
        p.write_text("0\n")
        result = read_mesh_bounds(str(p))
        assert result is None
