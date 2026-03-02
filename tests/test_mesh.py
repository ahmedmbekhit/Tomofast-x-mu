"""Tests for tomofast_x_mu.mesh module."""

import numpy as np
import pytest

from tomofast_x_mu.mesh import make_edges, build_mesh_per_blob


class TestMakeEdges:
    def test_core_region(self):
        edges, widths = make_edges(core_min=0.0, d_core=1.0, n_core=5, npad=0, r=1.0)
        assert len(widths) == 5
        np.testing.assert_allclose(widths, 1.0)
        np.testing.assert_allclose(edges, [0, 1, 2, 3, 4, 5])

    def test_with_padding(self):
        edges, widths = make_edges(core_min=10.0, d_core=2.0, n_core=3, npad=2, r=1.5)
        # Total cells = 2 (left pad) + 3 (core) + 2 (right pad) = 7
        assert len(widths) == 7
        assert len(edges) == 8

    def test_padding_growth(self):
        _, widths = make_edges(core_min=0.0, d_core=1.0, n_core=4, npad=3, r=2.0)
        # Left padding: d_core * r^2, d_core * r^1, d_core * r^0
        left = widths[:3]
        np.testing.assert_allclose(left, [4.0, 2.0, 1.0])
        # Right padding: d_core * r^0, d_core * r^1, d_core * r^2
        right = widths[-3:]
        np.testing.assert_allclose(right, [1.0, 2.0, 4.0])

    def test_edges_monotonic(self):
        edges, _ = make_edges(core_min=100.0, d_core=4.4, n_core=10, npad=5, r=1.15)
        assert np.all(np.diff(edges) > 0)

    def test_core_min_position(self):
        edges, widths = make_edges(core_min=50.0, d_core=2.0, n_core=4, npad=2, r=1.5)
        # Sum of left padding widths
        left_sum = widths[:2].sum()
        assert edges[0] == pytest.approx(50.0 - left_sum)
        # First core edge should be at core_min
        assert edges[2] == pytest.approx(50.0)


class TestBuildMeshPerBlob:
    def test_builds_meshes(self, tmp_path):
        # Create a windows file
        win_path = tmp_path / "blob_windows.txt"
        win_path.write_text("# blob_id x_min x_max y_min y_max\n1 0.0 44.0 0.0 44.0\n")

        out_dir = tmp_path / "meshgrids"
        summary_path = tmp_path / "summary.txt"

        results = build_mesh_per_blob(
            str(win_path),
            str(out_dir),
            str(summary_path),
            dx_core=4.4,
            dy_core=4.4,
            npad_x=3,
            npad_y=3,
            rx=1.15,
            ry=1.15,
            nz=5,
            dz0=4.4,
            rz=1.0,
        )

        assert len(results) == 1
        assert results[0]["blob_id"] == 1
        assert (out_dir / "Meshgrid_Blob_001.txt").exists()
        assert summary_path.exists()

    def test_mesh_file_format(self, tmp_path):
        win_path = tmp_path / "windows.txt"
        win_path.write_text("# header\n1 0.0 22.0 0.0 22.0\n")

        out_dir = tmp_path / "meshes"
        summary_path = tmp_path / "sum.txt"

        build_mesh_per_blob(
            str(win_path),
            str(out_dir),
            str(summary_path),
            npad_x=2,
            npad_y=2,
            nz=3,
        )

        mesh_file = out_dir / "Meshgrid_Blob_001.txt"
        with open(mesh_file) as f:
            header = f.readline().strip()
            n_cells = int(header)
            lines = f.readlines()
        assert len(lines) == n_cells
