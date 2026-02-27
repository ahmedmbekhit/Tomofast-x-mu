"""Command line interface for tomofast_x_mu.

This module intentionally keeps dependencies minimal (stdlib argparse).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    """Walk upwards from *start* to locate a project root (pyproject.toml)."""
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return None


def _cmd_version(_args: argparse.Namespace) -> int:
    v: str | None = None

    # 1) Preferred: installed dist metadata
    try:
        from importlib.metadata import version

        v = version("tomofast_x_mu")
    except Exception:
        v = None

    # 2) Fallback: parse pyproject.toml (source checkout)
    if v is None:
        root = _find_repo_root(Path.cwd()) or _find_repo_root(Path(__file__))
        if root is not None:
            pyproject = root / "pyproject.toml"
            if pyproject.exists():
                for line in pyproject.read_text().splitlines():
                    s = line.strip()
                    if s.startswith("version") and "=" in s:
                        # naive TOML parse: version = "0.1.0"
                        parts = s.split("=", 1)
                        rhs = parts[1].strip().strip('"').strip("'")
                        if rhs:
                            v = rhs
                            break

    print(v or "unknown")
    return 0


def _cmd_diagram(_args: argparse.Namespace) -> int:
    # Prefer locating the file inside a source checkout.
    root = _find_repo_root(Path.cwd()) or _find_repo_root(Path(__file__))

    candidates: list[Path] = []
    if root is not None:
        candidates.extend(
            [
                root / "docs" / "architecture.md",
                root / "ARCHITECTURE.md",
            ]
        )

    for p in candidates:
        if p.exists():
            print(str(p.resolve()))
            return 0

    # Last resort: print a helpful relative name.
    print("docs/architecture.md")
    return 1


def _cmd_smoke(args: argparse.Namespace) -> int:
    from tomofast_x_mu.detect import detect_anomalies
    from tomofast_x_mu.format import format_tomofast_obs
    from tomofast_x_mu.io import load_qdm_mat
    from tomofast_x_mu.mesh import build_mesh_per_blob
    from tomofast_x_mu.parfile import create_inversion_parfiles

    root = _find_repo_root(Path.cwd()) or _find_repo_root(Path(__file__))

    if args.mat is None:
        if root is None:
            raise SystemExit("Could not locate repo root; please pass --mat")
        data_dir = root / "real_qdm_data"
        mats = sorted(data_dir.glob("*.mat"))
        if not mats:
            raise SystemExit(f"No .mat files found in {data_dir}")
        mat_path = mats[0]
    else:
        mat_path = Path(args.mat)

    out_dir = Path(args.out).resolve() if args.out else Path.cwd() / "tomofast_x_mu_smoke_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "qdm.csv"
    obs_path = out_dir / "obs" / "Mag_Inversion.obs"
    obs_path.parent.mkdir(parents=True, exist_ok=True)

    windows_path = out_dir / "blob_windows.txt"
    summary_path = out_dir / "blob_window_summary.txt"
    blob_obs_dir = out_dir / "blob_obs"

    mesh_dir = out_dir / "meshgrids"
    mesh_summary = out_dir / "mesh_summary.txt"

    par_dir = out_dir / "parfiles"

    # 1) .mat -> CSV
    _ = load_qdm_mat(mat_path, save_csv=True, csv_path=csv_path, h_units=args.h_units)

    # 2) CSV -> Tomofast .obs
    _ = format_tomofast_obs(str(csv_path), str(obs_path))

    # 3) Detect anomaly windows + per-blob obs
    blobs = detect_anomalies(
        str(obs_path),
        threshold_factor=args.threshold_factor,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        min_blob_cells=args.min_blob_cells,
        max_blob_cells=args.max_blob_cells,
        nz_fixed=args.nz_fixed,
        out_obs_dir=str(blob_obs_dir),
        windows_path=str(windows_path),
        summary_path=str(summary_path),
    )

    # 4) Build meshes
    _ = build_mesh_per_blob(
        str(windows_path),
        str(mesh_dir),
        str(mesh_summary),
        npad_x=args.npad_x,
        npad_y=args.npad_y,
        nz=args.nz,
        dz0=args.dz0,
    )

    # 5) Create parfiles
    par_paths = create_inversion_parfiles(str(mesh_summary), str(par_dir))

    print(f"mat: {mat_path}")
    print(f"out: {out_dir}")
    print(f"blobs: {len(blobs)}")
    print(f"parfiles: {len(par_paths)}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tomofast_x_mu", description="Tomofast-x-µ utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("version", help="Print package version")
    sp.set_defaults(func=_cmd_version)

    sp = sub.add_parser("diagram", help="Print path to architecture diagram markdown")
    sp.set_defaults(func=_cmd_diagram)

    sp = sub.add_parser("smoke", help="Run a lightweight pipeline smoke test")
    sp.add_argument("--mat", type=str, default=None, help="Path to input .mat (defaults to first in real_qdm_data)")
    sp.add_argument("--out", type=str, default=None, help="Output directory (default: ./tomofast_x_mu_smoke_out)")
    sp.add_argument(
        "--h-units",
        type=str,
        default="auto",
        choices=["m", "um", "auto"],
        help=(
            "Units of scalar h in the .mat file. "
            "Use 'auto' (default) to infer (common: step in m, h in um)."
        ),
    )

    sp.add_argument("--threshold-factor", type=float, default=2.75)
    sp.add_argument("--pad-x", type=int, default=40)
    sp.add_argument("--pad-y", type=int, default=10)
    sp.add_argument("--min-blob-cells", type=int, default=150)
    sp.add_argument("--max-blob-cells", type=int, default=100000)
    sp.add_argument("--nz-fixed", type=int, default=120)

    sp.add_argument("--npad-x", type=int, default=20)
    sp.add_argument("--npad-y", type=int, default=20)
    sp.add_argument("--nz", type=int, default=120)
    sp.add_argument("--dz0", type=float, default=4.4)

    sp.set_defaults(func=_cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
