"""Tests for the tomofast_x_mu CLI."""

from __future__ import annotations

from pathlib import Path

from tomofast_x_mu.cli import main


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not locate repo root")


def test_cli_version_prints_something(capsys):
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out


def test_cli_diagram_prints_existing_path(capsys, monkeypatch):
    repo_root = _find_repo_root(Path(__file__))
    monkeypatch.chdir(repo_root)

    rc = main(["diagram"])
    assert rc == 0

    out = capsys.readouterr().out.strip()
    p = Path(out)
    assert p.exists()
