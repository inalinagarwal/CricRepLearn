"""Smoke tests for the 3-mode app entrypoints."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_help() -> None:
    from cric_rep_learn.app.cli import main
    import sys

    with pytest.raises(SystemExit) as exc:
        sys.argv = ["cric", "--help"]
        main()
    assert exc.value.code == 0


def test_player_dive_smoke() -> None:
    canonical = Path("artifacts/canonical")
    if not (canonical / "deliveries.parquet").exists():
        pytest.skip("canonical artifacts missing")
    from cric_rep_learn.app.services import run_player_dive

    result = run_player_dive(
        batter="Chris Gayle",
        bowlers="Mohammad Hafeez,Wahab Riaz,Shaheen Shah Afridi,Shadab Khan,Haris Rauf",
        venue="Rawalpindi",
    )
    assert result["mode"] == "player_dive"
    assert result.get("expected_runs") is not None
    assert result.get("attack")
