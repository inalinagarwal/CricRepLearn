"""Partnership (striker × non-striker) rate tilts from co-batter familiarity."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_partnership_index(
    co_batters_path: Path | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """
    Map unordered (player_a, player_b) → familiarity stats.

    Familiar pairs get a mild SR lift and dismissal dampening in the sim.
    """
    path = co_batters_path or Path("artifacts/co-batters/co_batters.parquet")
    if not path.exists():
        return {}
    frame = pq.read_table(path).to_pandas()
    index: dict[tuple[str, str], dict[str, float]] = {}
    for row in frame.to_dict(orient="records"):
        a, b = str(row["player_a"]), str(row["player_b"])
        balls = float(row["balls_together"])
        familiar = min(1.0, math.log1p(balls) / math.log1p(400.0))
        index[(a, b)] = {
            "balls_together": balls,
            "matches_together": float(row.get("matches_together") or 0),
            "familiarity": familiar,
            "sr_mult": 1.0 + 0.05 * familiar,
            "dismiss_mult": 1.0 - 0.04 * familiar,
        }
    return index


def partnership_tilt(
    striker_id: str,
    non_striker_id: str,
    *,
    index: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    if not index or not striker_id or not non_striker_id:
        return {"sr_mult": 1.0, "dismiss_mult": 1.0, "familiarity": 0.0, "balls_together": 0.0}
    key = (min(striker_id, non_striker_id), max(striker_id, non_striker_id))
    stats = index.get(key)
    if not stats:
        return {"sr_mult": 1.0, "dismiss_mult": 1.0, "familiarity": 0.0, "balls_together": 0.0}
    return {
        "sr_mult": float(stats["sr_mult"]),
        "dismiss_mult": float(stats["dismiss_mult"]),
        "familiarity": float(stats["familiarity"]),
        "balls_together": float(stats["balls_together"]),
    }
