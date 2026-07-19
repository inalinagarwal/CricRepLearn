"""Venue batting/bowling character for fantasy role tilt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def venue_scoring_profile(
    canonical_dir: Path,
    venue: str | None,
    *,
    min_balls: float = 3000.0,
) -> dict[str, Any]:
    """
    Compare venue SR / dismissal rate to global train.

    Returns ``tilt`` in roughly [-1, +1]:
      + → batting-favoured (prefer more BAT)
      - → bowling-favoured (prefer more BOWL)
    """
    if not venue:
        return {
            "venue": None,
            "tilt": 0.0,
            "label": "balanced",
            "target_roles": {"WK": 1, "BAT": 4, "AR": 2, "BOWL": 4},
            "constraints": {"min_bowl": 3, "max_bat": 5, "min_bat": 3, "max_bowl": 5},
            "note": "no venue → balanced target 1-4-2-4",
        }

    deliveries = _escape(canonical_dir / "deliveries.parquet")
    matches = _escape(canonical_dir / "matches.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    venue_esc = venue.replace("'", "''")
    connection = duckdb.connect()
    try:
        global_row = connection.execute(
            f"""
            SELECT
                AVG(d.runs_batter)::DOUBLE AS sr,
                AVG(CASE WHEN d.batter_dismissed THEN 1.0 ELSE 0.0 END)::DOUBLE AS dismiss
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
            """
        ).fetchone()
        venue_row = connection.execute(
            f"""
            SELECT
                COUNT(*)::DOUBLE AS balls,
                AVG(d.runs_batter)::DOUBLE AS sr,
                AVG(CASE WHEN d.batter_dismissed THEN 1.0 ELSE 0.0 END)::DOUBLE AS dismiss
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{matches}') m USING (match_id)
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
              AND (
                    m.venue ILIKE '%{venue_esc}%'
                 OR m.city ILIKE '%{venue_esc}%'
              )
            """
        ).fetchone()
    finally:
        connection.close()

    g_sr = float(global_row[0] or 1.2)
    g_dismiss = float(global_row[1] or 0.05)
    balls = float(venue_row[0] or 0.0)
    if balls < min_balls:
        # Try similar-condition cluster via venue_similarity if sparse.
        from cric_rep_learn.players.venue_similarity import resolve_venues

        resolved = resolve_venues(canonical_dir, venue, include_similar=True)
        return {
            "venue": venue,
            "tilt": 0.0,
            "label": "balanced",
            "balls": balls,
            "target_roles": {"WK": 1, "BAT": 4, "AR": 2, "BOWL": 4},
            "constraints": {"min_bowl": 3, "max_bat": 5, "min_bat": 3, "max_bowl": 5},
            "venue_resolution": resolved,
            "note": f"sparse venue evidence ({balls:.0f} balls) → balanced",
        }

    v_sr = float(venue_row[1])
    v_dismiss = float(venue_row[2])
    # Positive when scoring is high and wickets are scarce.
    sr_z = (v_sr - g_sr) / max(g_sr, 1e-6)
    dismiss_z = (v_dismiss - g_dismiss) / max(g_dismiss, 1e-6)
    tilt = float(max(-1.0, min(1.0, 2.5 * sr_z - 2.0 * dismiss_z)))

    if tilt >= 0.25:
        label = "batting"
        target = {"WK": 1, "BAT": 5, "AR": 2, "BOWL": 3}
        constraints = {"min_bowl": 3, "max_bat": 6, "min_bat": 4, "max_bowl": 4}
    elif tilt <= -0.25:
        label = "bowling"
        target = {"WK": 1, "BAT": 3, "AR": 2, "BOWL": 5}
        constraints = {"min_bowl": 4, "max_bat": 4, "min_bat": 3, "max_bowl": 5}
    else:
        label = "balanced"
        target = {"WK": 1, "BAT": 4, "AR": 2, "BOWL": 4}
        constraints = {"min_bowl": 3, "max_bat": 5, "min_bat": 3, "max_bowl": 5}

    return {
        "venue": venue,
        "tilt": tilt,
        "label": label,
        "balls": balls,
        "venue_sr": v_sr,
        "venue_dismiss": v_dismiss,
        "global_sr": g_sr,
        "global_dismiss": g_dismiss,
        "target_roles": target,
        "constraints": constraints,
        "note": (
            f"{label} venue (tilt={tilt:+.2f}): target "
            f"{target['WK']}-{target['BAT']}-{target['AR']}-{target['BOWL']} "
            f"(WK-BAT-AR-BOWL)"
        ),
    }
