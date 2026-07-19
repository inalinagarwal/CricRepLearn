"""Chase-target pressure and chasing-team confidence from train outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def rrr_bucket(required_rate: float) -> str:
    if required_rate < 1.0:
        return "rr_lt_1"
    if required_rate < 1.5:
        return "rr_1_1.5"
    if required_rate < 2.0:
        return "rr_1.5_2"
    if required_rate < 2.5:
        return "rr_2_2.5"
    return "rr_ge_2.5"


def wicket_bucket(wickets: int) -> str:
    if wickets <= 2:
        return "w0_2"
    if wickets <= 5:
        return "w3_5"
    if wickets <= 7:
        return "w6_7"
    return "w8_plus"


def build_chase_impacts(canonical_dir: Path) -> dict[str, Any]:
    """
    Estimate chase SR / dismissal multipliers and win-confidence by state.

    State = (required_run_rate bucket, wickets-down bucket). Confidence is the
    empirical P(chase wins | state) on train matches.
    """
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    matches = _escape(canonical_dir / "matches.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    connection = duckdb.connect()
    try:
        baseline = connection.execute(
            f"""
            SELECT
                AVG(d.runs_batter)::DOUBLE AS sr,
                AVG(CASE WHEN d.batter_dismissed THEN 1.0 ELSE 0.0 END)::DOUBLE AS dismiss
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND d.innings > 1
              AND d.target_runs IS NOT NULL
              AND (d.is_legal OR d.extras_noballs > 0)
            """
        ).fetchone()
        g_sr = float(baseline[0] or 1.2)
        g_dismiss = float(baseline[1] or 0.05)

        frame = connection.execute(
            f"""
            WITH chase_balls AS (
                SELECT
                    d.match_id,
                    d.batting_team,
                    d.legal_balls_before,
                    d.score_before,
                    d.wickets_before,
                    d.target_runs,
                    d.scheduled_balls,
                    d.runs_batter,
                    d.batter_dismissed,
                    CASE
                        WHEN (d.scheduled_balls - d.legal_balls_before) <= 0 THEN NULL
                        ELSE (d.target_runs - d.score_before)::DOUBLE
                             / (d.scheduled_balls - d.legal_balls_before)
                    END AS required_rate
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND d.innings > 1
                  AND d.target_runs IS NOT NULL
                  AND (d.is_legal OR d.extras_noballs > 0)
            ),
            winners AS (
                SELECT match_id, winner
                FROM read_parquet('{matches}')
            ),
            labeled AS (
                SELECT
                    c.*,
                    CASE
                        WHEN c.required_rate IS NULL THEN NULL
                        WHEN c.required_rate < 1.0 THEN 'rr_lt_1'
                        WHEN c.required_rate < 1.5 THEN 'rr_1_1.5'
                        WHEN c.required_rate < 2.0 THEN 'rr_1.5_2'
                        WHEN c.required_rate < 2.5 THEN 'rr_2_2.5'
                        ELSE 'rr_ge_2.5'
                    END AS rrr_bucket,
                    CASE
                        WHEN c.wickets_before <= 2 THEN 'w0_2'
                        WHEN c.wickets_before <= 5 THEN 'w3_5'
                        WHEN c.wickets_before <= 7 THEN 'w6_7'
                        ELSE 'w8_plus'
                    END AS wicket_bucket,
                    CASE WHEN w.winner = c.batting_team THEN 1.0 ELSE 0.0 END AS chase_won
                FROM chase_balls c
                JOIN winners w USING (match_id)
                WHERE c.required_rate IS NOT NULL
                  AND c.required_rate >= 0
            )
            SELECT
                rrr_bucket,
                wicket_bucket,
                COUNT(*)::DOUBLE AS balls,
                AVG(runs_batter)::DOUBLE AS sr,
                AVG(CASE WHEN batter_dismissed THEN 1.0 ELSE 0.0 END)::DOUBLE AS dismiss,
                AVG(chase_won)::DOUBLE AS win_confidence,
                COUNT(DISTINCT match_id)::DOUBLE AS matches
            FROM labeled
            GROUP BY 1, 2
            HAVING COUNT(*) >= 2000
            """
        ).fetchdf()
    finally:
        connection.close()

    cells: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = f"{row['rrr_bucket']}|{row['wicket_bucket']}"
        cells[key] = {
            "rrr_bucket": row["rrr_bucket"],
            "wicket_bucket": row["wicket_bucket"],
            "balls": float(row["balls"]),
            "matches": float(row["matches"]),
            "sr": float(row["sr"]),
            "dismissal_rate": float(row["dismiss"]),
            "sr_mult": float(row["sr"] / g_sr) if g_sr else 1.0,
            "dismiss_mult": float(row["dismiss"] / g_dismiss) if g_dismiss else 1.0,
            "win_confidence": float(row["win_confidence"]),
        }

    # Marginal RRR-only fallbacks.
    rrr_only: dict[str, dict[str, float]] = {}
    for row in frame.to_dict(orient="records"):
        bucket = row["rrr_bucket"]
        slot = rrr_only.setdefault(
            bucket, {"balls": 0.0, "runs": 0.0, "dismissals": 0.0, "wins": 0.0}
        )
        balls = float(row["balls"])
        slot["balls"] += balls
        slot["runs"] += float(row["sr"]) * balls
        slot["dismissals"] += float(row["dismiss"]) * balls
        slot["wins"] += float(row["win_confidence"]) * balls
    for bucket, slot in rrr_only.items():
        balls = slot["balls"]
        sr = slot["runs"] / balls
        dismiss = slot["dismissals"] / balls
        rrr_only[bucket] = {
            "balls": balls,
            "sr": sr,
            "dismissal_rate": dismiss,
            "sr_mult": sr / g_sr if g_sr else 1.0,
            "dismiss_mult": dismiss / g_dismiss if g_dismiss else 1.0,
            "win_confidence": slot["wins"] / balls,
        }

    return {
        "baseline": {"sr": g_sr, "dismissal_rate": g_dismiss},
        "cells": cells,
        "rrr_marginal": rrr_only,
        "method": (
            "Train chase deliveries: multipliers vs chase baseline by "
            "(required_rate bucket × wickets). win_confidence = P(chasing team wins | state)."
        ),
    }


def save_chase_impacts(canonical_dir: Path, output_path: Path) -> dict[str, Any]:
    impacts = build_chase_impacts(canonical_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(impacts, indent=2) + "\n", encoding="utf-8")
    return impacts


def load_chase_impacts(
    path: Path | None = None,
    *,
    canonical_dir: Path | None = None,
) -> dict[str, Any]:
    default = Path("artifacts/baselines/chase_impacts.json")
    path = path or default
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if canonical_dir is None:
        canonical_dir = Path("artifacts/canonical")
    return save_chase_impacts(canonical_dir, path)


def apply_chase_pressure(
    *,
    sr: float,
    dismissal_rate: float,
    target: float,
    score: float,
    legal_balls: int,
    wickets: int,
    scheduled_balls: int = 120,
    impacts: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Tilt rates using historical chase state; return confidence score.

    Also nudges SR toward the required rate when confidence is low and RRR is high.
    """
    balls_left = max(scheduled_balls - legal_balls, 0)
    runs_needed = target - score
    if impacts is None or balls_left <= 0:
        return {
            "expected_sr": sr,
            "dismissal_rate": dismissal_rate,
            "win_confidence": None,
            "required_rate": None,
            "runs_needed": runs_needed,
            "balls_left": balls_left,
            "notes": [],
        }
    if runs_needed <= 0:
        return {
            "expected_sr": sr,
            "dismissal_rate": dismissal_rate,
            "win_confidence": 1.0,
            "required_rate": 0.0,
            "runs_needed": runs_needed,
            "balls_left": balls_left,
            "notes": ["target_reached"],
        }

    required_rate = runs_needed / balls_left
    rr = rrr_bucket(required_rate)
    wk = wicket_bucket(wickets)
    cell = (impacts.get("cells") or {}).get(f"{rr}|{wk}")
    marg = (impacts.get("rrr_marginal") or {}).get(rr)
    stats = cell or marg
    notes: list[str] = []
    conf = None
    if stats:
        sr *= float(stats.get("sr_mult", 1.0))
        dismissal_rate *= float(stats.get("dismiss_mult", 1.0))
        conf = float(stats.get("win_confidence", 0.5))
        notes.append(
            f"chase[{rr}|{wk}] conf={conf:.2f} "
            f"sr×{stats.get('sr_mult', 1):.3f} out×{stats.get('dismiss_mult', 1):.3f}"
        )
        # Low-confidence chases: push SR toward required rate (capped).
        if conf < 0.45 and required_rate > sr:
            blend = min(0.55, (0.45 - conf) * 1.2)
            target_sr = min(required_rate, 2.8)
            sr = (1 - blend) * sr + blend * target_sr
            notes.append(f"aggression_blend={blend:.2f}")

    return {
        "expected_sr": float(max(sr, 0.05)),
        "dismissal_rate": float(min(max(dismissal_rate, 1e-4), 0.40)),
        "win_confidence": conf,
        "required_rate": float(required_rate),
        "runs_needed": float(runs_needed),
        "balls_left": int(balls_left),
        "rrr_bucket": rr,
        "wicket_bucket": wk,
        "notes": notes,
    }
