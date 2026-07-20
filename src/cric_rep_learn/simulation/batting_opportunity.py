"""Batting-order opportunity tilt from train balls-faced shares."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


_ORDER_SHARE_CACHE: dict[str, list[float]] = {}

# Soft blend toward empirical order shares (0 = keep MC, 1 = full prior).
ORDER_BLEND = 0.35


def load_batting_order_ball_shares(
    canonical_dir: Path,
) -> list[float]:
    """
    Train mean share of faced balls by batting position (1..11).

    Position = order of first appearance in the innings among batters who faced.
    """
    key = str(canonical_dir.resolve())
    if key in _ORDER_SHARE_CACHE:
        return _ORDER_SHARE_CACHE[key]

    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            WITH faced AS (
                SELECT
                    d.match_id,
                    d.innings,
                    d.batter_id,
                    MIN(d.attempt_index_in_innings) AS first_ball,
                    SUM(CASE WHEN d.is_legal OR d.extras_noballs > 0 THEN 1 ELSE 0 END)
                        AS balls
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND d.innings IN (1, 2)
                  AND (d.is_legal OR d.extras_noballs > 0)
                GROUP BY 1, 2, 3
            ),
            ordered AS (
                SELECT
                    match_id,
                    innings,
                    batter_id,
                    balls,
                    ROW_NUMBER() OVER (
                        PARTITION BY match_id, innings
                        ORDER BY first_ball
                    ) AS batting_order
                FROM faced
            )
            SELECT
                batting_order,
                AVG(balls)::DOUBLE AS mean_balls
            FROM ordered
            WHERE batting_order BETWEEN 1 AND 11
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
    finally:
        connection.close()

    means = [0.0] * 11
    for row in frame.to_dict(orient="records"):
        idx = int(row["batting_order"]) - 1
        if 0 <= idx < 11:
            means[idx] = float(row["mean_balls"])
    total = sum(means)
    if total <= 0:
        shares = [1.0 / 11.0] * 11
    else:
        shares = [m / total for m in means]
    _ORDER_SHARE_CACHE[key] = shares
    return shares


def apply_batting_order_opportunity(
    batters: list[dict[str, Any]],
    *,
    shares: list[float] | None = None,
    blend: float = ORDER_BLEND,
) -> list[dict[str, Any]]:
    """
    Soft-redistribute expected balls (and scale runs/boundaries) toward train
    batting-order shares while preserving innings total balls and total runs.
    """
    if not batters or blend <= 0:
        return batters
    n = len(batters)
    shares = shares or ([1.0 / n] * n)
    shares = list(shares[:n]) + [0.0] * max(0, n - len(shares))
    share_sum = sum(shares) or 1.0
    shares = [s / share_sum for s in shares]

    balls = np.asarray([float(b.get("expected_balls") or 0.0) for b in batters])
    runs = np.asarray([float(b.get("expected_runs") or 0.0) for b in batters])
    fours = np.asarray([float(b.get("expected_fours") or 0.0) for b in batters])
    sixes = np.asarray([float(b.get("expected_sixes") or 0.0) for b in batters])
    total_balls = float(balls.sum())
    if total_balls <= 1e-9:
        return batters

    target = np.asarray(shares) * total_balls
    blended = (1.0 - blend) * balls + blend * target
    # Preserve total balls exactly.
    blended *= total_balls / max(float(blended.sum()), 1e-9)

    out: list[dict[str, Any]] = []
    for i, row in enumerate(batters):
        old_b = float(balls[i])
        new_b = float(blended[i])
        scale = (new_b / old_b) if old_b > 1e-9 else 1.0
        # Keep SR roughly stable when reallocating opportunity.
        new_runs = float(runs[i] * scale)
        new_fours = float(fours[i] * scale)
        new_sixes = float(sixes[i] * scale)
        updated = {
            **row,
            "expected_balls": new_b,
            "expected_runs": new_runs,
            "expected_fours": new_fours,
            "expected_sixes": new_sixes,
            "batting_order": i + 1,
            "balls_order_scale": scale,
        }
        # Scale run quantiles if present.
        for key in ("runs_p10", "runs_p50", "runs_p90"):
            if row.get(key) is not None:
                updated[key] = float(row[key]) * scale
        for key in ("balls_p10", "balls_p50", "balls_p90"):
            if row.get(key) is not None:
                updated[key] = float(row[key]) * scale
        out.append(updated)

    # Renormalize runs to preserve innings total (avoid score inflation).
    new_runs_sum = sum(float(b["expected_runs"]) for b in out)
    old_runs_sum = float(runs.sum())
    if new_runs_sum > 1e-9 and old_runs_sum > 1e-9:
        rscale = old_runs_sum / new_runs_sum
        for b in out:
            b["expected_runs"] = float(b["expected_runs"]) * rscale
            b["expected_fours"] = float(b["expected_fours"]) * rscale
            b["expected_sixes"] = float(b["expected_sixes"]) * rscale
            for key in ("runs_p10", "runs_p50", "runs_p90"):
                if b.get(key) is not None:
                    b[key] = float(b[key]) * rscale
    return out
