"""Train-calibrated run outcome sampler P(0/1/2/4/6 | SR bucket)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


OUTCOMES = [0, 1, 2, 4, 6]
DEFAULT_PATH = Path("artifacts/baselines/run_outcome_by_sr.json")

# Fallback if artifact missing (old heuristic template).
_FALLBACK = {
    "sr_lt_0.8": [0.50, 0.35, 0.08, 0.05, 0.02],
    "sr_0.8_1.1": [0.42, 0.36, 0.10, 0.08, 0.04],
    "sr_1.1_1.4": [0.36, 0.34, 0.12, 0.12, 0.06],
    "sr_1.4_1.8": [0.30, 0.30, 0.12, 0.18, 0.10],
    "sr_ge_1.8": [0.24, 0.26, 0.12, 0.22, 0.16],
}


def sr_bucket(sr: float) -> str:
    if sr < 0.8:
        return "sr_lt_0.8"
    if sr < 1.1:
        return "sr_0.8_1.1"
    if sr < 1.4:
        return "sr_1.1_1.4"
    if sr < 1.8:
        return "sr_1.4_1.8"
    return "sr_ge_1.8"


def build_run_outcome_table(canonical_dir: Path) -> dict[str, Any]:
    deliveries = str((canonical_dir / "deliveries.parquet").resolve()).replace("'", "''")
    split = str((canonical_dir / "split_manifest.parquet").resolve()).replace("'", "''")
    connection = duckdb.connect()
    try:
        # Bucket each faced ball by innings SR before the delivery.
        frame = connection.execute(
            f"""
            WITH faced AS (
                SELECT
                    CASE
                        WHEN d.legal_balls_before <= 0 THEN 1.2
                        ELSE d.score_before::DOUBLE / d.legal_balls_before
                    END AS innings_sr,
                    d.runs_batter
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                  AND d.runs_batter IN (0, 1, 2, 4, 6)
            ),
            labeled AS (
                SELECT
                    CASE
                        WHEN innings_sr < 0.8 THEN 'sr_lt_0.8'
                        WHEN innings_sr < 1.1 THEN 'sr_0.8_1.1'
                        WHEN innings_sr < 1.4 THEN 'sr_1.1_1.4'
                        WHEN innings_sr < 1.8 THEN 'sr_1.4_1.8'
                        ELSE 'sr_ge_1.8'
                    END AS bucket,
                    runs_batter
                FROM faced
            )
            SELECT
                bucket,
                runs_batter,
                COUNT(*)::DOUBLE AS n
            FROM labeled
            GROUP BY 1, 2
            """
        ).fetchdf()
    finally:
        connection.close()

    tables: dict[str, list[float]] = {}
    for bucket in _FALLBACK:
        counts = {o: 0.0 for o in OUTCOMES}
        sub = frame[frame["bucket"] == bucket]
        for row in sub.to_dict(orient="records"):
            counts[int(row["runs_batter"])] = float(row["n"])
        total = sum(counts.values())
        if total < 500:
            tables[bucket] = list(_FALLBACK[bucket])
        else:
            tables[bucket] = [counts[o] / total for o in OUTCOMES]
    return {
        "outcomes": OUTCOMES,
        "buckets": tables,
        "method": "P(runs|innings_sr_before bucket) on train faced legal balls",
    }


def save_run_outcome_table(canonical_dir: Path, path: Path = DEFAULT_PATH) -> dict[str, Any]:
    table = build_run_outcome_table(canonical_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return table


def load_run_outcome_table(
    path: Path | None = None,
    *,
    canonical_dir: Path | None = None,
) -> dict[str, Any]:
    path = path or DEFAULT_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if canonical_dir is None:
        canonical_dir = Path("artifacts/canonical")
    return save_run_outcome_table(canonical_dir, path)


_TABLE: dict[str, Any] | None = None


def sample_runs(
    rng: np.random.Generator,
    expected_sr: float,
    *,
    table: dict[str, Any] | None = None,
) -> int:
    """Draw discrete runs with mean nudged toward expected_sr."""
    global _TABLE
    if table is None:
        if _TABLE is None:
            try:
                _TABLE = load_run_outcome_table()
            except Exception:  # noqa: BLE001
                _TABLE = {"outcomes": OUTCOMES, "buckets": _FALLBACK}
        table = _TABLE
    sr = float(max(0.05, min(expected_sr, 3.0)))
    bucket = sr_bucket(sr)
    probs = np.asarray(
        (table.get("buckets") or _FALLBACK).get(bucket, _FALLBACK["sr_1.1_1.4"]),
        dtype=np.float64,
    )
    probs = probs / probs.sum()
    draw = int(rng.choice(OUTCOMES, p=probs))
    mean = float(np.dot(probs, OUTCOMES))
    if mean < sr - 0.15 and draw in (0, 1) and rng.random() < min(0.45, sr - mean):
        draw = 4 if rng.random() < 0.55 else 6
    elif mean > sr + 0.25 and draw in (4, 6) and rng.random() < min(0.45, mean - sr):
        draw = 1 if rng.random() < 0.7 else 0
    return draw
