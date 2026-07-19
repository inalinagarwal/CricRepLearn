"""Bowling attack scheduling across powerplay / middle / death."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


@dataclass
class BowlerSpell:
    player_id: str
    player_name: str
    max_overs: int = 4
    # Higher score => prefer bowling this phase (from train wicket/economy).
    phase_scores: dict[str, float] = field(default_factory=dict)
    phase_evidence: dict[str, dict[str, float]] = field(default_factory=dict)


def load_bowler_phase_profiles(
    canonical_dir: Path,
    bowler_ids: list[str],
    *,
    min_balls: int = 24,
    strength: float = 80.0,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Train bowling effectiveness by phase.

    score = shrunk_wicket_rate / (shrunk_sr_conceded + 0.35)
    Higher score => better to use in that phase (e.g. Bumrah at death).
    """
    if not bowler_ids:
        return {}
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    id_list = ", ".join(f"'{bid}'" for bid in bowler_ids)
    connection = duckdb.connect()
    try:
        globals_ = connection.execute(
            f"""
            SELECT
                d.phase,
                SUM(d.runs_batter)::DOUBLE / COUNT(*) AS sr,
                SUM(d.bowler_wicket_count)::DOUBLE / COUNT(*) AS wicket_rate
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
              AND d.phase IN ('powerplay', 'middle', 'death')
            GROUP BY 1
            """
        ).fetchdf()
        global_map = {
            row["phase"]: {
                "sr": float(row["sr"]),
                "wicket_rate": float(row["wicket_rate"]),
            }
            for row in globals_.to_dict(orient="records")
        }
        frame = connection.execute(
            f"""
            SELECT
                d.bowler_id,
                d.phase,
                SUM(d.runs_batter)::DOUBLE AS runs,
                COUNT(*)::DOUBLE AS balls,
                SUM(d.bowler_wicket_count)::DOUBLE AS wickets
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
              AND (d.is_legal OR d.extras_noballs > 0)
              AND d.phase IN ('powerplay', 'middle', 'death')
              AND d.bowler_id IN ({id_list})
            GROUP BY 1, 2
            """
        ).fetchdf()
    finally:
        connection.close()

    out: dict[str, dict[str, dict[str, float]]] = {bid: {} for bid in bowler_ids}
    for row in frame.to_dict(orient="records"):
        phase = row["phase"]
        balls = float(row["balls"])
        g = global_map.get(phase, {"sr": 1.2, "wicket_rate": 0.05})
        # Empirical Bayes shrink toward phase global.
        sr = (float(row["runs"]) + strength * g["sr"]) / (balls + strength)
        wicket_rate = (float(row["wickets"]) + strength * g["wicket_rate"]) / (
            balls + strength
        )
        score = wicket_rate / (sr + 0.35)
        out[row["bowler_id"]][phase] = {
            "balls": balls,
            "raw_sr": float(row["runs"] / balls) if balls else g["sr"],
            "raw_wicket_rate": float(row["wickets"] / balls) if balls else g["wicket_rate"],
            "sr": sr,
            "wicket_rate": wicket_rate,
            "score": score,
            "enough_evidence": float(balls >= min_balls),
        }
    # Fill missing phases with global-only scores.
    for bid in bowler_ids:
        for phase, g in global_map.items():
            if phase not in out[bid]:
                score = g["wicket_rate"] / (g["sr"] + 0.35)
                out[bid][phase] = {
                    "balls": 0.0,
                    "raw_sr": g["sr"],
                    "raw_wicket_rate": g["wicket_rate"],
                    "sr": g["sr"],
                    "wicket_rate": g["wicket_rate"],
                    "score": score,
                    "enough_evidence": 0.0,
                }
    return out


def attach_phase_profiles(
    attack: list[BowlerSpell],
    profiles: dict[str, dict[str, dict[str, float]]],
) -> list[BowlerSpell]:
    for bowler in attack:
        profile = profiles.get(bowler.player_id, {})
        bowler.phase_scores = {
            phase: float(stats["score"]) for phase, stats in profile.items()
        }
        bowler.phase_evidence = profile
    return attack


def build_over_schedule(
    attack: list[BowlerSpell],
    *,
    n_overs: int = 20,
) -> list[dict[str, Any]]:
    """
    Assign each over to a bowler under T20 constraints (max 4 overs).

    Ranking is by empirical phase score (death specialists like Bumrah rise
    for overs 16–19), not by list order heuristics.
    """
    if not attack:
        raise ValueError("attack must be non-empty")
    remaining = {b.player_id: b.max_overs for b in attack}
    by_id = {b.player_id: b for b in attack}
    schedule: list[dict[str, Any] | None] = [None] * n_overs

    def phase_for_over(over: int) -> str:
        if over < 6:
            return "powerplay"
        if over >= 16:
            return "death"
        return "middle"

    def pick(over: int, phase: str) -> str:
        ranked = sorted(
            attack,
            key=lambda b: (
                -b.phase_scores.get(phase, 0.0),
                -remaining[b.player_id],
                b.player_name,
            ),
        )
        prev = schedule[over - 1]["bowler_id"] if over > 0 and schedule[over - 1] else None
        for cand in ranked:
            if remaining[cand.player_id] <= 0:
                continue
            if cand.player_id == prev and any(
                remaining[b.player_id] > 0 and b.player_id != prev for b in attack
            ):
                continue
            return cand.player_id
        for cand in ranked:
            if remaining[cand.player_id] > 0:
                return cand.player_id
        raise RuntimeError("no overs remaining to allocate")

    for phase in ("death", "powerplay", "middle"):
        for over in range(n_overs):
            if schedule[over] is not None:
                continue
            if phase_for_over(over) != phase:
                continue
            bowler_id = pick(over, phase)
            remaining[bowler_id] -= 1
            b = by_id[bowler_id]
            schedule[over] = {
                "over": over,
                "phase": phase,
                "bowler_id": bowler_id,
                "bowler_name": b.player_name,
                "phase_score": b.phase_scores.get(phase),
            }

    assert all(slot is not None for slot in schedule)
    return [slot for slot in schedule if slot is not None]


def ball_bowler_schedule(over_schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand overs into 120 legal-ball slots."""
    balls: list[dict[str, Any]] = []
    for over in over_schedule:
        for ball_in_over in range(6):
            balls.append(
                {
                    **over,
                    "ball_in_over": ball_in_over,
                    "legal_balls_before": over["over"] * 6 + ball_in_over,
                }
            )
    return balls
