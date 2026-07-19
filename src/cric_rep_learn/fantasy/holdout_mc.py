"""Reconstruct holdout lineups and predict fantasy stats via short HB MC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from cric_rep_learn.data.player_attributes import load_attributes_index
from cric_rep_learn.simulation.attack import (
    BowlerSpell,
    assign_over_quotas_from_actual,
    configure_attack,
    load_bowler_phase_profiles,
)
from cric_rep_learn.simulation.chase import load_chase_impacts
from cric_rep_learn.simulation.innings import simulate_innings
from cric_rep_learn.simulation.partnership import load_partnership_index
from cric_rep_learn.simulation.priors import InningsRateModel


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


@dataclass(frozen=True, slots=True)
class HoldoutMatchSetup:
    match_id: str
    venue: str | None
    match_date: str | None
    first_team: str
    chase_team: str
    first_lineup: list[dict[str, str]]
    chase_lineup: list[dict[str, str]]
    first_attack: list[BowlerSpell]
    chase_attack: list[BowlerSpell]


def reconstruct_holdout_matches(
    canonical_dir: Path,
    *,
    splits: tuple[str, ...] = ("validation",),
    match_ids: list[str] | None = None,
    max_matches: int = 100,
    seed: int = 7,
    min_batters: int = 8,
    min_bowlers: int = 5,
    attributes: dict[str, dict[str, Any]] | None = None,
    opportunity: str = "scheduled",
) -> list[HoldoutMatchSetup]:
    """
    Rebuild XI batting order (first appearance) and bowling attacks from
    faced balls / legal overs on holdout matches.

    ``opportunity``:
      - ``scheduled`` (default): top-5 by legal balls + cricket-aware list-order
        quotas (production Dream XI path).
      - ``actual``: include all bowlers who bowled; set ``max_overs`` from
        observed overs (capped at 4, padded/trimmed). Phase/death rules still
        schedule overs within those quotas.

    Require ≥5 bowlers so quotas can cover 20 overs (max 4 each).
    """
    if opportunity not in {"actual", "scheduled"}:
        raise ValueError("opportunity must be 'actual' or 'scheduled'")
    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    matches = _escape(canonical_dir / "matches.parquet")
    players = _escape(canonical_dir / "match_players.parquet")
    split_list = ", ".join(f"'{s}'" for s in splits)
    connection = duckdb.connect()
    try:
        meta = connection.execute(
            f"""
            SELECT m.match_id, m.venue, CAST(m.match_date AS VARCHAR) AS match_date
            FROM read_parquet('{matches}') m
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split IN ({split_list})
            """
        ).fetchdf()
        bat_order = connection.execute(
            f"""
            WITH faced AS (
                SELECT
                    d.match_id,
                    d.innings,
                    d.batting_team AS team,
                    d.batter_id AS player_id,
                    any_value(d.batter_name) AS player_name,
                    MIN(d.attempt_index_in_innings) AS first_ball
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split IN ({split_list})
                  AND NOT d.is_super_over
                  AND d.innings IN (1, 2)
                  AND (d.is_legal OR d.extras_noballs > 0)
                GROUP BY 1, 2, 3, 4
            )
            SELECT * FROM faced
            ORDER BY match_id, innings, first_ball
            """
        ).fetchdf()
        bowl_order = connection.execute(
            f"""
            SELECT
                d.match_id,
                d.innings,
                d.bowling_team AS team,
                d.bowler_id AS player_id,
                any_value(d.bowler_name) AS player_name,
                SUM(CASE WHEN d.is_legal THEN 1 ELSE 0 END)::DOUBLE AS legal_balls
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split IN ({split_list})
              AND NOT d.is_super_over
              AND d.innings IN (1, 2)
            GROUP BY 1, 2, 3, 4
            HAVING SUM(CASE WHEN d.is_legal THEN 1 ELSE 0 END) > 0
            ORDER BY match_id, innings, legal_balls DESC
            """
        ).fetchdf()
        squad = connection.execute(
            f"""
            SELECT match_id, team, player_id, player_name
            FROM read_parquet('{players}')
            WHERE listed_in_match_squad
            """
        ).fetchdf()
    finally:
        connection.close()

    meta_by = {row["match_id"]: row for row in meta.to_dict(orient="records")}
    candidate_ids = sorted(meta_by.keys())
    if match_ids is not None:
        keep = set(match_ids)
        candidate_ids = [m for m in candidate_ids if m in keep]

    bat_by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in bat_order.to_dict(orient="records"):
        bat_by.setdefault((r["match_id"], int(r["innings"])), []).append(r)
    bowl_by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in bowl_order.to_dict(orient="records"):
        bowl_by.setdefault((r["match_id"], int(r["innings"])), []).append(r)
    squad_by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in squad.to_dict(orient="records"):
        squad_by.setdefault((r["match_id"], r["team"]), []).append(r)

    complete: list[str] = []
    for mid in candidate_ids:
        b1 = bat_by.get((mid, 1), [])
        b2 = bat_by.get((mid, 2), [])
        w1 = bowl_by.get((mid, 1), [])
        w2 = bowl_by.get((mid, 2), [])
        if (
            len(b1) >= min_batters
            and len(b2) >= min_batters
            and len(w1) >= min_bowlers
            and len(w2) >= min_bowlers
        ):
            complete.append(mid)
    if not complete:
        complete = candidate_ids

    rng = np.random.default_rng(seed)
    if len(complete) > max_matches:
        complete = sorted(rng.choice(complete, size=max_matches, replace=False).tolist())

    all_bowler_ids: set[str] = set()
    for mid in complete:
        for inn in (1, 2):
            rows = bowl_by.get((mid, inn), [])
            take = rows if opportunity == "actual" else rows[:5]
            for row in take:
                all_bowler_ids.add(str(row["player_id"]))
    profiles = load_bowler_phase_profiles(canonical_dir, sorted(all_bowler_ids))

    def _lineup(
        mid: str, rows: list[dict[str, Any]], team: str
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            pid = str(row["player_id"])
            if pid in seen:
                continue
            seen.add(pid)
            out.append(
                {
                    "player_id": pid,
                    "player_name": str(row["player_name"]),
                    "batting_hand": "unknown",
                }
            )
            if len(out) >= 11:
                break
        if len(out) < 11:
            for row in squad_by.get((mid, team), []):
                pid = str(row["player_id"])
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(
                    {
                        "player_id": pid,
                        "player_name": str(row["player_name"]),
                        "batting_hand": "unknown",
                    }
                )
                if len(out) >= 11:
                    break
        return out

    def _attack(rows: list[dict[str, Any]]) -> list[BowlerSpell]:
        if opportunity == "scheduled":
            attack = [
                BowlerSpell(
                    player_id=str(row["player_id"]),
                    player_name=str(row["player_name"]),
                )
                for row in rows[:5]
            ]
            return configure_attack(
                attack, profiles=profiles, attributes=attributes, assign_quotas=True
            )
        # Actual opportunity: all bowlers who bowled, quotas from overs.
        attack = [
            BowlerSpell(
                player_id=str(row["player_id"]),
                player_name=str(row["player_name"]),
            )
            for row in rows
        ]
        actual = {
            str(row["player_id"]): float(row["legal_balls"]) / 6.0 for row in rows
        }
        assign_over_quotas_from_actual(attack, actual)
        return configure_attack(
            attack, profiles=profiles, attributes=attributes, assign_quotas=False
        )

    setups: list[HoldoutMatchSetup] = []
    for mid in complete:
        info = meta_by[mid]
        b1 = bat_by.get((mid, 1), [])
        b2 = bat_by.get((mid, 2), [])
        w1 = bowl_by.get((mid, 1), [])
        w2 = bowl_by.get((mid, 2), [])
        if not b1 or not b2 or not w1 or not w2:
            continue
        first_team = str(b1[0]["team"])
        chase_team = str(b2[0]["team"])
        first_lineup = _lineup(mid, b1, first_team)
        chase_lineup = _lineup(mid, b2, chase_team)
        if len(first_lineup) < 2 or len(chase_lineup) < 2:
            continue
        first_attack = _attack(w1)
        chase_attack = _attack(w2)
        if not first_attack or not chase_attack:
            continue
        setups.append(
            HoldoutMatchSetup(
                match_id=mid,
                venue=None if info.get("venue") is None else str(info["venue"]),
                match_date=None
                if info.get("match_date") is None
                else str(info["match_date"]),
                first_team=first_team,
                chase_team=chase_team,
                first_lineup=first_lineup,
                chase_lineup=chase_lineup,
                first_attack=first_attack,
                chase_attack=chase_attack,
            )
        )
    return setups


def _attach_hands(
    lineup: list[dict[str, str]],
    attributes: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    out = []
    for row in lineup:
        attrs = attributes.get(row["player_id"]) or {}
        out.append(
            {
                **row,
                "batting_hand": str(attrs.get("batting_hand") or "unknown"),
            }
        )
    return out


def _innings_expected_runs(innings: dict[str, Any]) -> float:
    """
    Read team expected runs from ``simulate_innings`` output.

    Summaries nest totals under ``team``; a top-level ``expected_runs`` key is
    accepted only as a fallback for older/slimmed payloads.
    """
    team = innings.get("team")
    if isinstance(team, dict) and team.get("expected_runs") is not None:
        return float(team["expected_runs"])
    if innings.get("expected_runs") is not None:
        return float(innings["expected_runs"])
    return 0.0


def _merge_innings_to_rows(
    *,
    match_id: str,
    first: dict[str, Any],
    chase: dict[str, Any],
    first_team: str,
    chase_team: str,
    first_lineup: list[dict[str, str]],
    chase_lineup: list[dict[str, str]],
) -> list[dict[str, Any]]:
    bat_first = {b["player_id"]: b for b in first["batters"]}
    bat_chase = {b["player_id"]: b for b in chase["batters"]}
    bowl_first = {b["player_id"]: b for b in first["bowlers"]}
    bowl_chase = {b["player_id"]: b for b in chase["bowlers"]}
    players: dict[str, dict[str, Any]] = {}
    for row in first_lineup:
        players[row["player_id"]] = {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "team": first_team,
        }
    for row in chase_lineup:
        players[row["player_id"]] = {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "team": chase_team,
        }
    for src, team in ((bowl_first, chase_team), (bowl_chase, first_team)):
        for pid, b in src.items():
            if pid not in players:
                players[pid] = {
                    "player_id": pid,
                    "player_name": b["player_name"],
                    "team": team,
                }

    rows: list[dict[str, Any]] = []
    for pid, meta in players.items():
        batting = bat_first.get(pid) or bat_chase.get(pid) or {}
        bowling = bowl_first.get(pid) or bowl_chase.get(pid) or {}
        overs = float(bowling.get("expected_overs") or 0.0)
        runs_conc = float(bowling.get("expected_runs_conceded") or 0.0)
        econ = bowling.get("expected_economy")
        if econ is None and overs > 0:
            econ = runs_conc / overs
        rows.append(
            {
                "match_id": match_id,
                "player_id": pid,
                "player_name": meta["player_name"],
                "team": meta["team"],
                "expected_runs": float(batting.get("expected_runs") or 0.0),
                "expected_balls": float(batting.get("expected_balls") or 0.0),
                "expected_fours": float(batting.get("expected_fours") or 0.0),
                "expected_sixes": float(batting.get("expected_sixes") or 0.0),
                "p_runs_ge30": float(batting.get("p_runs_ge30") or 0.0),
                "p_runs_ge50": float(batting.get("p_runs_ge50") or 0.0),
                "p_runs_ge100": float(batting.get("p_runs_ge100") or 0.0),
                "expected_wickets": float(bowling.get("expected_wickets") or 0.0),
                "expected_overs": overs,
                "expected_economy": econ,
                "p_wickets_ge3": float(bowling.get("p_wickets_ge3") or 0.0),
                "p_wickets_ge4": float(bowling.get("p_wickets_ge4") or 0.0),
                "p_wickets_ge5": float(bowling.get("p_wickets_ge5") or 0.0),
            }
        )
    return rows


def predict_holdout_via_mc(
    setups: list[HoldoutMatchSetup],
    *,
    canonical_dir: Path,
    attributes_path: Path,
    effects_path: Path,
    matchups_path: Path,
    chase_impacts_path: Path,
    co_batters_path: Path,
    weather_dir: Path | None = None,
    n_sims: int = 50,
    seed: int = 7,
) -> pd.DataFrame:
    """Run short HB MC per holdout match; return prediction frame."""
    attributes = load_attributes_index(attributes_path)
    chase_impacts = load_chase_impacts(chase_impacts_path, canonical_dir=canonical_dir)
    partnership_index = load_partnership_index(co_batters_path)
    rate_cache: dict[tuple[str | None, str], InningsRateModel] = {}

    def rates(venue: str | None, group: str, match_date: str | None) -> InningsRateModel:
        if weather_dir is None or match_date is None:
            key = (venue, group)
            if key not in rate_cache:
                rate_cache[key] = InningsRateModel(
                    canonical_dir=canonical_dir,
                    effects_path=effects_path,
                    matchups_path=matchups_path,
                    attributes=attributes,
                    venue=venue,
                    innings_group=group,
                )
            return rate_cache[key]
        return InningsRateModel(
            canonical_dir=canonical_dir,
            effects_path=effects_path,
            matchups_path=matchups_path,
            attributes=attributes,
            venue=venue,
            innings_group=group,
            match_date=match_date,
            weather_dir=weather_dir,
        )

    all_rows: list[dict[str, Any]] = []
    for i, setup in enumerate(setups):
        first_lineup = _attach_hands(setup.first_lineup, attributes)
        chase_lineup = _attach_hands(setup.chase_lineup, attributes)
        # Preserve holdout quotas (actual or scheduled); only refresh pace groups.
        configure_attack(setup.first_attack, attributes=attributes, assign_quotas=False)
        configure_attack(setup.chase_attack, attributes=attributes, assign_quotas=False)
        first = simulate_innings(
            lineup=first_lineup,
            attack=setup.first_attack,
            rates=rates(setup.venue, "first_innings", setup.match_date),
            n_sims=n_sims,
            seed=seed + i * 17,
            partnership_index=partnership_index,
        )
        # Must read nested team.expected_runs — top-level key is absent, so the
        # prior ``first.get("expected_runs") or 0`` collapsed every chase to
        # target=1 and wiped chase batting + bowling opportunity (~½ scale).
        target = _innings_expected_runs(first) + 1.0
        chase = simulate_innings(
            lineup=chase_lineup,
            attack=setup.chase_attack,
            rates=rates(setup.venue, "chase", setup.match_date),
            n_sims=n_sims,
            seed=seed + i * 17 + 1,
            target=target,
            chase_impacts=chase_impacts,
            partnership_index=partnership_index,
        )
        all_rows.extend(
            _merge_innings_to_rows(
                match_id=setup.match_id,
                first=first,
                chase=chase,
                first_team=setup.first_team,
                chase_team=setup.chase_team,
                first_lineup=first_lineup,
                chase_lineup=chase_lineup,
            )
        )
        if (i + 1) % 10 == 0 or i + 1 == len(setups):
            print(
                f"mc progress {i + 1}/{len(setups)} matches (n_sims={n_sims})",
                flush=True,
            )
    return pd.DataFrame(all_rows)
