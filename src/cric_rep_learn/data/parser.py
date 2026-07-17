"""Parse Cricsheet JSON into canonical match, player, delivery, and wicket rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION

BOWLER_CREDITED_KINDS = {
    "bowled",
    "caught",
    "caught and bowled",
    "hit the ball twice",
    "hit wicket",
    "lbw",
    "stumped",
}
NON_WICKET_KINDS = {"retired hurt", "absent hurt", "retired not out"}
SUPPORTED_T20_TYPES = {"T20", "IT20"}


class ParseError(ValueError):
    """Raised when a source match cannot be represented safely."""


@dataclass(slots=True)
class CanonicalMatch:
    match: dict[str, Any]
    innings: list[dict[str, Any]]
    match_players: list[dict[str, Any]]
    deliveries: list[dict[str, Any]]
    wickets: list[dict[str, Any]]
    replacements: list[dict[str, Any]]
    reviews: list[dict[str, Any]]


def _as_date(value: str) -> date:
    return date.fromisoformat(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _overs_to_balls(value: Any, balls_per_over: int) -> int | None:
    if value is None:
        return None
    raw = str(value)
    if "." not in raw:
        return int(raw) * balls_per_over
    completed, deliveries = raw.split(".", maxsplit=1)
    return int(completed) * balls_per_over + int(deliveries)


class CricsheetParser:
    """Strict parser for Cricsheet JSON data format 1.x."""

    def __init__(self, *, require_registry: bool = True, t20_only: bool = True):
        self.require_registry = require_registry
        self.t20_only = t20_only

    def parse_path(self, path: Path, *, input_root: Path | None = None) -> CanonicalMatch:
        payload = path.read_bytes()
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON in {path}: {exc}") from exc

        source_file = str(path.relative_to(input_root)) if input_root else str(path)
        source_dataset = path.parent.name
        return self.parse_dict(
            raw,
            match_id=path.stem,
            source_dataset=source_dataset,
            source_file=source_file,
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def parse_dict(
        self,
        raw: dict[str, Any],
        *,
        match_id: str,
        source_dataset: str = "fixture",
        source_file: str = "fixture.json",
        source_sha256: str = "",
    ) -> CanonicalMatch:
        meta = raw.get("meta", {})
        info = raw.get("info")
        innings_data = raw.get("innings")
        if not isinstance(info, dict) or not isinstance(innings_data, list):
            raise ParseError(f"{match_id}: missing info or innings")

        match_type = str(info.get("match_type", ""))
        if self.t20_only and match_type not in SUPPORTED_T20_TYPES:
            raise ParseError(f"{match_id}: unsupported match_type={match_type!r}")

        dates = info.get("dates") or []
        if not dates:
            raise ParseError(f"{match_id}: no match date")
        match_date = _as_date(str(dates[0]))
        end_date = _as_date(str(dates[-1]))

        teams = info.get("teams") or []
        if len(teams) != 2:
            raise ParseError(f"{match_id}: expected exactly two teams, got {teams!r}")

        registry = info.get("registry", {}).get("people", {})
        if self.require_registry and not registry:
            raise ParseError(f"{match_id}: missing info.registry.people")

        def player_id(name: str) -> str:
            identifier = registry.get(name)
            if identifier:
                return str(identifier)
            if self.require_registry:
                raise ParseError(f"{match_id}: {name!r} missing from people registry")
            digest = hashlib.sha256(name.casefold().encode()).hexdigest()[:16]
            return f"unregistered:{match_id}:{digest}"

        event = info.get("event") or {}
        outcome = info.get("outcome") or {}
        outcome_by = outcome.get("by") or {}
        toss = info.get("toss") or {}
        scheduled_overs = _optional_int(info.get("overs"))
        balls_per_over = int(info.get("balls_per_over", 6))
        scheduled_balls = scheduled_overs * balls_per_over if scheduled_overs is not None else None

        player_of_match = [player_id(name) for name in (info.get("player_of_match") or [])]
        official_players = info.get("players") or {}

        match_row = {
            "schema_version": SCHEMA_VERSION,
            "match_id": str(match_id),
            "match_date": match_date,
            "end_date": end_date,
            "data_version": str(meta.get("data_version", "")),
            "source_created_date": (
                _as_date(str(meta["created"])) if meta.get("created") else None
            ),
            "revision": int(meta.get("revision", 0)),
            "source_dataset": source_dataset,
            "source_file": source_file,
            "source_sha256": source_sha256,
            "gender": info.get("gender"),
            "team_type": info.get("team_type"),
            "match_type": match_type,
            "balls_per_over": balls_per_over,
            "scheduled_overs": scheduled_overs,
            "season": str(info.get("season", "")),
            "city": info.get("city"),
            "venue": info.get("venue"),
            "event_name": event.get("name"),
            "event_group": str(event["group"]) if event.get("group") is not None else None,
            "event_stage": (str(event["stage"]) if event.get("stage") is not None else None),
            "match_number": (
                str(event["match_number"]) if event.get("match_number") is not None else None
            ),
            "match_type_number": _optional_int(info.get("match_type_number")),
            "team_1": str(teams[0]),
            "team_2": str(teams[1]),
            "toss_winner": toss.get("winner"),
            "toss_decision": toss.get("decision"),
            "winner": outcome.get("winner"),
            "outcome_result": outcome.get("result"),
            "outcome_method": outcome.get("method"),
            "outcome_eliminator": outcome.get("eliminator"),
            "bowl_out_winner": outcome.get("bowl_out"),
            "win_by_runs": _optional_int(outcome_by.get("runs")),
            "win_by_wickets": _optional_int(outcome_by.get("wickets")),
            "player_of_match_ids": player_of_match,
            "has_official_lineup": bool(official_players),
            # Cricsheet currently supplies dates but not scheduled start time/timezone.
            "start_time_local": None,
            "timezone": None,
        }

        match_players: dict[tuple[str, str], dict[str, Any]] = {}

        def register_match_player(name: str, team: str, listed: bool) -> str:
            identifier = player_id(name)
            key = (team, identifier)
            existing = match_players.get(key)
            if existing:
                existing["listed_in_match_squad"] |= listed
            else:
                match_players[key] = {
                    "match_id": str(match_id),
                    "match_date": match_date,
                    "team": team,
                    "player_id": identifier,
                    "player_name": name,
                    "listed_in_match_squad": listed,
                }
            return identifier

        for team, names in official_players.items():
            for name in names:
                register_match_player(str(name), str(team), True)

        delivery_rows: list[dict[str, Any]] = []
        innings_rows: list[dict[str, Any]] = []
        wicket_rows: list[dict[str, Any]] = []
        replacement_rows: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []

        for innings_index, innings in enumerate(innings_data, start=1):
            batting_team = str(innings["team"])
            bowling_team = str(teams[1] if teams[0] == batting_team else teams[0])
            target = innings.get("target") or {}
            target_runs = _optional_int(target.get("runs"))
            target_overs_raw = str(target["overs"]) if target.get("overs") is not None else None
            target_balls = _overs_to_balls(target.get("overs"), balls_per_over)
            is_super_over = bool(innings.get("super_over", False))
            powerplays = innings.get("powerplays") or []
            penalty_runs = innings.get("penalty_runs") or {}
            penalty_runs_pre = int(penalty_runs.get("pre", 0))
            penalty_runs_post = int(penalty_runs.get("post", 0))
            absent_hurt_ids = [player_id(str(name)) for name in (innings.get("absent_hurt") or [])]

            innings_rows.append(
                {
                    "match_id": str(match_id),
                    "match_date": match_date,
                    "innings": innings_index,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "is_super_over": is_super_over,
                    "forfeited": bool(innings.get("forfeited", False)),
                    "target_runs": target_runs,
                    "target_overs_raw": target_overs_raw,
                    "target_balls": target_balls,
                    "penalty_runs_pre": penalty_runs_pre,
                    "penalty_runs_post": penalty_runs_post,
                    "absent_hurt_ids": absent_hurt_ids,
                    "powerplays_json": json.dumps(powerplays, sort_keys=True),
                    "miscounted_overs_json": json.dumps(
                        innings.get("miscounted_overs") or {}, sort_keys=True
                    ),
                }
            )

            score = penalty_runs_pre
            wickets_lost = 0
            legal_balls = 0
            attempts = 0

            for over in innings.get("overs", []):
                over_number = int(over["over"])
                legal_balls_in_over = 0
                for delivery_index, delivery in enumerate(over.get("deliveries", []), start=1):
                    attempts += 1
                    batter_name = str(delivery["batter"])
                    bowler_name = str(delivery["bowler"])
                    non_striker_name = str(delivery["non_striker"])
                    batter_id = register_match_player(batter_name, batting_team, False)
                    bowler_id = register_match_player(bowler_name, bowling_team, False)
                    non_striker_id = register_match_player(non_striker_name, batting_team, False)

                    extras = delivery.get("extras") or {}
                    runs = delivery.get("runs") or {}
                    is_legal = not extras.get("wides") and not extras.get("noballs")
                    ball_in_over = legal_balls_in_over + 1
                    actual_delivery = float(
                        delivery.get("actual_delivery", over_number + ball_in_over / 10)
                    )
                    source_ball_label = f"{over_number}.{ball_in_over}"

                    is_powerplay = self._is_powerplay(actual_delivery, powerplays)
                    phase, phase_source = self._phase(
                        legal_balls=legal_balls,
                        scheduled_balls=scheduled_balls,
                        is_powerplay=is_powerplay,
                    )

                    parsed_wickets = delivery.get("wickets") or []
                    bowler_wickets = 0
                    batter_dismissed = False
                    counted_wickets = 0
                    for wicket_index, wicket in enumerate(parsed_wickets, start=1):
                        kind = str(wicket["kind"])
                        out_name = str(wicket["player_out"])
                        out_id = register_match_player(out_name, batting_team, False)
                        credited = kind in BOWLER_CREDITED_KINDS
                        counted = kind not in NON_WICKET_KINDS
                        bowler_wickets += int(credited)
                        counted_wickets += int(counted)
                        batter_dismissed |= counted and out_id == batter_id

                        fielder_names: list[str] = []
                        fielder_ids: list[str] = []
                        unknown_fielder_count = 0
                        for fielder in wicket.get("fielders") or []:
                            # Format 1.1 uses strings; 1.2 may carry {"name": ...}.
                            if isinstance(fielder, dict):
                                raw_name = fielder.get("name")
                                if not raw_name:
                                    unknown_fielder_count += 1
                                    continue
                                name = str(raw_name)
                            else:
                                name = str(fielder)
                            fielder_names.append(name)
                            fielder_ids.append(register_match_player(name, bowling_team, False))

                        wicket_rows.append(
                            {
                                "match_id": str(match_id),
                                "match_date": match_date,
                                "innings": innings_index,
                                "over_number": over_number,
                                "delivery_index": delivery_index,
                                "wicket_index": wicket_index,
                                "player_out_id": out_id,
                                "player_out_name": out_name,
                                "kind": kind,
                                "credited_to_bowler": credited,
                                "fielder_ids": fielder_ids,
                                "fielder_names": fielder_names,
                                "unknown_fielder_count": unknown_fielder_count,
                            }
                        )

                    batter_runs = int(runs.get("batter", 0))
                    total_runs = int(runs.get("total", 0))
                    non_boundary = bool(runs.get("non_boundary", False))

                    for replacement_type, replacements in (
                        delivery.get("replacements") or {}
                    ).items():
                        for replacement_index, replacement in enumerate(replacements, start=1):
                            role = replacement.get("role")
                            team = replacement.get("team")
                            if team is None:
                                if role == "batter":
                                    team = batting_team
                                elif role == "bowler":
                                    team = bowling_team
                            in_name = str(replacement["in"])
                            in_id = register_match_player(in_name, str(team), False)
                            out_name = (
                                str(replacement["out"])
                                if replacement.get("out") is not None
                                else None
                            )
                            out_id = (
                                register_match_player(out_name, str(team), False)
                                if out_name is not None
                                else None
                            )
                            replacement_rows.append(
                                {
                                    "match_id": str(match_id),
                                    "match_date": match_date,
                                    "innings": innings_index,
                                    "over_number": over_number,
                                    "delivery_index": delivery_index,
                                    "replacement_type": replacement_type,
                                    "replacement_index": replacement_index,
                                    "team": team,
                                    "role": role,
                                    "reason": replacement.get("reason"),
                                    "player_in_id": in_id,
                                    "player_in_name": in_name,
                                    "player_out_id": out_id,
                                    "player_out_name": out_name,
                                }
                            )

                    review = delivery.get("review")
                    if review:
                        review_batter_name = str(review["batter"])
                        umpire_name = (
                            str(review["umpire"]) if review.get("umpire") is not None else None
                        )
                        review_rows.append(
                            {
                                "match_id": str(match_id),
                                "match_date": match_date,
                                "innings": innings_index,
                                "over_number": over_number,
                                "delivery_index": delivery_index,
                                "review_by": review.get("by"),
                                "batter_id": player_id(review_batter_name),
                                "batter_name": review_batter_name,
                                "decision": review.get("decision"),
                                "umpire_id": (
                                    player_id(umpire_name) if umpire_name is not None else None
                                ),
                                "umpire_name": umpire_name,
                                "umpires_call": bool(review.get("umpires_call", False)),
                            }
                        )

                    delivery_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "match_id": str(match_id),
                            "match_date": match_date,
                            "innings": innings_index,
                            "is_super_over": is_super_over,
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "target_runs": target_runs,
                            "target_overs_raw": target_overs_raw,
                            "target_balls": target_balls,
                            "over_number": over_number,
                            "delivery_index": delivery_index,
                            "attempt_index_in_innings": attempts,
                            "source_ball_label": source_ball_label,
                            "is_legal": is_legal,
                            "legal_balls_in_over_before": legal_balls_in_over,
                            "legal_balls_before": legal_balls,
                            "score_before": score,
                            "wickets_before": wickets_lost,
                            "scheduled_balls": scheduled_balls,
                            "phase": phase,
                            "phase_source": phase_source,
                            "is_powerplay": is_powerplay,
                            "batter_id": batter_id,
                            "batter_name": batter_name,
                            "bowler_id": bowler_id,
                            "bowler_name": bowler_name,
                            "non_striker_id": non_striker_id,
                            "non_striker_name": non_striker_name,
                            "runs_batter": batter_runs,
                            "runs_extras": int(runs.get("extras", 0)),
                            "runs_total": total_runs,
                            "non_boundary": non_boundary,
                            "is_boundary": batter_runs in {4, 6} and not non_boundary,
                            "extras_byes": int(extras.get("byes", 0)),
                            "extras_legbyes": int(extras.get("legbyes", 0)),
                            "extras_noballs": int(extras.get("noballs", 0)),
                            "extras_penalty": int(extras.get("penalty", 0)),
                            "extras_wides": int(extras.get("wides", 0)),
                            "wicket_count": counted_wickets,
                            "bowler_wicket_count": bowler_wickets,
                            "batter_dismissed": batter_dismissed,
                        }
                    )

                    score += total_runs
                    wickets_lost += counted_wickets
                    legal_balls += int(is_legal)
                    legal_balls_in_over += int(is_legal)

        return CanonicalMatch(
            match=match_row,
            innings=innings_rows,
            match_players=list(match_players.values()),
            deliveries=delivery_rows,
            wickets=wicket_rows,
            replacements=replacement_rows,
            reviews=review_rows,
        )

    @staticmethod
    def _is_powerplay(actual_delivery: float, powerplays: list[dict[str, Any]]) -> bool | None:
        if not powerplays:
            return None
        return any(
            float(powerplay["from"]) <= actual_delivery <= float(powerplay["to"])
            for powerplay in powerplays
            if powerplay.get("type") in {"mandatory", "batting", "fielding"}
        )

    @staticmethod
    def _phase(
        *,
        legal_balls: int,
        scheduled_balls: int | None,
        is_powerplay: bool | None,
    ) -> tuple[str, str]:
        if is_powerplay:
            return "powerplay", "recorded"

        if scheduled_balls == 120:
            if legal_balls < 36:
                return "powerplay", "standard_t20"
            if legal_balls >= 96:
                return "death", "standard_t20"
            return "middle", "standard_t20"

        if scheduled_balls:
            progress = legal_balls / scheduled_balls
            if progress < 0.30:
                return "powerplay", "proportional"
            if progress >= 0.80:
                return "death", "proportional"
            return "middle", "proportional"

        return "unknown", "unknown"
