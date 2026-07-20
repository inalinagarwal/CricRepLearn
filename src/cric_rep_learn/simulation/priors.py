"""Hierarchical Bayes delivery rates with phase and L/R adjustments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from cric_rep_learn.data.bowling_style import parse_bowling_style
from cric_rep_learn.players.player_effects import _posterior_rate, expected_runs_vs_bowler
from cric_rep_learn.players.forecast_vs_attack import expected_dismissal_rate_vs_bowler

_BOWLER_WICKET_PRIOR_CACHE: dict[str, dict[str, float]] = {}


def load_bowler_wicket_priors(
    canonical_dir: Path,
    *,
    strength: float = 120.0,
    min_balls: int = 60,
    clip: tuple[float, float] = (0.75, 1.35),
) -> dict[str, float]:
    """
    Train-only shrunk bowler wicket/ball rate as a multiplier vs global.

    Elite bowlers get >1, part-timers <1. Used to tilt dismissal hazards
    without reopening delivery-residual training.
    """
    key = str(canonical_dir.resolve())
    if key in _BOWLER_WICKET_PRIOR_CACHE:
        return _BOWLER_WICKET_PRIOR_CACHE[key]

    deliveries = _escape(canonical_dir / "deliveries.parquet")
    split = _escape(canonical_dir / "split_manifest.parquet")
    connection = duckdb.connect()
    try:
        global_row = connection.execute(
            f"""
            SELECT
                SUM(d.bowler_wicket_count)::DOUBLE AS wickets,
                SUM(CASE WHEN d.is_legal THEN 1 ELSE 0 END)::DOUBLE AS balls
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
            """
        ).fetchone()
        g_wk = float(global_row[0] or 0.0)
        g_balls = float(global_row[1] or 1.0)
        global_rate = g_wk / g_balls if g_balls > 0 else 0.05

        frame = connection.execute(
            f"""
            SELECT
                d.bowler_id,
                SUM(d.bowler_wicket_count)::DOUBLE AS wickets,
                SUM(CASE WHEN d.is_legal THEN 1 ELSE 0 END)::DOUBLE AS balls
            FROM read_parquet('{deliveries}') d
            JOIN read_parquet('{split}') s USING (match_id)
            WHERE s.split = 'train'
              AND NOT d.is_super_over
            GROUP BY 1
            """
        ).fetchdf()
    finally:
        connection.close()

    lo, hi = clip
    priors: dict[str, float] = {}
    for row in frame.to_dict(orient="records"):
        balls = float(row["balls"])
        if balls < min_balls:
            priors[str(row["bowler_id"])] = 1.0
            continue
        rate = _posterior_rate(
            float(row["wickets"]),
            balls,
            global_rate,
            strength,
        )
        mult = rate / global_rate if global_rate > 0 else 1.0
        priors[str(row["bowler_id"])] = float(min(max(mult, lo), hi))
    _BOWLER_WICKET_PRIOR_CACHE[key] = priors
    return priors


def _escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


class InningsRateModel:
    """
    Per-ball expected SR and dismissal hazard for (batter, bowler, phase).

    Hierarchy:
      overall HB matchup/archetype
        ↑ phase×batter×bowler (or phase×batter) when available
        ↑ venue(+similar) batter rates when available
        ↑ first-innings / chase batter rates when available
        × left/right handedness multiplier from train
    """

    def __init__(
        self,
        *,
        canonical_dir: Path,
        effects_path: Path,
        matchups_path: Path,
        attributes: dict[str, dict[str, Any]],
        venue: str | None = None,
        innings_group: str = "first_innings",
        match_date: str | None = None,
        weather_dir: Path | None = None,
        phase_strength: float = 60.0,
        context_strength: float = 80.0,
        matchup_strength: float | None = None,
        sparse_venue_balls: float = 40.0,
    ) -> None:
        self.canonical_dir = canonical_dir.resolve()
        self.attributes = attributes
        self.innings_group = innings_group
        if innings_group not in {"first_innings", "chase"}:
            raise ValueError("innings_group must be 'first_innings' or 'chase'")
        effects_frame = pq.read_table(effects_path).to_pandas()
        self.effects = {row["player_id"]: row for row in effects_frame.to_dict(orient="records")}
        matchups_frame = pq.read_table(matchups_path).to_pandas()
        self.matchups = {
            (row["batter_id"], row["bowler_id"]): {
                "runs": float(row["runs"]),
                "balls": float(row["balls"]),
                "dismissals": float(row["dismissals"]),
            }
            for row in matchups_frame.to_dict(orient="records")
        }
        smoothing = json.loads(
            (effects_path.parent / "smoothing.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (effects_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        self.global_sr = float(smoothing["global_sr"])
        self.global_dismiss = float(metadata.get("global_dismissal_rate", 0.05))
        self.matchup_strength = float(
            matchup_strength
            if matchup_strength is not None
            else smoothing["matchup_strength"]
        )
        self.archetype_strength = float(smoothing["archetype_strength"])
        self.phase_strength = phase_strength
        self.context_strength = context_strength
        self._phase_cache: dict[tuple[str, str, str], dict[str, float]] = {}
        self._batter_phase_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._context_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._hand_mult = self._build_handedness_multipliers()
        self.bowler_wicket_mult = load_bowler_wicket_priors(self.canonical_dir)
        self.venue = venue
        self.venue_resolution = None
        self.venue_clause = "TRUE"
        self.venue_scope = "none"
        self.match_date = match_date
        self.weather_dir = Path(weather_dir) if weather_dir else None
        self.weather_features = None
        self.weather_impacts = None
        self.weather_notes: list[str] = []
        if venue:
            self._configure_venue(venue, sparse_venue_balls)
        if self.weather_dir and match_date:
            self._configure_weather(match_date)

    def _configure_venue(self, venue: str, sparse_balls: float) -> None:
        from cric_rep_learn.players.venue_similarity import (
            resolve_venues,
            venue_sql_clause,
        )

        primary = resolve_venues(self.canonical_dir, venue, include_similar=False)
        if not primary["primary"]:
            self.venue_resolution = primary
            self.venue_scope = "unresolved"
            self.venue_clause = "FALSE"
            return
        clause = venue_sql_clause(primary["primary"])
        balls = self._count_venue_balls(clause)
        self.venue_scope = "primary"
        self.venue_resolution = primary
        if balls < sparse_balls:
            expanded = resolve_venues(self.canonical_dir, venue, include_similar=True)
            clause = venue_sql_clause(expanded["accepted"])
            self.venue_scope = "primary+similar_conditions"
            self.venue_resolution = expanded
        self.venue_clause = clause

    def _configure_weather(self, match_date: str) -> None:
        from datetime import date as date_cls

        from cric_rep_learn.weather import (
            apply_weather_multipliers,
            lookup_weather_features,
        )

        assert self.weather_dir is not None
        impacts_path = self.weather_dir / "weather_impacts.json"
        if impacts_path.exists():
            self.weather_impacts = json.loads(impacts_path.read_text(encoding="utf-8"))
        parsed = date_cls.fromisoformat(match_date)
        city = None
        if self.venue_resolution and self.venue_resolution.get("primary"):
            city = self.venue_resolution["primary"][0].get("city")
        self.weather_features = lookup_weather_features(
            self.weather_dir,
            venue=self.venue,
            city=city,
            match_date=parsed,
        )
        # Silence unused import warning path — multipliers applied in rates().
        _ = apply_weather_multipliers


    def _count_venue_balls(self, venue_clause: str) -> float:
        deliveries = _escape(self.canonical_dir / "deliveries.parquet")
        matches = _escape(self.canonical_dir / "matches.parquet")
        split = _escape(self.canonical_dir / "split_manifest.parquet")
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"""
                SELECT COUNT(*)::DOUBLE
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{matches}') m USING (match_id)
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                  AND {venue_clause}
                """
            ).fetchone()
        finally:
            connection.close()
        return float(row[0] if row else 0.0)

    def _build_handedness_multipliers(self) -> dict[tuple[str, str], dict[str, float]]:
        """Train-wide SR / dismiss multipliers for (batting_hand, bowling_arm)."""
        deliveries = _escape(self.canonical_dir / "deliveries.parquet")
        split = _escape(self.canonical_dir / "split_manifest.parquet")
        rows = [
            {
                "player_id": pid,
                "batting_hand": str(a.get("batting_hand") or "unknown"),
                "bowling_arm": str(a.get("bowling_arm") or "unknown"),
            }
            for pid, a in self.attributes.items()
        ]
        connection = duckdb.connect()
        try:
            connection.register("attr_hands", __import__("pandas").DataFrame(rows))
            global_row = connection.execute(
                f"""
                SELECT
                    SUM(d.runs_batter)::DOUBLE / COUNT(*) AS sr,
                    SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE
                        / COUNT(*) AS dismiss
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                """
            ).fetchone()
            g_sr, g_dismiss = float(global_row[0]), float(global_row[1])
            frame = connection.execute(
                f"""
                SELECT
                    COALESCE(bh.batting_hand, 'unknown') AS batting_hand,
                    COALESCE(bo.bowling_arm, 'unknown') AS bowling_arm,
                    SUM(d.runs_batter)::DOUBLE / COUNT(*) AS sr,
                    SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE
                        / COUNT(*) AS dismiss,
                    COUNT(*)::BIGINT AS balls
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                LEFT JOIN attr_hands bh ON d.batter_id = bh.player_id
                LEFT JOIN attr_hands bo ON d.bowler_id = bo.player_id
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                GROUP BY 1, 2
                HAVING COUNT(*) >= 500
                """
            ).fetchdf()
        finally:
            connection.close()

        out: dict[tuple[str, str], dict[str, float]] = {}
        for row in frame.to_dict(orient="records"):
            out[(row["batting_hand"], row["bowling_arm"])] = {
                "sr_mult": float(row["sr"] / g_sr) if g_sr else 1.0,
                "dismiss_mult": float(row["dismiss"] / g_dismiss) if g_dismiss else 1.0,
                "balls": float(row["balls"]),
            }
        self._hand_global = {"sr": g_sr, "dismiss": g_dismiss}
        return out

    def _bowler_attrs(self, bowler_id: str) -> dict[str, Any]:
        ba = dict(self.attributes.get(bowler_id, {}))
        parsed = parse_bowling_style(ba.get("bowling_style_raw"))
        if not ba.get("bowling_arm"):
            ba["bowling_arm"] = parsed.bowling_arm
        if not ba.get("pace_group"):
            ba["pace_group"] = parsed.pace_group
        return ba

    def _phase_matchup(
        self, batter_id: str, bowler_id: str, phase: str
    ) -> dict[str, float] | None:
        key = (batter_id, bowler_id, phase)
        if key in self._phase_cache:
            return self._phase_cache[key] or None
        deliveries = _escape(self.canonical_dir / "deliveries.parquet")
        split = _escape(self.canonical_dir / "split_manifest.parquet")
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"""
                SELECT
                    SUM(d.runs_batter)::DOUBLE AS runs,
                    COUNT(*)::DOUBLE AS balls,
                    SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                  AND d.batter_id = '{batter_id}'
                  AND d.bowler_id = '{bowler_id}'
                  AND d.phase = '{phase}'
                """
            ).fetchone()
        finally:
            connection.close()
        if not row or not row[1]:
            self._phase_cache[key] = {}
            return None
        stats = {
            "runs": float(row[0]),
            "balls": float(row[1]),
            "dismissals": float(row[2]),
        }
        self._phase_cache[key] = stats
        return stats

    def _batter_phase(self, batter_id: str, phase: str) -> dict[str, float] | None:
        key = (batter_id, phase)
        if key in self._batter_phase_cache:
            return self._batter_phase_cache[key] or None
        deliveries = _escape(self.canonical_dir / "deliveries.parquet")
        split = _escape(self.canonical_dir / "split_manifest.parquet")
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"""
                SELECT
                    SUM(d.runs_batter)::DOUBLE AS runs,
                    COUNT(*)::DOUBLE AS balls,
                    SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                  AND d.batter_id = '{batter_id}'
                  AND d.phase = '{phase}'
                """
            ).fetchone()
        finally:
            connection.close()
        if not row or not row[1]:
            self._batter_phase_cache[key] = {}
            return None
        stats = {
            "runs": float(row[0]),
            "balls": float(row[1]),
            "dismissals": float(row[2]),
        }
        self._batter_phase_cache[key] = stats
        return stats

    def _batter_context(self, batter_id: str, kind: str) -> dict[str, float] | None:
        """kind is 'venue' or 'innings_group'."""
        key = (batter_id, kind)
        if key in self._context_cache:
            return self._context_cache[key] or None
        deliveries = _escape(self.canonical_dir / "deliveries.parquet")
        matches = _escape(self.canonical_dir / "matches.parquet")
        split = _escape(self.canonical_dir / "split_manifest.parquet")
        if kind == "venue":
            if self.venue_scope in {"none", "unresolved"}:
                self._context_cache[key] = {}
                return None
            extra_join = f"JOIN read_parquet('{matches}') m USING (match_id)"
            extra_where = f"AND {self.venue_clause}"
        elif kind == "innings_group":
            extra_join = ""
            if self.innings_group == "first_innings":
                extra_where = "AND d.innings = 1"
            else:
                extra_where = "AND d.innings > 1"
        else:
            raise ValueError(kind)
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"""
                SELECT
                    SUM(d.runs_batter)::DOUBLE AS runs,
                    COUNT(*)::DOUBLE AS balls,
                    SUM(CASE WHEN d.batter_dismissed THEN 1 ELSE 0 END)::DOUBLE AS dismissals
                FROM read_parquet('{deliveries}') d
                JOIN read_parquet('{split}') s USING (match_id)
                {extra_join}
                WHERE s.split = 'train'
                  AND NOT d.is_super_over
                  AND (d.is_legal OR d.extras_noballs > 0)
                  AND d.batter_id = '{batter_id}'
                  {extra_where}
                """
            ).fetchone()
        finally:
            connection.close()
        if not row or not row[1]:
            self._context_cache[key] = {}
            return None
        stats = {
            "runs": float(row[0]),
            "balls": float(row[1]),
            "dismissals": float(row[2]),
        }
        self._context_cache[key] = stats
        return stats

    def rates(
        self,
        *,
        batter_id: str,
        bowler_id: str,
        phase: str,
        batting_hand: str | None = None,
    ) -> dict[str, Any]:
        ba = self._bowler_attrs(bowler_id)
        run_prior = expected_runs_vs_bowler(
            batter_id=batter_id,
            bowler_id=bowler_id,
            balls=1.0,
            effects=self.effects,
            matchups=self.matchups,
            bowler_attrs=ba,
            global_sr=self.global_sr,
            matchup_strength=self.matchup_strength,
            archetype_strength=self.archetype_strength,
        )
        dismiss_prior = expected_dismissal_rate_vs_bowler(
            batter_id=batter_id,
            bowler_id=bowler_id,
            effects=self.effects,
            matchups=self.matchups,
            global_dismiss=self.global_dismiss,
            matchup_strength=self.matchup_strength,
        )
        sr = float(run_prior["expected_sr"])
        p_out = float(dismiss_prior["dismissal_rate"])
        level = f"{run_prior['level']}|{dismiss_prior['level']}"

        # Phase shrink: matchup×phase → batter×phase → overall.
        parent_sr, parent_out = sr, p_out
        phase_mu = self._phase_matchup(batter_id, bowler_id, phase)
        if phase_mu and phase_mu["balls"] > 0:
            sr = _posterior_rate(
                phase_mu["runs"], phase_mu["balls"], parent_sr, self.phase_strength
            )
            p_out = _posterior_rate(
                phase_mu["dismissals"],
                phase_mu["balls"],
                parent_out,
                self.phase_strength,
            )
            level = f"phase_matchup→{level}"
        else:
            batter_phase = self._batter_phase(batter_id, phase)
            if batter_phase and batter_phase["balls"] > 0:
                sr = _posterior_rate(
                    batter_phase["runs"],
                    batter_phase["balls"],
                    parent_sr,
                    self.phase_strength,
                )
                p_out = _posterior_rate(
                    batter_phase["dismissals"],
                    batter_phase["balls"],
                    parent_out,
                    self.phase_strength,
                )
                level = f"phase_batter→{level}"

        for kind, label in (
            ("venue", f"venue({self.venue_scope})"),
            ("innings_group", f"innings({self.innings_group})"),
        ):
            if kind == "venue" and self.venue_scope in {"none", "unresolved"}:
                continue
            ctx = self._batter_context(batter_id, kind)
            if ctx and ctx["balls"] > 0:
                sr = _posterior_rate(
                    ctx["runs"], ctx["balls"], sr, self.context_strength
                )
                p_out = _posterior_rate(
                    ctx["dismissals"], ctx["balls"], p_out, self.context_strength
                )
                level = f"{label}→{level}"

        hand = batting_hand or str(
            (self.attributes.get(batter_id) or {}).get("batting_hand") or "unknown"
        )
        arm = str(ba.get("bowling_arm") or "unknown")
        mult = self._hand_mult.get((hand, arm))
        if mult:
            sr *= float(mult["sr_mult"])
            p_out *= float(mult["dismiss_mult"])
            level = f"handedness×{level}"

        if self.weather_features and self.weather_impacts:
            from cric_rep_learn.weather import apply_weather_multipliers

            sr, p_out, wx_notes = apply_weather_multipliers(
                sr=sr,
                dismissal_rate=p_out,
                features=self.weather_features,
                impacts=self.weather_impacts,
            )
            if wx_notes:
                level = "weather[" + ",".join(wx_notes) + "]→" + level
                self.weather_notes = wx_notes

        bowler_mult = float(self.bowler_wicket_mult.get(bowler_id, 1.0))
        if abs(bowler_mult - 1.0) > 1e-6:
            p_out *= bowler_mult
            level = f"bowler_wicket×{bowler_mult:.2f}→{level}"

        p_out = float(min(max(p_out, 1e-4), 0.35))
        sr = float(max(sr, 0.05))
        return {
            "expected_sr": sr,
            "dismissal_rate": p_out,
            "level": level,
            "phase": phase,
            "batting_hand": hand,
            "bowling_arm": arm,
            "innings_group": self.innings_group,
            "venue": self.venue,
            "venue_scope": self.venue_scope,
            "weather": self.weather_features,
        }
