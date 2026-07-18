"""Hierarchical empirical-Bayes baselines updated in chronological order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

N_RUN_CLASSES = 8  # 0..6 and the observed seven-run class
N_EXTRAS_CLASSES = 8  # 0..6 and 7+ (rare penalty events are capped)
N_LEGALITY_CLASSES = 3  # legal, wide, no-ball
BASELINE_LEVELS = (
    "global",
    "context",
    "player",
    "venue",
    "vs_pace",
    "vs_arm_pace",
    "vs_nation_arm_pace",
    "matchup",
)


@dataclass(frozen=True, slots=True)
class MatchContext:
    gender: str
    team_type: str
    venue: str


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Equivalent sample sizes used at each level of the hierarchy."""

    context: float = 2_000.0
    player: float = 200.0
    venue: float = 500.0
    vs_pace: float = 150.0
    vs_arm_pace: float = 100.0
    vs_nation_arm_pace: float = 80.0
    matchup: float = 60.0
    context_residual_weight: float = 0.25
    venue_max_weight: float = 0.25


@dataclass(slots=True)
class EventStats:
    n: int = 0
    batter_runs: np.ndarray = field(default_factory=lambda: np.zeros(N_RUN_CLASSES, dtype=np.int64))
    extras_runs: np.ndarray = field(
        default_factory=lambda: np.zeros(N_EXTRAS_CLASSES, dtype=np.int64)
    )
    legality: np.ndarray = field(
        default_factory=lambda: np.zeros(N_LEGALITY_CLASSES, dtype=np.int64)
    )
    batter_dismissals: int = 0
    bowler_wickets: int = 0

    def update(self, row: Mapping[str, Any]) -> None:
        self.n += 1
        self.batter_runs[min(int(row["runs_batter"]), N_RUN_CLASSES - 1)] += 1
        self.extras_runs[min(int(row["runs_extras"]), N_EXTRAS_CLASSES - 1)] += 1
        legality = 1 if int(row["extras_wides"]) > 0 else 2 if int(row["extras_noballs"]) > 0 else 0
        self.legality[legality] += 1
        self.batter_dismissals += int(bool(row["batter_dismissed"]))
        self.bowler_wickets += int(int(row["bowler_wicket_count"]) > 0)


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    batter_runs: np.ndarray
    extras_runs: np.ndarray
    legality: np.ndarray
    batter_dismissal: float
    bowler_wicket: float
    evidence: dict[str, int]

    @property
    def illegal_delivery(self) -> float:
        return float(1.0 - self.legality[0])


def _multiclass_posterior(
    counts: np.ndarray, n: int, prior: np.ndarray, strength: float
) -> np.ndarray:
    if n == 0:
        return prior.copy()
    return (counts.astype(np.float64) + strength * prior) / (n + strength)


def _binary_posterior(successes: int, n: int, prior: float, strength: float) -> float:
    if n == 0:
        return float(prior)
    return float((successes + strength * prior) / (n + strength))


def _reliability(n: int, strength: float) -> float:
    return float(n / (n + strength)) if n else 0.0


class HistoricalBaseline:
    """
    Stores only past aggregate evidence.

    Evaluation predicts all matches on a calendar date before calling ``update``
    for that date, so matches without known start times cannot leak into each
    other.

    When ``player_attributes`` is provided, batter×bowler-archetype levels sit
    between venue and direct matchup. The **prediction shrink path** is:

    ``venue → vs_pace → matchup``

    ``vs_arm_pace`` and ``vs_nation_arm_pace`` are still estimated (for coverage
    diagnostics and player-card fallbacks) but they do **not** feed matchup,
    because chaining through sparse nation/arm cells previously hurt log loss.

    Example for Rohit vs Starc: pace prior → Starc residual, with left-arm pace
    and Australia left-arm pace available as side diagnostics.
    """

    def __init__(
        self,
        smoothing: SmoothingConfig | None = None,
        player_attributes: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.smoothing = smoothing or SmoothingConfig()
        self.player_attributes = dict(player_attributes or {})
        self.global_stats = EventStats()
        self.context_stats: dict[tuple[Any, ...], EventStats] = {}
        self.batter_stats: dict[tuple[Any, ...], EventStats] = {}
        self.bowler_stats: dict[tuple[Any, ...], EventStats] = {}
        self.venue_stats: dict[tuple[Any, ...], EventStats] = {}
        self.vs_pace_stats: dict[tuple[Any, ...], EventStats] = {}
        self.vs_arm_pace_stats: dict[tuple[Any, ...], EventStats] = {}
        self.vs_nation_arm_pace_stats: dict[tuple[Any, ...], EventStats] = {}
        self.matchup_stats: dict[tuple[Any, ...], EventStats] = {}

    def _bowler_attrs(self, bowler_id: str) -> Mapping[str, Any]:
        return self.player_attributes.get(
            bowler_id,
            {
                "country": "unknown",
                "bowling_arm": "unknown",
                "pace_group": "unknown",
            },
        )

    def _keys(self, row: Mapping[str, Any], context: MatchContext) -> tuple[tuple[Any, ...], ...]:
        phase = str(row["phase"])
        super_over = bool(row["is_super_over"])
        innings_group = (
            "super_over" if super_over else "first_innings" if int(row["innings"]) == 1 else "chase"
        )
        wickets = int(row["wickets_before"])
        wickets_bucket = "0-2" if wickets <= 2 else "3-5" if wickets <= 5 else "6-9"
        context_key = (
            context.gender,
            context.team_type,
            innings_group,
            phase,
            wickets_bucket,
        )
        batter_id = str(row["batter_id"])
        bowler_id = str(row["bowler_id"])
        batter_key = (batter_id, phase, super_over)
        bowler_key = (bowler_id, phase, super_over)
        venue_key = (
            context.venue,
            context.gender,
            context.team_type,
            innings_group,
            phase,
            wickets_bucket,
        )
        attrs = self._bowler_attrs(bowler_id)
        country = str(attrs.get("country") or "unknown")
        arm = str(attrs.get("bowling_arm") or "unknown")
        pace = str(attrs.get("pace_group") or "unknown")
        vs_pace_key = (batter_id, pace, phase, super_over)
        vs_arm_pace_key = (batter_id, arm, pace, phase, super_over)
        vs_nation_arm_pace_key = (batter_id, country, arm, pace, phase, super_over)
        matchup_key = (batter_id, bowler_id, phase, super_over)
        return (
            context_key,
            batter_key,
            bowler_key,
            venue_key,
            vs_pace_key,
            vs_arm_pace_key,
            vs_nation_arm_pace_key,
            matchup_key,
        )

    @staticmethod
    def _stats(store: dict[tuple[Any, ...], EventStats], key: tuple[Any, ...]) -> EventStats:
        return store.get(key, EventStats())

    @staticmethod
    def _update_store(
        store: dict[tuple[Any, ...], EventStats],
        key: tuple[Any, ...],
        row: Mapping[str, Any],
    ) -> None:
        stats = store.get(key)
        if stats is None:
            stats = EventStats()
            store[key] = stats
        stats.update(row)

    def update(self, row: Mapping[str, Any], context: MatchContext) -> None:
        keys = self._keys(row, context)
        self.global_stats.update(row)
        for store, key in zip(
            (
                self.context_stats,
                self.batter_stats,
                self.bowler_stats,
                self.venue_stats,
                self.vs_pace_stats,
                self.vs_arm_pace_stats,
                self.vs_nation_arm_pace_stats,
                self.matchup_stats,
            ),
            keys,
            strict=True,
        ):
            self._update_store(store, key, row)

    def predict_all(
        self, row: Mapping[str, Any], context: MatchContext
    ) -> dict[str, BaselinePrediction]:
        (
            context_key,
            batter_key,
            bowler_key,
            venue_key,
            vs_pace_key,
            vs_arm_pace_key,
            vs_nation_arm_pace_key,
            matchup_key,
        ) = self._keys(row, context)
        context_stats = self._stats(self.context_stats, context_key)
        batter_stats = self._stats(self.batter_stats, batter_key)
        bowler_stats = self._stats(self.bowler_stats, bowler_key)
        venue_stats = self._stats(self.venue_stats, venue_key)
        vs_pace_stats = self._stats(self.vs_pace_stats, vs_pace_key)
        vs_arm_pace_stats = self._stats(self.vs_arm_pace_stats, vs_arm_pace_key)
        vs_nation_stats = self._stats(self.vs_nation_arm_pace_stats, vs_nation_arm_pace_key)
        matchup_stats = self._stats(self.matchup_stats, matchup_key)

        run_levels = self._multiclass_levels(
            "batter_runs",
            N_RUN_CLASSES,
            context_stats,
            batter_stats,
            bowler_stats,
            venue_stats,
            vs_pace_stats,
            vs_arm_pace_stats,
            vs_nation_stats,
            matchup_stats,
        )
        extras_levels = self._multiclass_levels(
            "extras_runs",
            N_EXTRAS_CLASSES,
            context_stats,
            batter_stats,
            bowler_stats,
            venue_stats,
            vs_pace_stats,
            vs_arm_pace_stats,
            vs_nation_stats,
            matchup_stats,
        )
        legality_levels = self._multiclass_levels(
            "legality",
            N_LEGALITY_CLASSES,
            context_stats,
            batter_stats,
            bowler_stats,
            venue_stats,
            vs_pace_stats,
            vs_arm_pace_stats,
            vs_nation_stats,
            matchup_stats,
        )
        dismissal_levels = self._binary_levels(
            "batter_dismissals",
            context_stats,
            batter_stats,
            bowler_stats,
            venue_stats,
            vs_pace_stats,
            vs_arm_pace_stats,
            vs_nation_stats,
            matchup_stats,
        )
        wicket_levels = self._binary_levels(
            "bowler_wickets",
            context_stats,
            batter_stats,
            bowler_stats,
            venue_stats,
            vs_pace_stats,
            vs_arm_pace_stats,
            vs_nation_stats,
            matchup_stats,
        )
        evidence = {
            "global": self.global_stats.n,
            "context": context_stats.n,
            "batter": batter_stats.n,
            "bowler": bowler_stats.n,
            "venue": venue_stats.n,
            "vs_pace": vs_pace_stats.n,
            "vs_arm_pace": vs_arm_pace_stats.n,
            "vs_nation_arm_pace": vs_nation_stats.n,
            "matchup": matchup_stats.n,
        }
        return {
            level: BaselinePrediction(
                batter_runs=run_levels[level],
                extras_runs=extras_levels[level],
                legality=legality_levels[level],
                batter_dismissal=dismissal_levels[level],
                bowler_wicket=wicket_levels[level],
                evidence=evidence,
            )
            for level in BASELINE_LEVELS
        }

    def _multiclass_levels(
        self,
        field_name: str,
        classes: int,
        context_stats: EventStats,
        batter_stats: EventStats,
        bowler_stats: EventStats,
        venue_stats: EventStats,
        vs_pace_stats: EventStats,
        vs_arm_pace_stats: EventStats,
        vs_nation_stats: EventStats,
        matchup_stats: EventStats,
    ) -> dict[str, np.ndarray]:
        global_counts = getattr(self.global_stats, field_name)
        global_probability = (global_counts.astype(np.float64) + 1.0) / (
            self.global_stats.n + classes
        )
        context_probability = _multiclass_posterior(
            getattr(context_stats, field_name),
            context_stats.n,
            global_probability,
            self.smoothing.context,
        )
        batter_probability = _multiclass_posterior(
            getattr(batter_stats, field_name),
            batter_stats.n,
            context_probability,
            self.smoothing.player,
        )
        bowler_probability = _multiclass_posterior(
            getattr(bowler_stats, field_name),
            bowler_stats.n,
            context_probability,
            self.smoothing.player,
        )
        player_probability = self._combine_player_probabilities(
            context_probability,
            batter_probability,
            bowler_probability,
            batter_stats.n,
            bowler_stats.n,
        )
        venue_probability = _multiclass_posterior(
            getattr(venue_stats, field_name),
            venue_stats.n,
            context_probability,
            self.smoothing.venue,
        )
        player_venue_probability = self._blend_venue(
            player_probability, venue_probability, venue_stats.n
        )
        vs_pace_probability = _multiclass_posterior(
            getattr(vs_pace_stats, field_name),
            vs_pace_stats.n,
            player_venue_probability,
            self.smoothing.vs_pace,
        )
        # Side diagnostics: arm/nation shrink from pace but do not parent matchup.
        vs_arm_pace_probability = _multiclass_posterior(
            getattr(vs_arm_pace_stats, field_name),
            vs_arm_pace_stats.n,
            vs_pace_probability,
            self.smoothing.vs_arm_pace,
        )
        vs_nation_probability = _multiclass_posterior(
            getattr(vs_nation_stats, field_name),
            vs_nation_stats.n,
            vs_arm_pace_probability,
            self.smoothing.vs_nation_arm_pace,
        )
        matchup_probability = _multiclass_posterior(
            getattr(matchup_stats, field_name),
            matchup_stats.n,
            vs_pace_probability,
            self.smoothing.matchup,
        )
        return {
            "global": global_probability,
            "context": context_probability,
            "player": player_probability,
            "venue": player_venue_probability,
            "vs_pace": vs_pace_probability,
            "vs_arm_pace": vs_arm_pace_probability,
            "vs_nation_arm_pace": vs_nation_probability,
            "matchup": matchup_probability,
        }

    def _binary_levels(
        self,
        field_name: str,
        context_stats: EventStats,
        batter_stats: EventStats,
        bowler_stats: EventStats,
        venue_stats: EventStats,
        vs_pace_stats: EventStats,
        vs_arm_pace_stats: EventStats,
        vs_nation_stats: EventStats,
        matchup_stats: EventStats,
    ) -> dict[str, float]:
        global_successes = int(getattr(self.global_stats, field_name))
        global_probability = (global_successes + 1.0) / (self.global_stats.n + 2.0)
        context_probability = _binary_posterior(
            int(getattr(context_stats, field_name)),
            context_stats.n,
            global_probability,
            self.smoothing.context,
        )
        batter_probability = _binary_posterior(
            int(getattr(batter_stats, field_name)),
            batter_stats.n,
            context_probability,
            self.smoothing.player,
        )
        bowler_probability = _binary_posterior(
            int(getattr(bowler_stats, field_name)),
            bowler_stats.n,
            context_probability,
            self.smoothing.player,
        )
        player_probability = float(
            self._combine_player_probabilities(
                context_probability,
                batter_probability,
                bowler_probability,
                batter_stats.n,
                bowler_stats.n,
            )
        )
        venue_probability = _binary_posterior(
            int(getattr(venue_stats, field_name)),
            venue_stats.n,
            context_probability,
            self.smoothing.venue,
        )
        player_venue_probability = float(
            self._blend_venue(player_probability, venue_probability, venue_stats.n)
        )
        vs_pace_probability = _binary_posterior(
            int(getattr(vs_pace_stats, field_name)),
            vs_pace_stats.n,
            player_venue_probability,
            self.smoothing.vs_pace,
        )
        vs_arm_pace_probability = _binary_posterior(
            int(getattr(vs_arm_pace_stats, field_name)),
            vs_arm_pace_stats.n,
            vs_pace_probability,
            self.smoothing.vs_arm_pace,
        )
        vs_nation_probability = _binary_posterior(
            int(getattr(vs_nation_stats, field_name)),
            vs_nation_stats.n,
            vs_arm_pace_probability,
            self.smoothing.vs_nation_arm_pace,
        )
        matchup_probability = _binary_posterior(
            int(getattr(matchup_stats, field_name)),
            matchup_stats.n,
            vs_pace_probability,
            self.smoothing.matchup,
        )
        return {
            "global": float(global_probability),
            "context": float(context_probability),
            "player": player_probability,
            "venue": player_venue_probability,
            "vs_pace": float(vs_pace_probability),
            "vs_arm_pace": float(vs_arm_pace_probability),
            "vs_nation_arm_pace": float(vs_nation_probability),
            "matchup": float(matchup_probability),
        }

    def _combine_player_probabilities(
        self,
        context_probability: np.ndarray | float,
        batter_probability: np.ndarray | float,
        bowler_probability: np.ndarray | float,
        batter_n: int,
        bowler_n: int,
    ) -> np.ndarray | float:
        batter_weight = _reliability(batter_n, self.smoothing.player)
        bowler_weight = _reliability(bowler_n, self.smoothing.player)
        context_weight = self.smoothing.context_residual_weight
        total_weight = context_weight + batter_weight + bowler_weight
        return (
            context_weight * context_probability
            + batter_weight * batter_probability
            + bowler_weight * bowler_probability
        ) / total_weight

    def _blend_venue(
        self,
        player_probability: np.ndarray | float,
        venue_probability: np.ndarray | float,
        venue_n: int,
    ) -> np.ndarray | float:
        venue_weight = self.smoothing.venue_max_weight * _reliability(venue_n, self.smoothing.venue)
        return (player_probability + venue_weight * venue_probability) / (1.0 + venue_weight)
