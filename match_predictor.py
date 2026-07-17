"""
Score any T20 fixture using trained batter/bowler embeddings.

Each player can appear as a batter and/or bowler (separate embeddings).
Predictions aggregate across opposition bowlers/batters, phases, and
typical in-innings context.
"""

from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    CHECKPOINT_PATH,
    N_OUTCOMES,
    OUTCOME_RUN_VALUES,
    PHASE_BALL_WEIGHTS,
    PROCESSED,
    load_vocab,
    resolve_league,
)
from export_embeddings import load_model
from model import CricketRepModel

INFERENCE_CONTEXT = PROCESSED / "inference_context.json"


@dataclass
class PlayerRole:
    name: str
    bats: bool = True
    bowls: bool = False


@dataclass
class TeamSquad:
    name: str
    players: List[PlayerRole] = field(default_factory=list)

    @property
    def batters(self) -> List[str]:
        return [p.name for p in self.players if p.bats]

    @property
    def bowlers(self) -> List[str]:
        return [p.name for p in self.players if p.bowls]


@dataclass
class MatchContext:
    team_a: TeamSquad
    team_b: TeamSquad
    venue: str
    league: str  # e.g. "t20i", "ipl", "t20s_json"


class MatchPredictor:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        self.device = torch.device("cpu")
        self.model, self.vocab = load_model(checkpoint_path)
        self.model.to(self.device)
        self.phase_states = self._load_phase_states()
        self._bowler_weights = self._load_bowler_weights()

    def _load_phase_states(self) -> Dict[int, np.ndarray]:
        """Scaled match-state medians per phase (precomputed at data prep time)."""
        if INFERENCE_CONTEXT.exists():
            with open(INFERENCE_CONTEXT, encoding="utf-8") as f:
                data = json.load(f)
            return {
                int(k): np.array(v, dtype=np.float32)
                for k, v in data["phase_states"].items()
            }

        # Fallback: typical raw T20 medians (unscaled — less accurate)
        raw = {
            0: [25.0, 0.0, 114.0, 7.0],
            1: [85.0, 2.0, 54.0, 8.5],
            2: [145.0, 5.0, 18.0, 10.0],
        }
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    def _load_bowler_weights(self) -> Dict[Tuple[int, int], float]:
        """Relative bowling share per (bowler_id, league_id) from sample."""
        try:
            df = pd.read_csv(
                "processed/train.csv",
                usecols=["bowler_id", "league_id"],
                nrows=500_000,
            )
        except Exception:
            return {}

        counts = df.groupby(["league_id", "bowler_id"]).size()
        weights = {}
        for league_id in counts.index.get_level_values(0).unique():
            league_counts = counts[league_id]
            total = league_counts.sum()
            for bowler_id, c in league_counts.items():
                weights[(int(bowler_id), int(league_id))] = c / total
        return weights

    def _resolve_id(
        self, name: str, mapping: dict, role: str
    ) -> Optional[int]:
        if name in mapping:
            return mapping[name]
        # fuzzy: case-insensitive exact
        lower = {k.lower(): v for k, v in mapping.items()}
        if name.lower() in lower:
            return lower[name.lower()]
        print(f"  [warn] {role} '{name}' not in training data — skipped")
        return None

    def _venue_id(self, venue: str) -> Optional[int]:
        m = self.vocab["venue_to_id"]
        if venue in m:
            return m[venue]
        lower = {k.lower(): v for k, v in m.items()}
        for key, vid in lower.items():
            if venue.lower() in key or key in venue.lower():
                return vid
        print(f"  [warn] venue '{venue}' not found — using index 0")
        return 0

    def _league_id(self, league: str) -> int:
        folder = resolve_league(league, self.vocab["league_to_id"])
        return self.vocab["league_to_id"][folder]

    def _expected_from_probs(self, probs: np.ndarray) -> dict:
        runs = sum(probs[i] * OUTCOME_RUN_VALUES[i] for i in range(N_OUTCOMES))
        wicket_p = float(probs[7])
        return {
            "exp_runs_per_ball": float(runs),
            "wicket_prob": wicket_p,
            "prob_dot": float(probs[0]),
            "prob_boundary": float(probs[4] + probs[6]),
        }

    @torch.no_grad()
    def _matchup_probs(
        self,
        batter_id: int,
        bowler_id: int,
        venue_id: int,
        league_id: int,
        phase: int,
    ) -> np.ndarray:
        numeric = torch.tensor(
            self.phase_states[phase], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        batter = torch.tensor([batter_id], device=self.device)
        bowler = torch.tensor([bowler_id], device=self.device)
        venue = torch.tensor([venue_id], device=self.device)
        league = torch.tensor([league_id], device=self.device)
        ph = torch.tensor([phase], device=self.device)

        probs = self.model.predict_probs(
            batter, bowler, venue, league, ph, numeric
        )
        return probs.squeeze(0).cpu().numpy()

    def score_batter_vs_bowlers(
        self,
        batter_name: str,
        opposition_bowlers: List[str],
        venue: str,
        league: str,
    ) -> Optional[dict]:
        batter_id = self._resolve_id(
            batter_name, self.vocab["batter_to_id"], "batter"
        )
        if batter_id is None:
            return None

        venue_id = self._venue_id(venue)
        league_id = self._league_id(league)

        bowler_ids = []
        bowler_names = []
        for name in opposition_bowlers:
            bid = self._resolve_id(name, self.vocab["bowler_to_id"], "bowler")
            if bid is not None:
                bowler_ids.append(bid)
                bowler_names.append(name)

        if not bowler_ids:
            return None

        phase_stats = []
        total_weight = 0.0
        agg_runs = 0.0
        agg_wicket = 0.0
        agg_boundary = 0.0

        for phase, phase_balls in enumerate(PHASE_BALL_WEIGHTS):
            phase_runs = 0.0
            phase_wicket = 0.0
            phase_boundary = 0.0
            phase_w = 0.0

            for bid, bname in zip(bowler_ids, bowler_names):
                w = self._bowler_weights.get((bid, league_id), 1.0 / len(bowler_ids))
                stats = self._expected_from_probs(
                    self._matchup_probs(batter_id, bid, venue_id, league_id, phase)
                )
                phase_runs += w * stats["exp_runs_per_ball"]
                phase_wicket += w * stats["wicket_prob"]
                phase_boundary += w * stats["prob_boundary"]
                phase_w += w

            if phase_w > 0:
                phase_runs /= phase_w
                phase_wicket /= phase_w
                phase_boundary /= phase_w

            phase_stats.append(
                {
                    "phase": ["powerplay", "middle", "death"][phase],
                    "exp_runs_per_ball": phase_runs,
                    "wicket_prob": phase_wicket,
                    "boundary_prob": phase_boundary,
                }
            )

            w_balls = phase_balls
            agg_runs += phase_runs * w_balls
            agg_wicket += phase_wicket * w_balls
            agg_boundary += phase_boundary * w_balls
            total_weight += w_balls

        return {
            "player": batter_name,
            "role": "batter",
            "exp_runs_per_ball": agg_runs / total_weight,
            "wicket_prob_per_ball": agg_wicket / total_weight,
            "boundary_prob_per_ball": agg_boundary / total_weight,
            "by_phase": phase_stats,
        }

    def score_bowler_vs_batters(
        self,
        bowler_name: str,
        opposition_batters: List[str],
        venue: str,
        league: str,
    ) -> Optional[dict]:
        bowler_id = self._resolve_id(
            bowler_name, self.vocab["bowler_to_id"], "bowler"
        )
        if bowler_id is None:
            return None

        venue_id = self._venue_id(venue)
        league_id = self._league_id(league)

        batter_ids = []
        for name in opposition_batters:
            bid = self._resolve_id(name, self.vocab["batter_to_id"], "batter")
            if bid is not None:
                batter_ids.append(bid)

        if not batter_ids:
            return None

        total_weight = 0.0
        agg_runs = 0.0
        agg_wicket = 0.0

        for phase, phase_balls in enumerate(PHASE_BALL_WEIGHTS):
            phase_runs = 0.0
            phase_wicket = 0.0
            n = len(batter_ids)

            for bid in batter_ids:
                stats = self._expected_from_probs(
                    self._matchup_probs(bid, bowler_id, venue_id, league_id, phase)
                )
                phase_runs += stats["exp_runs_per_ball"]
                phase_wicket += stats["wicket_prob"]

            phase_runs /= n
            phase_wicket /= n

            agg_runs += phase_runs * phase_balls
            agg_wicket += phase_wicket * phase_balls
            total_weight += phase_balls

        return {
            "player": bowler_name,
            "role": "bowler",
            "exp_runs_conceded_per_ball": agg_runs / total_weight,
            "wicket_prob_per_ball": agg_wicket / total_weight,
        }

    def predict_match(self, ctx: MatchContext) -> dict:
        venue = ctx.venue
        league = ctx.league

        results = {
            "venue": venue,
            "league": league,
            "team_a": ctx.team_a.name,
            "team_b": ctx.team_b.name,
            "batting_scores": [],
            "bowling_scores": [],
        }

        # Team A batters vs Team B bowlers
        for batter in ctx.team_a.batters:
            s = self.score_batter_vs_bowlers(
                batter, ctx.team_b.bowlers, venue, league
            )
            if s:
                results["batting_scores"].append({**s, "team": ctx.team_a.name})

        # Team B batters vs Team A bowlers
        for batter in ctx.team_b.batters:
            s = self.score_batter_vs_bowlers(
                batter, ctx.team_a.bowlers, venue, league
            )
            if s:
                results["batting_scores"].append({**s, "team": ctx.team_b.name})

        # Team A bowlers vs Team B batters
        for bowler in ctx.team_a.bowlers:
            s = self.score_bowler_vs_batters(
                bowler, ctx.team_b.batters, venue, league
            )
            if s:
                results["bowling_scores"].append({**s, "team": ctx.team_a.name})

        # Team B bowlers vs Team A batters
        for bowler in ctx.team_b.bowlers:
            s = self.score_bowler_vs_batters(
                bowler, ctx.team_a.batters, venue, league
            )
            if s:
                results["bowling_scores"].append({**s, "team": ctx.team_b.name})

        results["batting_scores"].sort(
            key=lambda x: x["exp_runs_per_ball"], reverse=True
        )
        results["bowling_scores"].sort(
            key=lambda x: x["wicket_prob_per_ball"], reverse=True
        )
        return results


def load_squad_from_json(path: Path) -> MatchContext:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def parse_team(data: dict) -> TeamSquad:
        players = []
        for p in data["players"]:
            if isinstance(p, str):
                players.append(PlayerRole(name=p, bats=True))
            else:
                players.append(
                    PlayerRole(
                        name=p["name"],
                        bats=p.get("bats", True),
                        bowls=p.get("bowls", False),
                    )
                )
        return TeamSquad(name=data["name"], players=players)

    return MatchContext(
        team_a=parse_team(raw["team_a"]),
        team_b=parse_team(raw["team_b"]),
        venue=raw["venue"],
        league=raw.get("league", "t20i"),
    )
