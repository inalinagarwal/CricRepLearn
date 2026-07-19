"""Innings state for the T20 simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatterInnings:
    player_id: str
    player_name: str
    batting_hand: str
    runs: float = 0.0
    balls: float = 0.0
    dismissals: float = 0.0
    entered: bool = False
    out: bool = False


@dataclass
class InningsState:
    scheduled_balls: int = 120
    legal_balls: int = 0
    score: float = 0.0
    wickets: int = 0
    striker: int = 0
    non_striker: int = 1
    next_batter: int = 2
    batters: list[BatterInnings] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None

    def active(self) -> bool:
        return (
            not self.finished
            and self.legal_balls < self.scheduled_balls
            and self.wickets < 10
            and self.striker is not None
        )

    def mark_finished(self, reason: str) -> None:
        self.finished = True
        self.finish_reason = reason

    def swap_strike(self) -> None:
        self.striker, self.non_striker = self.non_striker, self.striker

    def summary(self) -> dict[str, Any]:
        return {
            "runs": self.score,
            "wickets": self.wickets,
            "balls": self.legal_balls,
            "overs": f"{self.legal_balls // 6}.{self.legal_balls % 6}",
            "finish_reason": self.finish_reason,
            "batters": [
                {
                    "player_id": b.player_id,
                    "player_name": b.player_name,
                    "batting_hand": b.batting_hand,
                    "runs": b.runs,
                    "balls": b.balls,
                    "dismissals": b.dismissals,
                    "entered": b.entered,
                    "out": b.out,
                }
                for b in self.batters
            ],
        }
