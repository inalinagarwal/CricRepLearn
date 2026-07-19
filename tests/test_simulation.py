"""Tests for the T20 innings simulator."""

from __future__ import annotations

import pytest
import numpy as np

from cric_rep_learn.simulation.attack import BowlerSpell, build_over_schedule
from cric_rep_learn.simulation.phase import t20_phase
from cric_rep_learn.simulation.innings import _sample_runs, simulate_one_innings
from cric_rep_learn.simulation.priors import InningsRateModel


def test_t20_phase_cutoffs() -> None:
    assert t20_phase(0) == "powerplay"
    assert t20_phase(35) == "powerplay"
    assert t20_phase(36) == "middle"
    assert t20_phase(95) == "middle"
    assert t20_phase(96) == "death"


def test_over_schedule_respects_max_overs_and_phases() -> None:
    attack = [
        BowlerSpell(
            "a",
            "A",
            max_overs=4,
            phase_scores={"powerplay": 0.05, "middle": 0.04, "death": 0.03},
        ),
        BowlerSpell(
            "b",
            "B",
            max_overs=4,
            phase_scores={"powerplay": 0.04, "middle": 0.04, "death": 0.02},
        ),
        BowlerSpell(
            "c",
            "C",
            max_overs=4,
            phase_scores={"powerplay": 0.02, "middle": 0.05, "death": 0.03},
        ),
        BowlerSpell(
            "death_king",
            "DeathKing",
            max_overs=4,
            phase_scores={"powerplay": 0.03, "middle": 0.03, "death": 0.09},
        ),
        BowlerSpell(
            "e",
            "E",
            max_overs=4,
            phase_scores={"powerplay": 0.03, "middle": 0.04, "death": 0.03},
        ),
    ]
    schedule = build_over_schedule(attack)
    assert len(schedule) == 20
    counts: dict[str, int] = {}
    for row in schedule:
        counts[row["bowler_id"]] = counts.get(row["bowler_id"], 0) + 1
        assert 0 <= row["over"] < 20
    assert all(v <= 4 for v in counts.values())
    assert sum(counts.values()) == 20
    assert schedule[0]["phase"] == "powerplay"
    assert schedule[16]["phase"] == "death"
    # Highest death score should take the bulk of death overs (alternating constraint).
    death_bowlers = [row["bowler_id"] for row in schedule if row["phase"] == "death"]
    assert death_bowlers.count("death_king") >= 2
    assert max(death_bowlers.count(b) for b in set(death_bowlers)) == death_bowlers.count(
        "death_king"
    )



def test_sample_runs_in_legal_set() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert _sample_runs(rng, 1.3) in {0, 1, 2, 4, 6}


class _FakeRates(InningsRateModel):
    def __init__(self) -> None:  # noqa: D107
        pass

    def rates(self, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "expected_sr": 1.2,
            "dismissal_rate": 0.04,
            "level": "fake",
            "phase": kwargs.get("phase"),
            "batting_hand": "left",
            "bowling_arm": "right",
        }


def test_one_innings_finishes() -> None:
    lineup = [
        {"player_id": f"b{i}", "player_name": f"B{i}", "batting_hand": "left"}
        for i in range(11)
    ]
    attack = [
        BowlerSpell(
            f"w{i}",
            f"W{i}",
            max_overs=4,
            phase_scores={"powerplay": 0.04, "middle": 0.04, "death": 0.04},
        )
        for i in range(5)
    ]
    result = simulate_one_innings(
        lineup=lineup,
        attack=attack,
        rates=_FakeRates(),  # type: ignore[arg-type]
        rng=np.random.default_rng(1),
    )
    assert result["balls"] <= 120
    assert result["wickets"] <= 10
    assert result["finish_reason"] in {"overs_complete", "all_out", "incomplete"}
    assert "bowlers" in result
    assert "overs" in result
    assert sum(b["wickets"] for b in result["bowlers"]) == result["wickets"]
    assert sum(b["balls"] for b in result["bowlers"]) == result["balls"]
    assert abs(sum(b["runs"] for b in result["bowlers"]) - result["runs"]) < 1e-6
    assert abs(sum(o["runs"] for o in result["overs"]) - result["runs"]) < 1e-6
    assert abs(sum(o["wickets"] for o in result["overs"]) - result["wickets"]) < 1e-6
    assert all("partnership" in o for o in result["overs"])


def test_phase_weights_sum() -> None:
    from cric_rep_learn.simulation.phase_score import DEFAULT_PHASE_WEIGHTS, summarize_phases

    assert abs(sum(DEFAULT_PHASE_WEIGHTS.values()) - 1.0) < 1e-9
    overs = [
        {"over": i, "phase": "powerplay" if i < 6 else ("death" if i >= 16 else "middle"),
         "expected_runs": 8.0, "expected_wickets": 0.2}
        for i in range(20)
    ]
    phases = summarize_phases(overs)
    assert phases["powerplay"]["expected_runs"] == pytest.approx(48.0)
    assert phases["middle"]["expected_runs"] == pytest.approx(80.0)
    assert phases["death"]["expected_runs"] == pytest.approx(32.0)
    assert phases["phase_weighted_score"] > 0


def test_chase_pressure_tilts_and_stops_at_target() -> None:
    from cric_rep_learn.simulation.chase import apply_chase_pressure

    impacts = {
        "cells": {
            "rr_2_2.5|w0_2": {
                "sr_mult": 1.2,
                "dismiss_mult": 1.3,
                "win_confidence": 0.2,
            }
        },
        "rrr_marginal": {},
    }
    pressed = apply_chase_pressure(
        sr=1.0,
        dismissal_rate=0.05,
        target=150,
        score=20,
        legal_balls=60,
        wickets=1,
        impacts=impacts,
    )
    assert pressed["required_rate"] == pytest.approx(130 / 60)
    assert pressed["expected_sr"] > 1.0
    assert pressed["dismissal_rate"] > 0.05
    assert pressed["win_confidence"] == 0.2

    lineup = [
        {"player_id": f"b{i}", "player_name": f"B{i}", "batting_hand": "left"}
        for i in range(11)
    ]
    attack = [
        BowlerSpell(
            f"w{i}",
            f"W{i}",
            max_overs=4,
            phase_scores={"powerplay": 0.04, "middle": 0.04, "death": 0.04},
        )
        for i in range(5)
    ]

    class _Easy(_FakeRates):
        def rates(self, **kwargs):  # type: ignore[no-untyped-def]
            return {
                "expected_sr": 2.5,
                "dismissal_rate": 0.01,
                "level": "fake",
                "phase": kwargs.get("phase"),
                "batting_hand": "left",
                "bowling_arm": "right",
            }

    result = simulate_one_innings(
        lineup=lineup,
        attack=attack,
        rates=_Easy(),  # type: ignore[arg-type]
        rng=np.random.default_rng(2),
        target=40.0,
        chase_impacts=impacts,
    )
    assert result["finish_reason"] == "target_reached"
    assert result["chase_won"] is True
    assert result["runs"] >= 40
