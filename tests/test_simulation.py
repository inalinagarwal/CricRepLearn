"""Tests for the T20 innings simulator."""

from __future__ import annotations

import pytest
import numpy as np

from cric_rep_learn.simulation.attack import (
    BowlerSpell,
    assign_over_quotas,
    build_over_schedule,
    configure_attack,
)
from cric_rep_learn.simulation.phase import t20_phase
from cric_rep_learn.simulation.innings import _sample_runs, simulate_one_innings
from cric_rep_learn.simulation.priors import InningsRateModel


def test_t20_phase_cutoffs() -> None:
    assert t20_phase(0) == "powerplay"
    assert t20_phase(35) == "powerplay"
    assert t20_phase(36) == "middle"
    assert t20_phase(95) == "middle"
    assert t20_phase(96) == "death"


def _neutral_scores() -> dict[str, float]:
    return {"powerplay": 0.04, "middle": 0.04, "death": 0.04}


def test_assign_over_quotas_five_bowlers() -> None:
    attack = [BowlerSpell(f"w{i}", f"W{i}") for i in range(5)]
    assign_over_quotas(attack)
    assert [b.max_overs for b in attack] == [4, 4, 4, 4, 4]


def test_assign_over_quotas_six_bowlers_default() -> None:
    attack = [BowlerSpell(f"w{i}", f"W{i}") for i in range(6)]
    assign_over_quotas(attack)
    assert [b.max_overs for b in attack] == [4, 4, 4, 4, 2, 2]


def test_assign_over_quotas_six_bowlers_alt_plan() -> None:
    attack = [BowlerSpell(f"w{i}", f"W{i}") for i in range(6)]
    assign_over_quotas(attack, six_bowler_plan="4-4-4-3-3-2")
    assert [b.max_overs for b in attack] == [4, 4, 4, 3, 3, 2]


def test_assign_over_quotas_from_actual_matches_observed() -> None:
    from cric_rep_learn.simulation.attack import assign_over_quotas_from_actual

    attack = [
        BowlerSpell("a", "A"),
        BowlerSpell("b", "B"),
        BowlerSpell("c", "C"),
        BowlerSpell("d", "D"),
        BowlerSpell("e", "E"),
        BowlerSpell("f", "F"),
    ]
    assign_over_quotas_from_actual(
        attack,
        {"a": 4.0, "b": 4.0, "c": 4.0, "d": 3.0, "e": 3.0, "f": 2.0},
    )
    assert [b.max_overs for b in attack] == [4, 4, 4, 3, 3, 2]
    assert sum(b.max_overs for b in attack) == 20


def test_assign_over_quotas_from_actual_pads_short_innings() -> None:
    from cric_rep_learn.simulation.attack import assign_over_quotas_from_actual

    attack = [BowlerSpell(f"w{i}", f"W{i}") for i in range(5)]
    assign_over_quotas_from_actual(attack, {f"w{i}": 2.0 for i in range(5)})
    assert sum(b.max_overs for b in attack) == 20
    assert all(1 <= b.max_overs <= 4 for b in attack)


def test_configure_attack_can_preserve_quotas() -> None:
    attack = [
        BowlerSpell("p1", "P1", max_overs=3),
        BowlerSpell("s1", "S1", max_overs=4),
        BowlerSpell("p2", "P2", max_overs=3),
        BowlerSpell("p3", "P3", max_overs=4),
        BowlerSpell("s2", "S2", max_overs=3),
        BowlerSpell("p4", "P4", max_overs=3),
    ]
    attributes = {
        "p1": {"pace_group": "pace"},
        "s1": {"pace_group": "spin"},
        "p2": {"pace_group": "pace"},
        "p3": {"pace_group": "pace"},
        "s2": {"pace_group": "spin"},
        "p4": {"pace_group": "pace"},
    }
    configure_attack(attack, attributes=attributes, assign_quotas=False)
    assert [b.max_overs for b in attack] == [3, 4, 3, 4, 3, 3]
    assert attack[1].pace_group == "spin"


def test_over_schedule_respects_max_overs_and_phases() -> None:
    attack = [
        BowlerSpell(
            "a",
            "A",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.05, "middle": 0.04, "death": 0.03},
        ),
        BowlerSpell(
            "b",
            "B",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.04, "middle": 0.04, "death": 0.02},
        ),
        BowlerSpell(
            "c",
            "C",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.02, "middle": 0.05, "death": 0.03},
        ),
        BowlerSpell(
            "death_king",
            "DeathKing",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.03, "middle": 0.03, "death": 0.09},
        ),
        BowlerSpell(
            "e",
            "E",
            max_overs=4,
            pace_group="pace",
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


def test_death_overs_pace_only() -> None:
    attack = [
        BowlerSpell(
            "pace1",
            "Pace1",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "pace2",
            "Pace2",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "spin1",
            "Spin1",
            max_overs=4,
            pace_group="spin",
            phase_scores={"powerplay": 0.02, "middle": 0.08, "death": 0.10},
        ),
        BowlerSpell(
            "pace3",
            "Pace3",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "spin2",
            "Spin2",
            max_overs=4,
            pace_group="spin",
            phase_scores={"powerplay": 0.02, "middle": 0.07, "death": 0.09},
        ),
    ]
    schedule = build_over_schedule(attack)
    death = [row for row in schedule if row["phase"] == "death"]
    assert len(death) == 4
    assert all(row["pace_group"] == "pace" for row in death)
    assert all(row["bowler_id"] not in {"spin1", "spin2"} for row in death)


def test_top_spinner_bowled_before_death() -> None:
    # Rank-0/1/2 include a spinner who must finish quota in PP/middle.
    attack = [
        BowlerSpell(
            "pace1",
            "Pace1",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "pace2",
            "Pace2",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "axar",
            "Axar",
            max_overs=4,
            pace_group="spin",
            phase_scores={"powerplay": 0.03, "middle": 0.06, "death": 0.02},
        ),
        BowlerSpell(
            "pace3",
            "Pace3",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "sundar",
            "Sundar",
            max_overs=4,
            pace_group="spin",
            phase_scores={"powerplay": 0.03, "middle": 0.05, "death": 0.02},
        ),
    ]
    schedule = build_over_schedule(attack)
    by_id: dict[str, list[dict]] = {}
    for row in schedule:
        by_id.setdefault(row["bowler_id"], []).append(row)
    assert len(by_id["axar"]) == 4
    assert all(row["phase"] != "death" for row in by_id["axar"])
    # Top-3 spinner (rank 2) should be fully used before death.
    assert sum(1 for row in by_id["axar"] if row["phase"] == "middle") >= 2


def test_pp_spin_only_first_over_when_opener_is_spinner() -> None:
    attack = [
        BowlerSpell(
            "sundar",
            "Sundar",
            max_overs=4,
            pace_group="spin",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "pace1",
            "Pace1",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.08, "middle": 0.04, "death": 0.05},
        ),
        BowlerSpell(
            "pace2",
            "Pace2",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "pace3",
            "Pace3",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
        BowlerSpell(
            "pace4",
            "Pace4",
            max_overs=4,
            pace_group="pace",
            phase_scores=_neutral_scores(),
        ),
    ]
    schedule = build_over_schedule(attack)
    pp = [row for row in schedule if row["phase"] == "powerplay"]
    assert pp[0]["bowler_id"] == "sundar"
    assert all(row["bowler_id"] != "sundar" for row in pp[1:])
    assert all(row["pace_group"] == "pace" for row in pp[1:])


def test_configure_attack_sets_pace_and_quotas() -> None:
    attack = [
        BowlerSpell("p1", "P1"),
        BowlerSpell("s1", "S1"),
        BowlerSpell("p2", "P2"),
        BowlerSpell("p3", "P3"),
        BowlerSpell("s2", "S2"),
        BowlerSpell("p4", "P4"),
    ]
    attributes = {
        "p1": {"pace_group": "pace"},
        "s1": {"pace_group": "spin"},
        "p2": {"pace_group": "pace"},
        "p3": {"pace_group": "pace"},
        "s2": {"bowling_style_raw": "SLA Orthodox"},
        "p4": {"pace_group": "pace"},
    }
    configure_attack(attack, attributes=attributes)
    assert [b.max_overs for b in attack] == [4, 4, 4, 4, 2, 2]
    assert attack[0].pace_group == "pace"
    assert attack[1].pace_group == "spin"
    assert attack[4].pace_group == "spin"


def test_sample_runs_in_legal_set() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert _sample_runs(rng, 1.3) in {0, 1, 2, 4, 6}


def test_sample_runs_mean_tracks_sr() -> None:
    from cric_rep_learn.simulation.run_sampler import sample_runs

    rng = np.random.default_rng(1)
    table = {
        "outcomes": [0, 1, 2, 4, 6],
        "buckets": {"sr_1.4_1.8": [0.30, 0.30, 0.12, 0.18, 0.10]},
    }
    draws = [sample_runs(rng, 1.55, table=table) for _ in range(5000)]
    assert abs(float(np.mean(draws)) - 1.55) < 0.3


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
            pace_group="pace",
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


def test_mc_exports_haul_and_milestone_tail_probs() -> None:
    from cric_rep_learn.simulation.innings import simulate_innings

    lineup = [
        {"player_id": f"b{i}", "player_name": f"B{i}", "batting_hand": "left"}
        for i in range(11)
    ]
    attack = [
        BowlerSpell(
            f"w{i}",
            f"W{i}",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.04, "middle": 0.04, "death": 0.04},
        )
        for i in range(5)
    ]
    summary = simulate_innings(
        lineup=lineup,
        attack=attack,
        rates=_FakeRates(),  # type: ignore[arg-type]
        n_sims=80,
        seed=11,
    )
    assert "p_runs_ge30" in summary["batters"][0]
    assert "p_wickets_ge3" in summary["bowlers"][0]
    assert 0.0 <= summary["batters"][0]["p_runs_ge30"] <= 1.0
    assert 0.0 <= summary["bowlers"][0]["p_wickets_ge3"] <= 1.0
    # Survival probs are nested.
    for bowler in summary["bowlers"]:
        assert bowler["p_wickets_ge3"] >= bowler["p_wickets_ge4"] - 1e-12
        assert bowler["p_wickets_ge4"] >= bowler["p_wickets_ge5"] - 1e-12


def test_dismissal_spell_overdispersion_fattens_haul_tails() -> None:
    """Mean-preserving Gamma spell shocks raise rare multi-wicket haul rates."""
    import cric_rep_learn.simulation.innings as innings_mod
    from cric_rep_learn.simulation.innings import simulate_innings

    lineup = [
        {"player_id": f"b{i}", "player_name": f"B{i}", "batting_hand": "right"}
        for i in range(11)
    ]
    attack = [
        BowlerSpell(
            f"w{i}",
            f"W{i}",
            max_overs=4,
            pace_group="pace",
            phase_scores={"powerplay": 0.04, "middle": 0.04, "death": 0.04},
        )
        for i in range(5)
    ]

    class _Moderate(_FakeRates):
        def rates(self, **kwargs):  # type: ignore[no-untyped-def]
            return {
                "expected_sr": 1.2,
                "dismissal_rate": 0.045,
                "level": "fake",
                "phase": kwargs.get("phase"),
                "batting_hand": "right",
                "bowling_arm": "right",
            }

    rates = _Moderate()
    old_phi = innings_mod.DISMISSAL_SPELL_PHI
    old_sr = innings_mod.BATTER_SR_PHI

    def _haul_rate(phi: float, seeds: list[int]) -> tuple[float, float]:
        innings_mod.DISMISSAL_SPELL_PHI = phi
        innings_mod.BATTER_SR_PHI = 0.0
        p4s: list[float] = []
        means: list[float] = []
        for seed in seeds:
            summary = simulate_innings(
                lineup=lineup,
                attack=attack,
                rates=rates,  # type: ignore[arg-type]
                n_sims=200,
                seed=seed,
            )
            p4s.append(float(np.mean([b["p_wickets_ge4"] for b in summary["bowlers"]])))
            means.append(
                float(np.mean([b["expected_wickets"] for b in summary["bowlers"]]))
            )
        return float(np.mean(p4s)), float(np.mean(means))

    seeds = [3, 11, 29, 41, 57]
    try:
        thin_p4, thin_mean = _haul_rate(0.0, seeds)
        fat_p4, fat_mean = _haul_rate(1.5, seeds)
    finally:
        innings_mod.DISMISSAL_SPELL_PHI = old_phi
        innings_mod.BATTER_SR_PHI = old_sr

    assert fat_p4 > thin_p4 + 0.005
    assert abs(fat_mean - thin_mean) / max(thin_mean, 1e-6) < 0.25


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
            pace_group="pace",
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
