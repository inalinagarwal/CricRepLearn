"""Monte Carlo T20 innings simulator on hierarchical Bayes priors."""

from __future__ import annotations

from typing import Any

import numpy as np

from cric_rep_learn.simulation.attack import (
    BowlerSpell,
    ball_bowler_schedule,
    build_over_schedule,
)
from cric_rep_learn.simulation.phase import t20_phase
from cric_rep_learn.simulation.priors import InningsRateModel
from cric_rep_learn.simulation.state import BatterInnings, InningsState


def _sample_runs(rng: np.random.Generator, expected_sr: float) -> int:
    """Discrete run sample with mean ≈ expected_sr (0/1/2/4/6 only)."""
    # Mixture calibrated roughly to T20 scoring shapes.
    # Solve for weights favoring boundaries as SR rises.
    sr = float(max(0.05, min(expected_sr, 3.0)))
    # Base template then tilt toward 4/6 when SR is high.
    p0, p1, p2, p4, p6 = 0.38, 0.36, 0.10, 0.10, 0.06
    tilt = (sr - 1.2) * 0.12
    p4 = max(0.02, p4 + tilt)
    p6 = max(0.01, p6 + tilt * 0.7)
    p0 = max(0.15, p0 - tilt * 0.6)
    p1 = max(0.15, p1 - tilt * 0.3)
    total = p0 + p1 + p2 + p4 + p6
    probs = np.array([p0, p1, p2, p4, p6], dtype=np.float64) / total
    # Adjust mean toward target by rejection-light scaling via poisson mix.
    draws = rng.choice([0, 1, 2, 4, 6], p=probs)
    # One correction step: sometimes bump/drop to move mean.
    mean = float(np.dot(probs, [0, 1, 2, 4, 6]))
    if mean < sr - 0.15 and draws in (0, 1) and rng.random() < min(0.45, sr - mean):
        draws = 4 if rng.random() < 0.55 else 6
    elif mean > sr + 0.25 and draws in (4, 6) and rng.random() < min(0.45, mean - sr):
        draws = 1 if rng.random() < 0.7 else 0
    return int(draws)


def _new_state(lineup: list[dict[str, str]]) -> InningsState:
    batters = [
        BatterInnings(
            player_id=row["player_id"],
            player_name=row["player_name"],
            batting_hand=row.get("batting_hand") or "unknown",
        )
        for row in lineup
    ]
    if len(batters) < 2:
        raise ValueError("need at least two batters in the lineup")
    batters[0].entered = True
    batters[1].entered = True
    return InningsState(batters=batters, next_batter=2)


def simulate_one_innings(
    *,
    lineup: list[dict[str, str]],
    attack: list[BowlerSpell],
    rates: InningsRateModel,
    rng: np.random.Generator,
) -> dict[str, Any]:
    state = _new_state(lineup)
    over_sched = build_over_schedule(attack)
    ball_sched = ball_bowler_schedule(over_sched)
    rate_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    for slot in ball_sched:
        if not state.active():
            break
        phase = t20_phase(state.legal_balls)
        striker = state.batters[state.striker]
        bowler_id = slot["bowler_id"]
        cache_key = (striker.player_id, bowler_id, phase)
        if cache_key not in rate_cache:
            rate_cache[cache_key] = rates.rates(
                batter_id=striker.player_id,
                bowler_id=bowler_id,
                phase=phase,
                batting_hand=striker.batting_hand,
            )
        ball_rates = rate_cache[cache_key]
        striker.balls += 1.0

        if rng.random() < ball_rates["dismissal_rate"]:
            striker.out = True
            striker.dismissals += 1.0
            state.wickets += 1
            state.legal_balls += 1
            if state.wickets >= 10 or state.next_batter >= len(state.batters):
                state.mark_finished("all_out")
                break
            # New batter takes the striker's end.
            state.striker = state.next_batter
            state.batters[state.striker].entered = True
            state.next_batter += 1
            continue

        runs = _sample_runs(rng, ball_rates["expected_sr"])
        striker.runs += runs
        state.score += runs
        state.legal_balls += 1
        if runs % 2 == 1:
            state.swap_strike()
        # End of over: swap strike.
        if state.legal_balls % 6 == 0:
            state.swap_strike()

    if not state.finished:
        if state.legal_balls >= state.scheduled_balls:
            state.mark_finished("overs_complete")
        else:
            state.mark_finished("incomplete")

    summary = state.summary()
    summary["overs_bowled"] = [
        {"over": row["over"], "bowler_id": row["bowler_id"], "bowler_name": row["bowler_name"], "phase": row["phase"]}
        for row in over_sched
    ]
    return summary


def simulate_innings(
    *,
    lineup: list[dict[str, str]],
    attack: list[BowlerSpell],
    rates: InningsRateModel,
    n_sims: int = 400,
    seed: int = 7,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    runs = np.zeros(n_sims, dtype=np.float64)
    wickets = np.zeros(n_sims, dtype=np.float64)
    balls = np.zeros(n_sims, dtype=np.float64)
    batter_runs: dict[str, list[float]] = {
        row["player_id"]: [] for row in lineup
    }
    batter_balls: dict[str, list[float]] = {
        row["player_id"]: [] for row in lineup
    }
    sample_schedule = None

    for i in range(n_sims):
        result = simulate_one_innings(
            lineup=lineup, attack=attack, rates=rates, rng=rng
        )
        runs[i] = result["runs"]
        wickets[i] = result["wickets"]
        balls[i] = result["balls"]
        if sample_schedule is None:
            sample_schedule = result["overs_bowled"]
        for batter in result["batters"]:
            batter_runs[batter["player_id"]].append(batter["runs"])
            batter_balls[batter["player_id"]].append(batter["balls"])

    batter_summary = []
    for row in lineup:
        pid = row["player_id"]
        br = np.asarray(batter_runs[pid], dtype=np.float64)
        bb = np.asarray(batter_balls[pid], dtype=np.float64)
        batter_summary.append(
            {
                "player_id": pid,
                "player_name": row["player_name"],
                "batting_hand": row.get("batting_hand"),
                "expected_runs": float(br.mean()),
                "runs_p10": float(np.quantile(br, 0.10)),
                "runs_p50": float(np.quantile(br, 0.50)),
                "runs_p90": float(np.quantile(br, 0.90)),
                "expected_balls": float(bb.mean()),
                "p_batted": float(np.mean(bb > 0)),
            }
        )

    return {
        "n_sims": n_sims,
        "seed": seed,
        "team": {
            "expected_runs": float(runs.mean()),
            "runs_std": float(runs.std()),
            "runs_p10": float(np.quantile(runs, 0.10)),
            "runs_p50": float(np.quantile(runs, 0.50)),
            "runs_p90": float(np.quantile(runs, 0.90)),
            "expected_wickets": float(wickets.mean()),
            "expected_balls": float(balls.mean()),
        },
        "batters": batter_summary,
        "bowling_schedule": sample_schedule,
        "method": (
            "Monte Carlo T20 innings on HB matchup priors + phase shrink + "
            "L/R handedness multipliers; strike rotation; max 4 overs/bowler; "
            "bowler-attributable dismissals only"
        ),
    }
