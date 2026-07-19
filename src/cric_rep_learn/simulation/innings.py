"""Monte Carlo T20 innings simulator on hierarchical Bayes priors."""

from __future__ import annotations

from typing import Any

import numpy as np

from cric_rep_learn.simulation.attack import (
    BowlerSpell,
    ball_bowler_schedule,
    build_over_schedule,
)
from cric_rep_learn.simulation.chase import apply_chase_pressure
from cric_rep_learn.simulation.partnership import partnership_tilt
from cric_rep_learn.simulation.phase import t20_phase
from cric_rep_learn.simulation.phase_score import phase_for_over, summarize_phases
from cric_rep_learn.simulation.priors import InningsRateModel
from cric_rep_learn.simulation.run_sampler import sample_runs
from cric_rep_learn.simulation.state import BatterInnings, InningsState

# Mean-preserving Gamma overdispersion (E[mult]=1, Var=phi). Fattens haul /
# big-score tails without shifting ball-level mean rates.
DISMISSAL_SPELL_PHI = 0.90
BATTER_SR_PHI = 0.35


def _sample_runs(rng: np.random.Generator, expected_sr: float) -> int:
    """Backward-compatible wrapper around train-calibrated sampler."""
    return sample_runs(rng, expected_sr)


def _gamma_multipliers(
    rng: np.random.Generator,
    keys: list[str],
    *,
    phi: float,
) -> dict[str, float]:
    """Draw mean-1 Gamma multipliers; phi<=0 → all ones."""
    if phi <= 1e-12 or not keys:
        return {k: 1.0 for k in keys}
    shape = 1.0 / phi
    return {k: float(rng.gamma(shape, scale=phi)) for k in keys}


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


def _empty_over_row(over: int, bowler: dict[str, Any], pair: tuple[str, str]) -> dict[str, Any]:
    return {
        "over": over,
        "phase": phase_for_over(over),
        "bowler_id": bowler["bowler_id"],
        "bowler_name": bowler["bowler_name"],
        "runs": 0.0,
        "wickets": 0.0,
        "balls": 0.0,
        "striker_name": pair[0],
        "non_striker_name": pair[1],
        "partnership": f"{pair[0]} & {pair[1]}",
    }


def simulate_one_innings(
    *,
    lineup: list[dict[str, str]],
    attack: list[BowlerSpell],
    rates: InningsRateModel,
    rng: np.random.Generator,
    target: float | None = None,
    chase_impacts: dict[str, Any] | None = None,
    partnership_index: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    state = _new_state(lineup)
    over_sched = build_over_schedule(attack)
    ball_sched = ball_bowler_schedule(over_sched)
    rate_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    confidences: list[float] = []
    partnership_index = partnership_index or {}
    # Per-innings spell / form shocks → multi-wicket hauls and big knocks.
    dismiss_mult = _gamma_multipliers(
        rng, [b.player_id for b in attack], phi=DISMISSAL_SPELL_PHI
    )
    sr_mult = _gamma_multipliers(
        rng, [row["player_id"] for row in lineup], phi=BATTER_SR_PHI
    )
    bowler_figures: dict[str, dict[str, Any]] = {
        b.player_id: {
            "player_id": b.player_id,
            "player_name": b.player_name,
            "balls": 0.0,
            "runs": 0.0,
            "wickets": 0.0,
        }
        for b in attack
    }
    over_rows: list[dict[str, Any]] = []
    current_over: dict[str, Any] | None = None
    last_over_idx = -1

    for slot in ball_sched:
        if not state.active():
            break
        if target is not None and state.score >= target:
            state.mark_finished("target_reached")
            break

        over_idx = int(slot["over"])
        if over_idx != last_over_idx:
            if current_over is not None:
                over_rows.append(current_over)
            pair = (
                state.batters[state.striker].player_name,
                state.batters[state.non_striker].player_name,
            )
            current_over = _empty_over_row(over_idx, slot, pair)
            last_over_idx = over_idx

        phase = t20_phase(state.legal_balls)
        striker = state.batters[state.striker]
        non_striker = state.batters[state.non_striker]
        bowler_id = slot["bowler_id"]
        figures = bowler_figures[bowler_id]
        cache_key = (striker.player_id, bowler_id, phase)
        if cache_key not in rate_cache:
            rate_cache[cache_key] = rates.rates(
                batter_id=striker.player_id,
                bowler_id=bowler_id,
                phase=phase,
                batting_hand=striker.batting_hand,
            )
        ball_rates = dict(rate_cache[cache_key])

        # Two batters at the crease: mild familiarity tilt.
        tilt = partnership_tilt(
            striker.player_id,
            non_striker.player_id,
            index=partnership_index,
        )
        ball_rates["expected_sr"] = (
            float(ball_rates["expected_sr"])
            * tilt["sr_mult"]
            * sr_mult.get(striker.player_id, 1.0)
        )
        ball_rates["dismissal_rate"] = (
            float(ball_rates["dismissal_rate"])
            * tilt["dismiss_mult"]
            * dismiss_mult.get(bowler_id, 1.0)
        )

        # Wicket load: early collapses suppress SR and raise hazard.
        if state.wickets >= 3 and state.legal_balls < 60:
            load = min(0.20, 0.04 * (state.wickets - 2))
            ball_rates["expected_sr"] *= 1.0 - load
            ball_rates["dismissal_rate"] *= 1.0 + load * 0.8
        elif state.wickets >= 6:
            load = min(0.15, 0.03 * (state.wickets - 5))
            ball_rates["expected_sr"] *= 1.0 - load * 0.5
            ball_rates["dismissal_rate"] *= 1.0 + load

        if target is not None and chase_impacts is not None:
            pressed = apply_chase_pressure(
                sr=float(ball_rates["expected_sr"]),
                dismissal_rate=float(ball_rates["dismissal_rate"]),
                target=float(target),
                score=float(state.score),
                legal_balls=int(state.legal_balls),
                wickets=int(state.wickets),
                scheduled_balls=int(state.scheduled_balls),
                impacts=chase_impacts,
            )
            ball_rates["expected_sr"] = pressed["expected_sr"]
            ball_rates["dismissal_rate"] = pressed["dismissal_rate"]
            if pressed.get("win_confidence") is not None:
                confidences.append(float(pressed["win_confidence"]))

        ball_rates["dismissal_rate"] = float(
            min(max(float(ball_rates["dismissal_rate"]), 1e-4), 0.40)
        )
        ball_rates["expected_sr"] = float(
            min(max(float(ball_rates["expected_sr"]), 0.05), 3.5)
        )

        striker.balls += 1.0
        figures["balls"] += 1.0
        assert current_over is not None
        current_over["balls"] += 1.0

        if rng.random() < ball_rates["dismissal_rate"]:
            striker.out = True
            striker.dismissals += 1.0
            figures["wickets"] += 1.0
            current_over["wickets"] += 1.0
            state.wickets += 1
            state.legal_balls += 1
            if state.wickets >= 10 or state.next_batter >= len(state.batters):
                state.mark_finished("all_out")
                break
            state.striker = state.next_batter
            state.batters[state.striker].entered = True
            state.next_batter += 1
            continue

        runs = _sample_runs(rng, ball_rates["expected_sr"])
        striker.runs += runs
        if runs == 0:
            striker.dots += 1.0
        elif runs == 4:
            striker.fours += 1.0
        elif runs == 6:
            striker.sixes += 1.0
        figures["runs"] += runs
        current_over["runs"] += runs
        state.score += runs
        state.legal_balls += 1
        if target is not None and state.score >= target:
            state.mark_finished("target_reached")
            break
        if runs % 2 == 1:
            state.swap_strike()
        if state.legal_balls % 6 == 0:
            state.swap_strike()

    if current_over is not None:
        over_rows.append(current_over)

    if not state.finished:
        if target is not None and state.score >= target:
            state.mark_finished("target_reached")
        elif state.legal_balls >= state.scheduled_balls:
            state.mark_finished("overs_complete")
        else:
            state.mark_finished("incomplete")

    summary = state.summary()
    summary["overs"] = over_rows
    summary["overs_bowled"] = [
        {
            "over": row["over"],
            "bowler_id": row["bowler_id"],
            "bowler_name": row["bowler_name"],
            "phase": row["phase"],
        }
        for row in over_sched
    ]
    summary["bowlers"] = list(bowler_figures.values())
    summary["target"] = target
    summary["chase_won"] = (
        bool(target is not None and state.score >= target)
        if target is not None
        else None
    )
    summary["mean_win_confidence"] = (
        float(np.mean(confidences)) if confidences else None
    )
    return summary


def simulate_innings(
    *,
    lineup: list[dict[str, str]],
    attack: list[BowlerSpell],
    rates: InningsRateModel,
    n_sims: int = 400,
    seed: int = 7,
    target: float | None = None,
    chase_impacts: dict[str, Any] | None = None,
    partnership_index: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    runs = np.zeros(n_sims, dtype=np.float64)
    wickets = np.zeros(n_sims, dtype=np.float64)
    balls = np.zeros(n_sims, dtype=np.float64)
    chase_wins = np.zeros(n_sims, dtype=np.float64)
    confidences: list[float] = []
    batter_runs: dict[str, list[float]] = {row["player_id"]: [] for row in lineup}
    batter_balls: dict[str, list[float]] = {row["player_id"]: [] for row in lineup}
    batter_fours: dict[str, list[float]] = {row["player_id"]: [] for row in lineup}
    batter_sixes: dict[str, list[float]] = {row["player_id"]: [] for row in lineup}
    bowler_wickets: dict[str, list[float]] = {b.player_id: [] for b in attack}
    bowler_runs: dict[str, list[float]] = {b.player_id: [] for b in attack}
    bowler_balls: dict[str, list[float]] = {b.player_id: [] for b in attack}
    over_runs = np.zeros((n_sims, 20), dtype=np.float64)
    over_wickets = np.zeros((n_sims, 20), dtype=np.float64)
    over_played = np.zeros((n_sims, 20), dtype=np.float64)
    sample_schedule = None
    sample_partnerships: dict[int, str] = {}

    for i in range(n_sims):
        result = simulate_one_innings(
            lineup=lineup,
            attack=attack,
            rates=rates,
            rng=rng,
            target=target,
            chase_impacts=chase_impacts,
            partnership_index=partnership_index,
        )
        runs[i] = result["runs"]
        wickets[i] = result["wickets"]
        balls[i] = result["balls"]
        if result.get("chase_won") is not None:
            chase_wins[i] = 1.0 if result["chase_won"] else 0.0
        if result.get("mean_win_confidence") is not None:
            confidences.append(float(result["mean_win_confidence"]))
        if sample_schedule is None:
            sample_schedule = result["overs_bowled"]
        for over in result.get("overs") or []:
            idx = int(over["over"])
            if 0 <= idx < 20:
                over_runs[i, idx] = float(over["runs"])
                over_wickets[i, idx] = float(over["wickets"])
                over_played[i, idx] = 1.0
                if idx not in sample_partnerships:
                    sample_partnerships[idx] = str(over.get("partnership") or "")
        for batter in result["batters"]:
            batter_runs[batter["player_id"]].append(batter["runs"])
            batter_balls[batter["player_id"]].append(batter["balls"])
            batter_fours[batter["player_id"]].append(float(batter.get("fours") or 0.0))
            batter_sixes[batter["player_id"]].append(float(batter.get("sixes") or 0.0))
        for bowler in result["bowlers"]:
            pid = bowler["player_id"]
            bowler_wickets[pid].append(float(bowler["wickets"]))
            bowler_runs[pid].append(float(bowler["runs"]))
            bowler_balls[pid].append(float(bowler["balls"]))

    batter_summary = []
    for row in lineup:
        pid = row["player_id"]
        br = np.asarray(batter_runs[pid], dtype=np.float64)
        bb = np.asarray(batter_balls[pid], dtype=np.float64)
        bf = np.asarray(batter_fours[pid], dtype=np.float64)
        bs = np.asarray(batter_sixes[pid], dtype=np.float64)
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
                "expected_fours": float(bf.mean()),
                "expected_sixes": float(bs.mean()),
                "p_batted": float(np.mean(bb > 0)),
                "p_runs_ge30": float(np.mean(br >= 30.0)),
                "p_runs_ge50": float(np.mean(br >= 50.0)),
                "p_runs_ge100": float(np.mean(br >= 100.0)),
                "p_fours_ge4": float(np.mean(bf >= 4.0)),
                "p_sixes_ge2": float(np.mean(bs >= 2.0)),
            }
        )

    bowler_summary = []
    for bowler in attack:
        pid = bowler.player_id
        bw = np.asarray(bowler_wickets[pid], dtype=np.float64)
        bruns = np.asarray(bowler_runs[pid], dtype=np.float64)
        bb = np.asarray(bowler_balls[pid], dtype=np.float64)
        overs = bb / 6.0
        economy = np.divide(bruns, overs, out=np.zeros_like(bruns), where=overs > 0)
        bowler_summary.append(
            {
                "player_id": pid,
                "player_name": bowler.player_name,
                "expected_wickets": float(bw.mean()),
                "wickets_p10": float(np.quantile(bw, 0.10)),
                "wickets_p50": float(np.quantile(bw, 0.50)),
                "wickets_p90": float(np.quantile(bw, 0.90)),
                "p_wickets_ge2": float(np.mean(bw >= 2.0)),
                "p_wickets_ge3": float(np.mean(bw >= 3.0)),
                "p_wickets_ge4": float(np.mean(bw >= 4.0)),
                "p_wickets_ge5": float(np.mean(bw >= 5.0)),
                "expected_runs_conceded": float(bruns.mean()),
                "expected_balls": float(bb.mean()),
                "expected_overs": float(overs.mean()),
                "expected_economy": float(economy.mean()) if overs.mean() > 0 else None,
            }
        )

    over_summary = []
    schedule_by_over = {
        int(row["over"]): row for row in (sample_schedule or [])
    }
    for over in range(20):
        rr = over_runs[:, over]
        ww = over_wickets[:, over]
        played = over_played[:, over]
        sched = schedule_by_over.get(over, {})
        over_summary.append(
            {
                "over": over,
                "over_label": f"{over + 1}",
                "phase": phase_for_over(over),
                "bowler_id": sched.get("bowler_id"),
                "bowler_name": sched.get("bowler_name"),
                "expected_runs": float(rr.mean()),
                "runs_p10": float(np.quantile(rr, 0.10)),
                "runs_p50": float(np.quantile(rr, 0.50)),
                "runs_p90": float(np.quantile(rr, 0.90)),
                "expected_wickets": float(ww.mean()),
                "p_over_bowled": float(played.mean()),
                "sample_partnership": sample_partnerships.get(over),
            }
        )

    phases = summarize_phases(over_summary)
    method = (
        "Monte Carlo T20 innings on HB matchup priors + phase shrink + "
        "L/R handedness + partnership familiarity + wicket-load tilt; "
        "mean-preserving spell dismissal / batter SR overdispersion for "
        "haul and big-score tails; "
        "per-over runs/wickets; PP/middle/death weighted score; "
        "full XI batting; max 4 overs/bowler"
    )
    team: dict[str, Any] = {
        "expected_runs": float(runs.mean()),
        "runs_std": float(runs.std()),
        "runs_p10": float(np.quantile(runs, 0.10)),
        "runs_p50": float(np.quantile(runs, 0.50)),
        "runs_p90": float(np.quantile(runs, 0.90)),
        "expected_wickets": float(wickets.mean()),
        "expected_balls": float(balls.mean()),
        "phase_weighted_score": phases["phase_weighted_score"],
    }
    if target is not None:
        team["target"] = float(target)
        team["p_chase_win"] = float(chase_wins.mean())
        if confidences:
            team["mean_win_confidence"] = float(np.mean(confidences))
        method += (
            "; chase target pressure from train (RRR × wickets) "
            "with empirical chase win-confidence"
        )

    return {
        "n_sims": n_sims,
        "seed": seed,
        "team": team,
        "batters": batter_summary,
        "bowlers": bowler_summary,
        "overs": over_summary,
        "phases": phases,
        "bowling_schedule": sample_schedule,
        "method": method,
    }
