"""Joint T20 match simulation: first innings → chase with sampled target."""

from __future__ import annotations

from typing import Any

import numpy as np

from cric_rep_learn.simulation.attack import BowlerSpell
from cric_rep_learn.simulation.innings import simulate_innings, simulate_one_innings
from cric_rep_learn.simulation.priors import InningsRateModel


def simulate_match(
    *,
    first_lineup: list[dict[str, str]],
    first_attack: list[BowlerSpell],
    chase_lineup: list[dict[str, str]],
    chase_attack: list[BowlerSpell],
    first_rates: InningsRateModel,
    chase_rates: InningsRateModel,
    n_sims: int = 400,
    seed: int = 7,
    chase_impacts: dict[str, Any] | None = None,
    partnership_index: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    For each sim: play first innings, set target = score + 1, then chase.

    Aggregates win probabilities plus per-innings summaries (including overs /
    phases) from independent MC passes with shared seed structure.
    """
    rng = np.random.default_rng(seed)
    first_scores = np.zeros(n_sims, dtype=np.float64)
    chase_scores = np.zeros(n_sims, dtype=np.float64)
    targets = np.zeros(n_sims, dtype=np.float64)
    chase_wins = np.zeros(n_sims, dtype=np.float64)

    for i in range(n_sims):
        first = simulate_one_innings(
            lineup=first_lineup,
            attack=first_attack,
            rates=first_rates,
            rng=rng,
            partnership_index=partnership_index,
        )
        target = float(first["runs"]) + 1.0
        chase = simulate_one_innings(
            lineup=chase_lineup,
            attack=chase_attack,
            rates=chase_rates,
            rng=rng,
            target=target,
            chase_impacts=chase_impacts,
            partnership_index=partnership_index,
        )
        first_scores[i] = first["runs"]
        chase_scores[i] = chase["runs"]
        targets[i] = target
        chase_wins[i] = 1.0 if chase.get("chase_won") else 0.0

    # Rich marginal summaries (same seed → comparable schedules/rates).
    first_summary = simulate_innings(
        lineup=first_lineup,
        attack=first_attack,
        rates=first_rates,
        n_sims=n_sims,
        seed=seed,
        partnership_index=partnership_index,
    )
    # Chase summary uses expected first score as a representative target.
    rep_target = float(first_scores.mean()) + 1.0
    chase_summary = simulate_innings(
        lineup=chase_lineup,
        attack=chase_attack,
        rates=chase_rates,
        n_sims=n_sims,
        seed=seed + 1,
        target=rep_target,
        chase_impacts=chase_impacts,
        partnership_index=partnership_index,
    )

    p_chase = float(chase_wins.mean())
    return {
        "n_sims": n_sims,
        "seed": seed,
        "match": {
            "first_expected_runs": float(first_scores.mean()),
            "first_runs_p10": float(np.quantile(first_scores, 0.10)),
            "first_runs_p50": float(np.quantile(first_scores, 0.50)),
            "first_runs_p90": float(np.quantile(first_scores, 0.90)),
            "chase_expected_runs": float(chase_scores.mean()),
            "chase_runs_p10": float(np.quantile(chase_scores, 0.10)),
            "chase_runs_p50": float(np.quantile(chase_scores, 0.50)),
            "chase_runs_p90": float(np.quantile(chase_scores, 0.90)),
            "expected_target": float(targets.mean()),
            "p_chase_win": p_chase,
            "p_first_win": float(1.0 - p_chase),
            "margin_expected": float((chase_scores - targets).mean()),
        },
        "first_innings": first_summary,
        "chase_innings": chase_summary,
        "method": (
            "Joint MC: sample first innings → target=score+1 → chase with "
            "RRR×wickets pressure; partnership + phase-weighted over scores"
        ),
    }
