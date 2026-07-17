from __future__ import annotations

import numpy as np

from cric_rep_learn.baselines.evaluate import (
    EvidenceMetrics,
    MetricBundle,
    _process_day,
)
from cric_rep_learn.baselines.historical import HistoricalBaseline, MatchContext
from cric_rep_learn.baselines.historical import BASELINE_LEVELS
from cric_rep_learn.baselines.metrics import BinaryMetrics, MulticlassMetrics


def delivery(
    batter: str,
    bowler: str,
    *,
    runs: int = 0,
    dismissed: bool = False,
    bowler_wicket: bool = False,
    legal: bool = True,
) -> dict:
    return {
        "innings": 1,
        "wickets_before": 0,
        "phase": "middle",
        "is_super_over": False,
        "batter_id": batter,
        "bowler_id": bowler,
        "runs_batter": runs,
        "runs_extras": 0,
        "extras_wides": int(not legal),
        "extras_noballs": 0,
        "batter_dismissed": dismissed,
        "bowler_wicket_count": int(bowler_wicket),
    }


def test_unseen_baseline_falls_back_to_valid_global_probabilities() -> None:
    model = HistoricalBaseline()
    context = MatchContext(gender="male", team_type="international", venue="Ground")

    predictions = model.predict_all(delivery("new-batter", "new-bowler"), context)

    for prediction in predictions.values():
        assert np.isclose(prediction.batter_runs.sum(), 1.0)
        assert np.isclose(prediction.extras_runs.sum(), 1.0)
        assert np.isclose(prediction.legality.sum(), 1.0)
        assert prediction.batter_dismissal == 0.5
        assert prediction.bowler_wicket == 0.5
        assert np.isclose(prediction.illegal_delivery, 2.0 / 3.0)


def test_direct_matchup_is_shrunk_but_moves_toward_pair_evidence() -> None:
    model = HistoricalBaseline()
    context = MatchContext(gender="male", team_type="club", venue="Ground")

    for _ in range(200):
        model.update(delivery("batter-a", "bowler-c", runs=0), context)
        model.update(delivery("batter-d", "bowler-b", runs=0), context)
    for _ in range(30):
        model.update(delivery("batter-a", "bowler-b", runs=6), context)

    predictions = model.predict_all(delivery("batter-a", "bowler-b"), context)

    assert predictions["matchup"].batter_runs[6] > predictions["venue"].batter_runs[6]
    assert predictions["matchup"].batter_runs[6] < 1.0
    assert predictions["matchup"].evidence["matchup"] == 30


def test_streaming_metrics_reward_better_probabilities() -> None:
    good = MulticlassMetrics(classes=3)
    poor = MulticlassMetrics(classes=3)
    good.update(np.array([0.05, 0.90, 0.05]), target=1)
    poor.update(np.array([0.80, 0.10, 0.10]), target=1)

    assert good.as_dict()["log_loss"] < poor.as_dict()["log_loss"]
    assert good.as_dict()["brier_score"] < poor.as_dict()["brier_score"]

    binary = BinaryMetrics()
    binary.update(0.8, True)
    binary.update(0.2, False)
    assert binary.as_dict()["positive_rate"] == 0.5
    assert binary.as_dict()["brier_score"] < 0.05


def test_same_date_rows_are_all_predicted_before_history_updates() -> None:
    model = HistoricalBaseline()
    rows = []
    for match_id in ("one", "two"):
        row = delivery(f"batter-{match_id}", f"bowler-{match_id}")
        row.update(
            {
                "match_id": match_id,
                "split": "validation",
                "gender": "male",
                "team_type": "international",
                "venue": "Ground",
            }
        )
        rows.append(row)

    metrics = {"validation": {level: MetricBundle() for level in BASELINE_LEVELS}}
    evidence = {"validation": EvidenceMetrics()}
    _process_day(rows, model, metrics, evidence)

    global_runs = metrics["validation"]["global"].batter_runs.as_dict()
    assert np.isclose(global_runs["log_loss"], np.log(8.0))
    assert model.global_stats.n == 2
