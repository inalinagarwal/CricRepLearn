"""Tests for bowling-conditioned batting contribution."""

from __future__ import annotations

from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cric_rep_learn.contribution.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TOP_BOWLERS,
    EncodedStintDataset,
    _top_bowlers_for_stint,
    build_contribution_dataset,
)
from cric_rep_learn.contribution.model import BatterContributionModel, ModelConfig
from cric_rep_learn.contribution.train import TrainingConfig, _apply_ablation, contribution_loss


def test_ablation_zeros_batter_and_bowlers() -> None:
    categorical = torch.tensor([[3, 2, 1, 1, 1, 1, 1]])
    bowlers = torch.tensor([[4, 5, 0, 0]])
    cat, bowl = _apply_ablation(categorical, bowlers, "no_players")
    assert torch.equal(cat[:, 0], torch.zeros(1, dtype=torch.long))
    assert torch.equal(bowl, torch.zeros_like(bowlers))


def test_top_bowlers_weights_sum_to_one() -> None:
    idxs, weights = _top_bowlers_for_stint(
        [
            {"bowler_id": "a", "balls": 6},
            {"bowler_id": "b", "balls": 3},
            {"bowler_id": "c", "balls": 2},
            {"bowler_id": "d", "balls": 1},
            {"bowler_id": "e", "balls": 1},
        ],
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        top_k=4,
    )
    assert len(idxs) == 4
    assert abs(sum(weights) - 1.0) < 1e-6
    assert idxs[0] == 1


def test_contribution_model_shapes_and_zero_residual() -> None:
    model = BatterContributionModel(
        ModelConfig(
            n_batters=5,
            n_bowlers=6,
            n_venues=3,
            id_dropout=0.0,
            venue_dropout=0.0,
            bowler_dropout=0.0,
        )
    )
    categorical = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 1, 1, 2, 1]])
    numeric = torch.zeros((2, len(NUMERIC_COLUMNS)))
    baseline = torch.tensor([12.0, 8.0])
    bowler_idxs = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    bowler_weights = torch.tensor([[0.7, 0.3, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    outputs = model(categorical, numeric, baseline, bowler_idxs, bowler_weights)
    assert outputs["runs_pred"].shape == (2,)
    assert torch.allclose(outputs["runs_pred"], baseline)
    targets = torch.tensor([[10.0, 1.0], [3.0, 0.0]])
    loss = contribution_loss(outputs, targets, TrainingConfig())
    loss.backward()
    assert model.batting_embedding.weight.grad[1].abs().sum() > 0
    assert model.bowling_embedding.weight.grad[1].abs().sum() > 0


def test_build_contribution_dataset_includes_bowlers(tmp_path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()

    def delivery(
        *,
        match_id: str,
        match_date: date,
        batter_id: str,
        bowler_id: str,
        attempt: int,
        runs: int,
    ) -> dict:
        return {
            "match_id": match_id,
            "match_date": match_date,
            "innings": 1,
            "is_super_over": False,
            "batter_id": batter_id,
            "batter_name": batter_id,
            "bowler_id": bowler_id,
            "bowler_name": bowler_id,
            "attempt_index_in_innings": attempt,
            "phase": "powerplay",
            "score_before": attempt - 1,
            "wickets_before": 0,
            "runs_batter": runs,
            "batter_dismissed": False,
            "is_legal": True,
            "extras_noballs": 0,
        }

    deliveries = [
        delivery(
            match_id="m1",
            match_date=date(2020, 1, 1),
            batter_id="batter-train",
            bowler_id="bowler-train",
            attempt=index + 1,
            runs=1,
        )
        for index in range(5)
    ]
    deliveries.append(
        delivery(
            match_id="m2",
            match_date=date(2021, 1, 1),
            batter_id="batter-val",
            bowler_id="bowler-val",
            attempt=1,
            runs=4,
        )
    )
    pq.write_table(pa.Table.from_pylist(deliveries), canonical / "deliveries.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "match_id": "m1",
                    "match_date": date(2020, 1, 1),
                    "gender": "male",
                    "team_type": "international",
                    "venue": "Ground A",
                },
                {
                    "match_id": "m2",
                    "match_date": date(2021, 1, 1),
                    "gender": "male",
                    "team_type": "international",
                    "venue": "Ground B",
                },
            ]
        ),
        canonical / "matches.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"match_id": "m1", "split": "train"},
                {"match_id": "m2", "split": "validation"},
            ]
        ),
        canonical / "split_manifest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "player_id": "batter-train",
                    "player_name": "Train Batter",
                    "first_seen": date(2020, 1, 1),
                    "last_seen": date(2020, 1, 1),
                    "match_count": 1,
                },
                {
                    "player_id": "batter-val",
                    "player_name": "Val Batter",
                    "first_seen": date(2021, 1, 1),
                    "last_seen": date(2021, 1, 1),
                    "match_count": 1,
                },
                {
                    "player_id": "bowler-train",
                    "player_name": "Train Bowler",
                    "first_seen": date(2020, 1, 1),
                    "last_seen": date(2020, 1, 1),
                    "match_count": 1,
                },
                {
                    "player_id": "bowler-val",
                    "player_name": "Val Bowler",
                    "first_seen": date(2021, 1, 1),
                    "last_seen": date(2021, 1, 1),
                    "match_count": 1,
                },
            ]
        ),
        canonical / "player_aliases.parquet",
    )

    output = tmp_path / "contribution-data"
    manifest = build_contribution_dataset(canonical, output, overwrite=True, min_balls_eval=1)
    assert manifest["n_bowlers"] == 1
    assert manifest["objective"].startswith("bowling_conditioned")

    import json

    vocab = json.loads((output / "vocab.json").read_text())
    assert {row["player_id"] for row in vocab["bowlers"]} == {"UNK_BOWLER", "bowler-train"}

    val = EncodedStintDataset(output / "validation.parquet")
    categorical, numeric, baseline, bowler_idxs, bowler_weights, targets, balls, eligible = val[0]
    assert categorical.shape == (len(CATEGORICAL_COLUMNS),)
    assert numeric.shape == (len(NUMERIC_COLUMNS),)
    assert bowler_idxs.shape == (TOP_BOWLERS,)
    assert abs(float(bowler_weights.sum()) - 1.0) < 1e-5
    assert int(categorical[0].item()) == 0  # val batter -> UNK
    assert int(bowler_idxs[0].item()) == 0  # val bowler -> UNK
    assert targets[0].item() == 4.0
    assert balls.item() == 1.0
    assert eligible.item() is True
    assert float(baseline.item()) > 0
