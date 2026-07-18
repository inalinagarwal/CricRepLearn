"""Tests for expected batting contribution stints."""

from __future__ import annotations

import numpy as np
import torch

from cric_rep_learn.contribution.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    EncodedStintDataset,
    build_contribution_dataset,
)
from cric_rep_learn.contribution.model import BatterContributionModel, ModelConfig
from cric_rep_learn.contribution.train import TrainingConfig, _apply_ablation, contribution_loss


def test_ablation_zeros_batter_index() -> None:
    categorical = torch.tensor([[3, 2, 1, 1, 1, 1, 1], [4, 5, 1, 1, 1, 1, 1]])
    ablated = _apply_ablation(categorical, "no_players")
    assert torch.equal(ablated[:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(ablated[:, 1], categorical[:, 1])


def test_contribution_model_shapes_and_gradients() -> None:
    model = BatterContributionModel(
        ModelConfig(n_batters=5, n_venues=3, id_dropout=0.0, venue_dropout=0.0)
    )
    categorical = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 1, 1, 2, 1]])
    numeric = torch.zeros((2, len(NUMERIC_COLUMNS)))
    outputs = model(categorical, numeric)
    assert outputs["runs_pred"].shape == (2,)
    assert outputs["dismissal_logit"].shape == (2,)
    targets = torch.tensor([[10.0, 1.0], [3.0, 0.0]])
    loss = contribution_loss(outputs, targets, TrainingConfig())
    loss.backward()
    assert model.batting_embedding.weight.grad is not None
    assert model.batting_embedding.weight.grad[1].abs().sum() > 0


def test_build_contribution_dataset_train_only_vocab(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import date

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    deliveries = [
        {
            "schema_version": "1.0.0",
            "match_id": "m1",
            "match_date": date(2020, 1, 1),
            "innings": 1,
            "is_super_over": False,
            "batting_team": "A",
            "bowling_team": "B",
            "target_runs": None,
            "target_overs_raw": None,
            "target_balls": None,
            "over_number": 0,
            "delivery_index": index,
            "attempt_index_in_innings": index + 1,
            "source_ball_label": "0.1",
            "is_legal": True,
            "legal_balls_in_over_before": 0,
            "legal_balls_before": index,
            "score_before": index,
            "wickets_before": 0,
            "scheduled_balls": 120,
            "phase": "powerplay",
            "phase_source": "test",
            "is_powerplay": True,
            "batter_id": "batter-train",
            "batter_name": "Train Batter",
            "bowler_id": "bowler-1",
            "bowler_name": "Bowler",
            "non_striker_id": "ns",
            "non_striker_name": "NS",
            "runs_batter": 1,
            "runs_extras": 0,
            "runs_total": 1,
            "non_boundary": False,
            "is_boundary": False,
            "extras_byes": 0,
            "extras_legbyes": 0,
            "extras_noballs": 0,
            "extras_penalty": 0,
            "extras_wides": 0,
            "wicket_count": 0,
            "bowler_wicket_count": 0,
            "batter_dismissed": False,
        }
        for index in range(5)
    ]
    deliveries.append(
        {
            **deliveries[0],
            "match_id": "m2",
            "match_date": date(2021, 1, 1),
            "batter_id": "batter-val",
            "batter_name": "Val Batter",
            "attempt_index_in_innings": 1,
            "runs_batter": 4,
        }
    )
    pq.write_table(pa.Table.from_pylist(deliveries), canonical / "deliveries.parquet")
    matches = [
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
    # Minimal match columns used by builder
    for row in matches:
        row.setdefault("schema_version", "1.0.0")
    pq.write_table(pa.Table.from_pylist(matches), canonical / "matches.parquet")
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
            ]
        ),
        canonical / "player_aliases.parquet",
    )

    output = tmp_path / "contribution-data"
    manifest = build_contribution_dataset(canonical, output, overwrite=True, min_balls_eval=1)
    assert manifest["split_counts"]["train"] == 1
    assert manifest["split_counts"]["validation"] == 1

    import json

    vocab = json.loads((output / "vocab.json").read_text())
    batter_ids = {row["player_id"] for row in vocab["batters"]}
    assert "batter-train" in batter_ids
    assert "batter-val" not in batter_ids

    val = EncodedStintDataset(output / "validation.parquet")
    categorical, numeric, targets, balls, eligible = val[0]
    assert categorical.shape == (len(CATEGORICAL_COLUMNS),)
    assert numeric.shape == (len(NUMERIC_COLUMNS),)
    assert targets[0].item() == 4.0
    assert balls.item() == 1.0
    assert eligible.item() is True
    # Validation batter must map to UNK index 0
    assert int(categorical[0].item()) == 0
