from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cric_rep_learn.representations.data import (
    BINARY_TARGET_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_TARGET_COLUMNS,
    NUMERIC_COLUMNS,
    EncodedDeliveryDataset,
    build_model_dataset,
)
from cric_rep_learn.representations.model import ModelConfig, PlayerRepresentationModel
from cric_rep_learn.representations.train import TrainingConfig, multitask_loss


def test_dual_role_model_shapes_and_gradients() -> None:
    model = PlayerRepresentationModel(
        ModelConfig(
            n_batters=10,
            n_bowlers=8,
            n_venues=5,
            player_dim=8,
            venue_dim=4,
            context_dim=2,
            numeric_dim=4,
            hidden_dim=32,
            dropout=0.0,
            id_dropout=0.0,
        )
    )
    categorical = torch.tensor(
        [
            [1, 2, 1, 1, 1, 1, 1, 1],
            [3, 4, 2, 2, 2, 2, 2, 2],
        ],
        dtype=torch.long,
    )
    numeric = torch.zeros((2, 7), dtype=torch.float32)
    categorical_targets = torch.tensor([[4, 0, 0], [0, 1, 1]])
    binary_targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    outputs = model(categorical, numeric)
    assert outputs["runs_logits"].shape == (2, 8)
    assert outputs["extras_logits"].shape == (2, 8)
    assert outputs["legality_logits"].shape == (2, 3)
    assert outputs["batter_dismissal_logit"].shape == (2,)
    assert outputs["bowler_wicket_logit"].shape == (2,)

    loss = multitask_loss(outputs, categorical_targets, binary_targets, TrainingConfig())
    loss.backward()
    assert model.batting_embedding.weight.grad[1].abs().sum() > 0
    assert model.bowling_embedding.weight.grad[2].abs().sum() > 0
    assert model.batting_embedding.weight.grad[0].abs().sum() == 0
    assert model.bowling_embedding.weight.grad[0].abs().sum() == 0


def test_unknown_role_embeddings_are_learned_with_id_dropout() -> None:
    model = PlayerRepresentationModel(
        ModelConfig(
            n_batters=3,
            n_bowlers=3,
            n_venues=2,
            player_dim=4,
            venue_dim=2,
            context_dim=2,
            numeric_dim=2,
            hidden_dim=16,
            dropout=0.0,
            id_dropout=1.0,
        )
    )
    categorical = torch.ones((2, 8), dtype=torch.long)
    outputs = model(categorical, torch.zeros((2, 7)))
    sum(value.sum() for value in outputs.values()).backward()

    assert model.batting_embedding.weight.grad[0].abs().sum() > 0
    assert model.bowling_embedding.weight.grad[0].abs().sum() > 0


def test_encoded_delivery_dataset_tensor_layout(tmp_path) -> None:
    row = {}
    for column in CATEGORICAL_COLUMNS:
        row[column] = 1
    for column in NUMERIC_COLUMNS:
        row[column] = 0.5
    for column in CATEGORICAL_TARGET_COLUMNS:
        row[column] = 0
    for column in BINARY_TARGET_COLUMNS:
        row[column] = 0.0

    path = tmp_path / "encoded.parquet"
    pq.write_table(pa.Table.from_pylist([row, row]), path)
    dataset = EncodedDeliveryDataset(path)

    categorical, numeric, categorical_targets, binary_targets = dataset[0]
    assert len(dataset) == 2
    assert categorical.shape == (len(CATEGORICAL_COLUMNS),)
    assert numeric.shape == (len(NUMERIC_COLUMNS),)
    assert categorical_targets.shape == (len(CATEGORICAL_TARGET_COLUMNS),)
    assert binary_targets.shape == (len(BINARY_TARGET_COLUMNS),)
    assert categorical.dtype == torch.int64
    assert numeric.dtype == torch.float32
    assert np.isclose(float(numeric[0]), 0.5)


def test_model_data_vocab_is_fitted_on_training_only(tmp_path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "metadata.json").write_text('{"schema_version":"test"}')

    matches = [
        {
            "match_id": "train-match",
            "gender": "male",
            "team_type": "club",
            "venue": "Known Ground",
        },
        {
            "match_id": "validation-match",
            "gender": "male",
            "team_type": "club",
            "venue": "Future Ground",
        },
        {
            "match_id": "test-match",
            "gender": "male",
            "team_type": "club",
            "venue": "Future Ground",
        },
    ]
    pq.write_table(pa.Table.from_pylist(matches), canonical / "matches.parquet")
    splits = [
        {"match_id": "train-match", "split": "train"},
        {"match_id": "validation-match", "split": "validation"},
        {"match_id": "test-match", "split": "test"},
    ]
    pq.write_table(pa.Table.from_pylist(splits), canonical / "split_manifest.parquet")
    aliases = [
        {
            "player_id": player_id,
            "player_name": player_id,
            "match_count": 1,
            "last_seen": "2020-01-01",
        }
        for player_id in ("known-batter", "known-bowler", "future-player")
    ]
    pq.write_table(pa.Table.from_pylist(aliases), canonical / "player_aliases.parquet")

    delivery_rows = []
    for index, (match_id, batter, bowler) in enumerate(
        [
            ("train-match", "known-batter", "known-bowler"),
            ("validation-match", "future-player", "known-batter"),
            ("test-match", "future-player", "known-bowler"),
        ]
    ):
        delivery_rows.append(
            {
                "match_id": match_id,
                "match_date": f"202{index}-01-01",
                "innings": 1,
                "is_super_over": False,
                "phase": "powerplay",
                "wickets_before": 0,
                "batter_id": batter,
                "bowler_id": bowler,
                "score_before": 0,
                "scheduled_balls": 120,
                "legal_balls_before": 0,
                "target_runs": None,
                "target_balls": None,
                "runs_batter": 0,
                "runs_extras": 0,
                "extras_wides": 0,
                "extras_noballs": 0,
                "batter_dismissed": False,
                "bowler_wicket_count": 0,
            }
        )
    pq.write_table(pa.Table.from_pylist(delivery_rows), canonical / "deliveries.parquet")

    output = tmp_path / "model-data"
    manifest = build_model_dataset(canonical, output)
    validation = pq.read_table(
        output / "validation.parquet", columns=["batter_idx", "bowler_idx", "venue_idx"]
    ).to_pylist()[0]

    assert manifest["batters"] == 1
    assert manifest["bowlers"] == 1
    assert validation["batter_idx"] == 0
    assert validation["bowler_idx"] == 0
    assert validation["venue_idx"] == 0
