"""Evaluate contribution checkpoints and compare ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import EncodedStintDataset
from .model import BatterContributionModel, ModelConfig
from .train import TrainingConfig, run_epoch


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    *,
    split: str = "validation",
    ablation: str = "none",
    batch_size: int = 2048,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if split == "test":
        raise ValueError("test split is reserved until architecture is frozen")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BatterContributionModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    from .train import choose_device

    device = choose_device(device_name)
    model.to(device)
    dataset = EncodedStintDataset(data_dir / f"{split}.parquet")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    config = TrainingConfig(ablation=ablation)
    metrics = run_epoch(
        model,
        loader,
        device,
        config,
        optimizer=None,
        label=f"evaluate {split}",
    )
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "ablation": ablation,
        "best_epoch": checkpoint.get("best_epoch"),
        "metrics": metrics,
    }


def compare_histories(full_history: Path, no_players_history: Path) -> dict[str, Any]:
    full = json.loads(full_history.read_text(encoding="utf-8"))
    none = json.loads(no_players_history.read_text(encoding="utf-8"))
    full_mae = float(full["best_validation_runs_mae_min3"])
    none_mae = float(none["best_validation_runs_mae_min3"])
    gap = none_mae - full_mae
    return {
        "full": {
            "best_epoch": full["best_epoch"],
            "runs_mae_min3": full_mae,
            "checkpoint": full["checkpoint"],
        },
        "no_players": {
            "best_epoch": none["best_epoch"],
            "runs_mae_min3": none_mae,
            "checkpoint": none["checkpoint"],
        },
        "embedding_gap_mae": gap,
        "passes_0_5_runs_gate": gap >= 0.5,
        "primary_metric": "validation runs MAE on stints with balls_faced >= 3",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("artifacts/contribution-data"))
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-full", type=Path, help="history.json for full model")
    parser.add_argument(
        "--compare-no-players", type=Path, help="history.json for no_players model"
    )
    args = parser.parse_args()

    if args.compare_full and args.compare_no_players:
        result = compare_histories(args.compare_full, args.compare_no_players)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless comparing histories")
        result = evaluate_checkpoint(
            args.checkpoint,
            args.data,
            split=args.split,
            ablation=args.ablation,
            device_name=args.device,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
