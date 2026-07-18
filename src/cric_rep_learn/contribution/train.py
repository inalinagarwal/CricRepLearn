"""Train expected batting contribution under fixed opportunity."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import EncodedStintDataset
from .model import BatterContributionModel, ModelConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 3
    num_workers: int = 0
    runs_weight: float = 1.0
    dismissal_weight: float = 0.25
    embedding_weight_decay: float = 1e-3
    ablation: str = "none"


def choose_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _apply_ablation(
    categorical: torch.Tensor,
    bowler_idxs: torch.Tensor,
    ablation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ablation == "none":
        return categorical, bowler_idxs
    categorical = categorical.clone()
    bowler_idxs = bowler_idxs.clone()
    if ablation in {"no_players", "no_batter"}:
        categorical[:, 0] = 0
    if ablation in {"no_players", "no_bowler"}:
        bowler_idxs.zero_()
    if ablation == "no_venue":
        categorical[:, 1] = 0
    if ablation not in {"none", "no_players", "no_batter", "no_bowler", "no_venue"}:
        raise ValueError(f"Unknown ablation {ablation!r}")
    return categorical, bowler_idxs


def contribution_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    runs_loss = F.smooth_l1_loss(outputs["runs_pred"], targets[:, 0])
    dismissal_loss = F.binary_cross_entropy_with_logits(
        outputs["dismissal_logit"], targets[:, 1]
    )
    return config.runs_weight * runs_loss + config.dismissal_weight * dismissal_loss


class ContributionMetrics:
    def __init__(self):
        self.n = 0
        self.eval_n = 0
        self.loss_sum = 0.0
        self.mae_sum = 0.0
        self.eval_mae_sum = 0.0
        self.baseline_mae_sum = 0.0
        self.eval_baseline_mae_sum = 0.0
        self.rmse_sum = 0.0
        self.dismissal_brier_sum = 0.0
        self.runs_sum = 0.0
        self.pred_sum = 0.0

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss: torch.Tensor,
        eval_eligible: torch.Tensor,
        baseline_runs: torch.Tensor,
    ) -> None:
        batch = targets.shape[0]
        self.n += batch
        self.loss_sum += float(loss) * batch
        runs = targets[:, 0]
        pred = outputs["runs_pred"]
        abs_err = torch.abs(pred - runs)
        baseline_err = torch.abs(baseline_runs - runs)
        self.mae_sum += float(abs_err.sum())
        self.baseline_mae_sum += float(baseline_err.sum())
        self.rmse_sum += float(torch.square(pred - runs).sum())
        self.runs_sum += float(runs.sum())
        self.pred_sum += float(pred.sum())
        dismissal_prob = outputs["dismissal_logit"].sigmoid()
        self.dismissal_brier_sum += float(
            torch.square(dismissal_prob - targets[:, 1]).sum()
        )
        if eval_eligible.any():
            self.eval_n += int(eval_eligible.sum())
            self.eval_mae_sum += float(abs_err[eval_eligible].sum())
            self.eval_baseline_mae_sum += float(baseline_err[eval_eligible].sum())

    def as_dict(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        result: dict[str, float | int] = {
            "n": self.n,
            "loss": self.loss_sum / self.n,
            "runs_mae": self.mae_sum / self.n,
            "baseline_runs_mae": self.baseline_mae_sum / self.n,
            "runs_rmse": (self.rmse_sum / self.n) ** 0.5,
            "dismissal_brier": self.dismissal_brier_sum / self.n,
            "mean_runs": self.runs_sum / self.n,
            "mean_pred_runs": self.pred_sum / self.n,
        }
        if self.eval_n:
            result["eval_n"] = self.eval_n
            result["runs_mae_min3"] = self.eval_mae_sum / self.eval_n
            result["baseline_runs_mae_min3"] = self.eval_baseline_mae_sum / self.eval_n
        return result


def run_epoch(
    model: BatterContributionModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
    label: str,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    metrics = ContributionMetrics()
    total_batches = len(loader)
    report_every = max(1, total_batches // 10)
    for batch_index, batch in enumerate(loader, start=1):
        (
            categorical,
            numeric,
            baseline,
            bowler_idxs,
            bowler_weights,
            targets,
            _balls,
            eval_eligible,
        ) = batch
        categorical, bowler_idxs = _apply_ablation(
            categorical.to(device), bowler_idxs.to(device), config.ablation
        )
        numeric = numeric.to(device)
        baseline = baseline.to(device)
        bowler_weights = bowler_weights.to(device)
        targets = targets.to(device)
        eval_eligible = eval_eligible.to(device)
        outputs = model(categorical, numeric, baseline, bowler_idxs, bowler_weights)
        loss = contribution_loss(outputs, targets, config)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics.update(outputs, targets, loss.detach(), eval_eligible, baseline)
        if batch_index == total_batches or batch_index % report_every == 0:
            print(
                f"\r{label}: {100.0 * batch_index / total_batches:6.2f}% "
                f"({batch_index}/{total_batches} batches)",
                end="",
                flush=True,
            )
    print(flush=True)
    return metrics.as_dict()


def train_contribution(
    data_dir: Path,
    output_dir: Path,
    *,
    training_config: TrainingConfig | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    config = training_config or TrainingConfig()
    seed_everything(config.seed)
    device = choose_device(device_name)

    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    train_dataset = EncodedStintDataset(data_dir / "train.parquet")
    validation_dataset = EncodedStintDataset(data_dir / "validation.parquet")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_config = ModelConfig(
        n_batters=len(vocab["batters"]),
        n_bowlers=len(vocab["bowlers"]),
        n_venues=len(vocab["venues"]),
    )
    model = BatterContributionModel(model_config).to(device)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": config.embedding_weight_decay},
        ],
        lr=config.learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_epoch = 0
    patience_left = config.patience

    print(
        f"device={device} train={len(train_dataset):,} "
        f"validation={len(validation_dataset):,} "
        f"batters={model_config.n_batters - 1:,} "
        f"bowlers={model_config.n_bowlers - 1:,} "
        f"ablation={config.ablation}"
    )
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config,
            optimizer=optimizer,
            label=f"epoch {epoch}/{config.epochs} train",
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            config,
            optimizer=None,
            label=f"epoch {epoch}/{config.epochs} validation",
        )
        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        selection_mae = float(
            validation_metrics.get("runs_mae_min3", validation_metrics["runs_mae"])
        )
        if selection_mae < best_mae:
            best_mae = selection_mae
            best_epoch = epoch
            patience_left = config.patience
            torch.save(
                {
                    "model_config": model_config.as_dict(),
                    "training_config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "best_epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    result = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_runs_mae_min3": best_mae,
        "checkpoint": str(checkpoint_path),
        "ablation": config.ablation,
        "history": history,
    }
    (output_dir / "history.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/contribution-data"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/checkpoints/contribution-bat")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--ablation",
        choices=("none", "no_players", "no_batter", "no_bowler", "no_venue"),
        default="none",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train_contribution(
                args.data,
                args.output,
                training_config=TrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    ablation=args.ablation,
                ),
                device_name=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
