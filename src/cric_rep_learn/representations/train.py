"""Train dual-role player representations on chronological delivery data."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import EncodedDeliveryDataset
from .model import ModelConfig, PlayerRepresentationModel


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 8
    batch_size: int = 4096
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 2
    num_workers: int = 0
    runs_weight: float = 1.0
    extras_weight: float = 1.0
    legality_weight: float = 1.0
    batter_dismissal_weight: float = 1.0
    bowler_wicket_weight: float = 1.0
    embedding_weight_decay: float = 1e-3
    ablation: str = "none"
    use_baseline_residual: bool = True


def _apply_training_ablation(
    categorical: torch.Tensor, ablation: str
) -> torch.Tensor:
    if ablation == "none":
        return categorical
    categorical = categorical.clone()
    if ablation in {"no_batter", "no_players"}:
        categorical[:, 0] = 0
    if ablation in {"no_bowler", "no_players"}:
        categorical[:, 1] = 0
    if ablation == "no_venue":
        categorical[:, 2] = 0
    if ablation not in {"none", "no_batter", "no_bowler", "no_players", "no_venue"}:
        raise ValueError(f"Unknown ablation {ablation!r}")
    return categorical


class NeuralMetrics:
    def __init__(self):
        self.n = 0
        self.loss_sum = 0.0
        self.runs_log_loss_sum = 0.0
        self.runs_brier_sum = 0.0
        self.runs_expected_mae_sum = 0.0
        self.extras_log_loss_sum = 0.0
        self.legality_log_loss_sum = 0.0
        self.dismissal_log_loss_sum = 0.0
        self.dismissal_brier_sum = 0.0
        self.wicket_log_loss_sum = 0.0
        self.wicket_brier_sum = 0.0

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        categorical_targets: torch.Tensor,
        binary_targets: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        batch_size = categorical_targets.shape[0]
        self.n += batch_size
        self.loss_sum += float(loss) * batch_size

        runs_target = categorical_targets[:, 0]
        extras_target = categorical_targets[:, 1]
        legality_target = categorical_targets[:, 2]
        dismissal_target = binary_targets[:, 0]
        wicket_target = binary_targets[:, 1]

        runs_probability = outputs["runs_logits"].softmax(dim=-1)
        dismissal_probability = outputs["batter_dismissal_logit"].sigmoid()
        wicket_probability = outputs["bowler_wicket_logit"].sigmoid()

        self.runs_log_loss_sum += float(
            F.cross_entropy(outputs["runs_logits"], runs_target, reduction="sum")
        )
        self.extras_log_loss_sum += float(
            F.cross_entropy(outputs["extras_logits"], extras_target, reduction="sum")
        )
        self.legality_log_loss_sum += float(
            F.cross_entropy(outputs["legality_logits"], legality_target, reduction="sum")
        )

        runs_one_hot = F.one_hot(runs_target, num_classes=8).float()
        self.runs_brier_sum += float(torch.square(runs_probability - runs_one_hot).sum())
        run_values = torch.arange(8, device=runs_probability.device)
        expected_runs = runs_probability @ run_values.float()
        self.runs_expected_mae_sum += float(torch.abs(expected_runs - runs_target.float()).sum())

        self.dismissal_log_loss_sum += float(
            F.binary_cross_entropy_with_logits(
                outputs["batter_dismissal_logit"],
                dismissal_target,
                reduction="sum",
            )
        )
        self.wicket_log_loss_sum += float(
            F.binary_cross_entropy_with_logits(
                outputs["bowler_wicket_logit"], wicket_target, reduction="sum"
            )
        )
        self.dismissal_brier_sum += float(
            torch.square(dismissal_probability - dismissal_target).sum()
        )
        self.wicket_brier_sum += float(torch.square(wicket_probability - wicket_target).sum())

    def as_dict(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "total_loss": self.loss_sum / self.n,
            "runs_log_loss": self.runs_log_loss_sum / self.n,
            "runs_brier": self.runs_brier_sum / self.n,
            "runs_expected_mae": self.runs_expected_mae_sum / self.n,
            "extras_log_loss": self.extras_log_loss_sum / self.n,
            "legality_log_loss": self.legality_log_loss_sum / self.n,
            "batter_dismissal_log_loss": self.dismissal_log_loss_sum / self.n,
            "batter_dismissal_brier": self.dismissal_brier_sum / self.n,
            "bowler_wicket_log_loss": self.wicket_log_loss_sum / self.n,
            "bowler_wicket_brier": self.wicket_brier_sum / self.n,
        }


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    categorical_targets: torch.Tensor,
    binary_targets: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    runs_loss = F.cross_entropy(outputs["runs_logits"], categorical_targets[:, 0])
    extras_loss = F.cross_entropy(outputs["extras_logits"], categorical_targets[:, 1])
    legality_loss = F.cross_entropy(outputs["legality_logits"], categorical_targets[:, 2])
    dismissal_loss = F.binary_cross_entropy_with_logits(
        outputs["batter_dismissal_logit"], binary_targets[:, 0]
    )
    wicket_loss = F.binary_cross_entropy_with_logits(
        outputs["bowler_wicket_logit"], binary_targets[:, 1]
    )
    return (
        config.runs_weight * runs_loss
        + config.extras_weight * extras_loss
        + config.legality_weight * legality_loss
        + config.batter_dismissal_weight * dismissal_loss
        + config.bowler_wicket_weight * wicket_loss
    )


def run_epoch(
    model: PlayerRepresentationModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    label: str = "epoch",
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    metrics = NeuralMetrics()
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    report_every = max(1, math.ceil(total_batches / 20))

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        categorical, numeric, baseline, categorical_targets, binary_targets = (
            tensor.to(device, non_blocking=True) for tensor in batch
        )
        categorical = _apply_training_ablation(categorical, config.ablation)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(categorical, numeric, baseline)
            loss = multitask_loss(outputs, categorical_targets, binary_targets, config)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        metrics.update(outputs, categorical_targets, binary_targets, loss.detach())
        completed = batch_index + 1
        if completed % report_every == 0 or completed == total_batches:
            percentage = 100.0 * completed / total_batches
            print(
                f"\r{label}: {percentage:6.2f}% ({completed:,}/{total_batches:,} batches)",
                end="\n" if completed == total_batches else "",
                flush=True,
            )

    return metrics.as_dict()


def _baseline_validation_runs_log_loss(
    path: Path | None, *, level: str = "context"
) -> float | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["metrics"]["validation"][level]["batter_runs"]["log_loss"])


def train_representations(
    model_data_dir: Path,
    output_dir: Path,
    *,
    training_config: TrainingConfig | None = None,
    baseline_metrics_path: Path | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    config = training_config or TrainingConfig()
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(device_name)

    vocab = json.loads((model_data_dir / "vocab.json").read_text(encoding="utf-8"))
    manifest = json.loads((model_data_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline_level = str(manifest.get("baseline_features", {}).get("level", "context"))
    train_dataset = EncodedDeliveryDataset(model_data_dir / "train.parquet")
    validation_dataset = EncodedDeliveryDataset(model_data_dir / "validation.parquet")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_config = ModelConfig(
        n_batters=len(vocab["batters"]) + 1,
        n_bowlers=len(vocab["bowlers"]) + 1,
        n_venues=len(vocab["venues"]) + 1,
        use_baseline_residual=config.use_baseline_residual,
    )
    model = PlayerRepresentationModel(model_config).to(device)
    embedding_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if "_embedding." in name:
            embedding_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": embedding_parameters,
                "weight_decay": config.embedding_weight_decay,
            },
            {"params": other_parameters, "weight_decay": config.weight_decay},
        ],
        lr=config.learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history = []
    best_loss = math.inf
    epochs_without_improvement = 0
    baseline_runs_log_loss = _baseline_validation_runs_log_loss(
        baseline_metrics_path, level=baseline_level
    )

    print(
        f"device={device} train={len(train_dataset):,} "
        f"validation={len(validation_dataset):,} "
        f"batters={model_config.n_batters - 1:,} bowlers={model_config.n_bowlers - 1:,} "
        f"ablation={config.ablation} residual={config.use_baseline_residual} "
        f"baseline_level={baseline_level}"
    )
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config,
            optimizer=optimizer,
            max_batches=max_train_batches,
            label=f"epoch {epoch}/{config.epochs} train",
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            config,
            optimizer=None,
            max_batches=max_validation_batches,
            label=f"epoch {epoch}/{config.epochs} validation",
        )
        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        validation_loss = float(validation_metrics["total_loss"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "model_config": model_config.as_dict(),
                    "training_config": asdict(config),
                    "model_data_manifest": manifest,
                    "vocab": vocab,
                    "best_epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "baseline_level": baseline_level,
                    "baseline_validation_runs_log_loss": baseline_runs_log_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    result = {
        "device": str(device),
        "best_validation_loss": best_loss,
        "checkpoint": str(checkpoint_path),
        "baseline_level": baseline_level,
        "baseline_validation_runs_log_loss": baseline_runs_log_loss,
        "history": history,
    }
    (output_dir / "history.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", type=Path, default=Path("artifacts/model-data"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/checkpoints/representations")
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("artifacts/baselines/metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "no_players", "no_batter", "no_bowler", "no_venue"),
        default="none",
    )
    parser.add_argument(
        "--no-baseline-residual",
        action="store_true",
        help="Train standalone logits instead of residual-over-baseline",
    )
    args = parser.parse_args()

    result = train_representations(
        args.model_data,
        args.output,
        training_config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            ablation=args.ablation,
            use_baseline_residual=not args.no_baseline_residual,
        ),
        baseline_metrics_path=args.baseline_metrics,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
