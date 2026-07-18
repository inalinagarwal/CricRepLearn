"""Validation analysis for residual representation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from cric_rep_learn.baselines.metrics import BinaryMetrics, MulticlassMetrics

from .data import EncodedDeliveryDataset
from .model import ModelConfig, PlayerRepresentationModel
from .train import TrainingConfig, choose_device, multitask_loss

PHASE_LABELS = {0: "unknown", 1: "powerplay", 2: "middle", 3: "death"}
GENDER_LABELS = {0: "unknown", 1: "male", 2: "female"}
TEAM_TYPE_LABELS = {0: "unknown", 1: "club", 2: "international"}
INNINGS_LABELS = {0: "unknown", 1: "first_innings", 2: "chase", 3: "super_over"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_bucket(log1p_value: float) -> str:
    evidence = float(np.expm1(log1p_value))
    if evidence < 0.5:
        return "0"
    if evidence < 10.5:
        return "1-10"
    if evidence < 50.5:
        return "11-50"
    if evidence < 200.5:
        return "51-200"
    return "200+"


@dataclass(slots=True)
class HeadMetrics:
    runs: MulticlassMetrics = field(default_factory=lambda: MulticlassMetrics(8))
    extras: MulticlassMetrics = field(default_factory=lambda: MulticlassMetrics(8))
    legality: MulticlassMetrics = field(default_factory=lambda: MulticlassMetrics(3))
    batter_dismissal: BinaryMetrics = field(default_factory=BinaryMetrics)
    bowler_wicket: BinaryMetrics = field(default_factory=BinaryMetrics)
    loss_sum: float = 0.0
    n: int = 0

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

        runs_probability = outputs["runs_logits"].softmax(dim=-1).cpu().numpy()
        extras_probability = outputs["extras_logits"].softmax(dim=-1).cpu().numpy()
        legality_probability = outputs["legality_logits"].softmax(dim=-1).cpu().numpy()
        dismissal_probability = outputs["batter_dismissal_logit"].sigmoid().cpu().numpy()
        wicket_probability = outputs["bowler_wicket_logit"].sigmoid().cpu().numpy()

        runs_target = categorical_targets[:, 0].cpu().numpy()
        extras_target = categorical_targets[:, 1].cpu().numpy()
        legality_target = categorical_targets[:, 2].cpu().numpy()
        dismissal_target = binary_targets[:, 0].cpu().numpy()
        wicket_target = binary_targets[:, 1].cpu().numpy()

        for index in range(batch_size):
            self.runs.update(runs_probability[index], int(runs_target[index]))
            self.extras.update(extras_probability[index], int(extras_target[index]))
            self.legality.update(legality_probability[index], int(legality_target[index]))
            self.batter_dismissal.update(
                float(dismissal_probability[index]), bool(dismissal_target[index])
            )
            self.bowler_wicket.update(
                float(wicket_probability[index]), bool(wicket_target[index])
            )

    def update_baseline(
        self,
        baseline: torch.Tensor,
        categorical_targets: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> None:
        batch_size = categorical_targets.shape[0]
        self.n += batch_size
        baseline_np = baseline.cpu().numpy()
        runs_target = categorical_targets[:, 0].cpu().numpy()
        extras_target = categorical_targets[:, 1].cpu().numpy()
        legality_target = categorical_targets[:, 2].cpu().numpy()
        dismissal_target = binary_targets[:, 0].cpu().numpy()
        wicket_target = binary_targets[:, 1].cpu().numpy()

        for index in range(batch_size):
            row = baseline_np[index]
            self.runs.update(row[:8], int(runs_target[index]))
            self.extras.update(row[8:16], int(extras_target[index]))
            self.legality.update(row[16:19], int(legality_target[index]))
            self.batter_dismissal.update(float(row[19]), bool(dismissal_target[index]))
            self.bowler_wicket.update(float(row[20]), bool(wicket_target[index]))
            # Approximate multitask loss under equal head weights.
            self.loss_sum += (
                -np.log(max(float(row[int(runs_target[index])]), 1e-12))
                - np.log(max(float(row[8 + int(extras_target[index])]), 1e-12))
                - np.log(max(float(row[16 + int(legality_target[index])]), 1e-12))
                - (
                    float(dismissal_target[index]) * np.log(max(float(row[19]), 1e-12))
                    + (1.0 - float(dismissal_target[index]))
                    * np.log(max(1.0 - float(row[19]), 1e-12))
                )
                - (
                    float(wicket_target[index]) * np.log(max(float(row[20]), 1e-12))
                    + (1.0 - float(wicket_target[index]))
                    * np.log(max(1.0 - float(row[20]), 1e-12))
                )
            )

    def as_dict(self) -> dict[str, Any]:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "total_loss": self.loss_sum / self.n,
            "batter_runs": self.runs.as_dict(),
            "extras_runs": self.extras.as_dict(),
            "legality": self.legality.as_dict(),
            "batter_dismissal": self.batter_dismissal.as_dict(),
            "bowler_wicket": self.bowler_wicket.as_dict(),
        }


AblationFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _identity_ablation(
    categorical: torch.Tensor, numeric: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return categorical, numeric, baseline


def _mask_batter(
    categorical: torch.Tensor, numeric: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical = categorical.clone()
    categorical[:, 0] = 0
    return categorical, numeric, baseline


def _mask_bowler(
    categorical: torch.Tensor, numeric: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical = categorical.clone()
    categorical[:, 1] = 0
    return categorical, numeric, baseline


def _mask_players(
    categorical: torch.Tensor, numeric: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical = categorical.clone()
    categorical[:, 0] = 0
    categorical[:, 1] = 0
    return categorical, numeric, baseline


def _mask_venue(
    categorical: torch.Tensor, numeric: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical = categorical.clone()
    categorical[:, 2] = 0
    return categorical, numeric, baseline


ABLATIONS: dict[str, AblationFn | None] = {
    "neural": _identity_ablation,
    "baseline_only": None,
    "mask_batter": _mask_batter,
    "mask_bowler": _mask_bowler,
    "mask_players": _mask_players,
    "mask_venue": _mask_venue,
}


@dataclass(slots=True)
class SubgroupStore:
    buckets: dict[str, HeadMetrics] = field(default_factory=lambda: defaultdict(HeadMetrics))

    def update_row(
        self,
        key: str,
        *,
        runs_probability: np.ndarray,
        extras_probability: np.ndarray,
        legality_probability: np.ndarray,
        dismissal_probability: float,
        wicket_probability: float,
        runs_target: int,
        extras_target: int,
        legality_target: int,
        dismissal_target: bool,
        wicket_target: bool,
        loss: float,
    ) -> None:
        metrics = self.buckets[key]
        metrics.n += 1
        metrics.loss_sum += loss
        metrics.runs.update(runs_probability, runs_target)
        metrics.extras.update(extras_probability, extras_target)
        metrics.legality.update(legality_probability, legality_target)
        metrics.batter_dismissal.update(dismissal_probability, dismissal_target)
        metrics.bowler_wicket.update(wicket_probability, wicket_target)

    def as_dict(self) -> dict[str, Any]:
        return {key: metrics.as_dict() for key, metrics in sorted(self.buckets.items())}


def _row_multitask_loss(
    runs_probability: np.ndarray,
    extras_probability: np.ndarray,
    legality_probability: np.ndarray,
    dismissal_probability: float,
    wicket_probability: float,
    runs_target: int,
    extras_target: int,
    legality_target: int,
    dismissal_target: float,
    wicket_target: float,
) -> float:
    dismissal_probability = float(np.clip(dismissal_probability, 1e-12, 1.0 - 1e-12))
    wicket_probability = float(np.clip(wicket_probability, 1e-12, 1.0 - 1e-12))
    return float(
        -np.log(max(float(runs_probability[runs_target]), 1e-12))
        - np.log(max(float(extras_probability[extras_target]), 1e-12))
        - np.log(max(float(legality_probability[legality_target]), 1e-12))
        - (
            dismissal_target * np.log(dismissal_probability)
            + (1.0 - dismissal_target) * np.log(1.0 - dismissal_probability)
        )
        - (
            wicket_target * np.log(wicket_probability)
            + (1.0 - wicket_target) * np.log(1.0 - wicket_probability)
        )
    )


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    model_data_dir: Path,
    *,
    split: str = "validation",
    batch_size: int = 4096,
    device_name: str = "auto",
    ablations: list[str] | None = None,
    baseline_metrics_path: Path | None = None,
) -> dict[str, Any]:
    if split == "test":
        raise ValueError(
            "Neural test evaluation is reserved until architecture is frozen; "
            "use --split validation"
        )
    if split != "validation":
        raise ValueError(f"Unsupported split {split!r}; expected validation")

    selected = ablations or list(ABLATIONS)
    unknown = sorted(set(selected) - set(ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}")

    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PlayerRepresentationModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = EncodedDeliveryDataset(model_data_dir / f"{split}.parquet")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    training_config = TrainingConfig()

    overall: dict[str, HeadMetrics] = {name: HeadMetrics() for name in selected}
    subgroups = {
        "cold_start": SubgroupStore(),
        "phase": SubgroupStore(),
        "gender": SubgroupStore(),
        "team_type": SubgroupStore(),
        "innings_group": SubgroupStore(),
        "batter_evidence": SubgroupStore(),
        "bowler_evidence": SubgroupStore(),
        "matchup_evidence": SubgroupStore(),
        "venue_evidence": SubgroupStore(),
    }

    for batch in loader:
        categorical, numeric, baseline, categorical_targets, binary_targets = (
            tensor.to(device) for tensor in batch
        )

        if "baseline_only" in selected:
            overall["baseline_only"].update_baseline(
                baseline, categorical_targets, binary_targets
            )

        for name in selected:
            if name == "baseline_only":
                continue
            transform = ABLATIONS[name]
            assert transform is not None
            cat_in, num_in, base_in = transform(categorical, numeric, baseline)
            outputs = model(cat_in, num_in, base_in)
            loss = multitask_loss(
                outputs, categorical_targets, binary_targets, training_config
            )
            overall[name].update(
                outputs,
                categorical_targets,
                binary_targets,
                loss,
            )

            if name != "neural":
                continue

            runs_probability = outputs["runs_logits"].softmax(dim=-1).cpu().numpy()
            extras_probability = outputs["extras_logits"].softmax(dim=-1).cpu().numpy()
            legality_probability = outputs["legality_logits"].softmax(dim=-1).cpu().numpy()
            dismissal_probability = (
                outputs["batter_dismissal_logit"].sigmoid().cpu().numpy()
            )
            wicket_probability = outputs["bowler_wicket_logit"].sigmoid().cpu().numpy()

            categorical_np = categorical.cpu().numpy()
            numeric_np = numeric.cpu().numpy()
            categorical_targets_np = categorical_targets.cpu().numpy()
            binary_targets_np = binary_targets.cpu().numpy()

            for index in range(categorical_np.shape[0]):
                unknown_flags = []
                if categorical_np[index, 0] == 0:
                    unknown_flags.append("unknown_batter")
                if categorical_np[index, 1] == 0:
                    unknown_flags.append("unknown_bowler")
                if categorical_np[index, 2] == 0:
                    unknown_flags.append("unknown_venue")
                cold_key = "+".join(unknown_flags) if unknown_flags else "all_known"

                row_loss = _row_multitask_loss(
                    runs_probability[index],
                    extras_probability[index],
                    legality_probability[index],
                    float(dismissal_probability[index]),
                    float(wicket_probability[index]),
                    int(categorical_targets_np[index, 0]),
                    int(categorical_targets_np[index, 1]),
                    int(categorical_targets_np[index, 2]),
                    float(binary_targets_np[index, 0]),
                    float(binary_targets_np[index, 1]),
                )
                payload = {
                    "runs_probability": runs_probability[index],
                    "extras_probability": extras_probability[index],
                    "legality_probability": legality_probability[index],
                    "dismissal_probability": float(dismissal_probability[index]),
                    "wicket_probability": float(wicket_probability[index]),
                    "runs_target": int(categorical_targets_np[index, 0]),
                    "extras_target": int(categorical_targets_np[index, 1]),
                    "legality_target": int(categorical_targets_np[index, 2]),
                    "dismissal_target": bool(binary_targets_np[index, 0]),
                    "wicket_target": bool(binary_targets_np[index, 1]),
                    "loss": row_loss,
                }
                subgroups["cold_start"].update_row(cold_key, **payload)
                subgroups["phase"].update_row(
                    PHASE_LABELS.get(int(categorical_np[index, 3]), "unknown"), **payload
                )
                subgroups["gender"].update_row(
                    GENDER_LABELS.get(int(categorical_np[index, 4]), "unknown"), **payload
                )
                subgroups["team_type"].update_row(
                    TEAM_TYPE_LABELS.get(int(categorical_np[index, 5]), "unknown"),
                    **payload,
                )
                subgroups["innings_group"].update_row(
                    INNINGS_LABELS.get(int(categorical_np[index, 6]), "unknown"),
                    **payload,
                )
                # Evidence features occupy the last four numeric columns.
                subgroups["batter_evidence"].update_row(
                    _evidence_bucket(float(numeric_np[index, -4])), **payload
                )
                subgroups["bowler_evidence"].update_row(
                    _evidence_bucket(float(numeric_np[index, -3])), **payload
                )
                subgroups["venue_evidence"].update_row(
                    _evidence_bucket(float(numeric_np[index, -2])), **payload
                )
                subgroups["matchup_evidence"].update_row(
                    _evidence_bucket(float(numeric_np[index, -1])), **payload
                )

    neural = overall["neural"].as_dict() if "neural" in overall else {}
    baseline = overall["baseline_only"].as_dict() if "baseline_only" in overall else {}
    comparison = {}
    if neural and baseline:
        comparison = {
            "runs_log_loss_delta": (
                neural["batter_runs"]["log_loss"] - baseline["batter_runs"]["log_loss"]
            ),
            "runs_brier_delta": (
                neural["batter_runs"]["brier_score"]
                - baseline["batter_runs"]["brier_score"]
            ),
            "runs_expected_mae_delta": (
                neural["batter_runs"]["expected_value_mae"]
                - baseline["batter_runs"]["expected_value_mae"]
            ),
            "runs_ece_delta": (
                neural["batter_runs"]["top_label_ece"]
                - baseline["batter_runs"]["top_label_ece"]
            ),
            "extras_log_loss_delta": (
                neural["extras_runs"]["log_loss"] - baseline["extras_runs"]["log_loss"]
            ),
            "legality_log_loss_delta": (
                neural["legality"]["log_loss"] - baseline["legality"]["log_loss"]
            ),
            "dismissal_log_loss_delta": (
                neural["batter_dismissal"]["log_loss"]
                - baseline["batter_dismissal"]["log_loss"]
            ),
            "wicket_log_loss_delta": (
                neural["bowler_wicket"]["log_loss"] - baseline["bowler_wicket"]["log_loss"]
            ),
            "total_loss_delta": neural["total_loss"] - baseline["total_loss"],
        }

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_epoch": checkpoint.get("best_epoch"),
        "model_config": checkpoint.get("model_config"),
        "baseline_metrics_path": (
            str(baseline_metrics_path) if baseline_metrics_path else None
        ),
        "policy": {
            "test_evaluation": "blocked",
            "ablations": "inference-time input masking on frozen residual checkpoint",
            "subgroups": "delivery-weighted validation only",
        },
        "overall": {name: metrics.as_dict() for name, metrics in overall.items()},
        "comparison_vs_baseline_only": comparison,
        "subgroups": {
            name: store.as_dict() for name, store in subgroups.items() if "neural" in selected
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/representations-residual-mps-user/best.pt"),
    )
    parser.add_argument("--model-data", type=Path, default=Path("artifacts/model-data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/validation.json"))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=tuple(ABLATIONS),
        default=list(ABLATIONS),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("artifacts/baselines/metrics.json"),
    )
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        args.model_data,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
        ablations=args.ablations,
        baseline_metrics_path=args.baseline_metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {
        "checkpoint": result["checkpoint"],
        "best_epoch": result["best_epoch"],
        "overall_runs_log_loss": {
            name: values["batter_runs"]["log_loss"]
            for name, values in result["overall"].items()
            if values.get("n")
        },
        "comparison_vs_baseline_only": result["comparison_vs_baseline_only"],
        "cold_start_runs_log_loss": {
            key: values["batter_runs"]["log_loss"]
            for key, values in result["subgroups"].get("cold_start", {}).items()
            if values.get("n")
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
