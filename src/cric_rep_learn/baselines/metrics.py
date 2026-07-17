"""Streaming probabilistic metrics for baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPSILON = 1e-12


@dataclass(slots=True)
class MulticlassMetrics:
    classes: int
    n: int = 0
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    expected_value_absolute_error_sum: float = 0.0
    calibration_count: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.int64))
    calibration_confidence_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.float64)
    )
    calibration_accuracy_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.float64)
    )

    def update(self, probability: np.ndarray, target: int) -> None:
        probability = np.asarray(probability, dtype=np.float64)
        if probability.shape != (self.classes,):
            raise ValueError(
                f"Expected probability shape {(self.classes,)}, got {probability.shape}"
            )
        if not 0 <= target < self.classes:
            raise ValueError(f"Target {target} outside 0..{self.classes - 1}")

        self.n += 1
        self.log_loss_sum -= float(np.log(np.clip(probability[target], EPSILON, 1.0)))
        one_hot = np.zeros(self.classes, dtype=np.float64)
        one_hot[target] = 1.0
        self.brier_sum += float(np.square(probability - one_hot).sum())
        expected_value = float(probability @ np.arange(self.classes))
        self.expected_value_absolute_error_sum += abs(expected_value - target)

        confidence = float(probability.max())
        predicted = int(probability.argmax())
        bin_index = min(int(confidence * 10), 9)
        self.calibration_count[bin_index] += 1
        self.calibration_confidence_sum[bin_index] += confidence
        self.calibration_accuracy_sum[bin_index] += float(predicted == target)

    def as_dict(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "log_loss": self.log_loss_sum / self.n,
            "brier_score": self.brier_sum / self.n,
            "expected_value_mae": self.expected_value_absolute_error_sum / self.n,
            "top_label_ece": self._ece(),
        }

    def _ece(self) -> float:
        error = 0.0
        for index, count in enumerate(self.calibration_count):
            if not count:
                continue
            confidence = self.calibration_confidence_sum[index] / count
            accuracy = self.calibration_accuracy_sum[index] / count
            error += (count / self.n) * abs(confidence - accuracy)
        return float(error)


@dataclass(slots=True)
class BinaryMetrics:
    n: int = 0
    positive_count: int = 0
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    calibration_count: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.int64))
    calibration_probability_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.float64)
    )
    calibration_target_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(10, dtype=np.float64)
    )

    def update(self, probability: float, target: bool | int) -> None:
        probability = float(np.clip(probability, EPSILON, 1.0 - EPSILON))
        target_value = int(bool(target))
        self.n += 1
        self.positive_count += target_value
        self.log_loss_sum -= target_value * np.log(probability) + (1 - target_value) * np.log(
            1.0 - probability
        )
        self.brier_sum += (probability - target_value) ** 2

        bin_index = min(int(probability * 10), 9)
        self.calibration_count[bin_index] += 1
        self.calibration_probability_sum[bin_index] += probability
        self.calibration_target_sum[bin_index] += target_value

    def as_dict(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "positive_rate": self.positive_count / self.n,
            "log_loss": self.log_loss_sum / self.n,
            "brier_score": self.brier_sum / self.n,
            "ece": self._ece(),
        }

    def _ece(self) -> float:
        error = 0.0
        for index, count in enumerate(self.calibration_count):
            if not count:
                continue
            probability = self.calibration_probability_sum[index] / count
            observed = self.calibration_target_sum[index] / count
            error += (count / self.n) * abs(probability - observed)
        return float(error)
